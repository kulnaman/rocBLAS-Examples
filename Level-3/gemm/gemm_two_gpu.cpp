#include "helpers.hpp"
#include <hip/hip_runtime.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void run_sgemm_on_gpu(int                       gpu_id,
                      const std::vector<float>& hA,
                      const std::vector<float>& hB,
                      std::vector<float>&       hC,
                      rocblas_int               M,
                      rocblas_int               N,
                      rocblas_int               K,
                      rocblas_int               lda,
                      rocblas_int               ldb,
                      rocblas_int               ldc,
                      float                     hAlpha,
                      float                     hBeta)
{
    // Set the current device to the GPU identified by gpu_id
    hipSetDevice(gpu_id);
    std::cout << "Running on GPU " << gpu_id << std::endl;

    // using rocblas API
    rocblas_handle handle;
    rocblas_status rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    // Allocate memory on device
    helpers::DeviceVector<float> dA(hA.size());
    helpers::DeviceVector<float> dB(hB.size());
    helpers::DeviceVector<float> dC(hC.size());

    if(!dA || !dB || !dC)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(float) * hA.size(), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(float) * hB.size(), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(float) * hC.size(), hipMemcpyHostToDevice));

    // Enable passing alpha and beta parameters from host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    // Timing start
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    // Perform the sgemm operation
    for(int i=0;i< 40;i++){
    rstatus = rocblas_sgemm(handle,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            M,
                            N,
                            K,
                            &hAlpha,
                            dA,
                            lda,
                            dB,
                            ldb,
                            &hBeta,
                            dC,
                            ldc);
    CHECK_ROCBLAS_STATUS(rstatus);}
    hipDeviceSynchronize();
    // Timing end
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU " << gpu_id << " execution time: " << milliseconds << " ms\n";

    // Fetch device memory results
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(float) * hC.size(), hipMemcpyDeviceToHost));

    // Release device memory and destroy handle
    rocblas_destroy_handle(handle);
    hipEventDestroy(start);
    hipEventDestroy(stop);
}

int main(int argc, char** argv)
{
    helpers::ArgParser options("MNKab");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    typedef float dataType;

    rocblas_int M = options.M;
    rocblas_int N = options.N;
    rocblas_int K = options.K;

    float hAlpha = options.alpha;
    float hBeta  = options.beta;

    const rocblas_int lda = M;
    const rocblas_int ldb = K;
    const rocblas_int ldc = M;

    const size_t          sizeA = K * lda;
    const size_t          sizeB = N * ldb;
    const size_t          sizeC = N * ldc;
    std::vector<dataType> hA(sizeA, 1);
    std::vector<dataType> hB(sizeB, 1);
    std::vector<dataType> hC(sizeC, 1);
    std::vector<dataType> hGold(sizeC);
    // Initialize matrix B and matrix C
    // helpers::matIdentity(hB.data(), K, N, ldb);
    // helpers::matIdentity(hB.data(), K, N, ldb);
    hGold = hC;
    std::cout<< "Problem Size A "<< K << "x" << lda << " Problem Size B " << N << "x" << ldb <<std::endl; 
    int num_gpus;
    hipGetDeviceCount(&num_gpus);
    // Vector to hold threads
    std::vector<std::thread> threads;

    for(int gpu_id = 0; gpu_id < num_gpus; ++gpu_id)
    {
        // Launch a thread for each GPU
        threads.emplace_back(run_sgemm_on_gpu,
                             gpu_id,
                             std::ref(hA),
                             std::ref(hB),
                             std::ref(hC),
                             M,
                             N,
                             K,
                             lda,
                             ldb,
                             ldc,
                             hAlpha,
                             hBeta);
    }

    // Join the threads (wait for all to complete)
    for(auto& t : threads)
    {
        t.join();
    }
    // Get the number of available GPUs

    // Compare results from the last GPU with CPU results (optional)
    // helpers::matMatMult<dataType>(
    //     hAlpha, hBeta, M, N, K, hA.data(), 1, lda, hB.data(), 1, ldb, hGold.data(), 1, ldc);
    // dataType maxRelativeError = helpers::maxRelativeError(hC, hGold);
    // dataType eps              = std::numeric_limits<dataType>::epsilon();
    // dataType tolerance        = 10;
    // if(maxRelativeError > eps * tolerance)
    // {
    //     std::cout << "Final GPU results verification FAILED: Max relative error = "
    //               << maxRelativeError << std::endl;
    // }
    // else
    // {
    //     std::cout << "Final GPU results verification PASSED" << std::endl;
    // }
    //
    return EXIT_SUCCESS;
}
