#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include "src/runner.cuh"
#include "src/utils.cuh"


void prepare_matrices(
    float *&A, float *&B, float *&C, float *&C_ref,
    float *&A_d, float *&B_d, float *&C_d, float *&C_ref_d,
    size_t matrix_size
) {
    A     = (float *)malloc(sizeof(float) * matrix_size * matrix_size);
    B     = (float *)malloc(sizeof(float) * matrix_size * matrix_size);
    C     = (float *)malloc(sizeof(float) * matrix_size * matrix_size);
    C_ref = (float *)malloc(sizeof(float) * matrix_size * matrix_size); 

    randomize_matrix(A, matrix_size * matrix_size);
    randomize_matrix(B, matrix_size * matrix_size);
    randomize_matrix(C, matrix_size * matrix_size);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&A_d, sizeof(float) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&B_d, sizeof(float) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&C_d, sizeof(float) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&C_ref_d, sizeof(float) * matrix_size * matrix_size));

    CHECK_CUDA_ERROR(cudaMemcpy(A_d, A, sizeof(float) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B, sizeof(float) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_d, C, sizeof(float) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_ref_d, C, sizeof(float) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
}


void free_matrices(
    float *&A, float *&B, float *&C, float *&C_ref,
    float *&A_d, float *&B_d, float *&C_d, float *&C_ref_d
) {
    free(A);
    free(B);
    free(C);
    free(C_ref);

    CHECK_CUDA_ERROR(cudaFree(A_d));
    CHECK_CUDA_ERROR(cudaFree(B_d));
    CHECK_CUDA_ERROR(cudaFree(C_d));
    CHECK_CUDA_ERROR(cudaFree(C_ref_d));
}


int main(int argc, char **argv) {
    int kernel_num = get_kernel_input(argc, argv);

    // Set CUDA context.
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    
    // Create a handle to be used for using cublas.
    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle)) {
        std::cerr << "Error encountered when creating a cublas handler." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Generate matrices.
    // std::vector<size_t> const MATRIX_SIZE = {128, 256, 512, 1024, 2048, 4096, 8192};
    std::vector<size_t> const MATRIX_SIZE = {4096};

    size_t const max_matrix_size = MATRIX_SIZE[MATRIX_SIZE.size() - 1];
    std::cout << "Max matrix size: " << max_matrix_size << std::endl;

    // Define matmul parameters: C = α * AB + β * C.
    float const alpha{0.5};
    float const beta{3.0};

    // Prepare host and device matrices variables.
    float *A{nullptr}, *B{nullptr}, *C{nullptr}, *C_ref{nullptr};
    float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr}, *C_ref_d{nullptr};
    prepare_matrices(A, B, C, C_ref, A_d, B_d, C_d, C_ref_d, max_matrix_size);

    int const repeat_times{50};
    float elapsed_time_ms;
    cudaEvent_t start, end;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&end));
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    for (size_t size : MATRIX_SIZE) {
        size_t M{size}, N{size}, K{size};

        std::cout << "dimensions(m=n=k) " << M
                  << ", alpha: " << alpha
                  << ", beta: "  << beta << std::endl;
        
        // Verify the correctness by comparing against cuBLAS if kernel number != 0.
        if (kernel_num != 0) {
            run_kernel(0, M, N, K, alpha, A_d, B_d, beta, C_ref_d, cublas_handle);
            run_kernel(kernel_num, M, N, K, alpha, A_d, B_d, beta, C_d, cublas_handle);
            CHECK_LAST_CUDA_ERROR();

            CHECK_CUDA_ERROR(cudaMemcpy(C_ref, C_ref_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(C, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

            if (!verify_matrix(C_ref, C, M * N)) {
                std::cerr << "The kernel function implementation is not correct compared to cuBLAS results." << std::endl;
                exit(EXIT_FAILURE);
            }
        } else {
            // Run kernel 0 as a warmup.
            run_kernel(0, M, N, K, alpha, A_d, B_d, beta, C_ref_d, cublas_handle);
        }

        // Measure elapsed time.
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
        
        // Run the kernel repeatedly; since we already verify the results, we don't reset C here to save time.
        for (int i{0}; i < repeat_times; ++i)
            run_kernel(kernel_num, M, N, K, alpha, A_d, B_d, beta, C_d, cublas_handle);

        CHECK_CUDA_ERROR(cudaEventRecord(end, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_ms, start, end));

        CHECK_LAST_CUDA_ERROR();

        float mean_elapsed_time_ms{elapsed_time_ms / repeat_times};
        size_t flops{2 * M * N * K + 2 * M * N};
        float gflops{flops * 1.0e-6f / mean_elapsed_time_ms};

        printf(
            "average elapsed time: (%7.3f) ms, performance: (%7.1f) GFLOPS.\n",
            mean_elapsed_time_ms, gflops);
        std::cout << std::string(50, '=') << std::endl;
        fflush(stdout);

        // Make C_d and C_ref_d equal to prepare for another iteration.
        CHECK_CUDA_ERROR(cudaMemcpy(C_d, C_ref_d, sizeof(float) * M * N, cudaMemcpyDeviceToDevice));
    }

    free_matrices(A, B, C, C_ref, A_d, B_d, C_d, C_ref_d);

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(end));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    cublasDestroy(cublas_handle);

    return 0;
}
