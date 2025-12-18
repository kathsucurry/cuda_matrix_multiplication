#pragma once

#include <cassert>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "host_utils.cuh"
#include "kernels.cuh"


void run_kernel(
    int kernel_num, int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C, cublasHandle_t handle
);


void run_kernel(
    int kernel_num, int M, int N, int K, float alpha, __nv_bfloat16 *A, __nv_bfloat16 *B,
    float beta, float *C, cublasHandle_t handle
);


template <typename T>
void run_naive(int M, int N, int K, float alpha, T *A, T *B, float beta, float *C) {
    constexpr uint BLOCK_DIM{32};
    assert((N % BLOCK_DIM == 0) && (M % BLOCK_DIM == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

    naive_gemm<T, BLOCK_DIM><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}

template <typename T>
void run_sharedmem_block_tiling(int M, int N, int K, float alpha, T *A, T *B, float beta, float *C) {
    constexpr uint BLOCK_DIM{32};
    assert((N % BLOCK_DIM == 0) && (M % BLOCK_DIM == 0) && (K % BLOCK_DIM == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

    block_tiling_gemm<T, BLOCK_DIM><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


template <typename T>
void run_vectorize(int M, int N, int K, float alpha, T *A, T *B, float beta, float *C) {
    constexpr uint BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{BLOCK_DIM}, BN{BLOCK_DIM};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{16}, TN{8};

    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * BLOCK_DIM) == 0) && ((BK * BN) % (4 * BLOCK_DIM) == 0));

    static_assert(BK % 4 == 0);
    static_assert((TM % 4 == 0) && (TN % 4 == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BM * BN / (TM * TN));
    vectorize_gemm<T, BM * BN / (TM * TN), BM, BN, BK, TM, TN>
    <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


template <typename T>
void run_warptiling(int M, int N, int K, float alpha, T *A, T *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{16}, TN{8};
    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    constexpr uint FACTOR{4 * sizeof(float) / sizeof(T)};
    static_assert(((BK * BM) % (FACTOR * NUM_THREADS) == 0) && ((BK * BN) % (FACTOR * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    static_assert(WM * WN / 32 == TM * TN);

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling_gemm<T, NUM_THREADS, BM, BN, BK, WM, WN, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


template <typename T>
void run_warptiling_subdivided(int M, int N, int K, float alpha, T *A, T *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{8}, TN{4};
    static_assert(TN % 4 == 0);
    
    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * NUM_THREADS) == 0) && ((BK * BN) % (4 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    constexpr uint WNITER{2};
    static_assert((WM * WN) % (32 * TM * TN * WNITER) == 0);
    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    static_assert((WM % WMITER == 0) & (WN % WNITER == 0));

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling_subdivided_gemm<T, NUM_THREADS, BM, BN, BK, WM, WN, WNITER, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


template <typename T>
void measure_performance(
    std::vector<uint> const MATRIX_SIZE,
    cublasHandle_t cublas_handle,
    int const kernel_num,
    T     *&A,
    T     *&B,
    float *&C,
    float *&C_ref,
    T     *&A_d,
    T     *&B_d,
    float *&C_d,
    float *&C_ref_d
) {
    // Define matmul parameters: C = α * AB + β * C.
    constexpr float alpha{0.5f};
    constexpr float beta{3.0f};

    constexpr int repeat_times{50};
    float elapsed_time_ms;
    
    cudaEvent_t start, end;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&end));
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    for (size_t size : MATRIX_SIZE) {
        size_t const M{size};
        size_t const N{size};
        size_t const K{size}; 

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

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(end));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

