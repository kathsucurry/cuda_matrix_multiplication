#include <assert.h>

#include "kernels.cuh"
#include "runner.cuh"


void run_cublas(
    cublasHandle_t handle, int M, int N, int K, float alpha,
    float *A, float *B, float beta, float *C
) {
    // Recall that:
    // 1) All matrices are stored in row-major order.
    // 2) cuBLAS uses column-major order (let's say C* = A* x B*).
    // So we want (C*)^T = C = (A* x B*)^T
    // --> (B*)^T x (A*)^T --> B x A.
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                 K, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
}


void run_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    constexpr size_t BLOCK_DIM{32};
    assert((N % BLOCK_DIM == 0) && (M % BLOCK_DIM == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

    naive_gemm<BLOCK_DIM><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_sharedmem_block_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    constexpr size_t BLOCK_DIM{32};
    assert((N % BLOCK_DIM == 0) && (M % BLOCK_DIM == 0) && (K % BLOCK_DIM == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

    block_tiling_gemm<BLOCK_DIM><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_2d_thread_coarsening(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    size_t const BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    size_t const BM{BLOCK_DIM}, BN{BLOCK_DIM};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    size_t const TM{8}, TN{8};

    // Recall that BK determines the length of the tile and the shared memory size.
    size_t const BK{16};
    // For the sake of simplicity, we make sure that BK * BM is divisible by BLOCK_DIM.
    static_assert(((BK * BM) % BLOCK_DIM == 0) && ((BK * BN) % BLOCK_DIM == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BM * BN / (TM * TN));
    thread_coarsening_2d<BM, BN, BK, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_vectorize(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    size_t const BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    size_t const BM{BLOCK_DIM}, BN{BLOCK_DIM};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    size_t const TM{8}, TN{8};

    size_t const BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * BLOCK_DIM) == 0) && ((BK * BN) % (4 * BLOCK_DIM) == 0));

    static_assert(BK % 4 == 0);
    static_assert((TM % 4 == 0) && (TN % 4 == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BM * BN / (TM * TN));
    vectorize<BM, BN, BK, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_warptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    size_t const BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    size_t const BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    size_t const TM{8}, TN{16};
    size_t const BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * BLOCK_DIM) == 0) && ((BK * BN) % (4 * BLOCK_DIM) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    size_t const WM{64}, WN{64};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == BLOCK_DIM / 32);

    static_assert(WM * WN / 32 == TM * TN);

    dim3 block_dim(BLOCK_DIM);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling<BM, BN, BK, WM, WN, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_warptiling_subdivided(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    size_t const BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    size_t const BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    size_t const TM{4}, TN{4};
    size_t const BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * BLOCK_DIM) == 0) && ((BK * BN) % (4 * BLOCK_DIM) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    size_t const WM{64}, WN{64};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == BLOCK_DIM / 32);

    size_t const WNITER{1};
    static_assert((WM * WN) % (32 * TM * TN * WNITER) == 0);
    size_t const WMITER{WM * WN / (32 * TM * TN * WNITER)};
    static_assert((WM % WMITER == 0) & (WN % WNITER == 0));

    dim3 block_dim(BLOCK_DIM);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling_subdivided<BM, BN, BK, WM, WN, WNITER, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_kernel(
    int kernel_num, int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C, cublasHandle_t handle
) {
    switch (kernel_num) {
        case 0:
            run_cublas(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            run_naive(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            run_sharedmem_block_tiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            run_2d_thread_coarsening(M, N, K, alpha, A, B, beta, C);
            break;
        case 4:
            assert((N % 4 == 0) && (K % 4 == 0));
            run_vectorize(M, N, K, alpha, A, B, beta, C);
            break;
        case 5:
            run_warptiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 6:
            run_warptiling_subdivided(M, N, K, alpha, A, B, beta, C);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number.");
    }
}