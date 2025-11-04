#include "kernels.cuh"
#include "runner.cuh"


void run_cublas(
    cublasHandle_t handle, int M, int N, int K, float alpha,
    float *A, float *B, float beta, float *C
) {
    // Recall that:
    // 1) Both A and C are row-major while B is column-major.
    // 2) cuBLAS uses column-major order (let's say C* = A* x B*).
    // So we want (C*)^T = C = (A* x B*)^T
    // --> (B*)^T x (A*)^T --> B^T x A.
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                 K, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
}


void run_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    size_t const BLOCK_DIM{32};
    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

    naive_gemm<<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_sharedmem_block_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    size_t const BLOCK_DIM{32};
    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

    shared_memory_block_gemm<BLOCK_DIM><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_2d_thread_coarsening(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    size_t const BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    size_t const BM{BLOCK_DIM}, BN{BLOCK_DIM};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    size_t const TM{8}, TN{8};

    // The total number of threads per block corresponds to the number of cells in shared memory:
    // BM * BK = BN * BK = BM * BN / (TM * TN).
    // So BK = BM * BN / (TM * TN * BM) given that BM = BN.
    size_t const BK{BN / (TM * TN)};

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BM * BN / (TM * TN));
    thread_coarsening_2d<BM, BN, BK, TM, TN>
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
        default:
            throw std::invalid_argument("Invalid kernel number.");
    }
}