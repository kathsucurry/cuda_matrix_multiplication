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
    size_t const NUM_THREADS{32};
    dim3 grid_dim(CEIL_DIV(M, NUM_THREADS), CEIL_DIV(N, NUM_THREADS));
    dim3 block_dim(NUM_THREADS, NUM_THREADS, 1);

    naive_gemm<<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
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
        default:
            throw std::invalid_argument("Invalid kernel number.");
    }
}