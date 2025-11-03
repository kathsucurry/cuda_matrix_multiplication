#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>


void run_cublas(
    cublasHandle_t handle, int m, int n, int k, float alpha,
    float *A, float *B, float beta, float *C
);


void run_kernel(
    int kernel_num, int m, int n, int k, float alpha, float *A, float *B,
    float beta, float *C, cublasHandle_t handle
) {
    switch (kernel_num) {
        case 0:
            run_cublas(handle, m, n, k, alpha, A, B, beta, C);
            break;
        default:
            throw std::invalid_argument("Invalid kernel number.");
    }
}


void run_cublas(
    cublasHandle_t handle, int m, int n, int k, float alpha,
    float *A, float *B, float beta, float *C
) {
    // Recall that:
    // 1) Both A and C are row-major while B is column-major.
    // 2) cuBLAS uses column-major order (let's say C* = A* x B*).
    // So we want (C*)^T = C = (A* x B*)^T
    // --> (B*)^T x (A*)^T --> B^T x A.
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                 k, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
}