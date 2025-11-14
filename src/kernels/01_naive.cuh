#pragma once


template <size_t const BLOCK_DIM>
__global__ void naive_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    size_t const global_x_idx{blockIdx.x * BLOCK_DIM + threadIdx.x};
    size_t const global_y_idx{blockIdx.y * BLOCK_DIM + threadIdx.y};

    if (global_x_idx < N && global_y_idx < M) {
        float sum{0.0f};
    
        for (size_t k{0}; k < K; ++k) {
            sum += A[global_y_idx * K + k] * B[k * N + global_x_idx];
        }
    
        size_t const global_idx{global_y_idx * N + global_x_idx};
        C[global_idx] = alpha * sum + beta * C[global_idx];
    }
}