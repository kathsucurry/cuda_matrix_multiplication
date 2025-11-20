#pragma once


template <uint const BLOCK_DIM>
__global__ void __launch_bounds__(BLOCK_DIM * BLOCK_DIM) naive_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    uint const global_x_idx{blockIdx.x * BLOCK_DIM + threadIdx.x};
    uint const global_y_idx{blockIdx.y * BLOCK_DIM + threadIdx.y};

    if (global_x_idx < N && global_y_idx < M) {
        float sum{0.0f};
    
        for (uint k{0}; k < K; ++k) {
            sum += A[global_y_idx * K + k] * B[k * N + global_x_idx];
        }
    
        uint const global_idx{global_y_idx * N + global_x_idx};
        C[global_idx] = alpha * sum + beta * C[global_idx];
    }
}