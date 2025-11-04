#pragma once


__global__ void naive_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    size_t const global_x_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const global_y_idx{blockIdx.y * blockDim.y + threadIdx.y};
    float sum{0.0f};

    for (size_t k{0}; k < K; ++k) {
        sum += A[global_y_idx * K + k] * B[global_x_idx * K + k];
    }

    size_t const global_idx{global_y_idx * N + global_x_idx};
    C[global_idx] = alpha * sum + beta * C[global_idx];
}