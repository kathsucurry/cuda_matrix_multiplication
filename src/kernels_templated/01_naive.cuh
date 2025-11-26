#pragma once

#include "../traits.cuh"


template <typename T, typename U, uint const BLOCK_DIM>
__global__ void __launch_bounds__(BLOCK_DIM * BLOCK_DIM) naive_gemm(
    int M, int N, int K, 
    float alpha,
    T *__restrict__ A,
    T *__restrict__ B,
    float beta,
    U *__restrict__ C
) {
    using traits = float_traits<T>;
    using compute_t = typename traits::compute_t;

    using traits_c = float_traits<U>;

    uint const global_x_idx{blockIdx.x * BLOCK_DIM + threadIdx.x};
    uint const global_y_idx{blockIdx.y * BLOCK_DIM + threadIdx.y};

    if (global_x_idx < N && global_y_idx < M) {
        compute_t sum{0.0f};
    
        for (uint k{0}; k < K; ++k) {
            sum += traits::to_compute(A[global_y_idx * K + k]) *
                   traits::to_compute(B[k * N + global_x_idx]);
        }
    
        uint const global_idx{global_y_idx * N + global_x_idx};
        C[global_idx] = traits_c::from_compute(alpha * sum + beta * traits_c::to_compute(C[global_idx]));
    }
}
