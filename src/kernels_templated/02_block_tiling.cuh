#pragma once

#include "../traits.cuh"


/**
 * Corresponds to kernel 3: shared memory cache - blocking in Simon's post.
 * 
 * For each block, allocate 2 * block size number of cells of shared memory. For
 * instance, if the block size is 32 x 32, then the total allocated shared memory is 2 x 32 x 32.
 */
template <typename T, uint const BLOCK_DIM>
__global__ void __launch_bounds__(BLOCK_DIM * BLOCK_DIM) block_tiling_gemm(
    int M, int N, int K, float alpha,
    T const *__restrict__ A,
    T const *__restrict__ B,
    float   beta,
    float   *__restrict__ C
) {
    using traits = float_traits<T>;

    __shared__ T As[BLOCK_DIM * BLOCK_DIM];
    __shared__ T Bs[BLOCK_DIM * BLOCK_DIM];

    {
        uint const block_row_offset{blockIdx.y * BLOCK_DIM};
        uint const block_col_offset{blockIdx.x * BLOCK_DIM};

        // Shift A, B, and C with the block offsets.
        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
    }

    float sum{0.0f};

    for (int k_offset{0}; k_offset < K; k_offset += BLOCK_DIM) {
        // Stage 1: shared-memory stores.
        As[threadIdx.y * BLOCK_DIM + threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
        Bs[threadIdx.y * BLOCK_DIM + threadIdx.x] = B[threadIdx.y * N + threadIdx.x];
        __syncthreads();

        A += BLOCK_DIM;
        B += BLOCK_DIM * N;

        // Stage 2: dot-product computation.
        for (int dot_idx{0}; dot_idx < BLOCK_DIM; ++dot_idx) {
            sum += traits::to_compute(As[threadIdx.y * BLOCK_DIM + dot_idx]) *
                   traits::to_compute(Bs[dot_idx * BLOCK_DIM + threadIdx.x]);
        }
        __syncthreads();
    }

    // Stage 3: epilogue; output stores.
    C[threadIdx.y * N + threadIdx.x] = alpha * sum + beta * C[threadIdx.y * N + threadIdx.x];
}