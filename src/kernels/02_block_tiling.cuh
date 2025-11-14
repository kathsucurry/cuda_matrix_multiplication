#pragma once


/**
 * Corresponds to kernel 3: shared memory cache - blocking in Simon's post.
 * 
 * For each block, allocate 2 * block size number of cells of shared memory. For
 * instance, if the block size is 32 x 32, then the total allocated shared memory is 2 x 32 x 32.
 */
template <size_t const BLOCK_DIM>
__global__ void block_tiling_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BLOCK_DIM * BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM * BLOCK_DIM];

    size_t const block_row_offset{blockIdx.y * BLOCK_DIM};
    size_t const block_col_offset{blockIdx.x * BLOCK_DIM};

    float sum{0.0f};

    for (size_t k_offset{0}; k_offset < K; k_offset += BLOCK_DIM) {
        As[threadIdx.y * BLOCK_DIM + threadIdx.x] = A[(block_row_offset + threadIdx.y) * K + (threadIdx.x + k_offset)];
        Bs[threadIdx.y * BLOCK_DIM + threadIdx.x] = B[(k_offset + threadIdx.y) * N + (block_col_offset + threadIdx.x)];
        __syncthreads();

        // Execute the dot product; recall that BLOCK_DIM and BLOCK_DIM are the same here.
        for (size_t dot_idx{0}; dot_idx < BLOCK_DIM; ++dot_idx) {
            sum += As[threadIdx.y * BLOCK_DIM + dot_idx] * Bs[dot_idx * BLOCK_DIM + threadIdx.x];
        }
        __syncthreads();
    }

    size_t const global_idx{(block_row_offset + threadIdx.y) * N + (block_col_offset + threadIdx.x)};
    C[global_idx] = alpha * sum + beta * C[global_idx];
}