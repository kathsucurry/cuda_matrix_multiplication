#pragma once


/**
 * Corresponds to kernel 3: shared memory cache - blocking in Simon's post.
 * 
 * For each block, allocate 2 * block size number of cells of shared memory. For
 * instance, if the block size is 32 x 32, then the total allocated shared memory is 2 x 32 x 32.
 */
template <size_t BLOCK_DIM>
__global__ void shared_memory_block_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // Recall that blockDim.x == blockDim.y.    
    __shared__ float As[BLOCK_DIM * BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM * BLOCK_DIM];

    size_t const block_row_offset{blockIdx.y * BLOCK_DIM};
    size_t const block_col_offset{blockIdx.x * BLOCK_DIM};

    // Shift A and B according to the block indices.
    A += block_row_offset * K;
    B += block_col_offset * K;

    float temp_value{0.0f};

    for (size_t k_offset{0}; k_offset < K; k_offset += BLOCK_DIM) {
        As[threadIdx.y * BLOCK_DIM + threadIdx.x] = A[threadIdx.y * K + threadIdx.x + k_offset];
        Bs[threadIdx.y * BLOCK_DIM + threadIdx.x] = B[threadIdx.x * K + threadIdx.y + k_offset];
        __syncthreads();

        // Execute the dot product.
        for (size_t dot_idx{0}; dot_idx < BLOCK_DIM; ++dot_idx) {
            temp_value += As[threadIdx.y * BLOCK_DIM + dot_idx] * Bs[dot_idx * BLOCK_DIM + threadIdx.x];
        }
        __syncthreads();
    }

    size_t const global_idx{(block_row_offset + threadIdx.y) * N + block_col_offset + threadIdx.x};
    C[global_idx] = alpha * temp_value + beta * C[global_idx];
}