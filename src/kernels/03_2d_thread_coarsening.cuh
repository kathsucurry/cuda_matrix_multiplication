#pragma once


/**
 * Corresponds to kernel 5: 2d blocktiling in Simon's post.
 * 
 * Each thread computes the results of TC_M * TC_N cells.
 */
template <size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void thread_coarsening_2d(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float out_values[TM * TN] = {0.0f};

    size_t const threadIdx_x{threadIdx.x % (BN / TN)};
    size_t const threadIdx_y{threadIdx.x / (BN / TN)};

    size_t const block_row_offset{blockIdx.y * BM};
    size_t const block_col_offset{blockIdx.x * BN};

    size_t const LOAD_ITER_M{(BK * BM) / blockDim.x};
    size_t const LOAD_ITER_N{(BK * BN) / blockDim.x};

    // For storing into shared memory.
    size_t const A_block_row_idx{threadIdx.x / BK};
    size_t const B_trans_block_row_idx{threadIdx.x / BK};

    for (size_t k_offset{0}; k_offset < K; k_offset += BK) {
        size_t const A_block_col_idx{threadIdx.x % BK};
        size_t const B_trans_block_col_idx{threadIdx.x % BK};

        for (size_t iter_m{0}; iter_m < LOAD_ITER_M; ++iter_m) {
            As[threadIdx.x + (iter_m * blockDim.x)] = A[
                (block_row_offset + (A_block_row_idx + (BM / LOAD_ITER_M * iter_m))) * K +
                A_block_col_idx + k_offset];
        }
        for (size_t iter_n{0}; iter_n < LOAD_ITER_N; ++iter_n) {
            Bs[B_trans_block_col_idx * BN + B_trans_block_row_idx + (BN / LOAD_ITER_N * iter_n)] = B[
                (block_col_offset + (B_trans_block_row_idx + (BN / LOAD_ITER_N) * iter_n)) * K +
                B_trans_block_col_idx + k_offset];
        }
        __syncthreads();

        // Execute the dot product.
        for (size_t k{0}; k < BK; ++k) {
            for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
                for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                    out_values[tile_y_idx * TN + tile_x_idx] +=
                        As[(threadIdx_y * TM + tile_y_idx) * BK + k] *
                        Bs[k * BN + (threadIdx_x * TN + tile_x_idx)];
                }
            }
        }
        __syncthreads();
    }

    for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
            size_t const cell_row_idx{block_row_offset + threadIdx_y * TM + tile_y_idx};
            size_t const cell_col_idx{block_col_offset + threadIdx_x * TN + tile_x_idx};
            C[cell_row_idx * N + cell_col_idx] =
                alpha * out_values[tile_y_idx * TN + tile_x_idx] +
                beta * C[cell_row_idx * N + cell_col_idx];
        }
    }
}
