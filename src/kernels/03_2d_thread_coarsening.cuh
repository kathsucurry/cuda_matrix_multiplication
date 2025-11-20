#pragma once


/**
 * Corresponds to kernel 5: 2d blocktiling in Simon's post.
 * 
 * Each thread computes the results of TM * TN cells.
 */
template <size_t NUM_THREADS, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void __launch_bounds__(NUM_THREADS) thread_coarsening_2d_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // CHANGE 1: each thread now computes TM x TN tile.
    float out_values[TM * TN] = {0.0f};

    // CHANGE 2: since the block now has 1D layout, obtain the x and y coordinates
    // within BM x BN (for dot product computation & output stages).
    size_t const threadIdx_x{threadIdx.x % (BN / TN)};
    size_t const threadIdx_y{threadIdx.x / (BN / TN)};

    {
        size_t const block_row_offset{blockIdx.y * BM};
        size_t const block_col_offset{blockIdx.x * BN};
        
        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
    }

    // CHANGE 3.1: compute the number of iterations each thread store shared memory elements;
    // used only in the shared-memory stores.
    constexpr size_t stride_A{NUM_THREADS / BK};
    constexpr size_t stride_B{NUM_THREADS / BN};

    // CHANGE 3.2: obtain the row and column indices of A and B during shared-memory stores.
    size_t const A_block_row_idx{threadIdx.x / BK};
    size_t const A_block_col_idx{threadIdx.x % BK};
    size_t const B_block_row_idx{threadIdx.x / BN};
    size_t const B_block_col_idx{threadIdx.x % BN};

    for (size_t k_offset{0}; k_offset < K; k_offset += BK) {
        // Stage 1: shared-memory stores.
        for (size_t A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
            As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx] =
                A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx];
        }
        for (size_t B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
            Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx] =
                B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // Stage 2: dot-product computation.
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

    // Stage 3: output stores.
    for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
            size_t C_idx{};
            {
                size_t const cell_row_idx{threadIdx_y * TM + tile_y_idx};
                size_t const cell_col_idx{threadIdx_x * TN + tile_x_idx};
                C_idx = cell_row_idx * N + cell_col_idx;
            }
            C[C_idx] =
                alpha * out_values[tile_y_idx * TN + tile_x_idx] +
                beta * C[C_idx];
        }
    }
}
