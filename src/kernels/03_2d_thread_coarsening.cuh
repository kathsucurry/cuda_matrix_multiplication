#pragma once


/**
 * Corresponds to kernel 5: 2d blocktiling in Simon's post.
 * 
 * Each thread computes the results of TM * TN cells.
 */
template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK, uint const TM, uint const TN>
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
    uint const threadIdx_x{threadIdx.x % (BN / TN)};
    uint const threadIdx_y{threadIdx.x / (BN / TN)};

    {
        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};
        
        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
    }

    // CHANGE 3.1: compute the number of iterations each thread store shared memory elements;
    // used only in the shared-memory stores.
    constexpr uint stride_A{NUM_THREADS / BK};
    constexpr uint stride_B{NUM_THREADS / BN};

    // CHANGE 3.2: obtain the row and column indices of A and B during shared-memory stores.
    uint const A_block_row_idx{threadIdx.x / BK};
    uint const A_block_col_idx{threadIdx.x % BK};
    uint const B_block_row_idx{threadIdx.x / BN};
    uint const B_block_col_idx{threadIdx.x % BN};

    for (int k_offset{0}; k_offset < K; k_offset += BK) {
        // Stage 1: shared-memory stores.
        for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
            As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx] =
                A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx];
        }
        for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
            Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx] =
                B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // Stage 2: dot-product computation.
        for (int k{0}; k < BK; ++k) {
            for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
                for (int tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                    out_values[tile_y_idx * TN + tile_x_idx] +=
                        As[(threadIdx_y * TM + tile_y_idx) * BK + k] *
                        Bs[k * BN + (threadIdx_x * TN + tile_x_idx)];
                }
            }
        }
        __syncthreads();
    }

    // Stage 3: output stores.
    for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (int tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
            uint C_idx{};
            {
                uint const cell_row_idx{threadIdx_y * TM + tile_y_idx};
                uint const cell_col_idx{threadIdx_x * TN + tile_x_idx};
                C_idx = cell_row_idx * N + cell_col_idx;
            }
            C[C_idx] =
                alpha * out_values[tile_y_idx * TN + tile_x_idx] +
                beta * C[C_idx];
        }
    }
}
