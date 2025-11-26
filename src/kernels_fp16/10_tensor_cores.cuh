#pragma once

#include <cuda_bf16.h>
#include <mma.h>


namespace wt_tc {
    
template <uint const BM, uint const BN, uint const BK>
__device__ void load_from_gmem(
    __nv_bfloat16 *__restrict__ A, __nv_bfloat16 *__restrict__ B, int N, int K,
    __nv_bfloat16 *__restrict__ As, __nv_bfloat16 *__restrict__ Bs,
    uint A_block_row_idx, uint A_block_col_idx,
    uint B_block_row_idx, uint B_block_col_idx,
    uint stride_A, uint stride_B
) {
    for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        reinterpret_cast<__nv_bfloat162 *>(&As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx])[0] =
            reinterpret_cast<__nv_bfloat162 *>(&A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx])[0];
    }
        
    for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        reinterpret_cast<__nv_bfloat162 *>(&Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx])[0] =
            reinterpret_cast<__nv_bfloat162 *>(&B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx])[0];
    }
}

};


template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
          uint const WM, uint const WN, uint const WMMA_M, uint const WMMA_N, uint const WMMA_K>
__global__ void __launch_bounds__(NUM_THREADS) tensor_cores_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[BM * BK];
    __shared__ __nv_bfloat16 Bs[BK * BN];

    constexpr uint NUM_WMMA_M{WM / WMMA_M};
    constexpr uint NUM_WMMA_N{WN / WMMA_N};

    uint const lane_idx{threadIdx.x % 32};
    uint const warp_idx{threadIdx.x / 32};
    uint const warp_row_offset{(warp_idx / (BN / WN)) * WM};
    uint const warp_col_offset{(warp_idx % (BN / WN)) * WN};
    uint const thread_row_offset{(lane_idx / (NUM_WMMA_N)) * WMMA_M};
    uint const thread_col_offset{(lane_idx % (NUM_WMMA_N)) * WMMA_N};

    {
        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
    }

    constexpr uint stride_A{(NUM_THREADS << 1) / BK};
    constexpr uint stride_B{(NUM_THREADS << 1) / BN};

    // For storing into shared memory.
    uint const A_block_row_idx{threadIdx.x / (BK >> 1)};
    uint const A_block_col_idx{(threadIdx.x % (BK >> 1)) << 1};
    uint const B_block_row_idx{threadIdx.x / (BN >> 1)};
    uint const B_block_col_idx{(threadIdx.x % (BN >> 1)) << 1};

    // Declare fragments.
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frags[NUM_WMMA_M];
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frags[NUM_WMMA_N];
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N];
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    

    // Initialize the accumulator fragments.
    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx)
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx)
            nvcuda::wmma::fill_fragment(acc_frags[wmma_row_idx][wmma_col_idx], 0.0f);

    for (int k_offset{0}; k_offset < K; k_offset += BK) {
        wt_tc::load_from_gmem<BM, BN, BK>(
            A, B, N, K,
            As, Bs,
            A_block_row_idx, A_block_col_idx,
            B_block_row_idx, B_block_col_idx,
            stride_A, stride_B
        );
        __syncthreads();

        A += BK;
        B += BK * N;

        // Execute the dot product.
        for (uint k_offset{0u}; k_offset < BK; k_offset += WMMA_K) {
            for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
                nvcuda::wmma::load_matrix_sync(
                    a_frags[wmma_row_idx],
                    &As[(warp_row_offset + thread_row_offset + wmma_row_idx * WMMA_M) * BK + k_offset],
                    BK
                );

                for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
                    nvcuda::wmma::load_matrix_sync(
                        b_frags[wmma_col_idx],
                        &Bs[k_offset * BN + (warp_col_offset + thread_col_offset + wmma_col_idx * WMMA_N)],
                        BN
                    );
    
                    // Perform matrix multiplication.
                    nvcuda::wmma::mma_sync(
                        acc_frags[wmma_row_idx][wmma_col_idx],
                        a_frags[wmma_row_idx],
                        b_frags[wmma_col_idx],
                        acc_frags[wmma_row_idx][wmma_col_idx]
                    );
                }
            }
        }
        __syncthreads();
    }

    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
            uint const C_row_offset{warp_row_offset + thread_row_offset + wmma_row_idx * WMMA_M};
            uint const C_col_offset{warp_col_offset + thread_col_offset + wmma_col_idx * WMMA_N};

            for (int i{0}; i < c_frag.num_elements; ++i) {
                uint const row{i / WMMA_N};
                uint const col{i % WMMA_N};
                uint const C_idx{(C_row_offset + row) * N + (C_col_offset + col)};

                c_frag.x[i] = __bfloat162float(C[C_idx]);
                c_frag.x[i] = alpha * acc_frags[wmma_row_idx][wmma_col_idx].x[i] + beta * c_frag.x[i];
                C[C_idx] = __float2bfloat16(c_frag.x[i]);
            }
        }
    }
}
