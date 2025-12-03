#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>


namespace wt_tc_memcpy_async {
    
template <uint const BM, uint const BN, uint const BK>
__device__ __forceinline__ void load_from_gmem(
    __nv_bfloat16 *__restrict__ A, __nv_bfloat16 *__restrict__ B, int N, int K,
    __nv_bfloat16 *__restrict__ As, __nv_bfloat16 *__restrict__ Bs,
    uint A_block_row_idx, uint A_block_col_idx,
    uint B_block_row_idx, uint B_block_col_idx,
    uint stride_A, uint stride_B
) {
    for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        __pipeline_memcpy_async(
            &As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx],
            &A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx],
            8 * sizeof(__nv_bfloat16)
        );
    }
    __pipeline_commit();
        
    for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        __pipeline_memcpy_async(
            &Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx],
            &B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx],
            8 * sizeof(__nv_bfloat16)
        );
    }
    __pipeline_commit();
}

template <uint const BN, uint const BK,
          uint const WMMA_M, uint const WMMA_N, uint const WMMA_K,
          uint const NUM_WMMA_M, uint const NUM_WMMA_N>
__device__ void compute_gemm(
    __nv_bfloat16 *__restrict__ As,
    __nv_bfloat16 *__restrict__ Bs,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frag,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frag,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N],
    uint warp_row_offset, uint warp_col_offset
) {
    for (uint wmma_k_offset{0u}; wmma_k_offset < BK; wmma_k_offset += WMMA_K) {
        for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
            nvcuda::wmma::load_matrix_sync(
                a_frag,
                &As[(warp_row_offset + wmma_row_idx * WMMA_M) * BK + wmma_k_offset],
                BK
            );
            for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
                nvcuda::wmma::load_matrix_sync(
                    b_frag,
                    &Bs[wmma_k_offset * BN + (warp_col_offset + wmma_col_idx * WMMA_N)],
                    BN
                );

                // Perform matrix multiplication.
                nvcuda::wmma::mma_sync(
                    acc_frags[wmma_row_idx][wmma_col_idx],
                    a_frag,
                    b_frag,
                    acc_frags[wmma_row_idx][wmma_col_idx]
                );
            }
        }
    }
}

};


template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
          uint const WM, uint const WN, uint const WMMA_M, uint const WMMA_N, uint const WMMA_K>
__global__ void __launch_bounds__(NUM_THREADS) tensor_cores_memcpy_async_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[2][BM * BK];
    __shared__ __nv_bfloat16 Bs[2][BK * BN];

    constexpr uint NUM_WMMA_M{WM / WMMA_M};
    constexpr uint NUM_WMMA_N{WN / WMMA_N};

    uint const warp_idx{threadIdx.x / 32};
    uint const warp_row_offset{(warp_idx / (BN / WN)) * WM};
    uint const warp_col_offset{(warp_idx % (BN / WN)) * WN};

    {
        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
    }

    constexpr uint stride_A{(NUM_THREADS << 3) / BK};
    constexpr uint stride_B{(NUM_THREADS << 3) / BN};

    // For storing into shared memory.
    uint const A_block_row_idx{threadIdx.x / (BK >> 3)};
    uint const A_block_col_idx{(threadIdx.x % (BK >> 3)) << 3};
    uint const B_block_row_idx{threadIdx.x / (BN >> 3)};
    uint const B_block_col_idx{(threadIdx.x % (BN >> 3)) << 3};

    // Declare fragments.
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N];
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the accumulator fragments.
    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx)
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx)
            nvcuda::wmma::fill_fragment(acc_frags[wmma_row_idx][wmma_col_idx], 0.0f);

    int current{0};

    wt_tc_memcpy_async::load_from_gmem<BM, BN, BK>(
        A, B, N, K,
        As[current], Bs[current],
        A_block_row_idx, A_block_col_idx,
        B_block_row_idx, B_block_col_idx,
        stride_A, stride_B
    );
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int k_offset{0}; k_offset < K - BK; k_offset += BK) {
        A += BK;
        B += BK * N;

        wt_tc_memcpy_async::load_from_gmem<BM, BN, BK>(
            A, B, N, K,
            As[current ^ 1], Bs[current ^ 1],
            A_block_row_idx, A_block_col_idx,
            B_block_row_idx, B_block_col_idx,
            stride_A, stride_B
        );
        __pipeline_wait_prior(BM / stride_A + BK / stride_B);
        __syncthreads();

        // Execute the dot product.
        wt_tc_memcpy_async::compute_gemm<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
            As[current], Bs[current], a_frag, b_frag, acc_frags, warp_row_offset, warp_col_offset);
        __syncthreads();

        current ^= 1;
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    wt_tc_memcpy_async::compute_gemm<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
            As[current], Bs[current], a_frag, b_frag, acc_frags, warp_row_offset, warp_col_offset);
     __syncthreads();

    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
            uint const C_row_offset{warp_row_offset + wmma_row_idx * WMMA_M};
            uint const C_col_offset{warp_col_offset + wmma_col_idx * WMMA_N};

            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[C_row_offset * N + C_col_offset],
                N,
                nvcuda::wmma::mem_row_major
            );

            for (int i{0}; i < c_frag.num_elements; ++i) {
                c_frag.x[i] = alpha * acc_frags[wmma_row_idx][wmma_col_idx].x[i] + beta * c_frag.x[i];
            }

            nvcuda::wmma::store_matrix_sync(
                &C[C_row_offset * N + C_col_offset], c_frag, N, nvcuda::wmma::mem_row_major
            );
        }
    }
}
