#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

#include "kernels_templated/01_naive.cuh"

#include "kernels_fp32/02_block_tiling.cuh"
#include "kernels_fp32/03_2d_thread_coarsening.cuh"
#include "kernels_fp32/04_vectorize.cuh"
#include "kernels_fp32/05_warptiling.cuh"
#include "kernels_fp32/06_warptiling_subdivided.cuh"
#include "kernels_fp32/07_transpose_shared_a.cuh"
#include "kernels_fp32/08_resolved_bank_conflict.cuh"
#include "kernels_fp32/09_memcpy_async.cuh"

#include "kernels_fp16/04_vectorize.cuh"
#include "kernels_fp16/05_warptiling.cuh"
#include "kernels_fp16/09_memcpy_async.cuh"
#include "kernels_fp16/10_tensor_cores.cuh"
#include "kernels_fp16/11_tensor_cores_memcpy_async.cuh"
