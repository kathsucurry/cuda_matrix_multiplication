#pragma once

#include <cuda_runtime.h>

#include "host_utils.cuh"

#include "kernels_templated/01_naive.cuh"
#include "kernels_templated/02_block_tiling.cuh"
#include "kernels_templated/04_vectorize.cuh"
#include "kernels_templated/05_warptiling.cuh"
#include "kernels_templated/06_warptiling_subdivided.cuh"

#include "kernels_fp32/03_2d_thread_coarsening.cuh"
#include "kernels_fp32/07_transpose_shared_a.cuh"
#include "kernels_fp32/08_double_buffering.cuh"

#include "kernels_fp16/9_tensor_cores.cuh"
#include "kernels_fp16/10_tensor_cores_memcpy_async.cuh"
#include "kernels_fp16/11_tensor_cores_double_buffering.cuh"
#include "kernels_fp16/12_tensor_cores_three_level_pipeline.cuh"
#include "kernels_fp16/13_tensor_cores_mma.cuh"
#include "kernels_fp16/14_tensor_cores_mma_swizzled.cuh"
#include "kernels_fp16/15_tensor_cores_mma_three_level_pipeline.cuh"

