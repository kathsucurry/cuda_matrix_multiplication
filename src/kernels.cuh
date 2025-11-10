#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

#include "kernels/01_naive.cuh"
#include "kernels/02_shared_memory_block.cuh"
#include "kernels/03_2d_thread_coarsening.cuh"
#include "kernels/04_vectorize.cuh"
#include "kernels/05_warptiling.cuh"
#include "kernels/06_warptiling_subdivided.cuh"
