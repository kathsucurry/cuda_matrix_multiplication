#pragma once

#include <cuda_runtime.h>

#include "utils.cuh"

#include "kernels/00_naive.cuh"
#include "kernels/01_shared_memory_block.cuh"
#include "kernels/02_2d_thread_coarsening.cuh"



