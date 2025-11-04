#pragma once

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#define EPS 1e-4
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define KERNEL_NUM 3
#define CHECK_CUDA_ERROR(value) check_cuda_error((value), #value, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)

void check_cuda_error(cudaError_t error, char const *func, char const *file, int line);
void check_last_cuda_error(char const *file, int line);

int get_kernel_input(int argc, char **argv);

void randomize_matrix(float *matrix, size_t N);
bool verify_matrix(float *matrix_1, float *matrix_2, size_t N);
