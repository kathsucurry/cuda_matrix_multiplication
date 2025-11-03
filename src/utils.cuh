#pragma once

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sys/time.h>


#define CHECK_CUDA_ERROR(value) check((value), #value, __FILE__, __LINE__)
void check(cudaError_t error, char const *func, char const *file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA runtime error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(error) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const *file, int line) {
    cudaError_t const error{cudaGetLastError()};
    if (error != cudaSuccess) {
        std::cerr << "CUDA runtime error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


int get_kernel_input(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Please select a kernel (range 0 - *, 0 for NVIDIA cuBLAS)" << std::endl;
    }

    // Get kernel number.
    int kernel_num{std::stoi(argv[1])};
    if (kernel_num < 0 || kernel_num > 1) {
        std::cerr << "A valid kernel number should be between 0 and * inclusive." << std::endl;
        exit(EXIT_FAILURE);
    }

    return kernel_num;
}


void randomize_matrix(float *matrix, size_t N) {
    struct timeval time{};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);

    for (size_t i{0}; i < N; ++i) {
        float value = (float)(rand() % 5) + 0.01 * (rand() % 5);
        value = (rand() % 2 == 0) ? value : value * (-1.);
        matrix[i] = value;
    }
}
