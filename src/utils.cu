#include "utils.cuh"


void check_cuda_error(cudaError_t error, char const *func, char const *file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA runtime error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(error) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


void check_last_cuda_error(char const *file, int line) {
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
        exit(EXIT_FAILURE);
    }

    // Get kernel number.
    int kernel_num{std::stoi(argv[1])};
    if (kernel_num < 0 || kernel_num >= KERNEL_NUM) {
        std::cerr << "A valid kernel number should be between 0 and * inclusive." << std::endl;
        exit(EXIT_FAILURE);
    }

    return kernel_num;
}
