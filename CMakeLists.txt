cmake_minimum_required(VERSION 3.19)
project(MATMUL LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# Ensure cuda is available.
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 120)

# In debug mode, add debug symbols to device code.
# This disables most optimizations and kills performance.
# add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")
add_compile_options("--generate-line-info")

# Configure header file search paths.
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled.
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(matmul run_kernel.cu ${SRC})
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(matmul ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})