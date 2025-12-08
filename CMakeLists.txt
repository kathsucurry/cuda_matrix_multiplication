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
add_compile_options(
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--source-in-ptx>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g>
    $<$<CONFIG:Release>:--generate-line-info>
    $<$<CONFIG:Release>:--ptxas-options=-v>
)

# Configure header file search paths.
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled.
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(half_matmul run_kernel_fp16.cu ${SRC})
set_target_properties(half_matmul PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(half_matmul ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(single_matmul run_kernel_fp32.cu ${SRC})
set_target_properties(single_matmul PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(single_matmul ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}) 