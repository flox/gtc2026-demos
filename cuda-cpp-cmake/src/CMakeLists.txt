cmake_minimum_required(VERSION 3.18)
project(gpu_gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Broad architecture support: Turing through Hopper
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

add_executable(gpu-gemm gpu_gemm.cu)
target_compile_options(gpu-gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
