# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
cmake_minimum_required(VERSION 3.18)

# ==============================================================================
# 0. Source Definition
# ==============================================================================

# It is good practice to list headers here for IDE visibility,
# even though the compiler finds them via include paths.
set(SQRT_LIB_SOURCES
    sqrt_step.cu
    sqrt_step.h
)

# ==============================================================================
# 1. Standalone Bootstrap Logic
# ==============================================================================
# This block ONLY runs if you build this directory directly (e.g., for testing).
# If built from the project root, this block is skipped.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(STATUS "[Sqrt] Building in STANDALONE mode")
    
    project(StandaloneSqrt LANGUAGES CXX CUDA)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS CXX)

    # --- Architecture Selection Logic (Copied from Sieve) ---
    if(NOT GPU_TARGET)
        set(GPU_TARGET "native")
    endif()

    if(GPU_TARGET STREQUAL "native")
        set(CMAKE_CUDA_ARCHITECTURES all-major) 
    endif()

    if(GPU_TARGET STREQUAL "Turing" OR GPU_TARGET STREQUAL "75")
        set(TARGET_CC "75")
    elseif(GPU_TARGET STREQUAL "Ampere" OR GPU_TARGET STREQUAL "80")
        set(TARGET_CC "80")
    elseif(GPU_TARGET STREQUAL "Ada" OR GPU_TARGET STREQUAL "89")
        set(TARGET_CC "89")
    elseif(GPU_TARGET STREQUAL "Hopper" OR GPU_TARGET STREQUAL "90")
        set(TARGET_CC "90")
    elseif(GPU_TARGET STREQUAL "Blackwell" OR GPU_TARGET STREQUAL "120")
        set(TARGET_CC "120")
    elseif(GPU_TARGET STREQUAL "Orin" OR GPU_TARGET STREQUAL "orin" OR GPU_TARGET STREQUAL "87")
        set(TARGET_CC "87")
    elseif(GPU_TARGET STREQUAL "all")
        set(TARGET_CC "75;80;86;89;90;120")
    else()
        set(TARGET_CC ${GPU_TARGET})
    endif()
    
    set(CMAKE_CUDA_ARCHITECTURES ${TARGET_CC})
    message(STATUS "[Standalone] Configured GPU Architecture: ${GPU_TARGET} -> ${CMAKE_CUDA_ARCHITECTURES}")

    # --- Mock Global Dependencies ---

    # 1. flags
    if(NOT TARGET cudampqs_build_flags)
        add_library(cudampqs_build_flags INTERFACE)
        target_compile_options(cudampqs_build_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info --use_fast_math --expt-relaxed-constexpr>
        )
        target_link_libraries(cudampqs_build_flags INTERFACE CUDA::cudart OpenMP::OpenMP_CXX)
    endif()

    # 2. mpqs_common (Local)
    # Assuming the common library is one level up in "common"
    set(LOCAL_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common")
    
    if(EXISTS "${LOCAL_COMMON_DIR}/uint512.cuh")
        message(STATUS "[Standalone] Found common headers in: ${LOCAL_COMMON_DIR}")
        
        add_library(mpqs_common_local STATIC
            ${LOCAL_COMMON_DIR}/uint512.cuh
            ${LOCAL_COMMON_DIR}/math_utils.cuh
            ${LOCAL_COMMON_DIR}/montgomery.cuh
            ${LOCAL_COMMON_DIR}/logger/hpc_logger.cpp
        )
        
        target_link_libraries(mpqs_common_local PUBLIC CUDA::cudart)
        target_compile_features(mpqs_common_local PUBLIC cxx_std_20)
        set_target_properties(mpqs_common_local PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        
        target_include_directories(mpqs_common_local PUBLIC ${LOCAL_COMMON_DIR})
    else()
        message(FATAL_ERROR "[Standalone] Could not locate ../common/uint512.cuh. Please adjust LOCAL_COMMON_DIR.")
    endif()

    # Define the library target
    add_library(mpqs_sqrt STATIC ${SQRT_LIB_SOURCES})

    # Link against LOCAL common
    target_link_libraries(mpqs_sqrt PUBLIC mpqs_common_local cudampqs_build_flags)

else()
    # ==========================================================================
    # Project Mode (Standard Build)
    # ==========================================================================
    
    add_library(mpqs_sqrt STATIC ${SQRT_LIB_SOURCES})

    # Link against GLOBAL common
    target_link_libraries(mpqs_sqrt PUBLIC mpqs_common cudampqs_build_flags)

endif()

# ==============================================================================
# Shared Configuration
# ==============================================================================

target_include_directories(mpqs_sqrt PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(mpqs_sqrt PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ==============================================================================
# Benchmark Binary (Only built in Standalone or if explicitly requested)
# ==============================================================================

# If standalone, we assume we want to run the benchmark. 
# In project mode, you might wrap this in "if(BUILD_TESTING)"
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    
    # Path to benchmark in tests/ folder (relative to src/sqrt/)
    set(BENCHMARK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/sqrt_benchmark.cpp")

    if(EXISTS "${BENCHMARK_SRC}")
        add_executable(sqrt_benchmark ${BENCHMARK_SRC})
        
        target_link_libraries(sqrt_benchmark PRIVATE 
            mpqs_sqrt
            # mpqs_common is linked transitively via mpqs_sqrt
        )
        message(STATUS "[Standalone] Added benchmark target: sqrt_benchmark")
    else()
        message(WARNING "[Standalone] Could not find benchmark source at: ${BENCHMARK_SRC}")
    endif()

endif()
