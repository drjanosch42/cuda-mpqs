# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski

cmake_minimum_required(VERSION 3.18)

# ==============================================================================
# 0. Source Definition
# ==============================================================================

set(POSTPROC_LIB_SOURCES
    postprocessing.cu
    postprocessing.h
)

# ==============================================================================
# 1. Standalone Bootstrap Logic
# ==============================================================================
# This block ONLY runs if you build this directory directly (e.g., for testing).
# If built from the project root, this block is skipped.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(STATUS "[PostProc] Building in STANDALONE mode")
    
    project(StandalonePostProc LANGUAGES CXX CUDA)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS CXX)

    # --- Architecture Selection Logic (Copied from Sieve/Sqrt) ---
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

    # 1. Build Flags
    if(NOT TARGET cudampqs_build_flags)
        add_library(cudampqs_build_flags INTERFACE)
        target_compile_options(cudampqs_build_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info --use_fast_math --expt-relaxed-constexpr>
        )
        target_link_libraries(cudampqs_build_flags INTERFACE CUDA::cudart OpenMP::OpenMP_CXX)
    endif()

    # 2. mpqs_common (Local) - Includes Logger
    set(LOCAL_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common")
    if(EXISTS "${LOCAL_COMMON_DIR}/uint512.cuh")
        message(STATUS "[Standalone] Found common headers in: ${LOCAL_COMMON_DIR}")
        
        add_library(mpqs_common_local STATIC
            ${LOCAL_COMMON_DIR}/uint512.cuh
            ${LOCAL_COMMON_DIR}/math_utils.cuh
            ${LOCAL_COMMON_DIR}/logger/hpc_logger.cpp 
        )
        
        target_link_libraries(mpqs_common_local PUBLIC CUDA::cudart)
        target_compile_features(mpqs_common_local PUBLIC cxx_std_20)
        set_target_properties(mpqs_common_local PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_include_directories(mpqs_common_local PUBLIC ${LOCAL_COMMON_DIR})
    else()
        message(FATAL_ERROR "[Standalone] Could not locate ../common/uint512.cuh.")
    endif()

    # 3. mpqs_sieve (Local) - Necessary because PostProc depends on Sieve structs
    set(LOCAL_SIEVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sieve")
    if(EXISTS "${LOCAL_SIEVE_DIR}/Quadratic_Sieve/sievingDataStructs.h")
        message(STATUS "[Standalone] Building local Siever from: ${LOCAL_SIEVE_DIR}")
        
        add_library(mpqs_sieve_local STATIC
            ${LOCAL_SIEVE_DIR}/Quadratic_Sieve/kernel.cu
            ${LOCAL_SIEVE_DIR}/Quadratic_Sieve/primeAlgorithms.cu
            ${LOCAL_SIEVE_DIR}/Quadratic_Sieve/deviceSievingController.cpp
            ${LOCAL_SIEVE_DIR}/Quadratic_Sieve/debugDump.cu
        )
        
        target_link_libraries(mpqs_sieve_local PUBLIC mpqs_common_local cudampqs_build_flags)
        set_target_properties(mpqs_sieve_local PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_include_directories(mpqs_sieve_local PUBLIC ${LOCAL_SIEVE_DIR}/Quadratic_Sieve)
    else()
        message(FATAL_ERROR "[Standalone] Could not locate Siever sources.")
    endif()

    # Define the library target (using local deps)
    add_library(mpqs_postproc STATIC ${POSTPROC_LIB_SOURCES})
    target_link_libraries(mpqs_postproc PUBLIC mpqs_common_local mpqs_sieve_local cudampqs_build_flags)

else()
    # ==========================================================================
    # Project Mode (Standard Build)
    # ==========================================================================
    
    add_library(mpqs_postproc STATIC ${POSTPROC_LIB_SOURCES})

    # Link against GLOBAL common and sieve
    target_link_libraries(mpqs_postproc PUBLIC mpqs_common mpqs_sieve cudampqs_build_flags)

endif()

# ==============================================================================
# Shared Configuration
# ==============================================================================

target_compile_options(mpqs_postproc PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)  

set_target_properties(mpqs_postproc PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(mpqs_postproc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Stage 4 (branch-fixed character columns): postprocessing computes the per-relation
# branch character vector at relation birth via mpqs::matrix::branchCharBit
# (header-only __host__ __device__ inline in src/matrix/gpu_char_cols.cuh). Only the
# header is needed — no link dependency on mpqs_matrix is introduced. PRIVATE so it
# does not leak into postproc's public interface. Project-mode only (standalone
# postproc has no matrix sources).
if(NOT (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR))
    target_include_directories(mpqs_postproc PRIVATE
        ${CMAKE_SOURCE_DIR}/src/matrix
    )
endif()

# ==============================================================================
# Benchmark Binary (Only built in Standalone)
# ==============================================================================

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    
    # Path to benchmark in tests/ folder
    set(BENCHMARK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/sieving_benchmark.cpp")
    
    # We also need the fake generator from SQRT to make a valid N, 
    # but we don't want to pull in the whole SQRT library dependency chain.
    set(FAKE_GEN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../sqrt/fake_relation_generator.cpp")
      
    if(EXISTS "${BENCHMARK_SRC}" AND EXISTS "${FAKE_GEN_SRC}")
        
        add_executable(sieving_benchmark
            ${BENCHMARK_SRC}
            ${FAKE_GEN_SRC}
        )
	  
	set_target_properties(sieving_benchmark PROPERTIES 
	    CUDA_SEPARABLE_COMPILATION ON
	    CUDA_RESOLVE_DEVICE_SYMBOLS ON
	)
        
        target_link_libraries(sieving_benchmark PRIVATE 
            mpqs_postproc      # The library defined above
            mpqs_sieve_local   # The local siever we built
            mpqs_common_local  # The local common (logger)
        )

        # Include paths for the test to find headers
        target_include_directories(sieving_benchmark PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../sqrt  # For fake_relation_generator.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests
        )

        message(STATUS "[Standalone] Added benchmark target: sieving_benchmark")
    else()
        message(WARNING "[Standalone] Could not find benchmark sources. Skipped.")
    endif()

endif()
