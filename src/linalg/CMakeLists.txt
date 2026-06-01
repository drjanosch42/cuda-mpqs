cmake_minimum_required(VERSION 3.18)

# Enable if Block Wiedemann stage 2 fails because of too many kernel launches.
# This is known to be the case on Turing (SM 7.5) hardware, and is likely needed
# on Jetson Orin (SM 8.7) which has similar register file constraints (64 KB/SM).
# Recommended: -DENABLE_LINGEN_DEVICE_SYNC=ON for Jetson builds.
option(ENABLE_LINGEN_DEVICE_SYNC "Enables periodic CUDA stream synchronization in the lingen (BW stage 2) main loop." OFF)
option(ENABLE_CUDA_GRAPHS "Enable CUDA graph capture paths in the Block Wiedemann solver. \
Requires CUDA 10.1+ (basic graphs). When OFF, all graph code is compiled out." OFF)

# Single source of truth for the library version + a build-time git short hash.
# The release tooling (tools/release/extract_public.sh) greps cmake/version.cmake
# directly for LINALGBW_VERSION_{MAJOR,MINOR,PATCH}, so do not rename those.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake)

find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE LINALGBW_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(NOT LINALGBW_GIT_HASH)
    set(LINALGBW_GIT_HASH "unknown")
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bw_version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/bw_version.h
    @ONLY
)

# Tiny INTERFACE target exposing the generated header directory. Lets executables
# defined in subdirectories (e.g. cuda_spmm/bench/single_benchmark) include
# bw_version.h without re-deriving the path.
add_library(bw_version_header INTERFACE)
target_include_directories(bw_version_header INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# ==============================================================================
# 0. Standalone Bootstrap Logic
# ==============================================================================
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(STATUS "[Lingen] Building in STANDALONE mode")
    project(BlockWiedemannLingen VERSION ${LINALGBW_VERSION} LANGUAGES CXX CUDA)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)

    # Use the same architecture selection logic as the root project
    if(NOT GPU_TARGET)
        set(GPU_TARGET "native")
    endif()

    if(GPU_TARGET STREQUAL "native")
        # NOTE: all-major builds SM 50/60/70/80/90/100/120 but NOT minor revisions
        # like SM 87 (Jetson Orin). SM 80 PTX provides forward-compatible JIT for
        # SM 87, but for best performance use: -DGPU_TARGET=87
        set(CMAKE_CUDA_ARCHITECTURES all-major)
    else()
        # Mapping logic (simplified for brevity, match your root logic here)
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_TARGET})
    endif()

    find_package(CUDAToolkit REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS CXX)

    # Define flags if standalone
    add_library(bw_build_flags INTERFACE)
    if(CMAKE_CROSSCOMPILING)
        # Cross-compilation (e.g., x86 host → ARM/Jetson target): omit -march=native
        # because the host compiler's "native" refers to the host arch, not the target.
        target_compile_options(bw_build_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--use_fast_math;--expt-relaxed-constexpr>
        )
    else()
        # Native compilation (including ARM-native on Jetson): -march=native works correctly.
        target_compile_options(bw_build_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--use_fast_math;--expt-relaxed-constexpr>
        )
    endif()
    target_link_libraries(bw_build_flags INTERFACE CUDA::cudart OpenMP::OpenMP_CXX)

    # ==============================================================================
    # 1. Sub-Library: cuda_spmm
    # ==============================================================================
    add_subdirectory(cuda_spmm)
	  
    # ==============================================================================
    # 2. Lingen Operations Library
    # ==============================================================================
    add_library(lingen_ops STATIC
	src/lingen/operations/matmul_gf2.cu
	src/lingen/operations/poly_arithmetic.cu
        src/lingen/operations/poly_mat_vec.cu
        src/lingen/operations/bw_operator.cu
        src/lingen/stage1/krylov_kernels.cu
        src/lingen/stage1/krylov_generator.cu
        src/lingen/stage2/basecase_ops.cu
        src/lingen/stage2/basecase_solver.cu
        src/lingen/stage2/basecase_solver_reference.cu
        src/lingen/operations/poly_arith_engine.cu
        src/lingen/stage3/solution_reconstructor.cu
        src/lingen/bw_solver.cu
        src/lingen/bw_version.cpp
    )

    set_target_properties(lingen_ops PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    if(ENABLE_LINGEN_DEVICE_SYNC)
        target_compile_definitions(lingen_ops PRIVATE BASECASE_SOLVER_CUDA_STREAM_SYNC)
    endif()

    if(ENABLE_CUDA_GRAPHS)
        target_compile_definitions(lingen_ops PRIVATE BW_ENABLE_CUDA_GRAPHS)
    endif()

    # PUBLIC linking ensures the Master project gets spmm_core transitively
    target_link_libraries(lingen_ops PUBLIC
        bw_build_flags
        hpc_logger
        spmm_core
    )

    target_include_directories(lingen_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_spmm/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated
    )

    # ==============================================================================
    # 3. Benchmarks (Conditional for cleanliness)
    # ==============================================================================
    macro(add_lingen_bench name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE lingen_ops)
    endmacro()

    add_lingen_bench(bench_matmulgf2 benchmarks/bench_matmulgf2.cu)
    add_lingen_bench(bench_karatsuba benchmarks/bench_karatsuba.cu)
    add_lingen_bench(bench_lingen_apply_pi benchmarks/bench_lingen_apply_pi.cu)
    add_lingen_bench(bw_lingen_smoke benchmarks/bw_lingen_smoke.cu)
    add_lingen_bench(bw_lingen_bench benchmarks/bw_lingen_benchmark.cu)
    add_lingen_bench(test_s_disk_io tests/test_s_disk_io.cu)
	  
else()
    # If not standalone, we only build the library to increase build time
    message(STATUS "[Lingen] Building as part of parent project")

    # We skip the executables in cuda_spmm
    set(CUDA_SPMM_SKIP_BENCHMARKS ON CACHE BOOL "Skip cuda_spmm benchmarks")
      
    # Inherit cudampqs_build_flags if defined by parent, otherwise use local bw_build_flags
    if(TARGET cudampqs_build_flags)
        add_library(bw_build_flags INTERFACE)
        target_link_libraries(bw_build_flags INTERFACE cudampqs_build_flags)
    endif()
    # ==============================================================================
    # 1. Sub-Library: cuda_spmm
    # ==============================================================================
    add_subdirectory(cuda_spmm)
	  
    # ==============================================================================
    # 2. Lingen Operations Library (without heavy matrix poly arithmetic)
    # ==============================================================================
    add_library(lingen_ops STATIC
	src/lingen/operations/matmul_gf2.cu
	# src/lingen/operations/poly_arithmetic.cu
        src/lingen/operations/poly_mat_vec.cu
        src/lingen/operations/bw_operator.cu
        src/lingen/stage1/krylov_kernels.cu
        src/lingen/stage1/krylov_generator.cu
        src/lingen/stage2/basecase_ops.cu
        src/lingen/stage2/basecase_solver.cu
        src/lingen/stage2/basecase_solver_reference.cu
        # src/lingen/operations/poly_arith_engine.cu
        src/lingen/stage3/solution_reconstructor.cu
        src/lingen/bw_solver.cu
        src/lingen/bw_version.cpp
    )

    set_target_properties(lingen_ops PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    if(ENABLE_LINGEN_DEVICE_SYNC)
        target_compile_definitions(lingen_ops PRIVATE BASECASE_SOLVER_CUDA_STREAM_SYNC)
    endif()

    if(ENABLE_CUDA_GRAPHS)
        target_compile_definitions(lingen_ops PRIVATE BW_ENABLE_CUDA_GRAPHS)
    endif()

    # PUBLIC linking ensures the Master project gets spmm_core transitively
    target_link_libraries(lingen_ops PUBLIC
        bw_build_flags
        hpc_logger
        spmm_core
    )

    target_include_directories(lingen_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_spmm/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated
    )

endif()


