# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski

# Main Driver
add_executable(cuda-mpqs cuda-mpqs.cpp)

# Critical: Resolve device symbols because we have extensive RDC (Relocatable Device Code)
# across multiple static libraries (sieve, postproc, linalg, common).
set_target_properties(cuda-mpqs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cuda-mpqs PRIVATE
    mpqs_orchestrator
    cudampqs_cluster
    mpqs_autotune
    mpqs_sieve
    mpqs_postproc
    mpqs_matrix
    mpqs_sqrt
    lingen_ops
    cudampqs_build_flags
)

# Optional Sqrt specific benchmark (EXCLUDE_FROM_ALL: uses AoS Relation API, needs M2 update for SoA)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sqrt_benchmark.cpp")
    add_executable(bench_sqrt EXCLUDE_FROM_ALL sqrt_benchmark.cpp
        ${CMAKE_SOURCE_DIR}/src/sqrt/fake_relation_generator.cpp)
    set_target_properties(bench_sqrt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(bench_sqrt PRIVATE mpqs_sqrt mpqs_common cudampqs_build_flags)
endif()
