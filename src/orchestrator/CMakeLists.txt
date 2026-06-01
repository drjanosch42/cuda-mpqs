# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
# This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
# See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

add_library(mpqs_orchestrator STATIC
    orchestrator.cpp
)

target_link_libraries(mpqs_orchestrator PUBLIC
    mpqs_autotune
    mpqs_common
    mpqs_sieve
    mpqs_postproc
    mpqs_largeprimes
    mpqs_matrix
    lingen_ops
    mpqs_sqrt
    cudampqs_build_flags
)

target_include_directories(mpqs_orchestrator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/cluster
)

set_target_properties(mpqs_orchestrator PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)
