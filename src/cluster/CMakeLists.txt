# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
# This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
# See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

# Cluster v2 module
add_library(cudampqs_cluster STATIC
    cpu_lp.cu
    tcp_transport.cpp
    tcp_backend.cpp
    serialization.cpp
    work_pool.cpp
    chunk_scheduler.cpp
    async_network_data_tap.cpp
)

# Separable compilation required for CUDA (RDC) — Montgomery arithmetic uses
# __host__ __device__ functions from src/common/.
set_target_properties(cudampqs_cluster PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(cudampqs_cluster
    PUBLIC cudampqs_build_flags
    PUBLIC mpqs_common
    PUBLIC mpqs_sieve
)

# Make cluster headers available to downstream targets (#include "data_tap.h", etc.)
target_include_directories(cudampqs_cluster
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
