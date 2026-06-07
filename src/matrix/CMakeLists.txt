# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
add_library(mpqs_matrix STATIC
    matrix_constructor.cu
    gpu_singleton.cu
    gpu_singleton_packed.cu     # M9b: In-place packed singleton removal
    gpu_batch_merge.cu          # M9e: Batch-planned GPU merge execution
    gpu_gf2_extract.cu          # M9f: GF(2) extraction from packed merged CSR
    gpu_product_char_packed.cu  # M9f: Jacobi-only product char cols from merged sqrt_Q
    gpu_compact_packed.cu       # M10a: GPU compaction of post-merge scattered CSR
    gpu_char_cols.cu
    device_packed_csr.cu
    gpu_packed_expanded.cu
    expanded_matrix.cpp
    merge_filter.cpp
    merge_tree.cpp
    character_columns.cpp
    matrix_truncation.cpp
    preprocess.cpp
)

# Required for cross-object-file device symbol resolution (RDC).
set_target_properties(mpqs_matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(mpqs_matrix PUBLIC
    mpqs_common
    mpqs_sieve  # Stage 2: 64-bit NT primitives (Tonelli_Shanks_u64/jacobi_u64/is_prime_u64) + prime_algorithms.h include path
    cudampqs_build_flags
)

# Needs access to linalg types (HostMatrix)
target_include_directories(mpqs_matrix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/linalg/include # For lingen/types.h
    ${CMAKE_SOURCE_DIR}/src/linalg/cuda_spmm/include # For common.h (HostMatrix)
)
