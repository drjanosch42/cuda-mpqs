# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski

add_library(mpqs_largeprimes STATIC
    largeprime.cu
)

target_link_libraries(mpqs_largeprimes PUBLIC 
    mpqs_postproc
    mpqs_common 
    cudampqs_build_flags
)

target_include_directories(mpqs_largeprimes PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
