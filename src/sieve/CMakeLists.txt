# ==============================================================================
# Sieve module — integrated build (parent project only)
# ==============================================================================
#
# This CMakeLists.txt assumes it is invoked from the parent cuda-mpqs project,
# which provides:
#   - The `mpqs_common` target (uint512, math_utils, montgomery, logger)
#   - The `cudampqs_build_flags` interface library (compiler flags, CUDA::cudart, OpenMP)
#   - Global include directories for src/common/ and src/common/logger/
#
# Sources are located flat in this directory (formerly src/sieve/Quadratic_Sieve/).
# Header-only components (kernel.cuh, sieving_data_structs.h, common.h, graycode.cuh,
# json_helper.h, prime_algorithms.h, debug_dump.h, debug_dump.cuh,
# device_sieving_controller.h) are consumed via target_include_directories.

set(SIEVE_CORE_SOURCES
    kernel.cu
    prime_algorithms.cu
    device_sieving_controller.cpp
    debug_dump.cu
)

add_library(mpqs_sieve STATIC ${SIEVE_CORE_SOURCES})

target_include_directories(mpqs_sieve PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(mpqs_sieve PUBLIC mpqs_common cudampqs_build_flags)

set_target_properties(mpqs_sieve PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
