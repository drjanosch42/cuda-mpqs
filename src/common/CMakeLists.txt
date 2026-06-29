# src/common/CMakeLists.txt

# Standalone hpc_logger target — must be defined before src/linalg is processed,
# so linalg's if(NOT TARGET hpc_logger) guard skips its own stale copy.
if(NOT TARGET hpc_logger)
    add_library(hpc_logger STATIC logger/hpc_logger.cpp)
    target_include_directories(hpc_logger PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/logger
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(hpc_logger PUBLIC cudampqs_build_flags)
endif()

set(COMMON_SOURCES
    uint512.cuh
    math_utils.cuh
    montgomery.cuh
    uint128_helper.cuh
    mpqs_soa.cu
    relation_io.cpp
    sieve_checkpoint.cpp
)

add_library(mpqs_common ${COMMON_SOURCES})

set_target_properties(mpqs_common PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(mpqs_common PUBLIC cudampqs_build_flags hpc_logger)

# Expose headers to other modules
target_include_directories(mpqs_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
