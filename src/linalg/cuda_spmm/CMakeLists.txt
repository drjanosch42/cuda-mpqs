cmake_minimum_required(VERSION 3.18)

# ==============================================================================
# 0. Standalone Bootstrap Logic
# ==============================================================================
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(STATUS "[SpMM] Building in STANDALONE mode")
    project(spmm_gf2_nfs LANGUAGES CXX CUDA)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)

    # Re-use architecture selection logic if standalone
    if(NOT GPU_TARGET)
        set(GPU_TARGET "native")
    endif()
    # ... (Insert Architecture Mapping Logic here) ...
    set(CMAKE_CUDA_ARCHITECTURES all-major) # Default for standalone

    find_package(CUDAToolkit REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# ==============================================================================
# 1. Logger Definition (Shared Logic)
# ==============================================================================
# Use hpc_logger name consistently with the master project
if(NOT TARGET hpc_logger)
    # Search for logger in either linalg/utils or common/logger
    set(LOGGER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../utils/hpc_logger.cpp")
    if(NOT EXISTS ${LOGGER_SRC})
        # Fallback for full project structure
        set(LOGGER_SRC "${CMAKE_SOURCE_DIR}/src/common/logger/hpc_logger.cpp")
    endif()

    add_library(hpc_logger STATIC ${LOGGER_SRC})
    target_include_directories(hpc_logger PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/src/common/logger # Fallback path
    )
endif()

# ==============================================================================
# 2. SpMM Core Library
# ==============================================================================
add_library(spmm_core STATIC
    src/device_csr.cu
    src/device_format_convert.cu
    src/gpu_autotuner.cu
    src/bw_spmm_interface.cpp
    src/generator.cu
    src/preprocessing.cpp
    src/reordering.cpp
    src/kernels.cu
    src/m4rm_kernels.cu
    src/m4rm_host.cpp
    src/autotuner.cpp
    src/spmm_optimized.cpp
    src/verification.cpp
    src/benchmarksuite.cpp
)

target_compile_features(spmm_core PUBLIC cxx_std_17)
set_target_properties(spmm_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(spmm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link bw_build_flags if available from parent, else use local flags
target_link_libraries(spmm_core PUBLIC
    hpc_logger
    OpenMP::OpenMP_CXX
    CUDA::cudart
    $<$<TARGET_EXISTS:bw_build_flags>:bw_build_flags>
)

# ==============================================================================
# 3. Benchmarks
# ==============================================================================
if(NOT CUDA_SPMM_SKIP_BENCHMARKS)
    # We only build the benchmark binaries if
    # we are in standalone mode or in BW standalone mode
  
    add_executable(spmm_full_benchmark bench/full_benchmark.cpp)
    target_link_libraries(spmm_full_benchmark PRIVATE spmm_core)

    add_executable(spmm_single_benchmark bench/single_benchmark.cpp)
    target_link_libraries(spmm_single_benchmark PRIVATE spmm_core bw_version_header)

    add_executable(spmm_verify_interface bench/verify_interface.cpp)
    target_link_libraries(spmm_verify_interface PRIVATE spmm_core)

    # Validation tests
    add_executable(test_format_correctness tests/test_format_correctness.cu)
    target_link_libraries(test_format_correctness PRIVATE spmm_core)

    add_executable(test_autotuner_ab tests/test_autotuner_ab.cu)
    target_link_libraries(test_autotuner_ab PRIVATE spmm_core)

    add_executable(test_spmm_endtoend tests/test_spmm_endtoend.cu)
    target_link_libraries(test_spmm_endtoend PRIVATE spmm_core)

    add_executable(test_memory_leak tests/test_memory_leak.cu)
    target_link_libraries(test_memory_leak PRIVATE spmm_core)

    add_executable(test_golden_regression tests/test_golden_regression.cu)
    target_link_libraries(test_golden_regression PRIVATE spmm_core)

    # GPU autotune benchmark
    add_executable(bench_gpu_autotune benchmarks/bench_gpu_autotune.cu)
    target_link_libraries(bench_gpu_autotune PRIVATE spmm_core)

endif()
