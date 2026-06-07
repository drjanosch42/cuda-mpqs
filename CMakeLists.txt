cmake_minimum_required(VERSION 3.22)

include(cmake/version.cmake)

project(CUDAmpqs VERSION ${CUDAMPQS_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Capture git SHA and build date for version.h
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE CUDAMPQS_BUILD_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(NOT CUDAMPQS_BUILD_GIT_SHA)
    set(CUDAMPQS_BUILD_GIT_SHA "unknown")
endif()
string(TIMESTAMP CUDAMPQS_BUILD_DATE "%Y-%m-%d")

configure_file(
    ${CMAKE_SOURCE_DIR}/include/version.h.in
    ${CMAKE_BINARY_DIR}/include/version.h
    @ONLY
)

# Make the generated version.h accessible to all targets
include_directories(${CMAKE_BINARY_DIR}/include)

# ==============================================================================
# 1. Global Toolchain & Standards
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization config
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ==============================================================================
# 2. CUDA Architecture Selection / User Tunable Variable (Appears in ccmake / cmake-gui)
# ==============================================================================
set(GPU_TARGET "native" CACHE STRING 
    "Target GPU Architecture. Options: native, Blackwell, Hopper, Ampere, all, or specific SM (e.g., 86)")

  if(NOT GPU_TARGET)
    set(GPU_TARGET "native")
endif()

if(GPU_TARGET STREQUAL "native")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        # CMake 3.24+ supports native auto-detection
        set(ARCH_OPTS "native")
    else()
        # CMake < 3.24: detect GPU architecture via nvidia-smi or __nvcc_device_query
        set(_DETECTED_ARCH "")
        # Try nvidia-smi first (works on desktop + Jetson with JetPack 6.x)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _GPU_CC_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _GPU_CC_RESULT
            ERROR_QUIET
        )
        if(_GPU_CC_RESULT EQUAL 0 AND _GPU_CC_RAW)
            # Take first GPU if multi-GPU; output is e.g. "8.9\n8.9"
            string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _match "${_GPU_CC_RAW}")
            if(_match)
                set(_DETECTED_ARCH "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
            endif()
        endif()
        # Fallback: __nvcc_device_query (CUDA toolkit 11.6+)
        if(NOT _DETECTED_ARCH)
            execute_process(
                COMMAND __nvcc_device_query
                OUTPUT_VARIABLE _NVCC_QUERY_RAW
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _NVCC_QUERY_RESULT
                ERROR_QUIET
            )
            if(_NVCC_QUERY_RESULT EQUAL 0 AND _NVCC_QUERY_RAW)
                string(REGEX MATCH "([0-9]+)" _DETECTED_ARCH "${_NVCC_QUERY_RAW}")
            endif()
        endif()
        if(_DETECTED_ARCH)
            set(ARCH_OPTS "${_DETECTED_ARCH}")
            message(STATUS "Auto-detected GPU architecture: sm_${ARCH_OPTS} (CMake < 3.24 fallback)")
        else()
            message(WARNING "GPU auto-detection failed (no nvidia-smi or __nvcc_device_query). Using all-major.")
            set(ARCH_OPTS "all-major")
        endif()
    endif()
elseif(GPU_TARGET MATCHES "^(Turing|75)$")
    set(ARCH_OPTS "75")
elseif(GPU_TARGET MATCHES "^(Ampere|80)$")
    set(ARCH_OPTS "80")
elseif(GPU_TARGET MATCHES "^(Ada|89)$")
    set(ARCH_OPTS "89")
elseif(GPU_TARGET MATCHES "^(Hopper|90)$")
    set(ARCH_OPTS "90")
elseif(GPU_TARGET MATCHES "^(Blackwell|120)$")
    # Blackwell (RTX 50-series) is sm_120
    set(ARCH_OPTS "120")
elseif(GPU_TARGET MATCHES "^(Orin|orin|87)$")
    # Jetson AGX Orin / Orin NX is sm_87
    set(ARCH_OPTS "87")
elseif(GPU_TARGET STREQUAL "all")
    set(ARCH_OPTS "all-major")
else()
    set(ARCH_OPTS ${GPU_TARGET})
endif()

# ==============================================================================
# Apply Globally
# ==============================================================================
# This sets the property for ALL targets created after this point.
# It automatically handles passing the correct -gencode flags to nvcc 
# AND the correct arguments to the device linker (dlink).
set(CMAKE_CUDA_ARCHITECTURES ${ARCH_OPTS} CACHE STRING "" FORCE)      

message(STATUS "Configuring for GPU Architecture: ${GPU_TARGET} -> ${CMAKE_CUDA_ARCHITECTURES}")

# ==============================================================================
# 3. Global Build Flags (Interface Library)
# ==============================================================================
add_library(cudampqs_build_flags INTERFACE)

target_compile_options(cudampqs_build_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info --use_fast_math --expt-relaxed-constexpr -Xcompiler=-Wall>
)

target_link_libraries(cudampqs_build_flags INTERFACE
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

# Global Include Directories
# This allows #include "orchestrator.h" and "hpc_logger.h" to work everywhere
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/common)
include_directories(${CMAKE_SOURCE_DIR}/src/common/logger)
include_directories(${CMAKE_SOURCE_DIR}/src/linalg/include)
include_directories(${CMAKE_SOURCE_DIR}/src/linalg/include/lingen)

# ==============================================================================
# 3.5 Debug / Instrumentation Flags
# ==============================================================================
option(SIEVING_DEBUG_MODE "Enable debug mode" OFF)

if(SIEVING_DEBUG_MODE)
    message(STATUS "[cudampqs] SIEVING_DEBUG_MODE is ON")
    add_compile_definitions(SIEVING_DEBUG_FLAG)
endif()

if(LP_DEBUG)
    add_compile_definitions(LP_DEBUG)
endif()

# ==============================================================================
# 4. Add Sub-Modules
# ==============================================================================

# Order matters slightly for include path visibility, though linking handles dependencies
add_subdirectory(src/common)
add_subdirectory(src/sieve)
add_subdirectory(src/postprocessing)
add_subdirectory(src/largeprimes)
add_subdirectory(src/matrix)
add_subdirectory(src/linalg)
add_subdirectory(src/sqrt)
add_subdirectory(src/autotune)
add_subdirectory(src/orchestrator)
add_subdirectory(src/cluster)

# ==============================================================================
# 5. Tests / Benchmarks
# ==============================================================================
# Enable CTest at the top level so add_test() in tests/ is discoverable via
# `ctest --test-dir <build>` (root-scope enable_testing() generates the
# build-root CTestTestfile.cmake that includes the tests/ subdirectory).
enable_testing()
add_subdirectory(tests)
