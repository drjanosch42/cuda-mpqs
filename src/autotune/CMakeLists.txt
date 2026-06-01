# SPDX-License-Identifier: LGPL-3.0-only
add_library(mpqs_autotune STATIC
    auto_apply.cpp
    autotune.cpp
    autotune_history.cpp
    autotune_projection.cpp
    benign_history.cpp
    cost_models.cpp
    kernel_launch_validator.cpp
    kernel_param_optimizer.cpp
    runtime_estimator.cpp
    sieve_optimizer.cpp
)
target_link_libraries(mpqs_autotune PUBLIC mpqs_common cudampqs_build_flags mpqs_sieve)
target_include_directories(mpqs_autotune PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# PRIVATE include dirs needed by runtime_estimator.cpp, which includes orchestrator.h.
# orchestrator.h transitively includes headers from these modules.
# No link dep on mpqs_orchestrator — avoids circular CMake dependency
# (mpqs_orchestrator already links mpqs_autotune PUBLIC).
# All symbols resolve at final executable link time.
target_include_directories(mpqs_autotune PRIVATE
    ${CMAKE_SOURCE_DIR}/src/postprocessing
    ${CMAKE_SOURCE_DIR}/src/largeprimes
    ${CMAKE_SOURCE_DIR}/src/matrix
    ${CMAKE_SOURCE_DIR}/src/sqrt
    ${CMAKE_SOURCE_DIR}/src/linalg/src
    ${CMAKE_SOURCE_DIR}/src/linalg/cuda_spmm/include
)
