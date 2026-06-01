// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "logger/hpc_logger.h"

// Standard CUDA check: log error through HPCLogger + throw runtime_error.
// Module context comes from g_log_context.module (set by caller via LOG_SET_MODULE).
// Do NOT use in destructors — destructors must not throw. Use a try/catch wrapper instead.
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "CUDA error: " << cudaGetErrorString(err__) \
            << " at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error( \
            std::string("CUDA error: ") + cudaGetErrorString(err__) \
            + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
#endif

// Fatal variant: identical behavior (both throw; callers should not catch FATAL).
// Distinct name preserved for semantic documentation of call sites.
#ifndef CUDA_CHECK_FATAL
#define CUDA_CHECK_FATAL(call) CUDA_CHECK(call)
#endif
