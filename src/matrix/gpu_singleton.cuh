// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_singleton.cuh
#pragma once

#include "merge_filter.h"        // SingletonResult
#include "matrix_constructor.h"  // HostMatrixCSR

namespace mpqs {
namespace matrix {

/// GPU-accelerated singleton removal — drop-in replacement for
/// MergeFilterPipeline::removeSingletons().
///
/// Uploads the expanded CSR to device, runs an iterative atomic fixpoint
/// (parallel row-death detection), compacts the result, and downloads to host.
/// Returns a SingletonResult with identical fields and semantics to the CPU version.
///
/// Correctness: the singleton-free submatrix is unique regardless of removal order
/// (CPU sequential vs. GPU parallel). The GPU may remove more rows per iteration
/// than the CPU (column weights drop mid-iteration, causing additional rows to die),
/// but both converge to the same fixed point.
///
/// Memory budget: ~54 MB for RSA-110 scale (3% of RTX 5070 Ti VRAM).
/// On Jetson (SM 8.7, concurrentManagedAccess == 0), uses cudaMallocManaged.
SingletonResult gpuRemoveSingletons(const HostMatrixCSR& input);

} // namespace matrix
} // namespace mpqs
