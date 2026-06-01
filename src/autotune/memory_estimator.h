// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <cstddef>
#include <cstdint>

namespace mpqs::autotune {

/// GPU memory cost constants for the sieve + LP postprocessing pipeline.
///
/// These are per-element byte costs of the on-device buffers that dominate
/// memory pressure. Used by:
///   - `AutoApplyController::computeMemoryEstimate` (auto_apply.cpp), and
///   - `AutotuneController::printBufferRecommendations` (autotune.cpp)
///
/// to project total GPU memory consumption from the chosen buffer sizes
/// (accum / partial / persistent / witness / lp_output / hash directory).
///
/// Values from buffer-sizing audit §9.2; keep in sync with the SoA struct
/// definitions in src/postprocessing/ and src/largeprimes/.
namespace memory_costs {
    inline constexpr std::size_t DENSE_CANDIDATE_BYTES = 336;   // DenseCandidate struct
    inline constexpr std::size_t SOA_PER_REL_BYTES     = 253;   // RelationBatch SoA
    inline constexpr std::size_t WITNESS_PAYLOAD_BYTES = 128;   // Payload slab (16 × 8B)
    inline constexpr std::size_t WITNESS_DIR_BYTES     = 8;     // Per directory entry
    inline constexpr std::size_t LP_PIPELINE_BYTES     = 41;    // Sort arrays + status
} // namespace memory_costs

/// Hardware-derived minimum partial-buffer size (in relations).
///
/// The LP postprocessing pipeline cannot maintain throughput below this floor:
/// it amortizes the per-batch kernel-launch overhead across the partial buffer,
/// and modern NVIDIA GPUs need ~64K elements per launch to reach the
/// memory-bandwidth-bound regime. Going below 65536 leaves the GPU
/// launch-latency-bound and degrades end-to-end relation throughput.
///
/// Enforced as a clamp at history load time (autotune_history.cpp F2 filter)
/// and at apply time (auto_apply.cpp::mergeBufferParams).
inline constexpr std::uint64_t kMinPartialBufferSize = 65536;

} // namespace mpqs::autotune
