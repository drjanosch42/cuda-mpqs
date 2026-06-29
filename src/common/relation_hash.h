// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "mpqs_soa.h"
#include <cstdint>
#include <cstddef>

/// @file relation_hash.h
/// @brief SINGLE SOURCE OF TRUTH for the 64-bit relation dedup hash (m-sharedTU).
///
/// This formula is byte-for-byte identical to the GPU `compute_relation_hashes_soa`
/// (`src/postprocessing/postprocessing.cu:514`) and is shared by:
///   - the cluster `RelationAccumulator` (`src/cluster/accumulator.h`) — incremental
///     pooled dedup, and
///   - the solo checkpoint host-side dedup (`src/orchestrator/orchestrator.cpp`) — to
///     shrink the mid-sieve checkpoint file.
///
/// NEVER copy this formula into another translation unit — a divergent copy is a silent
/// dedup bug. Both the cluster and solo paths MUST call this one definition so they agree
/// with the GPU hash regardless of `--char_mode`.

namespace mpqs {

/// Compute the 64-bit relation dedup hash:
///   hash = (len << 48) | (exp_xor << 32) | body_xor
/// where body_xor folds factor_indices*MAGIC, sign, and val_2_exp.
///
/// Stage-4 invariant (M12-S5/S5b analogue): batch.char_bits is DELIBERATELY NOT folded
/// into this hash. char_bits is a deterministic function of (ax+b), so two relations with
/// identical factorization share identical char vectors — including it cannot change dedup
/// identity, and excluding it keeps this hash byte-for-byte equivalent to the GPU hash
/// regardless of char_mode.
///
/// Sign encoding (canonical, see matrix-module audit Appendix A): batch.signs[i] is uint8_t
/// with {1 = positive Q, 0xFF/other = negative Q}. The M11c encoding-agnostic "negative iff
/// != 1" extraction matches the GPU path in postprocessing.cu:594.
inline uint64_t computeRelationHash(const structures::HostRelationBatch& batch, size_t i) {
    constexpr uint32_t MAGIC = 0x9e3779b9;

    uint64_t start = batch.factor_offsets[i];
    uint64_t end   = batch.factor_offsets[i + 1];
    uint16_t len   = static_cast<uint16_t>(end - start);

    uint16_t exp_xor  = 0;
    uint32_t body_xor = 0;
    for (uint64_t k = start; k < end; k++) {
        exp_xor  ^= static_cast<uint16_t>(batch.factor_counts[k]);
        body_xor ^= (batch.factor_indices[k] * MAGIC);
    }

    int32_t sign_val  = (batch.signs[i] != 1u) ? -1 : 1;
    int32_t shift_val = sign_val * (1 << (batch.val_2_exps[i] & 0x1F));
    body_xor ^= static_cast<uint32_t>(shift_val);

    return (static_cast<uint64_t>(len) << 48)
         | (static_cast<uint64_t>(exp_xor) << 32)
         | static_cast<uint64_t>(body_xor);
}

} // namespace mpqs
