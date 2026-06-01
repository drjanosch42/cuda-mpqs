// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "mpqs_soa.h"
#include "uint512.cuh"
#include <string>
#include <vector>
#include <cstdint>

namespace mpqs::io {

/// v1 format (backward compatible): single HostRelationBatch with projected LP values.
/// Magic: "MPQS_SOA\0" (8 bytes).
bool serialize_v1(const std::string& path,
                  const structures::HostRelationBatch& batch);
bool deserialize_v1(const std::string& path,
                    structures::HostRelationBatch& batch);

/// v2 format: full smooths + raw partials + pipeline metadata.
/// Magic: "MPQS_V2\0" (8 bytes), version=2, flags bitfield.
struct V2Metadata {
    uint512 N;                         ///< Modulus (64 bytes little-endian)
    std::vector<uint32_t> factor_base; ///< Factor base primes
    uint64_t lp_bound = 0;            ///< LP bound used during sieving
    uint32_t sieve_bound = 0;         ///< M (sieve interval half-width)
};

bool serialize_v2(const std::string& path,
                  const structures::HostRelationBatch& full_smooths,
                  const structures::HostRelationBatch& partials,
                  const V2Metadata& meta);

bool deserialize_v2(const std::string& path,
                    structures::HostRelationBatch& full_smooths,
                    structures::HostRelationBatch& partials,
                    V2Metadata& meta);

/// Auto-detect format from magic bytes and deserialize accordingly.
/// Returns 1 for v1, 2 for v2, 0 on failure.
/// For v1: populates `batch` only (smooths/partials/meta untouched).
/// For v2: populates smooths, partials, meta (batch untouched).
int detect_and_deserialize(const std::string& path,
                           structures::HostRelationBatch& batch,
                           structures::HostRelationBatch& smooths,
                           structures::HostRelationBatch& partials,
                           V2Metadata& meta);

} // namespace mpqs::io
