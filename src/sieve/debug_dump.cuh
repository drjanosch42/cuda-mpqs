// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

// debug_dump.cuh
#pragma once
#include <cstdint>
#include "sieving_data_structs.h"

namespace mpqs::sieve {

struct DebugCandidateRecord {
    uint32_t global_slot;      // 0..(numBlocks*maxRelationsPerBlock-1)
    uint32_t writer_block;     // global_slot / maxRelationsPerBlock
    uint32_t slot_in_block;    // global_slot % maxRelationsPerBlock
    uint64_t device_addr;      // &devcandidateRelations[global_slot] as seen on device
    uint32_t poly_id;
    uint32_t sieve_offset;
    uint32_t num_factors;
    uint32_t factors[32];
    uint32_t is_nonempty;      // 1 if looks written, else 0
};

__global__ void debugPackCandidates(const candidateRelation* __restrict__ cand,
                                   uint32_t maxPerBlock,
                                   uint32_t numBlocks,
                                   DebugCandidateRecord* __restrict__ out)
{
    uint32_t total = maxPerBlock * numBlocks;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const candidateRelation& c = cand[tid];

    DebugCandidateRecord r{};
    r.global_slot   = tid;
    r.writer_block  = tid / maxPerBlock;
    r.slot_in_block = tid % maxPerBlock;
    r.device_addr   = (uint64_t)(uintptr_t)(&cand[tid]);
    r.poly_id       = c.poly_id;
    r.sieve_offset  = c.sieve_offset;
    r.num_factors   = c.num_factors;

    #pragma unroll
    for (int i = 0; i < 32; i++) r.factors[i] = c.factors[i];

    // "nonempty" heuristic: candidate buffer is cudaMemset(0) before sieveStep in your controller
    r.is_nonempty = (c.num_factors != 0) | (c.poly_id != 0) | (c.sieve_offset != 0);
    out[tid] = r;
}

} // namespace mpqs::sieve
