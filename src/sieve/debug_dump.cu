// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

// debug_dump.cu
#include <vector>
#include <cstdio>
#include <cinttypes>
#include <cuda_runtime.h>

#include "debug_dump.h"
#include "debug_dump.cuh"

namespace mpqs::sieve {

void dumpCandidatesJSON(DeviceSievingController& ctrl)
{
    const auto gs = ctrl.getGeneralConfig();     // add getters or pass configs explicitly
    const auto ss = ctrl.getSieveAndScanConfig();
    // const auto fs = ctrl.getFixedParams(); // currently unused
    const auto ds = ctrl.getDynamicParams();
    const auto fdata = ctrl.getFactoringData();

    const uint32_t numBlocks = ss.num_threadBlocks;
    const uint32_t maxPerBlock = gs.maxRelationsPerBlock;
    const uint32_t total = numBlocks * maxPerBlock;

    DebugCandidateRecord* d_out = nullptr;
    cudaMalloc(&d_out, total * sizeof(DebugCandidateRecord));

    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    debugPackCandidates<<<blocks, threads>>>(ctrl.getDevicePointers().dev_candidateRelations,
                                            maxPerBlock, numBlocks, d_out);
    cudaDeviceSynchronize();

    std::vector<DebugCandidateRecord> h(total);
    cudaMemcpy(h.data(), d_out, total * sizeof(DebugCandidateRecord), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    // Print ALL nonempty records; for one sieve step this should be “a few hundred”.
    for (const auto& r : h) {
        if (!r.is_nonempty) continue;

        // Reconstruct x index used by CPU checks: x = startIndex + sieve_offset
        int32_t x = ds.startIndex + (int32_t)r.sieve_offset;

        std::printf("{\"slot\":%u,\"block\":%u,\"slotInBlock\":%u,"
                    "\"devAddr\":\"0x%016" PRIx64 "\","
                    "\"polyId\":%u,\"sieveOffset\":%u,\"x\":%d,"
                    "\"numFactors\":%u,\"factors\":[",
                    r.global_slot, r.writer_block, r.slot_in_block,
                    r.device_addr, r.poly_id, r.sieve_offset, x, r.num_factors);

        // Print both factor-base index and prime value (odd primes only, as per your convention).
        uint32_t lim = (r.num_factors > 32u) ? 32u : r.num_factors;
        for (uint32_t i = 0; i < lim; i++) {
            uint32_t idx = r.factors[i];
            uint32_t p = (idx < fdata.factorBase.size()) ? fdata.factorBase[idx] : 0;
            std::printf("%s{\"idx\":%u,\"p\":%u}", (i ? "," : ""), idx, p);
        }
        std::printf("]}");

        if (r.num_factors > 32u) std::printf("  // WARNING: numFactors>32 (will overflow factors[32])");
        std::printf("\n");
    }
}

} // namespace mpqs::sieve
