// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "autotune_history.h"
#include "benign_history.h"

namespace mpqs { struct MPQSConfig; }
namespace mpqs::sieve { struct factoringData; }

namespace mpqs::autotune {

/// Result of auto-apply parameter selection.
struct AutoApplyResult {
    bool applied = false;              ///< True if any params were changed
    uint32_t fb_bound = 0;
    uint32_t sieve_bound = 0;
    uint64_t lp1_bound = 0;
    bool kernel_params_applied = false;

    // Buffer recommendations
    uint64_t accum_buffer_size = 0;
    uint64_t partial_buffer_size = 0;
    uint64_t persistent_buffer_size = 0;
    uint64_t lp1_max_witness_capacity = 0;
    uint64_t lp1_max_combined_output = 0;
    uint32_t lp1_hash_bits = 0;

    // Provenance tracking
    struct ParamSource {
        std::string name;       ///< e.g., "fb_bound"
        std::string source;     ///< "history", "benign", "user", "default"
        std::string detail;     ///< e.g., "exact match, conf=0.98"
    };
    std::vector<ParamSource> sources;

    double confidence = 0.0;
    size_t estimated_gpu_memory_mb = 0;
};

/// Lightweight controller that applies cached parameters from history
/// without running GPU probes. Fires after TuningStage() on every run.
class AutoApplyController {
public:
    AutoApplyController(MPQSConfig& config,
                        const mpqs::sieve::factoringData& f_data);
    AutoApplyResult apply();

private:
    void loadHistories();
    void mergeSieveParams();
    void mergeBufferParams();
    void computeMemoryEstimate();
    void printRecommendations();

    MPQSConfig& config_;
    const mpqs::sieve::factoringData& f_data_;
    AutoApplyResult result_;

    HistoryStore history_;
    BenignHistoryStore benign_;
    std::string N_hash_;
    uint32_t N_bits_ = 0;
    uint32_t N_digits_ = 0;
    std::string gpu_name_;
};

} // namespace mpqs::autotune
