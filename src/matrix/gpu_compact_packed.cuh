// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_compact_packed.cuh
//
// GPU compaction of a post-merge scattered DevicePackedCSR (M10a).
// Removes dead rows, resolves workspace indirection, remaps column indices
// to eliminate empty columns. Produces a fresh contiguous DevicePackedCSR
// for the next compact-merge cycle.

#pragma once

#include "device_packed_csr.cuh"   // DevicePackedCSR, PackedEntry, packed_col/exp/make
#include "gpu_batch_merge.cuh"     // DeviceMergeWorkspace, BatchMergeResult, ROW_DEAD, ROW_WS_BIT
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace mpqs {
namespace matrix {

// =========================================================================
// Result type
// =========================================================================

/// Result of gpuCompactPackedCSR(): a fresh contiguous packed CSR with
/// per-cycle row and column maps for sqrt reconstruction composition.
struct CompactResult {
    DevicePackedCSR csr;                    ///< Fresh, contiguous, no dead rows.

    /// row_map[new_r] = pre-compact row index (0..original_n_rows-1).
    /// Maps compacted row index back to the row it came from in the
    /// scattered CSR. Used by M10b to compose the cumulative row map.
    std::vector<uint32_t> row_map;

    /// col_map[new_col] = old_col index (before column remapping).
    /// Maps new column indices back to old column indices. Used by M10b
    /// to compose cumulative_col_map for singleton_col_map reconstruction.
    std::vector<uint32_t> col_map;

    /// Number of columns alive in the GF(2) projection (i.e. columns with at
    /// least one odd-exponent entry across all alive rows). Computed by the
    /// K4-GF2 kernel in gpuCompactPackedCSR(). Used by M12-S2 column-diversity
    /// floor in gpuCompactMergeCycles().
    uint32_t gf2_n_cols = 0;
};

// =========================================================================
// Host driver
// =========================================================================

/// Compact a post-merge scattered DevicePackedCSR into a fresh contiguous CSR.
///
/// Takes the original DevicePackedCSR and the BatchMergeResult (containing the
/// DeviceMergeWorkspace with row indirection and workspace rows). Produces a
/// fresh DevicePackedCSR with:
///   - Only alive rows (d_row_ptr[r] != ROW_DEAD).
///   - Workspace rows resolved to inline entries.
///   - Columns remapped to remove empty columns.
///   - Per-row metadata (sqrt_Q, signs, val_2_exps) preserved.
///
/// The caller must NOT free csr or merge before this function returns.
/// The caller is responsible for freeing them after receiving CompactResult.
///
/// @param csr   Post-merge DevicePackedCSR (original rows, some dead via row_ptr).
/// @param merge BatchMergeResult from gpuBatchMerge(): workspace + h_row_ptr.
/// @return CompactResult with fresh CSR, row_map, col_map.
CompactResult gpuCompactPackedCSR(
    const DevicePackedCSR& csr,
    const BatchMergeResult& merge);

// =========================================================================
// Multi-cycle compact-merge driver (M10b)
// =========================================================================

/// Result of the full multi-cycle compact-merge driver.
struct CompactMergeResult {
    DevicePackedCSR  final_csr;              ///< Final compacted CSR (ready for GF2 extraction)
    BatchMergeResult final_merge;            ///< Last merge pass result (workspace + h_row_ptr)

    /// Cumulative row map: final_compacted_row → original pre-cycle-0 relation index
    /// (in post-singleton row space). Composed across all compaction cycles.
    std::vector<uint32_t> cumulative_row_map;

    /// Cumulative column map: final_compacted_col → post-singleton column index.
    /// Composed across all compaction cycles.
    std::vector<uint32_t> cumulative_col_map;

    uint32_t total_merges = 0;  ///< Sum of merges across all cycles + final pass
    uint32_t cycles_run   = 0;  ///< Number of compact-merge cycles executed (excl. final pass)
};

/// Multi-cycle compact-merge driver.
///
/// Runs merge → compact → merge → compact → ... until convergence, then performs
/// a final merge pass (no compaction). Each cycle starts with a fresh, contiguous
/// DevicePackedCSR and an accurate inverted index — avoiding the workspace
/// exhaustion and stale-state abort cascade of a single-pass merge.
///
/// Termination criteria (M12-S2):
///   1. Convergence: cycle_merges < 0.02 × alive_rows
///   2. Budget exhausted: cycle >= max_cycles
///   3. GF(2) column-diversity floor: post-compaction GF(2) col count drops
///      below max(gf2_min_floor, gf2_floor_factor × initial_gf2_cols).
///      Stops the cascade that destroys algebraic structure on high-LP data.
///
/// Per-cycle budget: computeCycleBudget() = n_rows - ceil(truncation_factor × n_cols).
///
/// @param csr               Post-singleton packed CSR.
///                          Consumed on call — the function takes ownership via internal move.
/// @param mont              Montgomery context for N (for sqrt_Q multiplication).
/// @param k_max             Maximum column weight for higher-weight merges.
/// @param max_weight        Fill-in limit for higher-weight merges.
/// @param truncation_factor Per-cycle budget factor; 0 = no budget cap.
/// @param max_cycles        Hard limit on compact-merge cycles (default 5).
/// @param gf2_floor_factor  Stop cycling when GF(2) cols fall below this fraction
///                          of the initial GF(2) col count. Default 0.5.
/// @param gf2_min_floor     Absolute minimum GF(2) col floor. Default 8192.
/// @return CompactMergeResult with final CSR, final merge result, and cumulative maps.
CompactMergeResult gpuCompactMergeCycles(
    DevicePackedCSR csr,             // moved in; caller passes std::move(singleton.reduced)
    const MontgomeryContext& mont,
    uint32_t k_max             = 10,
    uint32_t max_weight        = 200,
    double   truncation_factor = 1.05,
    uint32_t max_cycles        = 5,
    double   gf2_floor_factor  = 0.5,
    uint32_t gf2_min_floor     = 8192);

} // namespace matrix
} // namespace mpqs
