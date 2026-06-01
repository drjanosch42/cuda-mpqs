// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_batch_merge.cuh
//
// Batch-planned GPU merge execution (M9e).
// CPU plans merge candidates from column weight arrays; GPU executes non-conflicting
// merges in parallel via atomicCAS row ownership + atomic_reserve_dual workspace
// allocation. Packed two-pointer merge with exponent addition replaces binary XOR;
// Montgomery sqrt_Q multiplication maintains per-row 1-partial metadata.
//
// No merge tree — merged rows carry sqrt_Q products directly for sqrt consumption.

#pragma once

#include "device_packed_csr.cuh"  // DevicePackedCSR, DevicePackedView, PackedEntry, etc.
#include "matrix_constructor.h"    // HostMatrixCSR
#include "montgomery.cuh"          // Montgomery
#include "uint512.cuh"             // uint512
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace mpqs {
namespace matrix {

// =========================================================================
// Types
// =========================================================================

/// A single merge operation: merge row r1 and r2, eliminating pivot_col.
struct MergeCandidate {
    uint32_t r1;         ///< First row (survives as merge result)
    uint32_t r2;         ///< Second row (consumed, marked DEAD)
    uint32_t pivot_col;  ///< Column being eliminated
};

/// Kernel-passable Montgomery context (POD subset of Montgomery class).
/// Constructed on host from existing Montgomery class, passed by value to kernels.
/// ~136 bytes: 2 × uint512 (128 bytes) + 1 × uint32_t (4 bytes) + padding.
struct MontgomeryContext {
    uint512  N;         ///< Modulus
    uint512  R2;        ///< R^2 mod N (for transform)
    uint32_t n_prime;   ///< -N^(-1) mod 2^32 (for CIOS reduction)

    /// Montgomery multiplication: (a * b * R^(-1)) mod N.
    /// CIOS algorithm, identical to Montgomery::mul().
    __device__ uint512 mul(const uint512& a, const uint512& b) const;

    /// Transform x to Montgomery form: x * R mod N.
    __device__ uint512 transform(const uint512& x) const {
        return mul(x, R2);
    }

    /// Reduce from Montgomery form: x * R^(-1) mod N.
    __device__ uint512 reduce(const uint512& x) const {
        uint512 one((uint32_t)1);
        return mul(x, one);
    }
};

/// Construct MontgomeryContext from existing Montgomery class.
MontgomeryContext makeMontgomeryContext(const math::Montgomery& mont);

// =========================================================================
// Workspace
// =========================================================================

/// Sentinel for dead rows in d_row_ptr.
static constexpr uint32_t ROW_DEAD = 0xFFFFFFFFu;
/// MSB flag indicating d_row_ptr points to workspace (not original CSR).
static constexpr uint32_t ROW_WS_BIT = 0x80000000u;

/// Device workspace for GPU batch merges. Append-only via bump allocator.
/// RAII: destructor frees all non-null device pointers.
struct DeviceMergeWorkspace {
    // Workspace CSR (append-only, for merged rows)
    PackedEntry* d_ws_entries     = nullptr;  ///< Flat array of merged row entries
    uint32_t*    d_ws_row_starts  = nullptr;  ///< [max_merged_rows] start offset per merged row
    uint32_t*    d_ws_row_lengths = nullptr;  ///< [max_merged_rows] length per merged row

    // Per-row metadata for merged rows
    uint512*     d_ws_sqrt_Q      = nullptr;  ///< [max_merged_rows]
    uint8_t*     d_ws_signs       = nullptr;  ///< [max_merged_rows]
    int32_t*     d_ws_val_2_exps  = nullptr;  ///< [max_merged_rows]

    // Bump allocator: packed uint64 = [uint32 row_count | uint32 entry_count]
    uint64_t*    d_dual_counter   = nullptr;

    // Row indirection: maps logical row → physical location
    //   MSB=0: original CSR row index → read from DevicePackedCSR
    //   MSB=1: workspace row index (index & 0x7FFFFFFF) → read from ws_*
    //   ROW_DEAD (0xFFFFFFFF): row consumed by merge
    uint32_t*    d_row_ptr        = nullptr;  ///< [n_total_rows]

    // Row ownership locks for atomicCAS conflict resolution
    uint32_t*    d_row_locks      = nullptr;  ///< [n_total_rows], 0 = free

    // Column weight array (maintained across rounds; recomputed from CSR at init)
    uint32_t*    d_col_weight     = nullptr;  ///< [n_cols]
    uint32_t*    d_gf2_col_weight = nullptr;  ///< [n_cols] GF(2) column weights (odd-exp only)

    // Abort count per kernel launch
    uint32_t*    d_abort_count    = nullptr;

    // Dimensions and capacities
    uint32_t     max_merged_rows   = 0;
    uint32_t     max_merged_entries = 0;
    uint32_t     n_total_rows      = 0;  ///< Original row count (for row_ptr/locks sizing)
    uint32_t     n_cols            = 0;
    bool         use_managed       = false;

    DeviceMergeWorkspace() = default;
    ~DeviceMergeWorkspace();

    DeviceMergeWorkspace(const DeviceMergeWorkspace&) = delete;
    DeviceMergeWorkspace& operator=(const DeviceMergeWorkspace&) = delete;

    // Movable (for returning from gpuBatchMerge)
    DeviceMergeWorkspace(DeviceMergeWorkspace&& other) noexcept;
    DeviceMergeWorkspace& operator=(DeviceMergeWorkspace&& other) noexcept;

    /// Allocate all workspace buffers.
    /// @param n_rows     Number of rows in post-singleton CSR (for row_ptr, locks).
    /// @param n_cols     Number of columns (for col_weight).
    /// @param est_merges Estimated merge count (~155K for RSA-110). Allocates 1.5× for safety.
    /// @param est_entries Estimated merged entry count. Allocates 1.5× for safety.
    /// @param jetson     Use cudaMallocManaged if true.
    void alloc(uint32_t n_rows, uint32_t n_cols,
               uint32_t est_merges, uint32_t est_entries, bool jetson);

    /// Initialize d_row_ptr to identity (row i → i) and d_row_locks to 0.
    /// Must be called after alloc() and before first merge round.
    void initRowState(uint32_t n_rows, cudaStream_t stream = 0);
};

// =========================================================================
// Merge planner (CPU-side)
// =========================================================================

/// A batch of merge candidates planned by the CPU for GPU execution.
struct MergePlan {
    std::vector<MergeCandidate> candidates;
};

/// Plan weight-2 merge candidates.
/// Scans h_col_weight for weight-2 columns, looks up containing rows via inverted index.
/// @param h_col_weight  Host column weight array [n_cols].
/// @param n_cols        Number of columns.
/// @param col_rows      Inverted index: col → list of alive rows containing it.
/// @param h_row_ptr     Host copy of d_row_ptr (for alive/dead check).
/// @return MergePlan with candidates. Empty plan = no more weight-2 columns.
MergePlan planWeight2Merges(
    const std::vector<uint32_t>& h_col_weight,
    uint32_t n_cols,
    const std::vector<std::vector<uint32_t>>& col_rows,
    const std::vector<uint32_t>& h_row_ptr);

/// Plan higher-weight (Markowitz) merge candidates.
/// For each column with weight in [3, k_max], selects the lightest row as pivot,
/// emits (other_row, pivot, col) for each other row sharing the column.
/// @param h_col_weight  Host column weight array.
/// @param n_cols        Number of columns.
/// @param k_max         Maximum column weight to process.
/// @param max_fill_in   Maximum merged row length (fill-in limit).
/// @param col_rows      Inverted index.
/// @param h_row_ptr     Host copy of d_row_ptr.
/// @param row_weights   Current row weights [n_rows] (for Markowitz pivot selection).
/// @return MergePlan with candidates. Empty plan = no more eligible columns.
MergePlan planHigherWeightMerges(
    const std::vector<uint32_t>& h_col_weight,
    uint32_t n_cols,
    uint32_t k_max,
    uint32_t max_fill_in,
    const std::vector<std::vector<uint32_t>>& col_rows,
    const std::vector<uint32_t>& h_row_ptr,
    const std::vector<uint32_t>& row_weights);

// =========================================================================
// Host driver
// =========================================================================

/// Result of the batch GPU merge phase.
struct BatchMergeResult {
    // Workspace ownership transferred to caller.
    DeviceMergeWorkspace workspace;

    // Host-side copies for downstream consumption (downloaded after all merge rounds)
    std::vector<uint32_t> h_row_ptr;      ///< Final row_ptr state
    std::vector<uint32_t> h_col_weight;   ///< Final column weights
    std::vector<uint32_t> h_gf2_col_weight;  ///< Final GF(2) column weights (odd-exp entries only)

    // Diagnostics
    uint32_t w2_merges    = 0;
    uint32_t hw_merges    = 0;
    uint32_t total_rounds = 0;
    uint32_t total_aborts = 0;
};

/// Run the full batch-planned GPU merge pipeline.
///
/// CPU plans merge batches, GPU executes in parallel. Weight-2 merges first,
/// then higher-weight (Markowitz) merges. The original DevicePackedCSR is NOT
/// modified — merged rows live in the workspace. d_row_ptr tracks indirection.
///
/// @param csr           Post-singleton (and optionally post-truncation) packed CSR.
///                      NOT consumed — remains valid for row reads during merges.
/// @param mont          Montgomery context for N (for sqrt_Q multiplication).
/// @param k_max         Maximum column weight for higher-weight merges (default 10).
/// @param max_weight    Fill-in limit for higher-weight merges (default 200).
/// @param max_total_merges  Budget cap: stop after this many total merges (0 = unlimited).
///                          Prevents over-merging below n_cols (underdetermined system).
/// @return BatchMergeResult with workspace, row_ptr, col_weight, diagnostics.
BatchMergeResult gpuBatchMerge(
    DevicePackedCSR& csr,
    const MontgomeryContext& mont,
    uint32_t k_max = 10,
    uint32_t max_weight = 200,
    uint32_t max_total_merges = 0);

} // namespace matrix
} // namespace mpqs
