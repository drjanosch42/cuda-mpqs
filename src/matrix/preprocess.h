// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/preprocess.h
#pragma once

#include "matrix_constructor.h"  // HostMatrixCSR
#include "merge_tree.h"          // MergeTree
#include "merge_filter.h"        // SingletonResult, MergeResult, MergeFilterPipeline
#include "uint512.cuh"           // uint512
#include "mpqs_soa.h"            // RelationBatchView
#include <vector>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Backend selection for matrix preprocessing.
enum class MatrixBackend { CPU, GPU, AUTO };

/// Unified result of the M2–M4 preprocessing pipeline.
/// Produced by both CPU and GPU backends — identical fields, identical semantics.
struct PreprocessResult {
    HostMatrixCSR reduced;               ///< Reduced CSR (post-merges, pre-char cols)
    std::vector<uint32_t> row_map;       ///< reduced_row → merge tree node index
    MergeTree merge_tree;                ///< Merge history for sqrt reconstruction

    // Diagnostics (for logging)
    uint32_t singletons_removed   = 0;  ///< Rows removed by singleton passes
    uint32_t singleton_iterations = 0;  ///< Fixpoint iterations for singleton removal
    uint32_t w2_merges            = 0;  ///< Weight-2 columns eliminated
    uint32_t hw_merges            = 0;  ///< Higher-weight columns eliminated
};

/// Run the full M2–M4 preprocessing pipeline on an expanded CSR matrix.
/// Routes through CPU or GPU backend based on `backend`.
///
/// Pipeline: removeSingletons → mergeWeight2 → mergeHigherWeight
///
/// The caller is responsible for:
///   1. Building the expanded matrix (ExpandedMatrixBuilder)
///   2. Applying matrix truncation (optional, post-preprocessing)
///   3. Computing and appending character columns (post-preprocessing)
///   4. Converting to HostMatrix for BW (ConvertFromCSR)
///
/// @param expanded   Expanded (F+2+L)-column CSR from ExpandedMatrixBuilder.
/// @param backend    CPU, GPU, or AUTO (GPU if available + matrix large enough).
/// @param k_max      Maximum column weight for higher-weight merges (default 10).
/// @param max_weight Fill-in limit for higher-weight merges (default 200).
/// @return PreprocessResult with reduced matrix, merge tree, and diagnostics.
PreprocessResult preprocessMatrix(
    const HostMatrixCSR& expanded,
    MatrixBackend backend   = MatrixBackend::CPU,
    uint32_t k_max          = 10,
    uint32_t max_weight     = 200);

// ============================================================================
// Preprocessing path overview
// ============================================================================
//
// Three preprocessing paths are active in this codebase. The correct path is
// selected automatically in orchestrator.cpp based on --matrix_mode and
// --matrix_backend flags.
//
// V1 CPU  (legacy + CPU backend)
//   Function: preprocessMatrix(..., MatrixBackend::CPU)
//   Pipeline: removeSingletons → mergeWeight2 → mergeHigherWeight
//   Used for: default matrix mode, RSA-120+, any --matrix_mode legacy run.
//   Output:   PreprocessResult (with MergeTree for sqrt reconstruction).
//
// V1 GPU hybrid  (legacy + GPU backend)
//   Function: preprocessMatrix(..., MatrixBackend::GPU or AUTO)
//   Pipeline: gpuRemoveSingletons (5 CUDA kernels) → CPU mergeWeight2/mergeHigherWeight
//   Used for: --matrix_backend gpu with legacy matrix mode.
//   Output:   PreprocessResult (same interface as V1 CPU).
//
// V2 packed GPU  (preprocess mode, M9v2)
//   Function: gpuPreprocessMatrix_packed(...)
//   Pipeline: gpuBuildPackedMatrix → gpuRemoveSingletons_packed → gpuBatchMerge ×N →
//             gpuCompactMergeCycles → gpuExtractGF2 → post-merge GF(2) singleton removal →
//             truncateMatrix → gpuProductCharCols_packed
//   Used for: --matrix_mode preprocess (LP fraction < --lp_preprocess_threshold).
//   Output:   PreprocessResultV2 (no MergeTree; sqrt uses merged_sqrt_Q directly).
//

// ============================================================================
// V2 packed pipeline (M9f): merged 1-partials, no merge tree
// ============================================================================

/// Unified result of the full GPU preprocessing pipeline (M9v2).
/// Carries merged 1-partial data for direct sqrt consumption — no merge tree.
struct PreprocessResultV2 {
    HostMatrixCSR reduced;               ///< GF(2) CSR for BW solver (odd-exponent filter)
    std::vector<uint32_t> row_map;       ///< reduced_row → merged_row logical index

    // Merged 1-partial data for sqrt consumption (replaces merge tree)
    std::vector<uint512>  merged_sqrt_Q;           ///< [n_merged_rows]
    std::vector<uint8_t>  merged_signs;            ///< [n_merged_rows]
    std::vector<int32_t>  merged_val_2_exps;       ///< [n_merged_rows]
    std::vector<uint32_t> merged_factor_offsets;    ///< [n_merged_rows + 1] CSR into merged factors
    std::vector<uint32_t> merged_factor_indices;    ///< [nnz_merged] column indices
    std::vector<uint8_t>  merged_factor_exponents;  ///< [nnz_merged] full exponents

    // Column mapping (needed for sqrt FB/LP discrimination)
    uint32_t fb_size = 0;                        ///< Factor base size (columns [2..fb_size+1] = FB)
    std::vector<uint32_t> singleton_col_map;     ///< post-singleton col → expanded col (from M9b)

    // Diagnostics
    uint32_t singletons_removed   = 0;
    uint32_t singleton_iterations = 0;
    uint32_t w2_merges            = 0;
    uint32_t hw_merges            = 0;
    uint32_t gf2_nnz              = 0;
};

/// Select merged rows from a BW kernel vector, without merge tree expansion.
/// For each set bit i in packed_bits (i < num_reduced_rows), row_map[i]
/// gives the merged row index. Returns the sorted list of merged row indices.
///
/// Replaces expandKernelVector() for the V2 pipeline.
///
/// @param packed_bits       BW solution: bit i set → reduced row i selected.
/// @param num_reduced_rows  Number of rows in the GF(2) matrix.
/// @param row_map           reduced_row → merged_row index.
/// @return Sorted vector of merged row indices.
std::vector<uint32_t> selectKernelVectorRows(
    const std::vector<uint64_t>& packed_bits,
    uint32_t num_reduced_rows,
    const std::vector<uint32_t>& row_map);

/// Run the full M9v2 packed GPU preprocessing pipeline.
/// Chains: M9a (packed CSR) → M9b (singleton) → M9e (GPU batch merges) →
///         M9f (GF(2) extract) → M11b (post-merge GF(2) singleton) →
///         M9c-post (truncation) → M9f (product char cols).
///
/// Solo mode only. Cluster mode uses the CPU pipeline.
///
/// @param smooth_view        Device-resident smooth relation batch.
/// @param n_smooth           Number of smooth relations.
/// @param partial_view       Device-resident partial (LP witness) batch.
/// @param n_partial          Number of partial relations.
/// @param fb_size            Factor base size.
/// @param N                  The composite being factored (for Montgomery context).
/// @param fb                 Factor base primes (for character column aux prime selection).
/// @param k_max              Maximum column weight for higher-weight merges (default 10).
/// @param max_weight         Fill-in limit (default 200).
/// @param truncation_factor  > 0 enables M9c-post truncation, 0 disables it.
/// @param compact_cycles     Max compact-merge cycles (default 5; 0 = single pass, no compaction).
/// @param truncation_excess  Excess rows over (n_cols + 32) at truncation. Default 200.
/// @param gf2_floor_factor   M12-S2 GF(2) col-diversity floor as fraction of
///                           the initial post-singleton GF(2) col count. Default 0.5.
/// @param gf2_min_floor      M12-S2 absolute lower bound on the GF(2) col floor. Default 8192.
/// @return PreprocessResultV2 with GF(2) CSR + merged 1-partial data.
PreprocessResultV2 gpuPreprocessMatrix_packed(
    const structures::RelationBatchView& smooth_view,
    uint64_t n_smooth,
    const structures::RelationBatchView& partial_view,
    uint64_t n_partial,
    uint32_t fb_size,
    const uint512& N,
    const std::vector<uint32_t>& fb,
    uint32_t k_max = 10,
    uint32_t max_weight = 200,
    double truncation_factor = 1.05,
    uint32_t compact_cycles = 5,
    uint32_t truncation_excess = 200,
    double gf2_floor_factor = 0.5,
    uint32_t gf2_min_floor = 8192);

} // namespace matrix
} // namespace mpqs
