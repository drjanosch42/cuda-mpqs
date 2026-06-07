// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/preprocess.cpp
#include "preprocess.h"
#include "gpu_singleton.cuh"
#include "gpu_packed_expanded.cuh"    // M9a: gpuBuildPackedMatrix
#include "gpu_singleton_packed.cuh"   // M9b: gpuRemoveSingletons_packed
#include "matrix_truncation.h"        // M9c (post-merge): truncateMatrix
#include "gpu_batch_merge.cuh"        // M9e: gpuBatchMerge, MontgomeryContext
#include "gpu_compact_packed.cuh"     // M10b: CompactMergeResult, gpuCompactMergeCycles
#include "gpu_gf2_extract.cuh"        // M9f: gpuExtractGF2
#include "gpu_product_char_packed.cuh" // M9f: gpuProductCharCols_packed
#include "character_columns.h"         // CharacterColumnComputer, AppendCharacterColumns
#include "montgomery.cuh"              // Montgomery
#include "cuda_check.h"
#include "hpc_logger.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>

namespace mpqs {
namespace matrix {

/// Resolve the effective backend, checking CUDA device availability.
/// Implementation detail — not exported in preprocess.h.
///
/// Resolution rules:
///   CPU:  always CPU (no CUDA API called)
///   GPU:  GPU if device found, else CPU with warning
///   AUTO: GPU if device found AND n_rows > 10000, else CPU
///         (below 10K rows kernel launch overhead dominates; CPU is faster)
static MatrixBackend resolveBackend(MatrixBackend requested, uint32_t n_rows) {
    if (requested == MatrixBackend::CPU) return MatrixBackend::CPU;

    // Verify CUDA device is available before selecting GPU or AUTO→GPU.
    int device_count = 0;
    cudaGetDeviceCount(&device_count);  // returns 0 on failure; no CUDA_CHECK needed
    if (device_count == 0) {
        LOG(LOG_WARNING) << "GPU matrix backend requested but no CUDA device found. "
                         << "Falling back to CPU.";
        return MatrixBackend::CPU;
    }

    if (requested == MatrixBackend::GPU) return MatrixBackend::GPU;

    // AUTO: use GPU only when the matrix is large enough to amortise kernel launch.
    if (n_rows > 10000) {
        LOG(LOG_INFO) << "Matrix backend AUTO: selecting GPU (n_rows=" << n_rows << ").";
        return MatrixBackend::GPU;
    }
    LOG(LOG_INFO) << "Matrix backend AUTO: selecting CPU (n_rows=" << n_rows
                  << " < 10K threshold).";
    return MatrixBackend::CPU;
}

PreprocessResult preprocessMatrix(
    const HostMatrixCSR& expanded,
    MatrixBackend backend,
    uint32_t k_max,
    uint32_t max_weight)
{
    LOG_SET_MODULE("Matrix");
    // M8d: resolve AUTO/GPU/CPU — checks cudaGetDeviceCount() for non-CPU backends.
    const MatrixBackend resolved = resolveBackend(backend, expanded.n_rows);

    if (resolved == MatrixBackend::GPU) {
        LOG(LOG_INFO) << "GPU singleton removal selected (n_rows=" << expanded.n_rows << ").";

        // M8b: GPU singleton removal fixpoint (5 CUDA kernels).
        auto sr = gpuRemoveSingletons(expanded);

        // Weight-2 and higher-weight merges remain on CPU (M3/M4).
        MergeFilterPipeline filter;
        auto w2 = filter.mergeWeight2(sr.reduced, sr.row_map);
        auto hw = filter.mergeHigherWeight(w2, k_max, max_weight);

        PreprocessResult res;
        res.reduced              = std::move(hw.reduced);
        res.row_map              = std::move(hw.row_map);
        res.merge_tree           = std::move(hw.merge_tree);
        res.singletons_removed   = sr.rows_removed;
        res.singleton_iterations = sr.iterations;
        res.w2_merges            = w2.merges_performed;
        res.hw_merges            = hw.merges_performed;
        return res;
    }

    // CPU path: backend == CPU, or AUTO with a small matrix (≤ 10K rows).
    MergeFilterPipeline filter;

    // M2: Singleton removal
    auto sr = filter.removeSingletons(expanded);

    // M3: Weight-2 merges
    auto w2 = filter.mergeWeight2(sr.reduced, sr.row_map);

    // M4: Higher-weight merges
    auto hw = filter.mergeHigherWeight(w2, k_max, max_weight);

    PreprocessResult result;
    result.reduced              = std::move(hw.reduced);
    result.row_map              = std::move(hw.row_map);
    result.merge_tree           = std::move(hw.merge_tree);
    result.singletons_removed   = sr.rows_removed;
    result.singleton_iterations = sr.iterations;
    result.w2_merges            = w2.merges_performed;
    result.hw_merges            = hw.merges_performed;
    return result;
}

// ============================================================================
// V2 packed pipeline (M9f)
// ============================================================================

std::vector<uint32_t> selectKernelVectorRows(
    const std::vector<uint64_t>& packed_bits,
    uint32_t num_reduced_rows,
    const std::vector<uint32_t>& row_map)
{
    std::vector<uint32_t> selected;
    for (uint32_t i = 0; i < num_reduced_rows; i++) {
        uint32_t word = i / 64;
        uint32_t bit  = i % 64;
        if (word < packed_bits.size() && (packed_bits[word] >> bit) & 1) {
            selected.push_back(row_map[i]);
        }
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

PreprocessResultV2 gpuPreprocessMatrix_packed(
    const structures::RelationBatchView& smooth_view,
    uint64_t n_smooth,
    const structures::RelationBatchView& partial_view,
    uint64_t n_partial,
    uint32_t fb_size,
    const uint512& N,
    const std::vector<uint32_t>& fb,
    uint32_t k_max,
    uint32_t max_weight,
    double truncation_factor,
    uint32_t compact_cycles,
    uint32_t truncation_excess,
    double gf2_floor_factor,
    uint32_t gf2_min_floor,
    CharMode char_mode,
    uint64_t lp1_bound)
{
    LOG_SET_MODULE("Matrix");
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    LOG(LOG_INFO) << "M9v2: Packed GPU preprocessing pipeline (solo mode).";
    LOG(LOG_INFO) << "  Smooth: " << n_smooth << ", Partial: " << n_partial
                  << ", FB: " << fb_size;

    // 1. [M9a] Build packed expanded CSR on device
    auto t_m9a = clock::now();
    auto expanded = gpuBuildPackedMatrix(smooth_view, n_smooth,
                                          partial_view, n_partial, fb_size);
    double m9a_ms = std::chrono::duration<double, std::milli>(clock::now() - t_m9a).count();
    LOG(LOG_INFO) << "  M9a: Packed CSR " << expanded.csr.n_rows << " x "
                  << expanded.csr.n_cols << " (NNZ=" << expanded.csr.nnz
                  << ", LP cols=" << expanded.n_lp_cols << ") [" << std::fixed
                  << std::setprecision(1) << m9a_ms << " ms].";

    // 2. [M9b] In-place packed singleton removal
    auto t_m9b = clock::now();
    auto singleton = gpuRemoveSingletons_packed(expanded.csr);
    double m9b_ms = std::chrono::duration<double, std::milli>(clock::now() - t_m9b).count();
    LOG(LOG_INFO) << "  M9b: Singleton removal: " << singleton.rows_removed << " rows removed ("
                  << singleton.iterations << " iters) → " << singleton.reduced.n_rows << " x "
                  << singleton.reduced.n_cols << " [" << std::fixed << std::setprecision(1)
                  << m9b_ms << " ms].";

    // 3. [M10b/M10c] Merge stage: compact-merge cycles (compact_cycles > 0)
    //    or single gpuBatchMerge pass (compact_cycles == 0, pre-M10 behavior).
    auto t_m9e = clock::now();
    math::Montgomery mont(N);
    auto mont_ctx = makeMontgomeryContext(mont);
    CompactMergeResult compact_merge;
    if (compact_cycles == 0) {
        // Single pass: no compaction, pre-M10 behavior.
        // gpuBatchMerge takes CSR by reference; move it into final_csr afterwards.
        auto merge = gpuBatchMerge(singleton.reduced, mont_ctx, k_max, max_weight, 0);
        uint32_t tot = merge.w2_merges + merge.hw_merges;
        std::vector<uint32_t> row_map(singleton.reduced.n_rows);
        std::vector<uint32_t> col_map(singleton.reduced.n_cols);
        for (uint32_t i = 0; i < row_map.size(); ++i) row_map[i] = i;
        for (uint32_t i = 0; i < col_map.size(); ++i) col_map[i] = i;
        compact_merge.final_csr          = std::move(singleton.reduced);
        compact_merge.final_merge        = std::move(merge);
        compact_merge.cumulative_row_map = std::move(row_map);
        compact_merge.cumulative_col_map = std::move(col_map);
        compact_merge.total_merges       = tot;
        compact_merge.cycles_run         = 0;
    } else {
        compact_merge = gpuCompactMergeCycles(
            std::move(singleton.reduced), mont_ctx, k_max, max_weight,
            truncation_factor, compact_cycles,
            gf2_floor_factor, gf2_min_floor);
    }
    // singleton.reduced is now invalid (moved); use compact_merge.final_csr going forward.
    double m9e_ms = std::chrono::duration<double, std::milli>(clock::now() - t_m9e).count();
    LOG(LOG_INFO) << "  M10b: Compact-merge: " << compact_merge.cycles_run
                  << " cycles, " << compact_merge.total_merges << " total merges → "
                  << compact_merge.final_csr.n_rows << " x " << compact_merge.final_csr.n_cols
                  << " [" << std::fixed << std::setprecision(1) << m9e_ms << " ms].";

    // GF(2) dimension estimate from final merge result
    {
        const auto& gf2_wt = compact_merge.final_merge.h_gf2_col_weight;
        uint32_t gf2_cols_final = 0;
        for (uint32_t c = 0; c < (uint32_t)gf2_wt.size(); ++c)
            if (gf2_wt[c] > 0) ++gf2_cols_final;
        uint32_t alive_rows = 0;
        for (uint32_t r = 0; r < (uint32_t)compact_merge.final_merge.h_row_ptr.size(); ++r)
            if (compact_merge.final_merge.h_row_ptr[r] != ROW_DEAD) ++alive_rows;
        LOG(LOG_INFO) << "  M11a: GF(2) estimate: " << alive_rows << " rows x "
                      << gf2_cols_final << " cols (packed: "
                      << compact_merge.final_csr.n_rows << " x "
                      << compact_merge.final_csr.n_cols << ").";
        if (alive_rows < gf2_cols_final) {
            LOG(LOG_WARNING) << "  M11a: GF(2) matrix underdetermined ("
                             << alive_rows << " rows < " << gf2_cols_final
                             << " cols). BW may find 0 solutions.";
        }
    }

    // 4. [M9f] GF(2) extraction
    auto t_gf2 = clock::now();
    auto gf2 = gpuExtractGF2(compact_merge.final_csr, compact_merge.final_merge);
    double gf2_ms = std::chrono::duration<double, std::milli>(clock::now() - t_gf2).count();
    LOG(LOG_INFO) << "  M9f: GF(2) extraction: " << gf2.gf2_csr.n_rows << " x "
                  << gf2.gf2_csr.n_cols << " (NNZ=" << gf2.gf2_nnz << ") ["
                  << std::fixed << std::setprecision(1) << gf2_ms << " ms].";

    // 4b. [M11b] Post-merge GF(2) singleton removal
    //     Merging may create GF(2) singletons not caught by the initial packed pass.
    //     Run CPU removeSingletons() on the host GF(2) CSR.
    {
        MergeFilterPipeline filter;
        auto sr = filter.removeSingletons(gf2.gf2_csr);
        if (sr.rows_removed > 0) {
            // Compose row maps: sr.row_map[i] indexes into gf2.gf2_csr,
            // gf2.row_map[j] maps gf2 row j to merged row index.
            std::vector<uint32_t> composed_row_map(sr.row_map.size());
            for (size_t i = 0; i < sr.row_map.size(); i++)
                composed_row_map[i] = gf2.row_map[sr.row_map[i]];

            gf2.gf2_csr = std::move(sr.reduced);
            gf2.row_map = std::move(composed_row_map);
            gf2.gf2_nnz = gf2.gf2_csr.row_offsets.back();

            LOG(LOG_INFO) << "  M11b: GF(2) singleton removal: " << sr.rows_removed
                          << " rows, " << sr.cols_removed << " cols removed → "
                          << gf2.gf2_csr.n_rows << " x " << gf2.gf2_csr.n_cols
                          << " (" << sr.iterations << " iters).";
        }
    }

    // 5. [M9c-post] Post-merge truncation (on host GF(2) CSR)
    //    Merges reduce the matrix first; truncation selects rows by coverage-greedy
    //    policy on the clean binary CSR from GF(2) extraction. Truncation runs
    //    *before* product char cols are appended, so the target accounts for the
    //    32 char cols via `n_extra_cols=32` (M12-S1).
    HostMatrixCSR* active_csr = &gf2.gf2_csr;
    std::vector<uint32_t>* active_row_map = &gf2.row_map;
    TruncationResult truncation;
    double m9c_ms = 0.0;
    if (truncation_factor > 0.0) {
        auto t_m9c = clock::now();
        constexpr uint32_t kCharColCount = 32;
        truncation = truncateMatrix(gf2.gf2_csr, gf2.row_map,
                                    truncation_factor,
                                    /*n_extra_cols=*/kCharColCount,
                                    /*k_excess=*/truncation_excess);
        m9c_ms = std::chrono::duration<double, std::milli>(clock::now() - t_m9c).count();
        if (truncation.rows_removed > 0) {
            active_csr = &truncation.truncated;
            active_row_map = &truncation.row_map;
            LOG(LOG_INFO) << "  M9c: Truncation: " << truncation.rows_removed << " rows, "
                          << truncation.cols_removed << " cols removed → "
                          << truncation.truncated.n_rows << " x " << truncation.truncated.n_cols
                          << " [" << std::fixed << std::setprecision(1) << m9c_ms << " ms].";
        }
    }

    // 6. [M9f] Character columns appended on the FINAL reduced rows.
    //    Two modes, selected by char_mode:
    //      NORM   : genus-blind norm symbol re-evaluated on the merged sqrt_Q
    //               (gpuProductCharCols_packed) — byte-identical to before.
    //      BRANCH : (Stage 6) the per-row branch char vector was XOR-composed through
    //               the packed reduction in lockstep with the Montgomery sqrt_Q product
    //               (seed in gpuBuildPackedMatrix → carried through singleton/merge/
    //               compaction via d_char_bits / d_ws_char_bits). Here we gather the
    //               composed vector for each alive row via the SAME ptr_val/ws_idx
    //               row-map gather as sqrt_Q, unpack its bits into the 32 columns, and
    //               append. NO symbol re-evaluation on a merged sqrt_Q (that is the
    //               norm bug). Bit-identical to computeProductCharacterColumns(BRANCH).
    auto t_char = clock::now();
    {
        // Collect merged sqrt_Q for alive rows into contiguous device array
        const auto& h_row_ptr = compact_merge.final_merge.h_row_ptr;
        const auto& workspace = compact_merge.final_merge.workspace;
        const uint32_t n_alive = static_cast<uint32_t>(active_row_map->size());

        if (char_mode == CharMode::NONE) {
            // --char_mode none (scientific null control): append zero character columns
            // so the reduced GF(2) CSR keeps exactly its FB(+LP) columns (k == 0). No
            // sqrt_Q gather, no aux-prime selection, no device Jacobi. Applies for any
            // n_alive (including the degenerate 0-row case, which needs no char cols).
            LOG(LOG_INFO) << "  M9f: Appended 0 char cols (--char_mode none, scientific "
                          << "control) → " << active_csr->n_rows << " x "
                          << active_csr->n_cols << ".";
        } else if (n_alive == 0) {
            LOG(LOG_WARNING) << "  GPU preprocess: degenerate matrix (0 alive rows). "
                             << "Skipping product char cols.";
        } else if (char_mode == CharMode::BRANCH) {
        // --- Stage 6 BRANCH path: gather composed d_char_bits + unpack 32 columns ---
        // Mirror the NORM sqrt_Q gather EXACTLY (same ptr_val / ROW_WS_BIT selector),
        // but over the char-vector arrays. The bits are already composed (XOR through
        // the reduction); we only select per-alive-row and unpack — no device Jacobi.
        uint64_t dual_counter_val = 0;
        CUDA_CHECK(cudaMemcpy(&dual_counter_val, workspace.d_dual_counter,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));
        uint32_t ws_row_count = static_cast<uint32_t>(dual_counter_val >> 32);

        std::vector<uint32_t> h_ws_char_bits(ws_row_count);
        if (ws_row_count > 0) {
            CUDA_CHECK(cudaMemcpy(h_ws_char_bits.data(), workspace.d_ws_char_bits,
                                  ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }

        std::vector<uint32_t> h_orig_char_bits(compact_merge.final_csr.n_rows);
        CUDA_CHECK(cudaMemcpy(h_orig_char_bits.data(), compact_merge.final_csr.d_char_bits,
                              compact_merge.final_csr.n_rows * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        // Gather the composed char vector per alive row (same row-map gather as sqrt_Q).
        std::vector<uint32_t> alive_char_bits(n_alive);
        for (uint32_t i = 0; i < n_alive; i++) {
            uint32_t merged_row = (*active_row_map)[i];
            uint32_t ptr_val = h_row_ptr[merged_row];
            if (ptr_val & ROW_WS_BIT) {
                uint32_t ws_idx = ptr_val & 0x7FFFFFFFu;
                alive_char_bits[i] = h_ws_char_bits[ws_idx];
            } else {
                alive_char_bits[i] = h_orig_char_bits[ptr_val];
            }
        }

        // Select branch aux primes to fix r (== column count); branch char bits are
        // already evaluated/composed, so only the count matters here. Matches the CPU
        // branch path (computeProductCharacterColumns BRANCH) and initBranchCharData.
        CharacterColumnComputer cc;
        cc.selectAuxPrimes(N, fb, CharMode::BRANCH, lp1_bound);
        const uint32_t k = static_cast<uint32_t>(cc.auxPrimes().size());

        // Unpack the composed bits into column-major CharacterColumns (bit j → col j),
        // identical to the CPU branch unpack in compute()/computeProductCharacterColumns.
        CharacterColumns branch_chars;
        branch_chars.aux_primes = cc.auxPrimes();
        branch_chars.k = k;
        branch_chars.columns.resize(k);
        for (uint32_t j = 0; j < k; ++j)
            branch_chars.columns[j].resize(n_alive, 0);
        for (uint32_t i = 0; i < n_alive; ++i) {
            const uint32_t cb = alive_char_bits[i];
            for (uint32_t j = 0; j < k; ++j)
                branch_chars.columns[j][i] = static_cast<uint8_t>((cb >> j) & 1u);
        }

        AppendCharacterColumns(*active_csr, branch_chars, n_alive);
        LOG(LOG_INFO) << "  M9f: Appended " << branch_chars.k
                      << " BRANCH char cols (XOR-composed via d_char_bits) → "
                      << active_csr->n_rows << " x " << active_csr->n_cols << ".";
        } else {
        // --- NORM path (default): byte-identical to before ---
        // Build contiguous device sqrt_Q array from alive rows
        // Download workspace sqrt_Q and original sqrt_Q as needed
        uint64_t dual_counter_val = 0;
        CUDA_CHECK(cudaMemcpy(&dual_counter_val, workspace.d_dual_counter,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));
        uint32_t ws_row_count = static_cast<uint32_t>(dual_counter_val >> 32);

        std::vector<uint512> h_ws_sqrt_Q(ws_row_count);
        if (ws_row_count > 0) {
            CUDA_CHECK(cudaMemcpy(h_ws_sqrt_Q.data(), workspace.d_ws_sqrt_Q,
                                  ws_row_count * sizeof(uint512), cudaMemcpyDeviceToHost));
        }

        std::vector<uint512> h_orig_sqrt_Q(compact_merge.final_csr.n_rows);
        CUDA_CHECK(cudaMemcpy(h_orig_sqrt_Q.data(), compact_merge.final_csr.d_sqrt_Q,
                              compact_merge.final_csr.n_rows * sizeof(uint512), cudaMemcpyDeviceToHost));

        // Build contiguous alive sqrt_Q
        std::vector<uint512> alive_sqrt_Q(n_alive);
        for (uint32_t i = 0; i < n_alive; i++) {
            uint32_t merged_row = (*active_row_map)[i];
            uint32_t ptr_val = h_row_ptr[merged_row];
            if (ptr_val & ROW_WS_BIT) {
                uint32_t ws_idx = ptr_val & 0x7FFFFFFFu;
                alive_sqrt_Q[i] = h_ws_sqrt_Q[ws_idx];
            } else {
                alive_sqrt_Q[i] = h_orig_sqrt_Q[ptr_val];
            }
        }

        // Upload to device for product char col kernel
        bool jetson = isJetsonDevice();
        uint512* d_alive_sqrt_Q = nullptr;
        if (jetson) { CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&d_alive_sqrt_Q), n_alive * sizeof(uint512))); }
        else        { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alive_sqrt_Q), n_alive * sizeof(uint512))); }
        CUDA_CHECK(cudaMemcpy(d_alive_sqrt_Q, alive_sqrt_Q.data(),
                              n_alive * sizeof(uint512), cudaMemcpyHostToDevice));

        // Select aux primes and compute product char cols.
        // NORM char mode: select with explicit NORM / lp1_bound=0 so selection is
        // byte-identical to today. Aux primes are stored 64-bit but stay < 2^32 in
        // NORM; the packed char kernel (uint32 aux primes) consumes a value-preserving
        // narrowed copy.
        CharacterColumnComputer cc;
        cc.selectAuxPrimes(N, fb, CharMode::NORM, /*lp1_bound=*/0);

        const std::vector<uint64_t>& aux_primes64 = cc.auxPrimes();
        std::vector<uint32_t> aux_primes32(aux_primes64.begin(), aux_primes64.end());

        auto product_chars = gpuProductCharCols_packed(
            d_alive_sqrt_Q, n_alive, aux_primes32, cc.nModQ());

        CUDA_CHECK(cudaFree(d_alive_sqrt_Q));

        // Append to GF(2) CSR
        AppendCharacterColumns(*active_csr, product_chars, n_alive);
        LOG(LOG_INFO) << "  M9f: Appended " << product_chars.k
                      << " product char cols → " << active_csr->n_rows
                      << " x " << active_csr->n_cols << ".";
        } // end n_alive > 0
    }
    double char_ms = std::chrono::duration<double, std::milli>(clock::now() - t_char).count();

    // 7. Download merged 1-partial data to host
    auto t_download = clock::now();
    PreprocessResultV2 result;
    result.reduced   = std::move(*active_csr);
    result.row_map   = std::move(*active_row_map);
    result.gf2_nnz   = gf2.gf2_nnz;
    result.fb_size              = fb_size;
    // Compose singleton_col_map with cumulative_col_map:
    //   result[c] = singleton.col_map[cumulative_col_map[c]]
    //   = compacted_col → post-singleton col → expanded col
    {
        const auto& cum_col = compact_merge.cumulative_col_map;
        result.singleton_col_map.resize(cum_col.size());
        for (uint32_t c = 0; c < cum_col.size(); ++c)
            result.singleton_col_map[c] = singleton.col_map[cum_col[c]];
    }
    result.singletons_removed   = singleton.rows_removed;
    result.singleton_iterations = singleton.iterations;
    result.w2_merges            = compact_merge.final_merge.w2_merges;
    result.hw_merges            = compact_merge.final_merge.hw_merges;

    {
        const auto& h_row_ptr = compact_merge.final_merge.h_row_ptr;
        const auto& workspace = compact_merge.final_merge.workspace;
        const uint32_t n_alive = static_cast<uint32_t>(result.row_map.size());

        // Download workspace metadata
        uint64_t dual_counter_val = 0;
        CUDA_CHECK(cudaMemcpy(&dual_counter_val, workspace.d_dual_counter,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));
        uint32_t ws_row_count   = static_cast<uint32_t>(dual_counter_val >> 32);
        uint32_t ws_entry_count = static_cast<uint32_t>(dual_counter_val & 0xFFFFFFFFu);

        // Download workspace arrays
        std::vector<uint512>  h_ws_sqrt_Q(ws_row_count);
        std::vector<uint8_t>  h_ws_signs(ws_row_count);
        std::vector<int32_t>  h_ws_val_2_exps(ws_row_count);
        std::vector<uint32_t> h_ws_row_starts(ws_row_count);
        std::vector<uint32_t> h_ws_row_lengths(ws_row_count);
        std::vector<PackedEntry> h_ws_entries(ws_entry_count);

        if (ws_row_count > 0) {
            CUDA_CHECK(cudaMemcpy(h_ws_sqrt_Q.data(), workspace.d_ws_sqrt_Q,
                                  ws_row_count * sizeof(uint512), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ws_signs.data(), workspace.d_ws_signs,
                                  ws_row_count * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ws_val_2_exps.data(), workspace.d_ws_val_2_exps,
                                  ws_row_count * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ws_row_starts.data(), workspace.d_ws_row_starts,
                                  ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ws_row_lengths.data(), workspace.d_ws_row_lengths,
                                  ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        if (ws_entry_count > 0) {
            CUDA_CHECK(cudaMemcpy(h_ws_entries.data(), workspace.d_ws_entries,
                                  ws_entry_count * sizeof(PackedEntry), cudaMemcpyDeviceToHost));
        }

        // Download original CSR metadata and entries (from final compacted CSR)
        std::vector<uint512>  h_orig_sqrt_Q(compact_merge.final_csr.n_rows);
        std::vector<uint8_t>  h_orig_signs(compact_merge.final_csr.n_rows);
        std::vector<int32_t>  h_orig_val_2_exps(compact_merge.final_csr.n_rows);
        std::vector<uint32_t> h_orig_row_offsets(compact_merge.final_csr.n_rows + 1);
        std::vector<PackedEntry> h_orig_entries(compact_merge.final_csr.nnz);

        CUDA_CHECK(cudaMemcpy(h_orig_sqrt_Q.data(), compact_merge.final_csr.d_sqrt_Q,
                              compact_merge.final_csr.n_rows * sizeof(uint512), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_orig_signs.data(), compact_merge.final_csr.d_signs,
                              compact_merge.final_csr.n_rows * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_orig_val_2_exps.data(), compact_merge.final_csr.d_val_2_exps,
                              compact_merge.final_csr.n_rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_orig_row_offsets.data(), compact_merge.final_csr.d_row_offsets,
                              (compact_merge.final_csr.n_rows + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (compact_merge.final_csr.nnz > 0) {
            CUDA_CHECK(cudaMemcpy(h_orig_entries.data(), compact_merge.final_csr.d_entries,
                                  compact_merge.final_csr.nnz * sizeof(PackedEntry), cudaMemcpyDeviceToHost));
        }

        // Build merged 1-partial data for sqrt
        result.merged_sqrt_Q.resize(n_alive);
        result.merged_signs.resize(n_alive);
        result.merged_val_2_exps.resize(n_alive);
        result.merged_factor_offsets.resize(n_alive + 1, 0);

        // First pass: count total entries to pre-allocate
        uint64_t total_merged_entries = 0;
        for (uint32_t i = 0; i < n_alive; i++) {
            uint32_t merged_row = result.row_map[i];
            uint32_t ptr_val = h_row_ptr[merged_row];
            if (ptr_val & ROW_WS_BIT) {
                uint32_t ws_idx = ptr_val & 0x7FFFFFFFu;
                total_merged_entries += h_ws_row_lengths[ws_idx];
            } else {
                total_merged_entries += h_orig_row_offsets[ptr_val + 1] - h_orig_row_offsets[ptr_val];
            }
        }

        result.merged_factor_indices.reserve(total_merged_entries);
        result.merged_factor_exponents.reserve(total_merged_entries);

        // Second pass: populate
        uint32_t factor_cursor = 0;
        for (uint32_t i = 0; i < n_alive; i++) {
            uint32_t merged_row = result.row_map[i];
            uint32_t ptr_val = h_row_ptr[merged_row];

            result.merged_factor_offsets[i] = factor_cursor;

            if (ptr_val & ROW_WS_BIT) {
                uint32_t ws_idx = ptr_val & 0x7FFFFFFFu;
                result.merged_sqrt_Q[i]     = h_ws_sqrt_Q[ws_idx];
                result.merged_signs[i]      = h_ws_signs[ws_idx];
                result.merged_val_2_exps[i] = h_ws_val_2_exps[ws_idx];

                uint32_t start = h_ws_row_starts[ws_idx];
                uint32_t len   = h_ws_row_lengths[ws_idx];
                for (uint32_t j = 0; j < len; j++) {
                    PackedEntry e = h_ws_entries[start + j];
                    result.merged_factor_indices.push_back(packed_col(e));
                    result.merged_factor_exponents.push_back(packed_exp(e));
                }
                factor_cursor += len;
            } else {
                result.merged_sqrt_Q[i]     = h_orig_sqrt_Q[ptr_val];
                result.merged_signs[i]      = h_orig_signs[ptr_val];
                result.merged_val_2_exps[i] = h_orig_val_2_exps[ptr_val];

                uint32_t begin = h_orig_row_offsets[ptr_val];
                uint32_t end   = h_orig_row_offsets[ptr_val + 1];
                for (uint32_t j = begin; j < end; j++) {
                    PackedEntry e = h_orig_entries[j];
                    result.merged_factor_indices.push_back(packed_col(e));
                    result.merged_factor_exponents.push_back(packed_exp(e));
                }
                factor_cursor += (end - begin);
            }
        }
        result.merged_factor_offsets[n_alive] = factor_cursor;
    }
    double download_ms = std::chrono::duration<double, std::milli>(clock::now() - t_download).count();

    // Compose active_row_map with cumulative_row_map (after download block uses raw indices):
    //   result[i] = cumulative_row_map[result.row_map[i]]
    //   = GF2/truncated row → compacted_row → original relation index
    {
        const auto& cum_rmap = compact_merge.cumulative_row_map;
        for (uint32_t i = 0; i < static_cast<uint32_t>(result.row_map.size()); ++i)
            result.row_map[i] = cum_rmap[result.row_map[i]];
    }

    double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_start).count();
    LOG(LOG_INFO) << "M10b: Packed preprocessing complete: "
                  << result.reduced.n_rows << " x " << result.reduced.n_cols
                  << " in " << std::fixed << std::setprecision(1) << total_ms << " ms"
                  << " (cycles:" << compact_cycles
                  << " M9a:" << std::setprecision(0) << m9a_ms
                  << " M9b:" << m9b_ms
                  << " M10b:" << m9e_ms
                  << " GF2:" << gf2_ms
                  << " post-trunc:" << m9c_ms
                  << " char:" << char_ms
                  << " dl:" << download_ms << " ms).";

    return result;
}

} // namespace matrix
} // namespace mpqs
