// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Stage 6 PARITY unit test for the M9v2 PACKED branch-fixed character propagation.
//
// The packed GPU reduction has NO merge tree: the per-row branch char vector is
// XOR-composed through two flat arrays exactly mirroring the Montgomery sqrt_Q
// product (src/matrix/{gpu_batch_merge,gpu_compact_packed,gpu_singleton_packed}.cu):
//   - ORIGINAL rows hold their seeded vector in d_char_bits (gpuBuildPackedMatrix E4),
//   - MERGED rows hold the XOR of their two constituents in d_ws_char_bits
//     (execute_merges_kernel: merged_char = resolve(r1) ^ resolve(r2)),
//   - the row indirection d_row_ptr selects between them by the ROW_WS_BIT MSB
//     (MSB=0 -> original index; MSB=1 -> workspace index = ptr & 0x7FFFFFFF),
//   - compaction RELOCATES rows (gather, no XOR),
//   - the FINAL gather at preprocess.cpp selects per alive row by the SAME
//     ptr_val/ws_idx selector and unpacks bit j of the composed vector into col j.
//
// This test reproduces that exact host-side gather/compose logic (the same code the
// device executes, run on the host over CPU-resident mirror arrays) and certifies it
// is BIT-FOR-BIT identical to the authoritative CPU merge-tree leaf-XOR oracle
// computeProductCharacterColumns(..., BRANCH) over the SAME logical merge structure,
// and to branchCharBit of the TRUE product field element (the Stage-3 genus reference).
//
//   (A) PACKED XOR-COMPOSE == CPU MERGE-TREE LEAF-XOR. For a controlled set of merges
//       (single rows, 2-leaf merges, nested 3-leaf merges, mixed smooth/partial), the
//       packed two-array compose + ptr_val/ws_idx gather yields, per alive row, the
//       same composed char vector as computeProductCharacterColumns(BRANCH).
//   (B) PACKED XOR-COMPOSE == GENUS REFERENCE. The same composed vector equals
//       branchCharBit of the product field element (XOR homomorphism, with multiplicity).
//   (C) NORM ORACLE GUARD. With all seeds 0 (norm mode), every composed/gathered vector
//       is exactly 0 — the packed propagation is a no-op for the byte-identical NORM path.
//
// Pure host test (no kernel launches); compiled as CUDA for the __host__ __device__
// math headers. Links mpqs_matrix (computeProductCharacterColumns, branchCharBit),
// mpqs_sieve (*_u64 primitives), mpqs_common (uint512 / HostRelationBatch).
//
// Exit code 0 iff every assertion passes (0 differing bits in all checks).

#include "character_columns.h"   // CharacterColumnComputer, computeProductCharacterColumns
#include "gpu_char_cols.cuh"     // branchCharBit, jacobi_symbol_dev_u64
#include "gpu_batch_merge.cuh"   // ROW_WS_BIT (the device selector constant)
#include "merge_tree.h"          // MergeTree
#include "mpqs_soa.h"            // HostRelationBatch
#include "uint128_helper.cuh"    // mul_mod, sub_mod
#include "uint512.cuh"           // uint512
#include "prime_algorithms.h"    // is_prime_u64, jacobi_u64, Tonelli_Shanks_u64

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using mpqs::matrix::CharMode;
using mpqs::matrix::CharacterColumns;
using mpqs::matrix::MergeTree;
using mpqs::matrix::computeProductCharacterColumns;
using mpqs::matrix::branchCharBit;
using mpqs::matrix::ROW_WS_BIT;
using mpqs::structures::HostRelationBatch;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) { ++g_failures;                                         \
        std::printf("  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
    }                                                                    \
} while (0)

// ---------------------------------------------------------------------------
// Toy aux-prime / relation model (shared shape with test_char_xor_propagation).
// ---------------------------------------------------------------------------
struct AuxPrimes {
    std::vector<uint64_t> q;
    std::vector<uint64_t> t_s;
    uint32_t k = 0;
};
struct ToyRel {
    std::vector<uint64_t> axb_mod_q;  // length k (signed (ax+b) mod q)
    uint32_t char_bits = 0;           // packed branch bits
};

static AuxPrimes makeAuxPrimes(const mpqs::uint512& N, uint64_t lp1_bound, uint32_t k) {
    AuxPrimes ap;
    ap.k = k;
    uint64_t cand = lp1_bound + 1;
    if ((cand & 1ull) == 0) ++cand;
    while (ap.q.size() < k) {
        if (mpqs::sieve::is_prime_u64(cand)) {
            uint64_t n_mod = N.mod_uint64(cand);
            if (mpqs::sieve::jacobi_u64(n_mod, cand) == 1) {
                uint64_t t = mpqs::sieve::Tonelli_Shanks_u64(n_mod, cand);
                if (t > cand - t) t = cand - t;  // canonical branch
                ap.q.push_back(cand);
                ap.t_s.push_back(t);
            }
        }
        cand += 2;
    }
    return ap;
}

static ToyRel makeRel(const AuxPrimes& ap, std::mt19937_64& rng) {
    ToyRel r;
    r.axb_mod_q.resize(ap.k);
    uint32_t cb = 0;
    for (uint32_t j = 0; j < ap.k; ++j) {
        std::uniform_int_distribution<uint64_t> dist(0, ap.q[j] - 1);
        uint64_t a = dist(rng);
        r.axb_mod_q[j] = a;
        cb |= ((uint32_t)branchCharBit(a, ap.t_s[j], ap.q[j])) << j;
    }
    r.char_bits = cb;
    return r;
}

// branchCharBit of the product field element of a SET of leaves (with multiplicity).
static int productBranchBit(const std::vector<const ToyRel*>& rels, const AuxPrimes& ap, uint32_t j) {
    uint64_t q = ap.q[j], ts = ap.t_s[j];
    uint64_t prod = 1;
    for (const ToyRel* r : rels) {
        uint64_t diff = mpqs::math::sub_mod(r->axb_mod_q[j], ts, q);
        prod = mpqs::math::mul_mod(prod, diff, q);
    }
    return (mpqs::matrix::jacobi_symbol_dev_u64(prod, q) == -1) ? 1 : 0;
}

static HostRelationBatch packBatch(const std::vector<ToyRel>& rels) {
    HostRelationBatch b;
    b.num_relations = rels.size();
    b.sqrt_Q.assign(rels.size(), mpqs::uint512((uint32_t)1));  // unused on BRANCH path
    b.char_bits.reserve(rels.size());
    for (const auto& r : rels) b.char_bits.push_back(r.char_bits);
    return b;
}

// ---------------------------------------------------------------------------
// Host mirror of the PACKED two-array char-vector reduction.
//
// d_char_bits      : composed vector for ORIGINAL rows (seeded; carries through
//                    singleton/compaction relocations unchanged).
// d_ws_char_bits   : composed vector for MERGED rows (= XOR of constituents).
// row_ptr          : indirection. MSB=0 -> orig index; MSB=1 -> ws index.
//
// resolveCharBits() is byte-identical to device resolve_char_bits() and to the
// preprocess.cpp final gather selection.
// ---------------------------------------------------------------------------
static uint32_t resolveCharBits(uint32_t ptr_val,
                                const std::vector<uint32_t>& d_char_bits,
                                const std::vector<uint32_t>& d_ws_char_bits) {
    if (ptr_val & ROW_WS_BIT) return d_ws_char_bits[ptr_val & 0x7FFFFFFFu];
    return d_char_bits[ptr_val];
}

// ---- (A)+(B)+(C) Packed compose/gather vs CPU merge-tree + genus reference ----
//
// Logical merge structure (mirrors test_char_xor_propagation merge-tree case):
//   ns=4 smooth leaves [0..3], np=3 partial leaves [4..6].
//   Reduced (alive) rows:
//     r0: leaf 0           (single smooth, original row)
//     r1: leaf 5           (single partial, original row)
//     r2: merge(1,2)       (2 smooth, workspace row)
//     r3: merge(3, merge(4,6))  (smooth + nested partials, depth 2, workspace row)
static void test_packed_compose_gather(uint32_t k, bool norm_mode) {
    int before = g_failures;
    mpqs::uint512 N((uint64_t)0x123456789ABCDEFull);
    if ((N % 2u) == 0) N = N + mpqs::uint512((uint32_t)1);
    AuxPrimes ap = makeAuxPrimes(N, 100000000000ull /*1e11*/, k);
    std::mt19937_64 rng(0xA11CE5EEDull + (norm_mode ? 1u : 0u));

    const size_t ns = 4, np = 3;
    std::vector<ToyRel> smooths, partials;
    for (size_t i = 0; i < ns; ++i) smooths.push_back(makeRel(ap, rng));
    for (size_t i = 0; i < np; ++i) partials.push_back(makeRel(ap, rng));

    // Seed the packed ORIGINAL-row array. In norm mode every seed is 0 (the Stage-4
    // placeholder), exactly as gpuBuildPackedMatrix seeds from the char_bits-zero view.
    // Original row layout = [smooth 0..3, partial 0..2] (gpuBuildPackedMatrix order).
    const uint32_t n_orig = static_cast<uint32_t>(ns + np);
    std::vector<uint32_t> d_char_bits(n_orig);
    for (uint32_t i = 0; i < ns; ++i)
        d_char_bits[i] = norm_mode ? 0u : smooths[i].char_bits;
    for (uint32_t i = 0; i < np; ++i)
        d_char_bits[ns + i] = norm_mode ? 0u : partials[i].char_bits;

    // Simulate the merge kernel: each merge writes a workspace row whose char vector is
    // the XOR of its two constituents resolved via row_ptr (orig or already-merged ws).
    std::vector<uint32_t> d_ws_char_bits;
    // row_ptr starts as identity over the n_orig original rows.
    std::vector<uint32_t> row_ptr(n_orig);
    for (uint32_t i = 0; i < n_orig; ++i) row_ptr[i] = i;

    auto doMerge = [&](uint32_t r1, uint32_t r2) {
        // resolve_char_bits via the current row_ptr (mirrors execute_merges_kernel)
        uint32_t cb1 = resolveCharBits(row_ptr[r1], d_char_bits, d_ws_char_bits);
        uint32_t cb2 = resolveCharBits(row_ptr[r2], d_char_bits, d_ws_char_bits);
        uint32_t slot = static_cast<uint32_t>(d_ws_char_bits.size());
        d_ws_char_bits.push_back(cb1 ^ cb2);          // merged_char = cb1 ^ cb2
        row_ptr[r1] = ROW_WS_BIT | slot;              // r1 -> workspace
        // r2 -> DEAD (not gathered); model by leaving it (never referenced again).
    };

    // merge(4,6) -> ws slot 0; row_ptr[4] points to it.
    doMerge(4u, 6u);
    // merge(3, 4) -> ws slot 1 (row_ptr[4] now resolves the nested ws row).
    doMerge(3u, 4u);
    // merge(1,2) -> ws slot 2.
    doMerge(1u, 2u);

    // Alive rows (the final reduced matrix) reference original rows 0, 5 and the
    // merged rows surviving at original-row slots 3 (=r3) and 1 (=r2).
    //   active_row_map[i] = the surviving original-row slot whose row_ptr is gathered.
    std::vector<uint32_t> active_row_map = {0u, 5u, 1u, 3u};
    // Expected leaf sets per alive row (with multiplicity) — the CPU oracle's view.
    std::vector<std::vector<uint32_t>> expected_leaves = {
        {0u}, {5u}, {1u, 2u}, {3u, 4u, 6u}};

    // FINAL gather (mirrors preprocess.cpp BRANCH gather + unpack).
    const uint32_t n_alive = static_cast<uint32_t>(active_row_map.size());
    std::vector<uint32_t> alive_char_bits(n_alive);
    for (uint32_t i = 0; i < n_alive; ++i)
        alive_char_bits[i] = resolveCharBits(row_ptr[active_row_map[i]],
                                             d_char_bits, d_ws_char_bits);

    // ----- Oracle 1: CPU merge-tree leaf-XOR via computeProductCharacterColumns -----
    HostRelationBatch sb = packBatch(smooths);
    HostRelationBatch pb = packBatch(partials);
    if (norm_mode) {  // norm-mode batches carry 0 char_bits (Stage-4 placeholder)
        for (auto& v : sb.char_bits) v = 0u;
        for (auto& v : pb.char_bits) v = 0u;
    }
    MergeTree tree;
    tree.num_leaves = static_cast<uint32_t>(ns + np);  // 7
    tree.internal_nodes.push_back({1u, 2u});  // global 7
    tree.internal_nodes.push_back({4u, 6u});  // global 8
    tree.internal_nodes.push_back({3u, 8u});  // global 9
    std::vector<uint32_t> tree_row_map = {0u, 5u, 7u, 9u};
    std::vector<uint32_t> fb;
    CharacterColumns cpu_cols = computeProductCharacterColumns(
        tree_row_map, tree, sb, pb, ns, N, fb,
        CharMode::BRANCH, /*lp1_bound=*/100000000000ull, /*k=*/ap.k);
    CHECK(cpu_cols.k == ap.k, "packed: CPU oracle k == requested k");

    auto leaf_rel = [&](uint32_t leaf) -> const ToyRel* {
        return (leaf < ns) ? &smooths[leaf] : &partials[leaf - ns];
    };

    int mism_cpu = 0, mism_ref = 0, mism_norm = 0;
    for (uint32_t i = 0; i < n_alive; ++i) {
        std::vector<const ToyRel*> rels;
        for (uint32_t leaf : expected_leaves[i]) rels.push_back(leaf_rel(leaf));
        for (uint32_t j = 0; j < ap.k; ++j) {
            int got = (alive_char_bits[i] >> j) & 1u;       // packed gathered bit
            int cpu = cpu_cols.columns[j][i];                // CPU merge-tree oracle
            if (got != cpu) ++mism_cpu;
            if (!norm_mode) {
                int ref = productBranchBit(rels, ap, j);     // genus reference
                if (got != ref) ++mism_ref;
            } else {
                if (got != 0) ++mism_norm;                   // norm: every bit 0
                if (cpu != 0) ++mism_norm;
            }
        }
    }

    if (norm_mode) {
        CHECK(mism_cpu == 0, "packed[norm]: gathered == CPU oracle (both all 0)");
        CHECK(mism_norm == 0, "packed[norm]: every composed/gathered bit is 0 (no-op)");
        std::printf("[%s] packed gather NORM: %u alive x %u primes; cpu_mism=%d norm_mism=%d\n",
                    (g_failures == before) ? "PASS" : "FAIL", n_alive, ap.k,
                    mism_cpu, mism_norm);
    } else {
        CHECK(mism_cpu == 0,
              "packed[branch]: gathered composed vector == CPU merge-tree leaf-XOR, bit-for-bit");
        CHECK(mism_ref == 0,
              "packed[branch]: gathered composed vector == branchCharBit of product field element");
        std::printf("[%s] packed gather BRANCH: %u alive x %u primes; cpu_mism=%d ref_mism=%d\n",
                    (g_failures == before) ? "PASS" : "FAIL", n_alive, ap.k,
                    mism_cpu, mism_ref);
    }
}

int main() {
    std::printf("=== test_packed_char_gather (Stage 6 packed propagation parity) ===\n");
    test_packed_compose_gather(/*k=*/32, /*norm_mode=*/false);  // (A)+(B)
    test_packed_compose_gather(/*k=*/32, /*norm_mode=*/true);   // (C) NORM oracle guard
    std::printf("---------------------------------------------------\n");
    std::printf("checks run: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures == 0) { std::printf("RESULT: PASS (0 failures)\n"); return 0; }
    std::printf("RESULT: FAIL (%d failures)\n", g_failures);
    return 1;
}
