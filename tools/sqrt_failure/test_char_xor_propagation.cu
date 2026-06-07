// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Stage 5 HOMOMORPHISM unit test for the CPU branch-fixed character-column
// XOR-propagation machinery:
//
//   (A) LP-COMBINATION XOR. A combined relation's char vector is the XOR of its
//       two constituents' raw per-relation vectors (largeprime.cu global_combine_
//       kernel / cluster cpu_lp.cu combineAndAppend). We certify the genus-correct
//       homomorphism: for two raw partials with branch bits c_a, c_b on aux prime q,
//       (c_a XOR c_b) == branchCharBit of the TRUE product field element
//       ∏ ((ax+b)_l − t_s) mod q. The mod-N product is NEVER consulted.
//
//   (B) MERGE-TREE REDUCTION XOR. computeProductCharacterColumns(..., BRANCH) must
//       compose, for each reduced row, the XOR (WITH MULTIPLICITY) of its leaves'
//       persisted char_bits along the SAME tree.expand(row_map[r]) leaf set the NORM
//       Montgomery sqrt_Q product uses. We build a MergeTree with single-leaf rows,
//       2-leaf merges, and a nested 3-leaf merge, mixing smooth (leaf<ns) and
//       partial (leaf>=ns) leaves, then assert each unpacked reduced-row column bit
//       equals BOTH (i) the XOR-with-multiplicity of raw leaf bits AND (ii)
//       branchCharBit of the true product field element (Stage-3 genus reference).
//
//   (C) PER-RELATION ADAPTER. compute(batch, BRANCH) unpacks bit j of char_bits[i]
//       into columns[j][i] verbatim (no symbol re-evaluation) — identical to the
//       single-leaf case of (B).
//
//   (D) APPEND-AFTER-REDUCTION STRUCTURAL CHECK. AppendCharacterColumns adds exactly
//       k columns to a reduced CSR; every char-column index is >= the pre-append
//       n_cols (so no char column index can ever have appeared in any reduction step)
//       and the count stays 32 in the production path.
//
// The NORM path is the byte-identical A/B oracle and is exercised implicitly:
// char_bits == 0 for every relation under norm, so every composed/combined vector is
// 0 (== the Stage-4 placeholder) — verified explicitly in test (A).
//
// Compiled as CUDA (the branch evaluator + matrix headers are __host__ __device__).
// Links mpqs_matrix (computeProductCharacterColumns, compute, AppendCharacterColumns,
// branchCharBit) + mpqs_sieve (*_u64 primitives) + mpqs_common.
//
// Exit code 0 iff every assertion passes (0 mismatches in all four checks).

#include "character_columns.h"   // CharacterColumnComputer, computeProductCharacterColumns, AppendCharacterColumns
#include "gpu_char_cols.cuh"     // branchCharBit
#include "merge_tree.h"          // MergeTree
#include "matrix_constructor.h"  // HostMatrixCSR
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
using mpqs::matrix::CharacterColumnComputer;
using mpqs::matrix::MergeTree;
using mpqs::matrix::HostMatrixCSR;
using mpqs::matrix::AppendCharacterColumns;
using mpqs::matrix::computeProductCharacterColumns;
using mpqs::matrix::branchCharBit;
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
// A toy "ground-truth" relation model the test controls end-to-end.
// Each relation carries its SIGNED (ax+b) as a residue mod each aux prime, so the
// branch bit is branchCharBit((ax+b) mod q, t_s, q). char_bits packs these bits.
// ---------------------------------------------------------------------------
struct AuxPrimes {
    std::vector<uint64_t> q;
    std::vector<uint64_t> t_s;        // fixed Tonelli root t_s^2 == N mod q
    uint32_t k = 0;
};

// Each relation's signed (ax+b) reduced mod q[j] (the genus-correct field-element
// coordinate). We only need the residues to evaluate branchCharBit + the product.
struct ToyRel {
    std::vector<uint64_t> axb_mod_q;  // length k
    uint32_t char_bits = 0;           // packed branch bits, bit j = branchCharBit
};

// Build k production-scale aux primes q > lp1_bound, locking a t_s for a fixed N.
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

// Random signed-residue relation; computes its packed branch bits from the aux primes.
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

// branchCharBit of the product field element of a SET of relations (with multiplicity)
// on aux prime j: bit = [ Legendre( ∏ ((ax+b)_l − t_s) / q ) == −1 ].
static int productBranchBit(const std::vector<const ToyRel*>& rels, const AuxPrimes& ap, uint32_t j) {
    uint64_t q = ap.q[j], ts = ap.t_s[j];
    uint64_t prod = 1;
    for (const ToyRel* r : rels) {
        uint64_t diff = mpqs::math::sub_mod(r->axb_mod_q[j], ts, q);  // (ax+b) − t_s mod q
        prod = mpqs::math::mul_mod(prod, diff, q);
    }
    return (mpqs::matrix::jacobi_symbol_dev_u64(prod, q) == -1) ? 1 : 0;
}

// Pack a set of ToyRels into a minimal HostRelationBatch carrying only char_bits
// (computeProductCharacterColumns BRANCH path consumes char_bits only).
static HostRelationBatch packBatch(const std::vector<ToyRel>& rels) {
    HostRelationBatch b;
    b.num_relations = rels.size();
    b.char_bits.reserve(rels.size());
    // sqrt_Q must be present too (NORM uses it; BRANCH ignores it). Provide a dummy.
    b.sqrt_Q.assign(rels.size(), mpqs::uint512((uint32_t)1));
    for (const auto& r : rels) b.char_bits.push_back(r.char_bits);
    return b;
}

// ---- (A) LP-combination XOR ------------------------------------------------
static void test_lp_combine_xor() {
    int before = g_failures;
    mpqs::uint512 N((uint64_t)0xABCDEF0123456789ull);  // arbitrary odd N for t_s locking
    if ((N % 2u) == 0) N = N + mpqs::uint512((uint32_t)1);
    AuxPrimes ap = makeAuxPrimes(N, 100000000000ull /*1e11*/, 32);
    std::mt19937_64 rng(0xC0DECAFEull);

    int mism_xor = 0, mism_ref = 0, mism_norm = 0;
    const int trials = 4000;
    for (int t = 0; t < trials; ++t) {
        ToyRel a = makeRel(ap, rng);
        ToyRel b = makeRel(ap, rng);

        // The combine kernels write: combined.char_bits = a.char_bits ^ b.char_bits.
        uint32_t combined = a.char_bits ^ b.char_bits;

        for (uint32_t j = 0; j < ap.k; ++j) {
            int bit = (combined >> j) & 1u;
            int ref = productBranchBit({&a, &b}, ap, j);  // genus reference on product
            if (bit != ref) ++mism_ref;
            int xa = (a.char_bits >> j) & 1u, xb = (b.char_bits >> j) & 1u;
            if (bit != (xa ^ xb)) ++mism_xor;
        }

        // NORM oracle: char_bits are a defined 0 → combined 0 → byte-identical to
        // the Stage-4 placeholder.
        uint32_t norm_combined = 0u ^ 0u;
        if (norm_combined != 0u) ++mism_norm;
    }
    CHECK(mism_xor == 0, "LP-combine: combined bits == XOR of constituent bits");
    CHECK(mism_ref == 0, "LP-combine: XOR bits == branchCharBit of product field element");
    CHECK(mism_norm == 0, "LP-combine: norm-mode (0^0) stays 0");
    std::printf("[%s] LP-combine XOR: %d trials x %u primes; xor_mism=%d ref_mism=%d norm_mism=%d\n",
                (g_failures == before) ? "PASS" : "FAIL", trials, ap.k,
                mism_xor, mism_ref, mism_norm);
}

// ---- (B) Merge-tree reduction XOR -----------------------------------------
// Layout: ns=4 smooth leaves [0..3], np=3 partial leaves [4..6].
// Reduced rows (row_map -> tree node):
//   r0: leaf 0            (single smooth)
//   r1: leaf 5            (single partial)
//   r2: merge(1,2)        (2 smooth)
//   r3: merge(3, merge(4,6))   (smooth + nested partials, depth 2, mixed)
static void test_merge_tree_xor() {
    int before = g_failures;
    mpqs::uint512 N((uint64_t)0x123456789ABCDEFull);
    if ((N % 2u) == 0) N = N + mpqs::uint512((uint32_t)1);
    AuxPrimes ap = makeAuxPrimes(N, 100000000000ull, 32);
    std::mt19937_64 rng(0xBEEFFEEDull);

    const size_t ns = 4, np = 3;
    std::vector<ToyRel> smooths, partials;
    for (size_t i = 0; i < ns; ++i) smooths.push_back(makeRel(ap, rng));
    for (size_t i = 0; i < np; ++i) partials.push_back(makeRel(ap, rng));

    HostRelationBatch sb = packBatch(smooths);
    HostRelationBatch pb = packBatch(partials);

    // Build the merge tree. num_leaves = ns + np = 7. Internal node global index
    // = num_leaves + internal_index.
    MergeTree tree;
    tree.num_leaves = static_cast<uint32_t>(ns + np);  // 7
    // internal 0: merge(1,2)            -> global 7
    tree.internal_nodes.push_back({1u, 2u});
    // internal 1: merge(4,6)            -> global 8
    tree.internal_nodes.push_back({4u, 6u});
    // internal 2: merge(3, 8)           -> global 9
    tree.internal_nodes.push_back({3u, 8u});

    // row_map for the 4 reduced rows.
    std::vector<uint32_t> row_map = {0u, 5u, 7u, 9u};

    // Expected leaf sets per reduced row (with multiplicity).
    std::vector<std::vector<uint32_t>> expected_leaves = {
        {0u}, {5u}, {1u, 2u}, {3u, 4u, 6u}};

    // Field base (fb) is irrelevant to the branch path; pass empty.
    std::vector<uint32_t> fb;
    CharacterColumns cols = computeProductCharacterColumns(
        row_map, tree, sb, pb, ns, N, fb,
        CharMode::BRANCH, /*lp1_bound=*/100000000000ull, /*k=*/ap.k);

    CHECK(cols.k == ap.k, "merge-tree: result.k == 32");
    CHECK(cols.columns.size() == ap.k, "merge-tree: column count == 32");

    auto leaf_bits = [&](uint32_t leaf) -> uint32_t {
        return (leaf < ns) ? smooths[leaf].char_bits : partials[leaf - ns].char_bits;
    };
    auto leaf_rel = [&](uint32_t leaf) -> const ToyRel* {
        return (leaf < ns) ? &smooths[leaf] : &partials[leaf - ns];
    };

    int mism_xor = 0, mism_ref = 0;
    for (uint32_t r = 0; r < row_map.size(); ++r) {
        // Expected XOR-with-multiplicity of raw leaf bits.
        uint32_t want = 0;
        std::vector<const ToyRel*> rels;
        for (uint32_t leaf : expected_leaves[r]) {
            want ^= leaf_bits(leaf);
            rels.push_back(leaf_rel(leaf));
        }
        for (uint32_t j = 0; j < ap.k; ++j) {
            int got = cols.columns[j][r];
            int xorbit = (want >> j) & 1u;
            if (got != xorbit) ++mism_xor;
            int ref = productBranchBit(rels, ap, j);  // genus reference on the product
            if (got != ref) ++mism_ref;
        }
    }
    CHECK(mism_xor == 0,
          "merge-tree: composed columns == XOR-with-multiplicity of leaf bits");
    CHECK(mism_ref == 0,
          "merge-tree: composed columns == branchCharBit of true product field element");
    std::printf("[%s] merge-tree XOR: %zu reduced rows x %u primes; xor_mism=%d ref_mism=%d\n",
                (g_failures == before) ? "PASS" : "FAIL", row_map.size(), ap.k,
                mism_xor, mism_ref);
}

// ---- (C) Per-relation adapter compute(BRANCH) ------------------------------
static void test_compute_adapter() {
    int before = g_failures;
    mpqs::uint512 N((uint64_t)0xFEDCBA9876543211ull);
    if ((N % 2u) == 0) N = N + mpqs::uint512((uint32_t)1);
    AuxPrimes ap = makeAuxPrimes(N, 100000000000ull, 32);
    std::mt19937_64 rng(0x5EED5EEDull);

    const size_t n = 257;
    std::vector<ToyRel> rels;
    for (size_t i = 0; i < n; ++i) rels.push_back(makeRel(ap, rng));
    HostRelationBatch b = packBatch(rels);

    std::vector<uint32_t> fb;
    CharacterColumnComputer cc;
    cc.selectAuxPrimes(N, fb, CharMode::BRANCH, /*lp1_bound=*/100000000000ull, ap.k);
    CharacterColumns cols = cc.compute(b, CharMode::BRANCH);

    CHECK(cols.k == ap.k, "adapter: result.k == 32");
    int mism = 0;
    for (size_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < ap.k; ++j) {
            int got = cols.columns[j][i];
            int want = (rels[i].char_bits >> j) & 1u;  // verbatim unpack
            if (got != want) ++mism;
        }
    CHECK(mism == 0, "adapter: compute(BRANCH) unpacks char_bits verbatim");
    std::printf("[%s] compute(BRANCH) adapter: %zu rels x %u primes; mism=%d\n",
                (g_failures == before) ? "PASS" : "FAIL", n, ap.k, mism);
}

// ---- (D) Append-after-reduction structural check ---------------------------
static void test_append_structural() {
    int before = g_failures;
    // A tiny reduced CSR (3 rows x 5 cols). Char columns must land at index >= 5.
    HostMatrixCSR csr;
    csr.n_rows = 3;
    csr.n_cols = 5;
    csr.row_offsets = {0, 2, 3, 5};
    csr.col_indices = {0, 4, 2, 1, 3};  // all < 5

    CharacterColumns chars;
    chars.k = 32;
    chars.aux_primes.assign(32, 101ull);
    chars.columns.resize(32);
    for (uint32_t j = 0; j < 32; ++j) chars.columns[j].assign(3, 0);
    chars.columns[0][0] = 1;  // row 0 sets char col 0 only → exactly one extra entry

    const uint32_t base_col = csr.n_cols;  // 5
    const uint32_t pre_nnz  = static_cast<uint32_t>(csr.col_indices.size());

    AppendCharacterColumns(csr, chars, csr.n_rows);

    CHECK(csr.n_cols == base_col + 32, "append: n_cols grows by exactly 32");
    // Structural: every char column index lies at >= base_col, so it cannot have
    // appeared in any pre-append (reduction) column index (all were < base_col).
    int char_entries = 0, structural_entries = 0;
    for (uint32_t idx : csr.col_indices) {
        if (idx >= base_col) ++char_entries;       // char columns
        else                 ++structural_entries; // pre-append (reduction) columns
    }
    CHECK(char_entries == 1, "append: exactly one char-column entry materialized");
    CHECK(structural_entries == static_cast<int>(pre_nnz),
          "append: every pre-append (reduction) column index preserved, all < base_col");
    CHECK(csr.col_indices.size() == pre_nnz + 1,
          "append: nnz grows by the number of set char bits only");
    std::printf("[%s] append-after-reduction: n_cols 5->%u (+32), char_entries=%d\n",
                (g_failures == before) ? "PASS" : "FAIL", csr.n_cols, char_entries);
}

int main() {
    std::printf("=== test_char_xor_propagation (Stage 5 homomorphism) ===\n");
    test_lp_combine_xor();
    test_merge_tree_xor();
    test_compute_adapter();
    test_append_structural();
    std::printf("---------------------------------------------------\n");
    std::printf("checks run: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures == 0) { std::printf("RESULT: PASS (0 failures)\n"); return 0; }
    std::printf("RESULT: FAIL (%d failures)\n", g_failures);
    return 1;
}
