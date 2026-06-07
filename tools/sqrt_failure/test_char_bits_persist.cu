// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Regression test for Stage 4 of the branch-fixed character-column work:
// birth-capture of the r-bit branch character vector and its persistence.
//
//   (a) PARITY — the birth-capture formula (the EXACT arithmetic processCandidate
//       runs: sign-aware (ax+b) mod q via calculate_sqrt_of_QX + uint512::mod_uint64
//       + sub_mod, then branchCharBit) is reproduced here on the host for a sample of
//       synthetic relations with known (a,b,x) provenance, and cross-checked against
//       an INDEPENDENT recompute that forms the signed field element ((ax+b) - t_s)
//       directly as a 128-bit residue and reads jacobi_u64. The two must agree.
//   (b) I/O ROUND-TRIP — a branch-mode batch (smooths + partials, each carrying a
//       char vector + the {r, aux_primes, t_s} metadata) is serialized via
//       relation_io::serialize_v2 and read back; char_bits and metadata must be
//       byte-identical. A norm-mode .v2 (FLAG_HAS_CHAR_BITS clear) must still load
//       with EMPTY char_bits and meta.has_char_bits == false (back-compat).
//   (c) CLUSTER WIRE — cluster::serializeRelationBatch / deserializeRelationBatch
//       round-trips each relation's char vector.
//
// CPU-only host test (no kernel launches); compiled as CUDA only so the
// __host__ __device__ math headers compile. Reuses Stage 1-3 primitives verbatim.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "relation_io.h"          // serialize_v2 / deserialize_v2, V2Metadata
#include "mpqs_soa.h"             // HostRelationBatch
#include "serialization.h"        // cluster wire (serializeRelationBatch)
#include "math_utils.cuh"         // calculate_sqrt_of_QX (5-arg, Stage 4)
#include "gpu_char_cols.cuh"      // branchCharBit (Stage 3)
#include "uint128_helper.cuh"     // sub_mod, mul_mod
#include "prime_algorithms.h"     // Tonelli_Shanks_u64, jacobi_u64, is_prime_u64 (Stage 1)
#include "uint512.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

using mpqs::uint512;
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
// Birth-capture under test: the EXACT formula processCandidate uses (Stage 4 §4.2).
// ---------------------------------------------------------------------------
static uint32_t captureCharBits(const uint512& a, const uint512& b, int32_t x,
                                const std::vector<uint64_t>& q_s,
                                const std::vector<uint64_t>& t_s) {
    uint512 sqrt_Q;
    int8_t sign_axb;
    mpqs::math::calculate_sqrt_of_QX(a, b, x, sqrt_Q, sign_axb);  // 5-arg form (Stage 4)
    uint32_t bits = 0u;
    for (uint32_t s = 0; s < q_s.size(); ++s) {
        uint64_t q  = q_s[s];
        uint64_t aq = sqrt_Q.mod_uint64(q);
        uint64_t axb_mod_q = (sign_axb == -1) ? mpqs::math::sub_mod(0, aq, q) : aq;
        bits |= ((uint32_t)mpqs::matrix::branchCharBit(axb_mod_q, t_s[s], q)) << s;
    }
    return bits;
}

// ---------------------------------------------------------------------------
// Independent oracle: form ((ax+b) - t_s) mod q directly from small (a,b,x) and
// read the rational Legendre symbol. (ax+b) is a small signed integer here, so we
// can reduce it mod q without uint512 — a genuinely separate code path.
// ---------------------------------------------------------------------------
static uint32_t oracleCharBits(int64_t a, int64_t b, int64_t x,
                               const std::vector<uint64_t>& q_s,
                               const std::vector<uint64_t>& t_s) {
    long long axb = (long long)a * (long long)x + (long long)b;  // signed (ax+b)
    uint32_t bits = 0u;
    for (uint32_t s = 0; s < q_s.size(); ++s) {
        int64_t q = (int64_t)q_s[s];
        int64_t m = axb % q;               // signed remainder in (-q, q)
        if (m < 0) m += q;                 // bring into [0, q)
        int64_t diff = m - (int64_t)t_s[s];
        if (diff < 0) diff += q;           // ((ax+b) - t_s) mod q
        int leg = mpqs::sieve::jacobi_u64((uint64_t)diff, (uint64_t)q);
        bits |= ((uint32_t)((leg == -1) ? 1 : 0)) << s;
    }
    return bits;
}

// ---------------------------------------------------------------------------
// Append one relation (single FB factor) to a HostRelationBatch with a given
// char_bits value. Maintains CSR validity.
// ---------------------------------------------------------------------------
static void addRel(HostRelationBatch& b, const uint512& sqrt_Q, uint8_t sign,
                   int32_t v2, unsigned __int128 lp, uint32_t fb_idx, uint8_t cnt,
                   uint32_t cbits) {
    if (b.factor_offsets.empty()) b.factor_offsets.push_back(0);
    b.sqrt_Q.push_back(sqrt_Q);
    b.signs.push_back(sign);
    b.val_2_exps.push_back(v2);
    b.large_primes.push_back(lp);
    b.char_bits.push_back(cbits);
    b.factor_indices.push_back(fb_idx);
    b.factor_counts.push_back(cnt);
    b.num_factors += 1;
    b.factor_offsets.push_back(b.num_factors);
    b.num_relations += 1;
}

// ---------------------------------------------------------------------------
// Optional structural check on a REAL GPU-captured .v2 file (argv[1]).
// NOTE: char_bits depends on the SIGN of (ax+b), which is discarded once a relation
// is reduced to sqrt_Q = |ax+b| (and signs[] stores sign_of_Q = sign(sqrt_Q^2 - N),
// a DIFFERENT quantity). Hence char_bits cannot be re-derived from persisted data
// alone — that is exactly why it is captured at birth and persisted. So this check
// verifies the PERSISTENCE CONTRACT (flag set, {r, aux_primes, t_s} present, char_bits
// vectors sized to the batches and non-trivially populated), not a recompute. The
// PARITY (birth-capture == oracle) is covered by the synthetic provenance test below.
// ---------------------------------------------------------------------------
static void checkRealFile(const char* path) {
    std::printf("--- Real-file persistence-contract check: %s\n", path);
    HostRelationBatch sm, pa, dummy;
    mpqs::io::V2Metadata meta;
    int ver = mpqs::io::detect_and_deserialize(path, dummy, sm, pa, meta);
    CHECK(ver == 2, "real file: deserialized as v2");
    CHECK(meta.has_char_bits, "real file: FLAG_HAS_CHAR_BITS set");
    CHECK(meta.r > 0 && meta.aux_primes.size() == meta.r && meta.t_s.size() == meta.r,
          "real file: aux-prime metadata present and consistent (r/q_s/t_s)");
    if (meta.r == 0) return;
    // Every aux prime must be odd (> lp1_bound) and each t_s in [0, q).
    bool meta_ok = true;
    for (uint32_t k = 0; k < meta.r; ++k)
        if ((meta.aux_primes[k] & 1ull) == 0 || meta.t_s[k] >= meta.aux_primes[k]) meta_ok = false;
    CHECK(meta_ok, "real file: aux primes odd, t_s in [0,q)");

    auto checkBatch = [&](const HostRelationBatch& b, const char* tag) {
        CHECK(b.char_bits.size() >= b.num_relations, "real file: char_bits sized to batch");
        size_t nonzero = 0;
        for (size_t i = 0; i < b.num_relations; ++i)
            if (b.char_bits[i] != 0u) ++nonzero;
        std::printf("    %s: %zu relations, %zu with non-zero char_bits\n",
                    tag, (size_t)b.num_relations, nonzero);
        // Branch mode over a non-degenerate N must yield SOME non-zero char vectors.
        CHECK(b.num_relations == 0 || nonzero > 0, "real file: char_bits non-trivially populated");
    };
    checkBatch(sm, "smooths");
    checkBatch(pa, "partials");
}

int main(int argc, char** argv) {
    std::printf("=== Stage 4: char-bit capture + persistence test ===\n");

    // --- Fixed branch aux primes / roots over a small odd composite N -------
    // Use a tiny N with t_s^2 == N mod q for each chosen prime so branchCharBit is
    // well-defined. We pick a handful of small odd primes q with (N|q) != 0 and the
    // exact Tonelli root of N mod q (or 0 when N is a QR with root 0 — avoided here).
    const uint64_t N = 1000003ULL * 1000033ULL;  // 1000036001099 (squarefree-ish toy)
    std::vector<uint64_t> q_s;
    std::vector<uint64_t> t_s;
    {
        uint64_t cand = 101;
        while (q_s.size() < 12) {
            if (mpqs::sieve::is_prime_u64(cand)) {
                uint64_t nmod = N % cand;
                if (nmod != 0 && mpqs::sieve::jacobi_u64(nmod, cand) == 1) {
                    uint64_t root = mpqs::sieve::Tonelli_Shanks_u64(nmod, cand);
                    // root^2 == N mod q expected
                    if (mpqs::math::mul_mod(root, root, cand) == nmod && root != 0) {
                        q_s.push_back(cand);
                        t_s.push_back(root);
                    }
                }
            }
            cand += 2;
        }
    }
    const uint32_t r = (uint32_t)q_s.size();
    CHECK(r == 12, "selected r=12 branch aux primes");

    // --- Build a branch-mode batch with known (a,b,x) provenance ------------
    // We use small a,b,x so the independent oracle can reduce (ax+b) without uint512.
    struct Prov { int64_t a, b, x; };
    std::vector<Prov> provs = {
        {17, 5, 3}, {17, 5, -3}, {23, -11, 7}, {23, -11, -7},
        {31, 13, 0}, {41, -29, 12}, {53, 47, -8}, {61, -1, 25},
    };

    HostRelationBatch smooths, partials;
    std::vector<uint32_t> expect_smooth_cb, expect_partial_cb;
    for (size_t i = 0; i < provs.size(); ++i) {
        uint512 a((uint64_t)(provs[i].a < 0 ? -provs[i].a : provs[i].a));
        // a is always positive in MPQS; keep it positive here.
        uint512 av((uint64_t)provs[i].a);  // provs.a > 0 in all rows above
        // b can be negative — encode via two's complement uint512 (sub from 0).
        uint512 bv;
        if (provs[i].b >= 0) bv = uint512((uint64_t)provs[i].b);
        else { uint512 zero; uint512 mag((uint64_t)(-provs[i].b)); zero.sub(mag); bv = zero; }

        uint512 sqrt_Q;
        int8_t sign_axb;
        mpqs::math::calculate_sqrt_of_QX(av, bv, (int32_t)provs[i].x, sqrt_Q, sign_axb);

        uint32_t cap = captureCharBits(av, bv, (int32_t)provs[i].x, q_s, t_s);
        uint32_t orc = oracleCharBits(provs[i].a, provs[i].b, provs[i].x, q_s, t_s);
        CHECK(cap == orc, "PARITY: birth-capture == independent oracle");

        if (i % 2 == 0) {
            addRel(smooths, sqrt_Q, sign_axb == -1 ? 0xFF : 1, 0, 1,
                   /*fb_idx=*/(uint32_t)(i + 1), 2, cap);
            expect_smooth_cb.push_back(cap);
        } else {
            addRel(partials, sqrt_Q, sign_axb == -1 ? 0xFF : 1, 1, (unsigned __int128)9973,
                   /*fb_idx=*/(uint32_t)(i + 1), 1, cap);
            expect_partial_cb.push_back(cap);
        }
    }
    (void)0;

    // --- (b) I/O ROUND-TRIP: branch-mode .v2 -------------------------------
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "stage4_char_bits_roundtrip.v2";
    {
        mpqs::io::V2Metadata meta;
        meta.N = uint512(N);
        meta.factor_base = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        meta.lp_bound = 1000000ULL;
        meta.sieve_bound = 65536u;
        meta.aux_primes = q_s;
        meta.t_s = t_s;
        meta.r = r;

        bool ok = mpqs::io::serialize_v2(tmp.string(), smooths, partials, meta);
        CHECK(ok, "serialize_v2 (branch mode) succeeded");

        HostRelationBatch sm2, pa2;
        mpqs::io::V2Metadata meta2;
        bool ok2 = mpqs::io::deserialize_v2(tmp.string(), sm2, pa2, meta2);
        CHECK(ok2, "deserialize_v2 (branch mode) succeeded");

        CHECK(meta2.has_char_bits == true, "round-trip: has_char_bits flag set");
        CHECK(meta2.r == r, "round-trip: r preserved");
        CHECK(meta2.aux_primes == q_s, "round-trip: aux_primes preserved");
        CHECK(meta2.t_s == t_s, "round-trip: t_s preserved");

        CHECK(sm2.char_bits.size() >= smooths.num_relations, "round-trip: smooth char_bits sized");
        CHECK(pa2.char_bits.size() >= partials.num_relations, "round-trip: partial char_bits sized");

        bool sm_ok = true;
        for (size_t i = 0; i < smooths.num_relations; ++i)
            if (sm2.char_bits[i] != expect_smooth_cb[i]) sm_ok = false;
        CHECK(sm_ok, "round-trip: smooth char_bits identical");

        bool pa_ok = true;
        for (size_t i = 0; i < partials.num_relations; ++i)
            if (pa2.char_bits[i] != expect_partial_cb[i]) pa_ok = false;
        CHECK(pa_ok, "round-trip: partial char_bits identical");
    }

    // --- (b) BACK-COMPAT: a norm-mode .v2 (no char flag) still loads --------
    fs::path tmp_norm = fs::temp_directory_path() / "stage4_norm_nochar.v2";
    {
        // Build norm-mode batches: char_bits left empty here? No — the SoA always
        // sizes char_bits. To emulate an OLD file we must NOT set meta.r, so the
        // serializer clears FLAG_HAS_CHAR_BITS and writes NO char_bits vector.
        HostRelationBatch nsm = smooths;   // copy (carries char_bits, but won't be written)
        HostRelationBatch npa = partials;
        mpqs::io::V2Metadata nmeta;
        nmeta.N = uint512(N);
        nmeta.factor_base = {2, 3, 5, 7};
        nmeta.lp_bound = 0;
        nmeta.sieve_bound = 32768u;
        // nmeta.r stays 0 → FLAG_HAS_CHAR_BITS NOT set → file has no char section.
        bool ok = mpqs::io::serialize_v2(tmp_norm.string(), nsm, npa, nmeta);
        CHECK(ok, "serialize_v2 (norm mode, no char flag) succeeded");

        HostRelationBatch sm2, pa2;
        mpqs::io::V2Metadata meta2;
        bool ok2 = mpqs::io::deserialize_v2(tmp_norm.string(), sm2, pa2, meta2);
        CHECK(ok2, "deserialize_v2 (norm mode) succeeded");
        CHECK(meta2.has_char_bits == false, "back-compat: has_char_bits false for norm file");
        CHECK(meta2.r == 0, "back-compat: r==0 for norm file");
        CHECK(sm2.char_bits.empty(), "back-compat: smooth char_bits empty (not parsed)");
        CHECK(pa2.char_bits.empty(), "back-compat: partial char_bits empty (not parsed)");
        // Core data still intact (proves we did NOT mis-parse trailing bytes).
        CHECK(sm2.num_relations == smooths.num_relations, "back-compat: smooth count intact");
        CHECK(pa2.num_relations == partials.num_relations, "back-compat: partial count intact");
        bool fb_ok = (sm2.factor_indices == smooths.factor_indices);
        CHECK(fb_ok, "back-compat: smooth factor indices intact");
    }

    // --- (c) CLUSTER WIRE round-trip ---------------------------------------
    {
        auto [buf, len] = mpqs::cluster::serializeRelationBatch(smooths, smooths.num_relations);
        CHECK(len > 0, "cluster serialize produced bytes");
        HostRelationBatch out;
        uint64_t n = mpqs::cluster::deserializeRelationBatch(buf.data(), len, out);
        CHECK(n == smooths.num_relations, "cluster: relation count round-trips");
        CHECK(out.char_bits.size() >= smooths.num_relations, "cluster: char_bits sized");
        bool cb_ok = true;
        for (size_t i = 0; i < smooths.num_relations; ++i)
            if (out.char_bits[i] != expect_smooth_cb[i]) cb_ok = false;
        CHECK(cb_ok, "cluster: char_bits identical after wire round-trip");
    }

    // Cleanup
    std::error_code ec;
    fs::remove(tmp, ec);
    fs::remove(tmp_norm, ec);

    // --- Optional: validate a REAL GPU-captured branch-mode .v2 file --------
    if (argc > 1) checkRealFile(argv[1]);

    std::printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
