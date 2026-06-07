// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Regression test for Stage 2 of the branch-fixed character-column work:
// CharacterColumnComputer::selectAuxPrimes() gated on CharMode {NORM, BRANCH}.
//
//   - BRANCH mode: aux primes are chosen > lp1_bound via a 64-bit walk, each with
//     a fixed Tonelli root t_s satisfying t_s^2 == N mod q (mod q). The test
//     verifies exactly k primes, every q is prime / > lp1_bound / not in FB /
//     (N|q)=+1, t_s < q and t_s^2 == N mod q, and that selection is deterministic.
//   - NORM mode: must reproduce the legacy start-at-3 uint32 selection bit-for-bit.
//     The test reimplements the legacy walk inline as an independent oracle and
//     asserts aux_primes_ / n_mod_q_ match exactly and that t_s_ stays empty.
//
// CPU-only host test (no kernel launches); compiled as CUDA only so the
// __host__ __device__ math headers compile. Reuses the Stage-1 64-bit primitives
// (Tonelli_Shanks_u64 / jacobi_u64 / is_prime_u64) and uint512::mod_uint64.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "character_columns.h"  // CharacterColumnComputer, CharMode
#include "gpu_char_cols.cuh"    // jacobi_symbol_dev (the exact legacy NORM-path symbol)
#include "prime_algorithms.h"   // Stage 1: Tonelli_Shanks_u64, jacobi_u64, is_prime_u64
#include "uint128_helper.cuh"   // mpqs::math::mul_mod
#include "uint512.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_set>
#include <vector>

using mpqs::uint512;
using mpqs::matrix::CharMode;
using mpqs::matrix::CharacterColumnComputer;
using mpqs::matrix::jacobi_symbol_dev;
using mpqs::sieve::Tonelli_Shanks_u64;
using mpqs::sieve::jacobi_u64;
using mpqs::sieve::is_prime_u64;
using mpqs::math::mul_mod;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) { ++g_failures;                                         \
        std::printf("  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
    }                                                                    \
} while (0)

// ---- Independent oracle: legacy is_prime_small (trial division) ------------
// Byte-faithful copy of the anonymous-namespace helper in character_columns.cpp.
static bool ref_is_prime_small(uint32_t candidate) {
    if (candidate < 2) return false;
    if (candidate == 2) return true;
    if ((candidate & 1u) == 0) return false;
    uint32_t limit = static_cast<uint32_t>(std::sqrt(static_cast<double>(candidate))) + 1;
    for (uint32_t d = 3; d <= limit; d += 2) {
        if (candidate % d == 0) return false;
    }
    return true;
}

// ---- Independent oracle: legacy NORM aux-prime selection -------------------
// Reproduces the pre-Stage-2 selectAuxPrimes() walk exactly: start at 3, step odd,
// is_prime_small, skip FB, require jacobi_symbol_dev(N%q, q) == +1.
static void ref_select_norm(const uint512& N,
                            const std::vector<uint32_t>& fb,
                            uint32_t k,
                            std::vector<uint64_t>& out_primes,
                            std::vector<uint32_t>& out_nmodq) {
    std::unordered_set<uint32_t> fb_set(fb.begin(), fb.end());
    out_primes.clear();
    out_nmodq.clear();
    for (uint32_t candidate = 3; out_primes.size() < k; candidate += 2) {
        if (!ref_is_prime_small(candidate)) continue;
        if (fb_set.count(candidate)) continue;
        uint32_t n_mod = static_cast<uint32_t>(N % candidate);
        if (jacobi_symbol_dev(n_mod, candidate) != 1) continue;
        out_primes.push_back(candidate);
        out_nmodq.push_back(n_mod);
    }
}

// ---- Per-N branch-mode acceptance --------------------------------------------
static void test_branch(const char* label, const uint512& N,
                        const std::vector<uint32_t>& fb, uint64_t lp1_bound) {
    int before = g_failures;
    const uint32_t k = 32;
    std::unordered_set<uint32_t> fb_set(fb.begin(), fb.end());

    CharacterColumnComputer cc;
    cc.selectAuxPrimes(N, fb, CharMode::BRANCH, lp1_bound, k);
    const std::vector<uint64_t>& primes = cc.auxPrimes();
    const std::vector<uint64_t>& ts     = cc.tS();

    CHECK(primes.size() == k, "branch: exactly k=32 aux primes");
    CHECK(ts.size() == k,     "branch: exactly k=32 Tonelli roots");
    // n_mod_q_ is the NORM representation; branch must NOT populate it.
    CHECK(cc.nModQ().empty(), "branch: n_mod_q_ left empty");

    bool strictly_increasing = true;
    for (size_t j = 0; j < primes.size(); ++j) {
        uint64_t q = primes[j];
        CHECK(is_prime_u64(q), "branch: q is prime (is_prime_u64)");
        CHECK(q > lp1_bound,   "branch: q > lp1_bound");
        // FB primes are all < lp1_bound; q > lp1_bound is automatically outside FB.
        CHECK(q > 0xFFFFFFFFull || fb_set.count(static_cast<uint32_t>(q)) == 0,
              "branch: q not in factor base");
        uint64_t n_mod = N.mod_uint64(q);
        CHECK(jacobi_u64(n_mod, q) == 1, "branch: (N|q) == +1");
        uint64_t t = ts[j];
        CHECK(t < q, "branch: t_s < q (canonicalized)");
        CHECK(mul_mod(t, t, q) == n_mod, "branch: t_s^2 == N mod q (mod q)");
        if (j > 0 && !(primes[j] > primes[j - 1])) strictly_increasing = false;
    }
    CHECK(strictly_increasing, "branch: aux primes strictly increasing (walk order)");

    // Determinism: a second call must yield identical primes AND identical roots.
    CharacterColumnComputer cc2;
    cc2.selectAuxPrimes(N, fb, CharMode::BRANCH, lp1_bound, k);
    bool det = (cc2.auxPrimes() == primes) && (cc2.tS() == ts);
    CHECK(det, "branch: selection deterministic across calls");

    std::printf("[%s] branch-mode aux selection (%s, lp1=%llu): %zu primes\n",
                g_failures == before ? "PASS" : "FAIL", label,
                (unsigned long long)lp1_bound, primes.size());
}

// ---- Per-N norm-mode parity -------------------------------------------------
static void test_norm(const char* label, const uint512& N,
                      const std::vector<uint32_t>& fb) {
    int before = g_failures;
    const uint32_t k = 32;

    CharacterColumnComputer cc;
    // lp1_bound is ignored in NORM mode; pass a non-zero value to prove it is unused.
    cc.selectAuxPrimes(N, fb, CharMode::NORM, /*lp1_bound=*/100000000000ull, k);

    std::vector<uint64_t> ref_primes;
    std::vector<uint32_t> ref_nmodq;
    ref_select_norm(N, fb, k, ref_primes, ref_nmodq);

    CHECK(cc.auxPrimes().size() == k, "norm: exactly k=32 aux primes");
    CHECK(cc.auxPrimes() == ref_primes, "norm: aux_primes_ == legacy oracle (bit-for-bit)");
    CHECK(cc.nModQ() == ref_nmodq,      "norm: n_mod_q_ == legacy oracle (bit-for-bit)");
    CHECK(cc.tS().empty(),              "norm: t_s_ left empty");

    std::printf("[%s] norm-mode parity (%s): %zu primes match legacy oracle\n",
                g_failures == before ? "PASS" : "FAIL", label, cc.auxPrimes().size());
}

int main() {
    std::printf("=== test_aux_prime_selection (Stage 2 regression) ===\n");

    // Targets: a 95-digit composite (c95_repo from imbalance_family.csv) and RSA-100.
    const uint512 N_c95(
        "12412366478083658690621278881413412216495301956213095531440291677880752164442010827407702520563");
    const uint512 N_rsa100(
        "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139");

    // Synthetic factor base of small odd primes (all < lp1_bound). Exercises the
    // FB-skip path in NORM mode; in BRANCH mode q > lp1_bound is outside it anyway.
    std::vector<uint32_t> fb;
    for (uint32_t c = 3; c < 5000; c += 2) {
        if (ref_is_prime_small(c)) fb.push_back(c);
    }

    const uint64_t lp1 = 100000000000ull;  // 1e11

    // --- Branch mode: > lp1_bound + fixed Tonelli root for >= 2 target N ---
    test_branch("c95_repo", N_c95, fb, lp1);
    test_branch("RSA-100", N_rsa100, fb, lp1);

    // --- Norm mode: byte-for-byte parity with legacy selection ---
    test_norm("c95_repo", N_c95, fb);
    test_norm("RSA-100", N_rsa100, fb);

    std::printf("---------------------------------------------------\n");
    std::printf("checks run: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures == 0) { std::printf("RESULT: PASS (0 failures)\n"); return 0; }
    std::printf("RESULT: FAIL (%d failures)\n", g_failures);
    return 1;
}
