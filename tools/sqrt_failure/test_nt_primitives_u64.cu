// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Regression test for the 64-bit, overflow-safe number-theory primitives
// (Stage 1 of the branch-fixed character-column work):
//   - mpqs::sieve::Tonelli_Shanks_u64
//   - mpqs::sieve::jacobi_u64
//   - mpqs::sieve::is_prime_u64
//   - mpqs::uint512::mod_uint64
//
// CPU-only host test (no kernel launches); compiled as CUDA only so the
// __host__ __device__ math headers (uint128_helper / uint512) compile. Each
// primitive is checked against an independent 128-bit oracle and (where
// applicable) cross-checked against the existing uint32 implementations.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "prime_algorithms.h"
#include "uint128_helper.cuh"
#include "uint512.cuh"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using mpqs::sieve::Tonelli_Shanks;
using mpqs::sieve::Tonelli_Shanks_u64;
using mpqs::sieve::jacobi;
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

// ---- Independent 128-bit oracle helpers ------------------------------------
static uint64_t ref_mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((unsigned __int128)a * b) % m);
}
static uint64_t ref_powmod(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1; b %= m;
    while (e) { if (e & 1) r = ref_mulmod(r, b, m); b = ref_mulmod(b, b, m); e >>= 1; }
    return r;
}
// Deterministic Miller-Rabin oracle, independent of the code under test.
static bool ref_isprime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull,31ull,37ull}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    uint64_t d = n - 1; int r = 0;
    while (!(d & 1)) { d >>= 1; ++r; }
    for (uint64_t a : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull,31ull,37ull}) {
        uint64_t x = ref_powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (int j = 0; j < r - 1; ++j) { x = ref_mulmod(x, x, n); if (x == n - 1) { comp = false; break; } }
        if (comp) return false;
    }
    return true;
}
static mpqs::uint512 mk512(const uint32_t (&l)[16]) { return mpqs::uint512(l); }
// Reference uint512 % d via long division over 32-bit limbs in a 128-bit accumulator.
static uint64_t ref_mod512(const uint32_t (&l)[16], uint64_t d) {
    if (d == 0) return 0;
    unsigned __int128 rem = 0;
    for (int i = 15; i >= 0; --i) rem = (((unsigned __int128)rem << 32) | l[i]) % d;
    return (uint64_t)rem;
}

int main() {
    std::mt19937_64 rng(0xC0FFEEull);
    std::printf("=== test_nt_primitives_u64 (Stage 1 regression) ===\n");

    // --- Criterion: jacobi_u64 spot checks + == uint32 jacobi over samples ---
    {
        int before = g_failures;
        CHECK(jacobi_u64(2, 7) == 1,  "(2|7) == +1");
        CHECK(jacobi_u64(3, 7) == -1, "(3|7) == -1");
        CHECK(jacobi_u64(1, 7) == 1,  "(1|7) == +1");
        CHECK(jacobi_u64(7, 7) == 0,  "(7|7) == 0 (gcd != 1)");
        CHECK(jacobi_u64(2, 15) == 1, "(2|15) == +1 (Jacobi)");
        CHECK(jacobi_u64(14, 15) == -1, "(14|15) == -1 (Jacobi)");

        std::uniform_int_distribution<uint32_t> dn(1, 0xFFFFFFFEu);
        int agree = 0, total = 200000;
        for (int k = 0; k < total; ++k) {
            uint32_t nn = dn(rng) | 1u;            // odd denominator
            if (nn == 1) nn = 3;
            uint32_t a = dn(rng);
            int e32 = jacobi(a, nn);
            int e64 = jacobi_u64((uint64_t)a, (uint64_t)nn);
            if (e32 == e64) ++agree;
            else { std::printf("  FAIL jacobi mismatch a=%u n=%u u32=%d u64=%d\n", a, nn, e32, e64); ++g_failures; ++g_checks; }
        }
        CHECK(agree == total, "jacobi_u64 == jacobi over 200k samples");
        std::printf("[%s] jacobi_u64 (spot checks + 200k uint32 cross-check)\n",
                    g_failures == before ? "PASS" : "FAIL");
    }

    // --- Criterion: is_prime_u64 vs 128-bit MR oracle (table + ~1e11 + Carmichael) ---
    {
        int before = g_failures;
        const uint64_t primes[] = {
            2, 3, 5, 7, 97, 7919, 104729,
            2147483647ull,            // 2^31 - 1
            1000000007ull,
            99999999977ull,           // ~1e11 prime
            100000000003ull,          // ~1e11 prime
            999999999989ull,          // ~1e12 prime
            18446744073709551557ull   // largest prime < 2^64
        };
        for (uint64_t p : primes) {
            CHECK(is_prime_u64(p), "is_prime_u64 true on known prime");
            CHECK(ref_isprime(p),  "oracle confirms prime");
        }
        const uint64_t composites[] = {
            0, 1, 4, 9, 100, 7917, 100000000000ull,
            561, 1105, 1729, 2465, 2821, 6601, 8911,   // Carmichael numbers
            41041, 825265, 321197185ull,               // larger Carmichael numbers
            1000003ull * 1000033ull,                   // ~1e12 semiprime
            999979ull * 999983ull,                     // ~1e12 semiprime
            2147483647ull * 2ull
        };
        for (uint64_t c : composites) {
            CHECK(!is_prime_u64(c), "is_prime_u64 false on known composite/Carmichael");
        }
        // Random 64-bit cross-check vs oracle.
        std::uniform_int_distribution<uint64_t> du;
        int agree = 0, total = 50000;
        for (int k = 0; k < total; ++k) {
            uint64_t n = du(rng);
            if (is_prime_u64(n) == ref_isprime(n)) ++agree;
            else { std::printf("  FAIL is_prime mismatch n=%llu\n", (unsigned long long)n); ++g_failures; ++g_checks; }
        }
        CHECK(agree == total, "is_prime_u64 == oracle over 50k random 64-bit values");
        // Dense sweep across [1e11, 1e11 + 1e4).
        int agree2 = 0, total2 = 0;
        for (uint64_t n = 100000000000ull; n < 100000010000ull; ++n) {
            ++total2;
            if (is_prime_u64(n) == ref_isprime(n)) ++agree2;
            else { std::printf("  FAIL is_prime mismatch near 1e11 n=%llu\n", (unsigned long long)n); ++g_failures; ++g_checks; }
        }
        CHECK(agree2 == total2, "is_prime_u64 == oracle over [1e11, 1e11+1e4)");
        std::printf("[%s] is_prime_u64 (table + Carmichaels + 50k random + 1e11 sweep)\n",
                    g_failures == before ? "PASS" : "FAIL");
    }

    // --- Criterion: Tonelli_Shanks_u64 — r^2 == n mod p for primes p > 2^32 ---
    {
        int before = g_failures;
        std::vector<uint64_t> bigprimes;
        std::uniform_int_distribution<uint64_t> dbig(4294967311ull, 200000000000ull);
        while (bigprimes.size() < 40) {
            uint64_t cand = dbig(rng) | 1ull;
            if (is_prime_u64(cand) && ref_isprime(cand)) bigprimes.push_back(cand);
        }
        std::uniform_int_distribution<uint64_t> dn;
        int verified = 0, attempted = 0;
        for (uint64_t p : bigprimes) {
            int found_for_p = 0;
            for (int tries = 0; tries < 64 && found_for_p < 5; ++tries) {
                uint64_t n = dn(rng) % p;
                if (n == 0) continue;
                if (jacobi_u64(n, p) != 1) continue;  // only residues
                ++attempted;
                uint64_t r = Tonelli_Shanks_u64(n, p);
                bool ok = (r != 0) && (mul_mod(r, r, p) == n % p);
                if (ok) { ++verified; ++found_for_p; }
                else { std::printf("  FAIL TS_u64 r^2 != n: n=%llu p=%llu r=%llu\n",
                                   (unsigned long long)n, (unsigned long long)p, (unsigned long long)r);
                        ++g_failures; ++g_checks; }
            }
        }
        CHECK(attempted > 100, "TS_u64 attempted enough residues (>100)");
        CHECK(verified == attempted, "TS_u64: mul_mod(r,r,p) == n%p for all residues over p>2^32");
        std::printf("[%s] Tonelli_Shanks_u64 r^2==n mod p (%d residues over 40 primes >2^32)\n",
                    g_failures == before ? "PASS" : "FAIL", attempted);
    }

    // --- Criterion: Tonelli_Shanks_u64 == uint32 Tonelli_Shanks (canonicalized) ---
    {
        int before = g_failures;
        const uint32_t smallprimes[] = {3,5,7,11,13,17,19,23,29,31,101,257,65537u,1000003u,2000003u};
        int agree = 0, total = 0;
        for (uint32_t p : smallprimes) {
            for (uint32_t n = 1; n < p && n < 5000; ++n) {
                if (jacobi(n, p) != 1) continue;
                ++total;
                uint32_t r32 = Tonelli_Shanks(n, p);
                uint64_t r64 = Tonelli_Shanks_u64((uint64_t)n, (uint64_t)p);
                uint64_t c32 = r32 < p - r32 ? r32 : p - r32;   // canonicalize sign
                uint64_t c64 = r64 < p - r64 ? r64 : p - r64;
                bool ok = (c32 == c64) && (mul_mod(r64, r64, p) == (uint64_t)n % p);
                if (ok) ++agree;
                else { std::printf("  FAIL TS cross n=%u p=%u r32=%u r64=%llu\n",
                                   n, p, r32, (unsigned long long)r64); ++g_failures; ++g_checks; }
            }
        }
        CHECK(total > 1000, "TS cross-check covered enough cases (>1000)");
        CHECK(agree == total, "TS_u64 == uint32 TS (canonicalized) for small primes");
        std::printf("[%s] Tonelli_Shanks_u64 vs uint32 cross-check (%d cases)\n",
                    g_failures == before ? "PASS" : "FAIL", total);
    }

    // --- Criterion: mod_uint64 vs 128-bit oracle; == mod_uint32 for d<2^32; d==0 ---
    {
        int before = g_failures;
        std::uniform_int_distribution<uint32_t> dlimb;
        std::uniform_int_distribution<uint64_t> dd_small(1, 0xFFFFFFFFull);    // < 2^32
        std::uniform_int_distribution<uint64_t> dd_big(1, 1000000000000ull);   // up to ~1e12
        int agree = 0, total = 0;
        for (int k = 0; k < 200000; ++k) {
            uint32_t l[16];
            for (int i = 0; i < 16; ++i) l[i] = dlimb(rng);
            mpqs::uint512 x = mk512(l);

            uint64_t db = dd_big(rng);
            uint64_t got = x.mod_uint64(db);
            uint64_t ref = ref_mod512(l, db);
            ++total; if (got == ref) ++agree;
            else { std::printf("  FAIL mod_uint64 big d=%llu got=%llu ref=%llu\n",
                               (unsigned long long)db, (unsigned long long)got, (unsigned long long)ref);
                   ++g_failures; ++g_checks; }

            uint32_t ds = (uint32_t)dd_small(rng);
            uint64_t g64 = x.mod_uint64((uint64_t)ds);
            uint32_t g32 = x.mod_uint32(ds);
            ++total; if (g64 == (uint64_t)g32) ++agree;
            else { std::printf("  FAIL mod_uint64 vs mod_uint32 d=%u u64=%llu u32=%u\n",
                               ds, (unsigned long long)g64, g32); ++g_failures; ++g_checks; }
        }
        {
            uint32_t l[16]; for (int i = 0; i < 16; ++i) l[i] = 0xDEADBEEFu;
            mpqs::uint512 x = mk512(l);
            CHECK(x.mod_uint64(0) == 0, "mod_uint64(0) == 0 edge case");
        }
        CHECK(agree == total, "mod_uint64 == oracle (big d) and == mod_uint32 (d<2^32) over 200k");
        std::printf("[%s] uint512::mod_uint64 (200k vs oracle + mod_uint32 + d==0)\n",
                    g_failures == before ? "PASS" : "FAIL");
    }

    std::printf("---------------------------------------------------\n");
    std::printf("checks run: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures == 0) { std::printf("RESULT: PASS (0 failures)\n"); return 0; }
    std::printf("RESULT: FAIL (%d failures)\n", g_failures);
    return 1;
}
