// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// =============================================================================
// relation_validator — standalone HOST-side exhaustive correctness checker
// =============================================================================
//
// Loads a saved relations file (.v2 or .soa via mpqs::io::detect_and_deserialize)
// and re-verifies every relation from scratch on the CPU. The pipeline normally
// trusts that the GPU sieve/postprocessing recorded a correct factorization; this
// tool independently re-derives it and, crucially, performs a PRIMALITY test on
// every recorded large prime.
//
// MOTIVATION (the pathology this rules out):
//   src/postprocessing/postprocessing.cu:368 accepts a residual cofactor as a
//   "large prime" solely because `remainder < lp1_bound`, with NO primality
//   test. A COMPOSITE cofactor passing that gate corrupts the congruence of
//   squares (we would be combining a relation whose "prime" is actually a
//   product of two primes), which can silently break the square-root stage.
//   The existing GPU validator (debug_validate_soa_kernel) trusts the LP blindly,
//   so it cannot catch this. This tool does.
//
// Per relation, for BOTH the smooths batch and the partials batch, it checks:
//
//   (1) ALGEBRAIC IDENTITY. Recompute Q = sqrt_Q^2 - N exactly (this is
//       (ax+b)^2 - N). Confirm sign(Q) matches the recorded `signs` byte, then
//       confirm |Q| == 2^(val_2_exps) * PROD(fb[idx]^count) * large_prime by
//       dividing |Q| down and checking the final residue == large_prime exactly.
//
//   (2) COMPLETENESS / MISSED-FACTOR. Independently trial-divide |Q| (after
//       removing 2^v2) by the ENTIRE factor base, re-deriving exponents from
//       scratch. Confirm the re-derived FB factorization matches the recorded
//       one and that the leftover cofactor equals large_prime. Flags any case
//       where an FB prime divides the recorded large_prime (the exact "we
//       work with a composite instead of a prime" pathology) or where a
//       recorded exponent disagrees.
//
//   (3) PRIMALITY of large_prime (partials only). Deterministic Miller-Rabin
//       (witness set valid for all n < 3.317e24, covering the entire u64 LP
//       range) with a BPSW fallback (strong base-2 PRP + strong Lucas) for any
//       larger value. Flags every COMPOSITE large prime.
//
//   (4) RANGE. Confirm max(factor_base) < large_prime <= lp_bound.
//
// CPU-only: no CUDA kernels are launched. The file is compiled as .cu purely so
// the __host__ __device__ math headers (which pull in <cuda_runtime.h> and rely
// on unsigned __int128) compile cleanly. Parallelised with OpenMP.
// =============================================================================

#include "relation_io.h"
#include "sieve_checkpoint.h"   // read sieve.ckpt (footer + trailer + v2 payload)
#include "mpqs_soa.h"
#include "uint512.cuh"
#include "math_utils.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <filesystem>

using mpqs::uint512;
using mpqs::structures::HostRelationBatch;

// =============================================================================
// Deterministic primality testing
// =============================================================================
//
// We operate on uint512 throughout (a large prime is at most ~lp_bound which is
// well under 2^64, but uint512 keeps the code uniform and overflow-proof for any
// conceivable bound). modpow / mul_mod from math_utils already use 1024-bit
// intermediates, so there is no overflow risk.

namespace {

// Strong-probable-prime test to a single base `a` (Miller-Rabin round).
// Precondition: n is odd, n > 3, and (d, s) satisfy n-1 = d * 2^s with d odd.
// Returns true if n is a strong probable prime to base a (i.e. NOT proven
// composite by this base); false proves n composite.
bool miller_rabin_round(const uint512& n, const uint512& n_minus_1,
                        const uint512& d, int s, const uint512& a) {
    uint512 x = mpqs::math::modpow(a, d, n);
    if (x.is_one() || x == n_minus_1) return true;
    for (int r = 1; r < s; ++r) {
        x.mul_mod(x, n);          // x = x^2 mod n
        if (x == n_minus_1) return true;
        if (x.is_one()) return false; // nontrivial sqrt of 1 -> composite
    }
    return false;
}

// Jacobi symbol (a / n), n odd positive. Operates on uint512.
// Returns -1, 0, or +1 as an int.
int jacobi_symbol(uint512 a, uint512 n) {
    a.mod(n);
    int result = 1;
    while (!a.is_zero()) {
        // Factor out powers of two from a.
        while ((a.limbs[0] & 1) == 0) {
            a.rshift(1);
            uint32_t n_mod_8 = n.limbs[0] & 7u;
            if (n_mod_8 == 3 || n_mod_8 == 5) result = -result;
        }
        // Reciprocity swap.
        std::swap(a, n);
        if ((a.limbs[0] & 3u) == 3u && (n.limbs[0] & 3u) == 3u) result = -result;
        a.mod(n);
    }
    if (n.is_one()) return result;
    return 0;
}

// Strong Lucas probable-prime test with Selfridge parameters (D, P=1, Q).
// Precondition: n is odd, n > 2, n not a perfect square, gcd(n, 2*Q) handled by
// caller's small-divisor pre-screen. Returns true if n is a strong Lucas PRP.
bool strong_lucas_prp(const uint512& n) {
    // 1. Selfridge: find first D in 5, -7, 9, -11, ... with Jacobi(D/n) == -1.
    //    Represent D as (sign, magnitude); Q = (1 - D) / 4.
    int64_t D = 0;
    int d_sign = 1;
    int64_t d_mag = 5;
    for (int i = 0; i < 1000; ++i) {
        // Build D mod n as a uint512 representative for the Jacobi computation.
        uint512 d_rep((uint64_t)d_mag);
        if (d_sign < 0) d_rep.negate_mod_inplace(n); // n - d_mag (mod n)
        int j = jacobi_symbol(d_rep, n);
        if (j == 0) {
            // gcd(D, n) > 1 with 1 < gcd < n  ->  n is composite (D < n here for
            // any realistic prime candidate). Treat as composite.
            return false;
        }
        if (j == -1) { D = d_sign * d_mag; break; }
        d_sign = -d_sign;
        d_mag += 2;
    }
    if (D == 0) return false; // No suitable D found (n likely a perfect square).

    // Q = (1 - D) / 4. With P = 1.
    // Compute Q mod n as uint512. (1 - D) is divisible by 4 by construction.
    int64_t Q_int = (1 - D) / 4;
    uint512 Qm;
    if (Q_int >= 0) {
        Qm = uint512((uint64_t)Q_int);
        Qm.mod(n);
    } else {
        Qm = uint512((uint64_t)(-Q_int));
        Qm.mod(n);
        Qm.negate_mod_inplace(n);
    }
    uint512 P((uint32_t)1); // P = 1

    // 2. n+1 = d * 2^s, d odd. Compute the Lucas sequence U_d, V_d via binary
    //    expansion of d (MSB to LSB), doubling formulas mod n.
    uint512 n_plus_1 = n;
    n_plus_1.add(uint512((uint32_t)1));
    uint512 dd = n_plus_1;
    int s = 0;
    while ((dd.limbs[0] & 1) == 0) { dd.rshift(1); ++s; }

    // U = 1, V = P, Qk = Q   (state for the highest set bit already consumed)
    uint512 U((uint32_t)1);
    uint512 V = P;
    uint512 Qk = Qm;

    int top = dd.msb();
    for (int i = top - 1; i >= 0; --i) {
        // Double: U_2k = U_k V_k ; V_2k = V_k^2 - 2 Q^k ; Q^2k = (Q^k)^2
        U.mul_mod(V, n);
        // V = V^2 - 2*Qk
        uint512 V2 = V; V2.mul_mod(V, n);
        uint512 twoQk = Qk; twoQk.add_mod(Qk, n);
        V2.sub_mod(twoQk, n);
        V = V2;
        Qk.mul_mod(Qk, n);

        if ((dd.limbs[i / 32] >> (i % 32)) & 1u) {
            // Add one (k -> k+1): with P=1,
            //   U' = (P*U + V) / 2 ; V' = (D*U + P*V) / 2 ; Qk *= Q
            uint512 Unew = U; Unew.add_mod(V, n);              // P*U + V (P=1)
            // D*U mod n: build |D| then sign-adjust
            uint512 Du = U;
            {
                uint512 absD((uint64_t)(D < 0 ? -D : D));
                Du.mul_mod(absD, n);
                if (D < 0) Du.negate_mod_inplace(n);
            }
            uint512 Vnew = Du; Vnew.add_mod(V, n);             // D*U + P*V (P=1)
            // Divide by 2 mod n (n odd): if odd, add n then >>1.
            auto half_mod = [&](uint512 v) -> uint512 {
                if (v.limbs[0] & 1) v.add(n); // safe: v < n, v+n < 2n < 2^512
                v.rshift(1);
                return v;
            };
            U = half_mod(Unew);
            V = half_mod(Vnew);
            Qk.mul_mod(Qm, n);
        }
    }

    // 3. Strong test: prime if U_d == 0 OR V_{d*2^r} == 0 for some 0 <= r < s.
    if (U.is_zero()) return true;
    if (V.is_zero()) return true;
    for (int r = 1; r < s; ++r) {
        // V_{2k} = V_k^2 - 2 Q^k ; advance Qk too.
        uint512 V2 = V; V2.mul_mod(V, n);
        uint512 twoQk = Qk; twoQk.add_mod(Qk, n);
        V2.sub_mod(twoQk, n);
        V = V2;
        if (V.is_zero()) return true;
        Qk.mul_mod(Qk, n);
    }
    return false;
}

// Perfect-square check (needed before the Lucas test, which is undefined for
// perfect squares). Uses uint512::sqrt (Newton) then verifies by squaring.
bool is_perfect_square(const uint512& n) {
    uint512 r = n.sqrt();
    // sqrt() (Newton) may be off by one in either direction; check r-1, r, r+1.
    uint512 a = r; a.mult(a);
    if (a == n) return true;
    uint512 up = r; up.add(uint512((uint32_t)1));
    uint512 u2 = up; u2.mult(up);
    if (u2 == n) return true;
    if (!r.is_zero()) {
        uint512 dn = r; dn.sub(uint512((uint32_t)1));
        uint512 d2 = dn; d2.mult(dn);
        if (d2 == n) return true;
    }
    return false;
}

// Deterministic primality test for n.
// Strategy:
//   * Trivial small cases / even check.
//   * Trial-divide by small primes (cheap composite screen + handles tiny n).
//   * Deterministic Miller-Rabin with the 12-base witness set
//       {2,3,5,7,11,13,17,19,23,29,31,37}
//     which is proven to give zero false positives for ALL n < 3.317e24
//     (> 2^81). lp_bound is < 2^64, so this branch is deterministic for every
//     large prime we can encounter.
//   * BPSW fallback (strong base-2 MR already covered above + strong Lucas) for
//     the (practically unreachable) case n >= 3.317e24 — no known composite
//     passes BPSW, and none exists below 2^64.
bool is_prime_deterministic(const uint512& n) {
    if (n < (uint32_t)2) return false;
    // Small primes for trial division and as the deterministic MR witness set.
    static const uint32_t small_primes[] = {
        2,3,5,7,11,13,17,19,23,29,31,37
    };
    for (uint32_t p : small_primes) {
        if (n == p) return true;
        if (n.mod_uint32(p) == 0) return false;
    }
    // n is now odd, coprime to all witnesses, and > 37.

    uint512 n_minus_1 = n;
    n_minus_1.sub(uint512((uint32_t)1));
    uint512 d = n_minus_1;
    int s = 0;
    while ((d.limbs[0] & 1) == 0) { d.rshift(1); ++s; }

    for (uint32_t p : small_primes) {
        uint512 a((uint32_t)p);
        if (!miller_rabin_round(n, n_minus_1, d, s, a)) return false;
    }

    // Deterministic threshold: 3.317e24. If n is below it, the 12-base MR pass
    // above is a proof of primality. Compare against 2^82 (> 3.317e24) as a
    // safe, cheap upper screen on the limbs.
    bool below_det_threshold = true;
    for (int i = 3; i < 16; ++i) if (n.limbs[i] != 0) { below_det_threshold = false; break; }
    // limbs[0..2] cover bits 0..95; 3.317e24 < 2^82 < 2^96, so if limbs[3..]==0
    // we are below 2^96. Tighten: require it under 2^81 by checking high bits of
    // limb[2]. 2^81 has bit 81 set -> limb[2] (bits 64..95) bit 17.
    if (below_det_threshold && (n.limbs[2] >> 18) == 0) {
        return true; // proven prime by deterministic MR (n < 2^82)
    }

    // BPSW fallback for very large n (unreachable for genuine LPs): strong
    // base-2 PRP is already implied by the MR pass above; add strong Lucas.
    if (is_perfect_square(n)) return false;
    return strong_lucas_prp(n);
}

// =============================================================================
// uint512 helpers
// =============================================================================

// |a - b| for non-negative a,b. (kept for clarity; not currently used)
// Build a uint512 from unsigned __int128.
uint512 u512_from_u128(unsigned __int128 v) { return uint512(v); }

// =============================================================================
// Per-relation validation
// =============================================================================

struct FailExample {
    size_t index;
    std::string detail;
};

struct ClassResult {
    std::atomic<size_t> count{0};
    std::mutex mtx;
    std::vector<FailExample> examples; // capped
    void record(size_t idx, const std::string& detail) {
        size_t prev = count.fetch_add(1);
        if (prev < 10) {
            std::lock_guard<std::mutex> g(mtx);
            if (examples.size() < 10) examples.push_back({idx, detail});
        }
    }
};

struct BatchResults {
    ClassResult sign_mismatch;
    ClassResult product_mismatch;
    ClassResult incomplete;     // missed FB prime / exponent disagreement
    ClassResult composite_lp;
    ClassResult out_of_range_lp;
    ClassResult structural;     // CSR / data inconsistency (e.g. bad offsets)
    ClassResult smooth_with_lp; // (smooths batch only) a "smooth" carrying lp>1
    std::atomic<size_t> checked{0};
    std::atomic<size_t> partials_with_lp{0};
};

// Validate a single relation `i` of `batch`.
// `is_smooths_batch` = true when validating the FULL-SMOOTHS batch (where every
// relation must have large_prime == 1). Any lp>1 there is a semantic anomaly:
// such a row is not a genuine smooth and is flagged in its own class rather than
// cascading into product/completeness errors.
void validate_relation(const HostRelationBatch& batch, size_t i,
                       const uint512& N,
                       const std::vector<uint32_t>& fb,
                       uint64_t lp_bound,
                       bool is_smooths_batch,
                       BatchResults& R) {
    // --- recompute Q = sqrt_Q^2 - N, with sign ---
    const uint512& sq = batch.sqrt_Q[i];
    uint512 sq2 = sq; sq2.mult(sq);     // (ax+b)^2

    int8_t computed_sign;
    uint512 absQ;
    if (sq2 < N) {
        computed_sign = -1;
        absQ = N; absQ.sub(sq2);        // N - (ax+b)^2
    } else {
        computed_sign = 1;
        absQ = sq2; absQ.sub(N);        // (ax+b)^2 - N
    }

    // recorded sign byte: postprocessing stores int8 sign_of_Q (+1 / -1) into a
    // uint8 field. Compare as signed.
    int8_t stored_sign = (int8_t)batch.signs[i];
    if (stored_sign != computed_sign) {
        R.sign_mismatch.record(i,
            "stored_sign=" + std::to_string((int)stored_sign) +
            " computed_sign=" + std::to_string((int)computed_sign) +
            " sqrt_Q=" + sq.to_string());
    }

    // --- CSR slice bounds ---
    uint64_t off0 = batch.factor_offsets[i];
    uint64_t off1 = batch.factor_offsets[i + 1];
    if (off1 < off0 || off1 > batch.factor_indices.size() ||
        off1 > batch.factor_counts.size()) {
        R.structural.record(i, "bad CSR offsets [" + std::to_string(off0) +
                               "," + std::to_string(off1) + ")");
        return;
    }

    unsigned __int128 lp = batch.large_primes[i];
    uint512 lp512 = u512_from_u128(lp);

    int32_t v2 = batch.val_2_exps[i];

    // A full smooth MUST have large_prime == 1. If the smooths batch carries a
    // row with lp > 1, the row is not a genuine smooth (e.g. dual-append leakage
    // or a leftover slot). Flag it distinctly and stop: the algebraic identity
    // and completeness checks below assume a correctly-classified relation, and
    // running them on such a row only produces noisy, misleading cascades.
    if (is_smooths_batch && lp > 1) {
        R.smooth_with_lp.record(i,
            "smooth has large_prime=" + lp512.to_string() +
            " (must be 1); v2=" + std::to_string(v2) +
            " sqrt_Q=" + sq.to_string());
        R.checked.fetch_add(1);
        return;
    }

    // ============================================================
    // CHECK 1: algebraic identity using the RECORDED factorization
    //   |Q| ?= 2^v2 * PROD(fb[idx]^cnt) * lp
    // Verified by dividing |Q| down and confirming the residue == lp exactly.
    // ============================================================
    {
        uint512 rem = absQ;
        bool clean = true;
        // remove 2^v2
        if (v2 < 0) { clean = false; }
        for (int k = 0; clean && k < v2; ++k) {
            if ((rem.limbs[0] & 1) != 0) { clean = false; break; }
            rem.rshift(1);
        }
        // remove recorded FB primes
        for (uint64_t k = off0; clean && k < off1; ++k) {
            uint32_t idx = batch.factor_indices[k];
            uint8_t cnt = batch.factor_counts[k];
            if (idx >= fb.size()) {
                R.structural.record(i, "factor_index " + std::to_string(idx) +
                                       " >= fb size " + std::to_string(fb.size()));
                clean = false; break;
            }
            uint32_t p = fb[idx];
            for (uint8_t e = 0; e < cnt; ++e) {
                if (rem.mod_uint32(p) != 0) { clean = false; break; }
                rem.div_uint32_inplace(p);
            }
        }
        if (!clean) {
            R.product_mismatch.record(i,
                "could not factor out recorded primes; sqrt_Q=" + sq.to_string());
        } else if (!(rem == lp512)) {
            R.product_mismatch.record(i,
                "residue=" + rem.to_string() + " != large_prime=" + lp512.to_string());
        }
    }

    // ============================================================
    // CHECK 2: completeness — independently re-derive the FB factorization of
    // |Q| (after removing 2^v2) over the ENTIRE factor base, and confirm:
    //   (a) every recorded FB exponent matches the re-derived exponent,
    //   (b) no additional FB prime divides |Q| beyond those recorded,
    //   (c) the final leftover cofactor equals large_prime,
    //   (d) — the key pathology — NO FB prime divides large_prime.
    //
    // (d) is the "we accepted a COMPOSITE cofactor whose small factor is in the
    // factor base" case: it means the sieve overlooked an FB hit, leaving a
    // composite masquerading as a large prime.
    // ============================================================
    {
        // (d) does any FB prime divide the recorded large prime?
        if (lp > 1) {
            for (uint32_t p : fb) {
                if (lp512.mod_uint32(p) == 0) {
                    R.incomplete.record(i,
                        "FB prime " + std::to_string(p) +
                        " divides large_prime=" + lp512.to_string() +
                        " (composite cofactor accepted as LP)");
                    break;
                }
            }
        }

        // Re-derive over the whole FB.
        uint512 rem = absQ;
        bool ok = true;
        // remove 2^v2 first (re-derive the 2-power independently and compare)
        int derived_v2 = 0;
        while (!rem.is_zero() && (rem.limbs[0] & 1) == 0) { rem.rshift(1); ++derived_v2; }
        if (derived_v2 != v2) {
            R.incomplete.record(i,
                "2-power mismatch: recorded v2=" + std::to_string(v2) +
                " derived=" + std::to_string(derived_v2));
            ok = false;
        }

        // Build a map idx->recorded exponent for comparison.
        // (factor base is small relative to the divisions; trial-divide directly.)
        // We accumulate derived exponents per fb index where p | rem.
        // To compare against recorded, we walk recorded list into a small lookup.
        // Recorded entries are few (<= 64), so linear scan is fine.
        if (ok) {
            // For each FB prime, divide out completely, counting exponent.
            for (size_t idx = 0; idx < fb.size(); ++idx) {
                uint32_t p = fb[idx];
                if (rem.mod_uint32(p) != 0) continue;
                uint32_t derived_cnt = 0;
                while (rem.mod_uint32(p) == 0) { rem.div_uint32_inplace(p); ++derived_cnt; }
                // find recorded count for this idx
                uint32_t recorded_cnt = 0;
                for (uint64_t k = off0; k < off1; ++k) {
                    if (batch.factor_indices[k] == idx) { recorded_cnt = batch.factor_counts[k]; break; }
                }
                if (recorded_cnt != derived_cnt) {
                    R.incomplete.record(i,
                        "FB prime " + std::to_string(p) + " exponent mismatch: recorded=" +
                        std::to_string(recorded_cnt) + " derived=" + std::to_string(derived_cnt));
                    ok = false;
                    break;
                }
            }
        }

        // leftover cofactor must equal the recorded large prime.
        if (ok && !(rem == lp512)) {
            R.incomplete.record(i,
                "leftover cofactor=" + rem.to_string() +
                " != recorded large_prime=" + lp512.to_string());
        }
    }

    // ============================================================
    // CHECK 3 & 4: primality and range of the large prime (partials only).
    // ============================================================
    if (lp > 1) {
        R.partials_with_lp.fetch_add(1);

        // RANGE
        uint32_t fb_max = fb.empty() ? 0 : fb.back();
        bool below_or_eq_bound = (lp_bound == 0) ? true : (lp <= (unsigned __int128)lp_bound);
        if (!(lp512 > (uint32_t)fb_max) || !below_or_eq_bound) {
            R.out_of_range_lp.record(i,
                "large_prime=" + lp512.to_string() +
                " fb_max=" + std::to_string(fb_max) +
                " lp_bound=" + std::to_string(lp_bound));
        }

        // PRIMALITY
        if (!is_prime_deterministic(lp512)) {
            // try to surface a small factor for the report
            std::string small = "";
            for (uint32_t p : {2u,3u,5u,7u,11u,13u,17u,19u,23u,29u,31u,37u,41u,43u,47u}) {
                if (lp512.mod_uint32(p) == 0) { small = " smallest_factor<=47:" + std::to_string(p); break; }
            }
            R.composite_lp.record(i,
                "COMPOSITE large_prime=" + lp512.to_string() + small);
        }
    }

    R.checked.fetch_add(1);
}

void run_batch(const char* label, const HostRelationBatch& batch,
               const uint512& N, const std::vector<uint32_t>& fb,
               uint64_t lp_bound, bool is_smooths_batch, BatchResults& R) {
    if (batch.factor_offsets.size() < batch.num_relations + 1) {
        std::fprintf(stderr,
            "[%s] WARNING: factor_offsets size %zu < num_relations+1 %zu — skipping\n",
            label, batch.factor_offsets.size(), batch.num_relations + 1);
        return;
    }
    const long long n = (long long)batch.num_relations;
    #pragma omp parallel for schedule(dynamic, 1024)
    for (long long i = 0; i < n; ++i) {
        validate_relation(batch, (size_t)i, N, fb, lp_bound, is_smooths_batch, R);
    }
}

void print_class(const char* name, ClassResult& c) {
    std::printf("    %-26s : %zu\n", name, c.count.load());
    for (const auto& e : c.examples) {
        std::printf("        [#%zu] %s\n", e.index, e.detail.c_str());
    }
}

// JSON escaping for example detail strings.
std::string json_escape(const std::string& s) {
    std::string o; o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            default:   o += c;      break;
        }
    }
    return o;
}

void emit_class_json(std::ostream& os, const char* name, ClassResult& c, bool last) {
    os << "      \"" << name << "\": { \"count\": " << c.count.load()
       << ", \"examples\": [";
    for (size_t k = 0; k < c.examples.size(); ++k) {
        if (k) os << ", ";
        os << "{ \"index\": " << c.examples[k].index
           << ", \"detail\": \"" << json_escape(c.examples[k].detail) << "\" }";
    }
    os << "] }" << (last ? "" : ",") << "\n";
}

void emit_batch_json(std::ostream& os, const char* label, BatchResults& R, bool last) {
    os << "    \"" << label << "\": {\n";
    os << "      \"checked\": " << R.checked.load() << ",\n";
    os << "      \"partials_with_lp\": " << R.partials_with_lp.load() << ",\n";
    emit_class_json(os, "sign_mismatch",   R.sign_mismatch, false);
    emit_class_json(os, "product_mismatch", R.product_mismatch, false);
    emit_class_json(os, "incomplete_missed_fb_prime", R.incomplete, false);
    emit_class_json(os, "composite_large_prime", R.composite_lp, false);
    emit_class_json(os, "out_of_range_lp", R.out_of_range_lp, false);
    emit_class_json(os, "structural", R.structural, false);
    emit_class_json(os, "smooth_with_large_prime", R.smooth_with_lp, true);
    os << "    }" << (last ? "" : ",") << "\n";
}

// -----------------------------------------------------------------------------
// Checkpoint detection: probe for the fixed EOF footer magic ("MPQS_CKFT").
// -----------------------------------------------------------------------------
bool looksLikeCheckpoint(const std::string& path) {
    std::error_code ec;
    uint64_t sz = std::filesystem::file_size(path, ec);
    if (ec || sz < mpqs::ckpt::CKPT_FOOTER_SIZE) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(static_cast<std::streamoff>(sz - mpqs::ckpt::CKPT_FOOTER_SIZE), std::ios::beg);
    char magic[9];
    f.read(magic, 9);
    return f && std::memcmp(magic, mpqs::ckpt::CKPT_FOOTER_MAGIC, 9) == 0;
}

} // namespace

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <relations.v2|relations.soa|sieve.ckpt|<checkpoint_dir>> [--out <summary.json>]\n"
            "\n"
            "Exhaustively re-validates saved relations on the CPU, with a\n"
            "deterministic primality test on every recorded large prime.\n"
            "Accepts a relations.v2/.soa file, a sieve.ckpt checkpoint (footer + trailer\n"
            "are validated and reported), or a checkpoint directory (picks sieve.ckpt,\n"
            "falling back to sieve.ckpt.prev).\n",
            argv[0]);
        return 2;
    }
    std::string path = argv[1];
    std::string out_path;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) { out_path = argv[++i]; }
        else { std::fprintf(stderr, "Unknown argument: %s\n", a.c_str()); return 2; }
    }

    HostRelationBatch v1batch, smooths, partials;
    mpqs::io::V2Metadata meta;
    int fmt = 0;

    // --- Checkpoint path: a directory, or a file carrying the EOF footer magic. ---
    bool is_ckpt = false;
    {
        std::error_code ec;
        is_ckpt = std::filesystem::is_directory(path, ec) || looksLikeCheckpoint(path);
    }
    if (is_ckpt) {
        mpqs::ckpt::CheckpointLoadResult res;
        bool ok = std::filesystem::is_directory(path)
            ? mpqs::ckpt::loadLatestCheckpoint(path, res)
            : mpqs::ckpt::readCheckpoint(path, res);
        if (!ok || !res.ok) {
            std::fprintf(stderr, "ERROR: failed to load checkpoint from %s (torn/incomplete?)\n",
                         path.c_str());
            return 1;
        }
        const auto& tr = res.trailer;
        std::printf("Loaded sieve.ckpt: footer + trailer OK (schema %u)\n",
                    mpqs::ckpt::CKPT_SCHEMA_VERSION);
        std::printf("  a_index (global_a_index) = %llu\n", (unsigned long long)tr.global_a_index);
        std::printf("  target_relations         = %llu\n", (unsigned long long)tr.target_relations);
        std::printf("  loaded_smooths_raw       = %llu\n", (unsigned long long)tr.loaded_smooths_raw);
        std::printf("  loaded_smooths_dedup     = %llu\n", (unsigned long long)tr.loaded_smooths_dedup);
        std::printf("  loaded_partials          = %llu\n", (unsigned long long)tr.loaded_partials);
        std::printf("  elapsed_sieve_sec        = %llu\n", (unsigned long long)tr.elapsed_sieve_sec);
        std::printf("  cluster_section_present  = %u\n", (unsigned)tr.cluster_section_present);
        // S3 cluster block: completedPrefixCursor (B2) + per-node initial-range high-water (M1).
        if (tr.cluster_section_present) {
            const auto& cb = res.cluster;
            std::printf("  [cluster] completed_prefix_cursor = %llu\n",
                        (unsigned long long)cb.completed_prefix_cursor);
            std::printf("  [cluster] node_count              = %zu\n",
                        cb.initial_high_water.size());
            for (size_t i = 0; i < cb.initial_high_water.size(); ++i)
                std::printf("  [cluster]   node %2zu initial-range high-water = %llu\n",
                            i, (unsigned long long)cb.initial_high_water[i]);
        }
        // Sanity: the deduped trailer count must equal the smooths actually in the payload.
        if (tr.loaded_smooths_dedup != res.smooths.num_relations) {
            std::fprintf(stderr,
                "ERROR: trailer loaded_smooths_dedup=%llu != payload smooths=%zu (corrupt)\n",
                (unsigned long long)tr.loaded_smooths_dedup, res.smooths.num_relations);
            return 1;
        }
        smooths = std::move(res.smooths);
        partials = std::move(res.partials);
        meta = std::move(res.meta);
        fmt = 2;
    } else {
        fmt = mpqs::io::detect_and_deserialize(path, v1batch, smooths, partials, meta);
        if (fmt == 0) {
            std::fprintf(stderr, "ERROR: failed to load relations from %s\n", path.c_str());
            return 1;
        }
    }

    std::vector<uint32_t> fb;
    uint512 N;
    uint64_t lp_bound = 0;
    bool have_smooths = false, have_partials = false;

    if (fmt == 2) {
        fb = meta.factor_base;
        N = meta.N;
        lp_bound = meta.lp_bound;
        have_smooths = have_partials = true;
        std::printf("Loaded v2: N=%s\n", N.to_string().c_str());
        std::printf("  factor_base size = %zu (max prime = %u)\n",
                    fb.size(), fb.empty() ? 0 : fb.back());
        std::printf("  lp_bound = %llu, sieve_bound(M) = %u\n",
                    (unsigned long long)lp_bound, meta.sieve_bound);
        std::printf("  smooths  = %zu relations\n", smooths.num_relations);
        std::printf("  partials = %zu relations\n", partials.num_relations);
    } else {
        // v1: single batch, no metadata / factor base. We can only do the LP
        // primality / range / sign reconstruction is impossible without N+FB.
        std::fprintf(stderr,
            "ERROR: v1 (.soa) format carries no N / factor base metadata; this\n"
            "tool requires v2 (full algebraic re-derivation needs N and the FB).\n"
            "Re-run the pipeline with v2 output, or point at the .v2 file.\n");
        return 1;
    }

    // Sort a copy of fb for max() correctness (it is already ascending, but be safe).
    if (!fb.empty() && !std::is_sorted(fb.begin(), fb.end())) {
        std::sort(fb.begin(), fb.end());
    }

    BatchResults Rs, Rp;
    if (have_smooths) {
        std::printf("\nValidating SMOOTHS (%zu)...\n", smooths.num_relations);
        run_batch("smooths", smooths, N, fb, lp_bound, /*is_smooths_batch=*/true, Rs);
    }
    if (have_partials) {
        std::printf("Validating PARTIALS (%zu)...\n", partials.num_relations);
        run_batch("partials", partials, N, fb, lp_bound, /*is_smooths_batch=*/false, Rp);
    }

    auto total = [](BatchResults& R) {
        return R.sign_mismatch.count.load() + R.product_mismatch.count.load() +
               R.incomplete.count.load() + R.composite_lp.count.load() +
               R.out_of_range_lp.count.load() + R.structural.count.load() +
               R.smooth_with_lp.count.load();
    };

    std::printf("\n================ SUMMARY ================\n");
    std::printf("SMOOTHS: checked=%zu  partials_with_lp=%zu\n",
                Rs.checked.load(), Rs.partials_with_lp.load());
    print_class("sign_mismatch", Rs.sign_mismatch);
    print_class("product_mismatch", Rs.product_mismatch);
    print_class("incomplete/missed_FB_prime", Rs.incomplete);
    print_class("COMPOSITE_large_prime", Rs.composite_lp);
    print_class("out_of_range_LP", Rs.out_of_range_lp);
    print_class("structural", Rs.structural);
    print_class("smooth_with_large_prime", Rs.smooth_with_lp);

    std::printf("\nPARTIALS: checked=%zu  partials_with_lp=%zu\n",
                Rp.checked.load(), Rp.partials_with_lp.load());
    print_class("sign_mismatch", Rp.sign_mismatch);
    print_class("product_mismatch", Rp.product_mismatch);
    print_class("incomplete/missed_FB_prime", Rp.incomplete);
    print_class("COMPOSITE_large_prime", Rp.composite_lp);
    print_class("out_of_range_LP", Rp.out_of_range_lp);
    print_class("structural", Rp.structural);
    print_class("smooth_with_large_prime", Rp.smooth_with_lp);

    size_t grand = total(Rs) + total(Rp);
    std::printf("\nTOTAL FAILURES (all classes, both batches): %zu\n", grand);
    std::printf("RESULT: %s\n", grand == 0 ? "CLEAN — all relations validate" : "FAILURES DETECTED");

    // --- machine-readable summary ---
    auto write_json = [&](std::ostream& os) {
        os << "{\n";
        os << "  \"file\": \"" << json_escape(path) << "\",\n";
        os << "  \"format\": " << fmt << ",\n";
        os << "  \"N\": \"" << N.to_string() << "\",\n";
        os << "  \"factor_base_size\": " << fb.size() << ",\n";
        os << "  \"factor_base_max\": " << (fb.empty() ? 0u : fb.back()) << ",\n";
        os << "  \"lp_bound\": " << lp_bound << ",\n";
        os << "  \"total_failures\": " << grand << ",\n";
        os << "  \"batches\": {\n";
        emit_batch_json(os, "smooths", Rs, false);
        emit_batch_json(os, "partials", Rp, true);
        os << "  }\n";
        os << "}\n";
    };

    if (!out_path.empty()) {
        std::ofstream jf(out_path);
        if (!jf) {
            std::fprintf(stderr, "WARNING: could not open --out path %s; writing JSON to stdout\n",
                         out_path.c_str());
            std::printf("\n--- JSON summary ---\n");
            write_json(std::cout);
        } else {
            write_json(jf);
            std::printf("\nJSON summary written to %s\n", out_path.c_str());
        }
    }

    return grand == 0 ? 0 : 3;
}
