// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/*
 * ============================================================================
 * REFACTORING SUMMARY & AUDIT REPORT
 * ============================================================================
 *
 * 1. CRITICAL BUG FIXES
 * ---------------------
 * - Fixed Undefined Behavior in `Tonelli_Shanks`: The original function lacked a
 *   return statement in the failure case (non-residue input). It now safely returns 0.
 * - Fixed Logic Error in `determineParams`: Removed a hardcoded statement
 *   (`fData->F = 3000000`) that overwrote the dynamic parameter selection logic.
 * - Fixed Memory Leak in `recalc_a`: Replaced raw pointer allocation (`new[]`)
 *   with `std::vector` to ensure RAII compliance and prevent leaks.
 * - Fixed Integer Overflow in Modular Exponentiation: The original `modular_pow`
 *   performed `(res * base) % mod` using 64-bit integers. If `mod > 2^32`,
 *   intermediate products could overflow. Replaced with `mul_mod_u128` helper.
 *
 * 2. TYPE SYSTEM OVERHAUL (Signed -> Unsigned)
 * --------------------------------------------
 * - Adopted strict Unsigned Arithmetic:
 *   - `mpqs::int512` (signed) replaced by `mpqs::uint512` (unsigned).
 *   - `long long int` replaced by strict fixed-width types:
 *     - `uint32_t`: For Factor Base primes (p), roots (r), and offsets (p < 2^32).
 *     - `uint64_t`: For intermediate calculations (squaring u32) and Hensel
 *       lifting (modulo p^2).
 * - Rationale: The Factor Base primes fit comfortably in 32 bits. Using `uint32_t`
 *   reduces memory bandwidth requirements by 50% compared to `long long`.
 *
 * 3. MATHEMATICAL CORRECTIONS
 * ---------------------------
 * - `liftRoot` (Hensel Lifting):
 *   - The original code relied on signed arithmetic `(r*r - N)` producing a
 *     negative result.
 *   - The new implementation uses explicit modular subtraction logic to handle
 *     `r^2 < N` correctly within the unsigned domain (modulo p^2).
 * - `findRoot` & Factor Base Generation:
 *   - Replaced ambiguous casts of BigInt division with `uint512::mod_uint32`,
 *     ensuring correct residue calculation without modifying the dividend.
 *
 * 4. PERFORMANCE IMPROVEMENTS
 * ---------------------------
 * - Bit Manipulation: Replaced iterative loops in `jacobi` and `advanceGray`
 *   with O(1) compiler intrinsics (`__builtin_ctz` / `__ffs`).
 * - BigInt Reductions: Optimized BigInt % small-int operations to avoid
 *   full division overhead.
 * - Architecture: Code organized into `mpqs::sieve` namespace to isolate
 *   polynomial selection logic from global MPQS utilities.
 *
 * ============================================================================
 */

#include "prime_algorithms.h"
#include "graycode.cuh"
#include "uint128_helper.cuh" // mpqs::math::{mul_mod,pow_mod} — u128-safe 64-bit modular ops
#include <cassert>
#include <bit>

namespace mpqs {
namespace sieve {

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * @brief Computes (a * b) % m safely using 128-bit precision.
 * Essential for modular exponentiation where intermediate product > 2^64.
 */
static inline uint64_t mul_mod_u128(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((unsigned __int128)(a) * b) % m);
}

// ============================================================================
// Public Implementation
// ============================================================================

/**
 * @brief Modular Exponentiation: $res \equiv base^{exp} \pmod{modulus}$.
 *
 * Algorithm: Right-to-left binary method (Square-and-Multiply).
 * Complexity: O(log exp).
 *
 * @param base Base $b < m$.
 * @param exp Exponent $e$.
 * @param modulus Modulus $m$.
 * @return $b^e \pmod m$.
 */
uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t modulus) {
    uint64_t res = 1;
    base %= modulus; // Ensure base is reduced

    while (exp > 0) {
        // If current bit is 1, multiply result
        if (exp % 2 == 1) {
            res = mul_mod_u128(res, base, modulus);
        }
        // Square the base
        base = mul_mod_u128(base, base, modulus);
        exp >>= 1;
    }
    return res;
}

/**
 * @brief Calculates the Jacobi Symbol $\left(\frac{a}{n}\right)$.
 *
 * Algorithm: Eisenstein's variation of the Euclidean algorithm.
 *
 * Properties:
 * 1. $\left(\frac{a}{n}\right) \in \{-1, 0, 1\}$.
 * 2. If $n$ is prime, equivalent to Legendre Symbol.
 * 3. $\left(\frac{a}{n}\right) = 0 \iff \gcd(a,n) \neq 1$.
 *
 * @param a The numerator.
 * @param n The denominator (must be odd).
 * @return 1, 0, or -1.
 */
int jacobi(uint32_t a, uint32_t n) {
    assert(n % 2 == 1); // Jacobi is defined for odd n

    a = a % n;
    int t = 1;

    while (a != 0) {
        // 1. Remove factors of 2 from a
        // Count trailing zeros
        int zeros = __builtin_ctz(a);
        a >>= zeros;

        // 2. If 2 appeared an odd number of times, check n mod 8
        // (-1)^((n^2-1)/8) is -1 if n = 3, 5 (mod 8)
        if (zeros % 2 == 1) {
            if (n % 8 == 3 || n % 8 == 5) {
                t = -t;
            }
        }

        // 3. Quadratic Reciprocity Law
        // If a, n = 3 (mod 4), flip sign
        if ((a % 4 == 3) && (n % 4 == 3)) {
            t = -t;
        }

        // 4. Euclidean Step: swap and mod
        uint32_t temp = n;
        n = a;
        a = temp % a;
    }

    if (n == 1) return t;
    return 0; // gcd(a, n) != 1
}

/**
 * @brief Tonelli-Shanks Algorithm.
 * Solves $x^2 \equiv n \pmod p$ where $p$ is an odd prime.
 *
 * Preconditions:
 * 1. $p$ is an odd prime.
 * 2. $\gcd(n, p) = 1$.
 *
 * @param n Quadratic residue.
 * @param p Prime modulus ($p < 2^{32}$).
 * @return Root $r$ such that $r^2 \equiv n \pmod p$. Returns 0 if no root exists.
 */
uint32_t Tonelli_Shanks(uint32_t n, uint32_t p) {
    // 0. Trivial checks
    if (n == 0) return 0;
    // Check Euler's Criterion: n^((p-1)/2) == 1 mod p
    if (pow_mod(n, (p - 1) / 2, p) != 1) {
        return 0; // Not a quadratic residue
    }

    // 1. Factor out powers of 2 from p-1
    // $p - 1 = Q \cdot 2^S$
    uint32_t Q = p - 1;
    uint32_t S = 0;
    while ((Q & 1) == 0) {
        Q >>= 1;
        S++;
    }

    // 2. Find a quadratic non-residue z
    // Since half of numbers are NR, this loop terminates quickly (avg 2 iterations)
    uint32_t z = 2;
    while (pow_mod(z, (p - 1) / 2, p) != p - 1) {
        z++;
    }

    // 3. Initialization
    // M = S
    // c = z^Q
    // t = n^Q
    // R = n^((Q+1)/2)
    uint32_t M = S;
    uint64_t c = pow_mod(z, Q, p); // u64 for intermediate math safety
    uint64_t t = pow_mod(n, Q, p);
    uint64_t R = pow_mod(n, (Q + 1) / 2, p);

    // 4. Main Loop
    while (true) {
        if (t == 0) return 0; // Should not happen if (n,p)=1
        if (t == 1) return (uint32_t)R;

        // Find smallest i (0 < i < M) such that t^(2^i) = 1 mod p
        uint32_t i = 0;
        uint64_t temp_t = t;
        for (i = 1; i < M; i++) {
            temp_t = (temp_t * temp_t) % p;
            if (temp_t == 1) break;
        }

        // If no such i found, something is wrong (input wasn't a residue?)
        if (i == M) return 0;

        // b = c^(2^(M-i-1))
        uint64_t b = c;
        // Repeated squaring M-i-1 times
        for (uint32_t j = 0; j < M - i - 1; j++) {
            b = (b * b) % p;
        }

        // Update variables
        M = i;
        c = (b * b) % p;
        t = (t * c) % p;
        R = (R * b) % p;
    }
}

/**
 * @brief 64-bit Jacobi symbol $\left(\frac{a}{n}\right)$ (== Legendre when n prime).
 *
 * Identical algorithm to the uint32 `jacobi` above, lifted to 64-bit operands.
 * Uses `__builtin_ctzll` for trailing-zero counting and 64-bit `% 8` / `% 4`
 * reductions. Required because aux primes $q_s > $ lp1_bound exceed $2^{32}$.
 *
 * @param a The numerator.
 * @param n The denominator (must be odd).
 * @return 1, 0, or -1.
 */
int jacobi_u64(uint64_t a, uint64_t n) {
    assert(n % 2 == 1); // Jacobi is defined for odd n

    a = a % n;
    int t = 1;

    while (a != 0) {
        // 1. Remove factors of 2 from a
        int zeros = __builtin_ctzll(a);
        a >>= zeros;

        // 2. If 2 appeared an odd number of times, check n mod 8
        if (zeros % 2 == 1) {
            if (n % 8 == 3 || n % 8 == 5) {
                t = -t;
            }
        }

        // 3. Quadratic Reciprocity Law
        if ((a % 4 == 3) && (n % 4 == 3)) {
            t = -t;
        }

        // 4. Euclidean Step: swap and mod
        uint64_t temp = n;
        n = a;
        a = temp % a;
    }

    if (n == 1) return t;
    return 0; // gcd(a, n) != 1
}

/**
 * @brief 64-bit Tonelli-Shanks: solves $x^2 \equiv n \pmod p$ for odd prime $p$.
 *
 * Structurally identical to the uint32 `Tonelli_Shanks` above, but every
 * `x^k mod p` / `t·t mod p` step is routed through the u128-safe primitives
 * `mpqs::math::pow_mod` / `mpqs::math::mul_mod`. The uint32 version's bare
 * `(x*x) % p` products overflow once $p > 2^{32}$; the 64-bit aux primes
 * ($q_s > $ lp1_bound) require the wide-multiply path here.
 *
 * Preconditions: $p$ is an odd prime, $\gcd(n,p)=1$.
 *
 * @param n Quadratic residue.
 * @param p Prime modulus (may exceed $2^{32}$).
 * @return Root $r$ such that $r^2 \equiv n \pmod p$. Returns 0 if no root exists.
 */
uint64_t Tonelli_Shanks_u64(uint64_t n, uint64_t p) {
    using mpqs::math::mul_mod;
    using mpqs::math::pow_mod;

    // 0. Trivial checks
    if (n == 0) return 0;
    // Euler's criterion: n^((p-1)/2) == 1 mod p
    if (pow_mod(n, (p - 1) / 2, p) != 1) {
        return 0; // Not a quadratic residue
    }

    // 1. Factor out powers of 2 from p-1:  p - 1 = Q * 2^S
    uint64_t Q = p - 1;
    uint64_t S = 0;
    while ((Q & 1) == 0) {
        Q >>= 1;
        S++;
    }

    // 2. Find a quadratic non-residue z
    uint64_t z = 2;
    while (pow_mod(z, (p - 1) / 2, p) != p - 1) {
        z++;
    }

    // 3. Initialization
    uint64_t M = S;
    uint64_t c = pow_mod(z, Q, p);
    uint64_t t = pow_mod(n, Q, p);
    uint64_t R = pow_mod(n, (Q + 1) / 2, p);

    // 4. Main Loop
    while (true) {
        if (t == 0) return 0; // Should not happen if (n,p)=1
        if (t == 1) return R;

        // Find smallest i (0 < i < M) such that t^(2^i) = 1 mod p
        uint64_t i = 0;
        uint64_t temp_t = t;
        for (i = 1; i < M; i++) {
            temp_t = mul_mod(temp_t, temp_t, p);
            if (temp_t == 1) break;
        }

        // If no such i found, input wasn't a residue.
        if (i == M) return 0;

        // b = c^(2^(M-i-1)) via repeated squaring
        uint64_t b = c;
        for (uint64_t j = 0; j < M - i - 1; j++) {
            b = mul_mod(b, b, p);
        }

        // Update variables
        M = i;
        c = mul_mod(b, b, p);
        t = mul_mod(t, c, p);
        R = mul_mod(R, b, p);
    }
}

/**
 * @brief Deterministic Miller-Rabin primality test, valid for all $n < 2^{64}$.
 *
 * Uses the fixed witness set {2,3,5,7,11,13,17,19,23,29,31,37}, which is
 * proven sufficient to deterministically classify every 64-bit integer.
 * All modular arithmetic uses the u128-safe `mpqs::math::{mul_mod,pow_mod}`.
 * (Trial division is infeasible for the ~1e11-scale aux primes.)
 *
 * @param n Candidate.
 * @return true iff n is prime.
 */
bool is_prime_u64(uint64_t n) {
    using mpqs::math::mul_mod;
    using mpqs::math::pow_mod;

    if (n < 2) return false;
    // Small-prime quick check / witness handling
    static const uint64_t witnesses[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (uint64_t w : witnesses) {
        if (n == w) return true;
        if (n % w == 0) return false;
    }

    // Write n - 1 = d * 2^r with d odd
    uint64_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }

    for (uint64_t a : witnesses) {
        // a < n is guaranteed here: all witnesses are small primes and any
        // n <= 37 already returned above, so n > 37 > a.
        uint64_t x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1) continue;

        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = mul_mod(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

/**
 * @brief Finds a root of the polynomial $(X - d)^2 \equiv N \pmod p$.
 *
 * Used in MPQS Sieve initialization.
 * Solving for X:
 * $(X - d)^2 \equiv N$
 * $X - d \equiv \pm \sqrt{N}$
 * $X \equiv d + \sqrt{N} \pmod p$
 *
 * Note: Returns only one root. The other is $2d - X \pmod p$?
 * Actually, for sieving we usually need both roots ($d \pm r$).
 * This function returns the primary root $r_{primary} = d + \text{TS}(\text{red\_N})$.
 *
 * @param d Polynomial shift $d$.
 * @param N The number being factored.
 * @param p Prime modulus.
 * @return One root $X \in [0, p-1]$.
 */
uint32_t findRoot(const mpqs::uint512& d, const mpqs::uint512& N, uint32_t p) {
    // 1. Reduce inputs modulo p
    // Uses optimized uint512 % uint32_t
    uint32_t reduced_N = N.mod_uint32(p);
    uint32_t reduced_d = d.mod_uint32(p);

    // 2. Solve y^2 = N mod p
    uint32_t sqrt_N = Tonelli_Shanks(reduced_N, p);

    // 3. Solve X = d + sqrt_N mod p
    // Use uint64 to prevent overflow during addition
    return (uint32_t)((uint64_t(reduced_d) + sqrt_N) % p);
}

// ============================================================================
// Factor Base Generation
// ============================================================================

/**
 * @brief Generates the Factor Base for MPQS.
 *
 * Algorithm:
 * 1. Sieve of Eratosthenes up to bound F.
 * 2. Identify primes p such that N is a quadratic residue mod p (Legendre symbol (N/p) = 1).
 * 3. Compute modular square roots of N mod p (Tonelli-Shanks).
 *
 * @param fData Pointer to factoring data structure.
 */
void generateFactorBase(factoringData* fData) {
    const mpqs::uint512& N = fData->N;
    uint32_t F = fData->F;

    // 1. Sieve of Eratosthenes
    // Optimization: Use vector<bool> or raw byte array for compactness
    std::vector<bool> isComposite(F + 1, false);

    isComposite[0] = true;
    isComposite[1] = true; // 0 and 1 are not prime

    // Mark evens
    for (uint32_t i = 4; i <= F; i += 2) {
        isComposite[i] = true;
    }

    // Sieve odds
    uint32_t sqrtF = (uint32_t)std::sqrt(F);
    for (uint32_t i = 3; i <= sqrtF; i += 2) {
        if (!isComposite[i]) {
            for (uint32_t j = i * i; j <= F; j += 2 * i) {
                isComposite[j] = true;
            }
        }
    }

    // 2. Populate Factor Base
    std::vector<uint32_t> F_B;
    std::vector<uint32_t> rootN;

    // Reserve memory to avoid reallocations
    // Pi(x) approx x/ln(x). For 3M, approx 216k primes. Half are residues.
    size_t est_size = (size_t)(F / log(F) * 0.6);
    F_B.reserve(est_size);
    rootN.reserve(est_size);

    // Include 2 in FB? Usually MPQS handles 2 separately or ensures N is odd.
    // The loop starts at 3, handling odd primes.
    for (uint32_t p = 3; p <= F; p += 2) {
        if (!isComposite[p]) {
            // Compute a = N mod p
            // Use fast modulo reduction from uint512
            uint32_t reduced_N = N.mod_uint32(p);

            // Check Legendre Symbol (reduced_N / p)
            if (jacobi(reduced_N, p) == 1) {
                F_B.push_back(p);

                // Solve x^2 = N mod p
                uint32_t root = Tonelli_Shanks(reduced_N, p);

                // Normalize root: choose the smaller of {r, p-r}
                // This ensures consistency (e.g., 0 <= root <= p/2)
                if (root > (p >> 1)) {
                    root = p - root;
                }
                rootN.push_back(root);
            }
        }
    }

    fData->size = (uint32_t)F_B.size();
    fData->factorBase = std::move(F_B);
    fData->rootN = std::move(rootN);
}

/**
 * @brief Determines optimal sieving parameters.
 *
 * Sets the Factor Base bound (F) and Sieve Interval (M) based on the
 * bit-length of N.
 *
 * Formula for 'a' target:
 * a_target \approx \sqrt{2N} / M
 */
void determineParams(factoringData* fData) {
    const mpqs::uint512& N = fData->N;
    int bits = N.msb(); // Most Significant Bit index

    // Empirical parameters based on magnitude of N
    if(fData->F == 0) {
        if (bits < 180) {
            fData->F = 60000;
	} else if (bits < 220) {
	    fData->F = 350000;
	} else if (bits < 235) {       // 66-70d: L(N) optimal ~450-550K
	    fData->F = 500000;
	} else if (bits < 250) {       // 70-75d: L(N) optimal ~550-850K
	    fData->F = 900000;
	} else if (bits < 280) {       // 75-84d: L(N) optimal ~850K-1.8M
	    fData->F = 1500000;
	} else if (bits < 340) {       // 84-102d: L(N) optimal ~1.8-5M
            fData->F = 3000000;
        } else if (bits < 370) { // RSA-110
            fData->F = 4000000;
        } else if (bits < 400) { // RSA-120
            fData->F = 5000000;
        } else if (bits < 440) { // RSA-130
            fData->F = 7500000;
        } else {                // RSA-140+
            fData->F = 10000000;
	}
    }

    if(fData->M == 0) {
        // Fixed Sieve Interval
        // M = 2^18 = 262144
        fData->M = 262144;
    }

    // Calculate a_target = sqrt(2 * N) / M
    // 1. 2 * N
    mpqs::uint512 twoN = N;
    twoN.lshift(1);

    // 2. sqrt(2 * N)
    mpqs::uint512 sqrt2N = twoN.sqrt();

    // 3. Divide by M
    sqrt2N.div_uint32_inplace(fData->M); // Performs division, sqrt2N becomes the quotient
    fData->a_target = sqrt2N; // Note: div_uint32_inplace updates the object to be the quotient
}

// ============================================================================
// Hypercube / Polynomial Utils
// ============================================================================

/**
 * @brief Generates a Hamiltonian path on a Hypercube.
 *
 * Used for the Knuth-Schroeppel algorithm to iterate through square-free
 * integers 'a' by changing only one prime factor at a time (Gray Code).
 *
 * @param direction Output array for dimension indices.
 * @param sign Output array for add/subtract direction (+1 or -1).
 * @param dim Dimension of the hypercube (number of factors in 'a').
 */
void generateHypercubePath(int* direction, int8_t* sign, int dim) {
    // Base case: midpoint 0
    int currentMidpoint = 0;

    // Iteratively build the path
    for (int d = 0; d < dim; d++) {
        // The new dimension is introduced at the midpoint
        direction[currentMidpoint] = d;
        sign[currentMidpoint] = 1; // Arbitrary start sign

        // Reflection step (Gray code property)
        // Copy the previous pattern in reverse order to the upper half
        for (int i = currentMidpoint + 1; i <= 2 * currentMidpoint; i++) {
            direction[i] = direction[2 * currentMidpoint - i];
            sign[i] = -sign[2 * currentMidpoint - i];
        }

        // Update midpoint for next iteration
        currentMidpoint = 2 * currentMidpoint + 1;
    }
}

// ============================================================================
// Modular Arithmetic & Number Theory
// ============================================================================

/**
 * @brief Hensel Lifting: Lifts a root from mod p to mod p^2.
 *
 * Solves $f(x) \equiv 0 \pmod{p^2}$ given root $r \pmod p$.
 * Here $f(x) = x^2 - N$.
 * Formula: $r_{new} = r - \frac{r^2 - N}{2r} \pmod{p^2}$.
 *
 * Implementation Details:
 * Uses strictly unsigned arithmetic with the formula:
 * $r_{new} = r - (r^2 - N) \cdot (2r)^{-1} \pmod{p^2}$.
 * The term $(2r)^{-1}$ is the inverse modulo p (valid due to Hensel lemma properties).
 *
 * @param r Root modulo p.
 * @param reduced_N The value N % p^2 (passed as u64).
 * @param p The prime modulus.
 * @return Root modulo p^2.
 */
uint64_t liftRoot(uint64_t r, uint64_t reduced_N, uint32_t p) {
    uint64_t p_sq = (uint64_t)p * p;

    // 1. Calculate Difference: diff = (r^2 - N) mod p^2
    //    We use unsigned logic: if r^2 < N, result is p^2 - (N - r^2)
    uint64_t r_sq = mul_mod_u128(r, r, p_sq);
    uint64_t diff;

    if (r_sq >= reduced_N) {
        diff = r_sq - reduced_N;
    } else {
        diff = p_sq - (reduced_N - r_sq);
    }

    // 2. Calculate Inverse: inv = (2r)^-1 mod p
    //    Note: We only need inverse mod p, because the diff term is a multiple of p.
    //    This significantly simplifies calculation (u32 vs u64).
    uint32_t two_r = (2 * (uint32_t)r) % p;
    uint32_t inv_p = modInv(two_r, p);

    // 3. Calculate Correction Term: term = (diff * inv) mod p^2
    //    Even though inv is mod p, (diff * inv) correctly adjusts the result mod p^2
    //    because diff is a multiple of p.
    uint64_t term = mul_mod_u128(diff, inv_p, p_sq);

    // 4. Subtract: res = (r - term) mod p^2
    if (r >= term) {
        return r - term;
    } else {
        return p_sq - (term - r);
    }
}

/**
 * @brief Computes Modular Inverse $a^{-1} \pmod p$.
 *
 * Uses Fermat's Little Theorem: $a^{p-2} \equiv a^{-1} \pmod p$.
 * Valid because p is prime in the Factor Base.
 *
 * @param a The number to invert.
 * @param p The prime modulus.
 * @return $a^{-1} \pmod p$.
 */
uint32_t modInv(uint32_t a, uint32_t p) {
    // Exponent = p - 2
    return (uint32_t)pow_mod(a, p - 2, p);
}

/**
 * @brief Computes Modular Inverse $a^{-1} \pmod{p^2}$.
 *
 * Uses Euler's Theorem: $a^{\phi(p^2)-1} \equiv a^{-1} \pmod{p^2}$.
 * $\phi(p^2) = p^2 - p$.
 * Exponent = $p^2 - p - 1$.
 *
 * @param a The number to invert.
 * @param p The prime modulus (result is mod p^2).
 * @return $a^{-1} \pmod{p^2}$.
 */
uint64_t modInvSquare(uint64_t a, uint32_t p) {
    uint64_t p_sq = (uint64_t)p * p;
    uint64_t exponent = p_sq - p - 1;

    return pow_mod(a, exponent, p_sq);
}

// ============================================================================
// Polynomial Selection: Initialization & State Management
// ============================================================================

/**
 * @brief Initializes the 'a' coefficient factors for MPQS.
 *
 * Algorithm:
 * Selects a subset of primes from the Factor Base such that their product is
 * approximately equal to a_target ($\approx \sqrt{2N}/M$).
 *
 * Methodology:
 * 1. Defines a search window in the Factor Base around index 150.
 * 2. Multiplies primes into a product until it exceeds the lower bound.
 * 3. Divides primes out if the product exceeds the upper bound.
 * 4. Stores the indices of the selected primes in `fData->a_factors`.
 *
 * @param fData Pointer to factoring data structure.
 */
void init_a_factors(factoringData* fData) {
    LOG_SET_SUBMODULE("InitAF");
    const std::vector<uint32_t>& factorBase = fData->factorBase;
    const mpqs::uint512& lowerBound = fData->a_target;

    // Upper bound is approx 1.05 * lowerBound
    mpqs::uint512 upperBound = lowerBound;
    upperBound.mult_uint32(100);
    upperBound.div_uint32_inplace(95);

    // Initial indices for the sliding window search
    // We skip small primes to improve sieving efficiency
    uint32_t currentUpperIndex = 150; // Index of prime to potentially include
    uint32_t currentLowerIndex = 150; // Index of prime to potentially exclude

    mpqs::uint512 product((uint32_t)1);

    // Sliding window loop to find product ~ a_target
    // Note: We access factorBase unsafely here (assuming size > 150),
    // real implementation should check bounds.
    while (product < lowerBound || product > upperBound) {
        if (currentUpperIndex >= factorBase.size() || currentLowerIndex >= factorBase.size()) break;
        if (product < lowerBound) {
            // Multiply by next available prime
            product.mult_uint32(factorBase[currentUpperIndex]);
            currentUpperIndex += 2; // Step 2 to allow for "gaps" in Knuth's strategy
        }
        if (product > upperBound) {
            // Divide by lowest included prime
            product.div_uint32_inplace(factorBase[currentLowerIndex]);
            currentLowerIndex += 2;
        }
    }

    // Populate the factor indices
    uint32_t numFactors = (currentUpperIndex - currentLowerIndex) / 2;
    std::vector<uint32_t> factors(numFactors);
    for (uint32_t i = 0; i < numFactors; i++) {
        factors[i] = currentLowerIndex + 2 * i;
    }

    // Debug Output (Consider removing for pure production HPC)

    LOG(LOG_DEBUG_2) << "Prime product: " << product.to_string();
    mpqs::uint512 newProd = product;
    newProd.mult_uint32(factorBase[currentUpperIndex]);
    newProd.div_uint32_inplace(factorBase[currentLowerIndex]);
    LOG(LOG_DEBUG_2) << "Number of factors: " << numFactors << ", Lowest Prime: " <<factorBase[currentLowerIndex];


    // Initialize Hypercube / Gray code parameters
    fData->max_a_index = 1 << factors.size();
    fData->current_a_index = 0;

    // Set pivot points for sliding the window later (advance_a_factors)
    if (factors.size() % 2 == 0) {
        fData->lowerHalfStart = currentLowerIndex + factors.size() - 2;
        fData->upperHalfStart = currentLowerIndex + factors.size();
    } else {
        fData->lowerHalfStart = currentLowerIndex + factors.size() - 2;
        fData->upperHalfStart = currentLowerIndex + factors.size() + 2;
    }

    fData->a_factors = std::move(factors);

    // Calculate initial 'a' and 'B' values
    recalc_a(fData);
}

/**
 * @brief Helper to shift factor indices in the Factor Base array.
 *
 * Used by `advance_a_factors` to slide the window of chosen primes when
 * the current Hypercube traversal is exhausted.
 *
 * @param indices The vector of prime indices to modify.
 * @param pivotPrime The index acting as the boundary/pivot.
 * @param direction +1 for shifting up, -1 for shifting down.
 * @param stepSize The stride (usually 2).
 */
void advance(std::vector<uint32_t>& indices, int pivotPrime, int direction, int stepSize) {
    int startIndex = 0;

    // Determine the split point based on parity of factor count
    if (indices.size() % 2 == 0) {
        startIndex = (indices.size() - 1 + direction) / 2;
    } else {
        startIndex = (indices.size() - 1 + 2 * direction) / 2;
    }

    // Shift to the Right (Higher indices)
    if (direction > 0) {
        for (size_t i = startIndex; i < indices.size() - 1; i++) {
            // Check if there is space to move current index
            if (indices[i] + stepSize < indices[i+1]) {
                indices[i] += stepSize;
                // Reset all indices to the left of the changed index to be contiguous
                for (size_t j = startIndex; j < i; j++) {
                    indices[j] = pivotPrime + stepSize * (int)(j - startIndex);
                }
                return;
            }
        }
        // If loop completes, shift the last element
        indices[indices.size() - 1] += stepSize;
    }

    // Shift to the Left (Lower indices)
    if (direction < 0) {
        for (size_t i = startIndex; i > 0; i--) {
            if ((int)indices[i] - stepSize > (int)indices[i-1]) {
                indices[i] -= stepSize;
                // Reset all indices to the right of the changed index
                for (size_t j = startIndex; j > i; j--) {
                    indices[j] = pivotPrime - stepSize * (int)(startIndex - j);
                }
                return;
            }
        }
        // If loop completes, shift the first element
        if (indices[0] < (uint32_t)stepSize) return;  // Prevent uint32_t underflow at FB boundary
        indices[0] -= stepSize;
    }
}

/**
 * @brief Advances the selection of 'a' factors.
 *
 * Strategy:
 * 1. Uses Gray codes to traverse the hypercube of the current set of primes.
 *    This allows changing 'a' by swapping just one prime for another (O(1) update).
 * 2. If the hypercube is exhausted, it slides the window of primes to find a
 *    new set, maintaining the product size ~ a_target.
 *
 * Visualization of sliding window strategy:
 * ("o" = prime included, "-" = prime skipped)
 *
 * -----ooooooo-----  (Initial state)
 * ----o-ooooo-o----  (Spread out)
 * ----oo-ooo-oo----
 * ----ooo-o-ooo----
 * ---o--ooooo--o---
 * ---o-o-ooo-o-o---
 *
 * @param fData Pointer to factoring data.
 * @param steps Number of Gray code steps to advance.
 */
void advance_a_factors(factoringData* fData, int steps) {
    std::vector<uint32_t>& bits = fData->a_factors;

    // Case 1: Standard traversal within the current Hypercube
    if (fData->current_a_index + steps < fData->max_a_index) {
        uint32_t indexToChange;
        for (int i = 0; i < steps; i++) {
            // Determine which bit flips in the Gray code sequence
            indexToChange = advanceGray(fData->current_a_index + i);

            // Adjust the index in the factor list based on parity
            // This effectively swaps a prime with its neighbor in the stride pattern
            if (bits[indexToChange] % 2 == 0) {
                bits[indexToChange]++;
            } else {
                bits[indexToChange]--;
            }
        }
        fData->current_a_index += steps;
        recalc_a(fData);
        return;
    }

    // Case 2: Hypercube exhausted, slide the window
    // First, reset any odd parities from the previous traversal
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i] % 2 == 1) {
            bits[i]--;
        }
    }

    // Slide the window outwards symmetrically
    advance(bits, fData->upperHalfStart, 1, 2);  // Move upper half right
    advance(bits, fData->lowerHalfStart, -1, 2); // Move lower half left

    // Reset Gray code index for the new set
    fData->current_a_index = 0;
    recalc_a(fData);
}

/**
 * @brief Re-calculates 'a' and 'B' values after factors change.
 *
 * When the set of primes composing 'a' changes, we must:
 * 1. Compute the new value of 'a'.
 * 2. Compute the new 'B' values used for solving the congruence $x^2 \equiv N \pmod a$.
 *
 * Optimization:
 * Uses a Prefix-Suffix product approach to calculate partial products efficiently.
 * For each $p_i$ in 'a', we need $P_i = (a / p_i)$.
 * The B value is derived from $P_i^{-1} \pmod{p_i}$.
 *
 * @param fData Pointer to factoring data.
 */
void recalc_a(factoringData* fData) {
    const std::vector<uint32_t>& factorBase = fData->factorBase;
    const std::vector<uint32_t>& a_factors = fData->a_factors;
    size_t size = a_factors.size();

    // Temp storage for prefix products
    // We use std::vector to manage memory safely (RAII)
    std::vector<mpqs::uint512> result(size);

    // 1. Compute Prefix Products
    // result[i] will momentarily hold p_0 * ... * p_{i-1}
    result[0] = mpqs::uint512((uint32_t)1);
    for (size_t i = 1; i < size; i++) {
        result[i] = result[i-1];
        result[i].mult_uint32(factorBase[a_factors[i-1]]);
    }

    // 2. Compute Suffix Products and final 'a'
    // Simultaneously update result[i] to be (Prefix * Suffix) = a / p_i
    mpqs::uint512 suffix((uint32_t)1);

    for (int i = (int)size - 1; i >= 0; i--) {
        result[i].mult(suffix);
        suffix.mult_uint32(factorBase[a_factors[i]]);
    }

    // Now 'suffix' holds the full product 'a'
    fData->a = suffix;

    // 3. Compute 'B' values
    // Gamma_i = ( (a/p_i)^-1 mod p_i ) * rootN[p_i] mod p_i
    std::vector<mpqs::uint512> new_B_values;
    new_B_values.reserve(size);

    for (size_t i = 0; i < size; i++) {
        uint32_t p = factorBase[a_factors[i]];

        // Calculate (a / p_i) mod p
        uint32_t a_div_p_mod = result[i].mod_uint32(p);

        // Calculate Modular Inverse
        uint32_t inv = modInv(a_div_p_mod, p);

        // Calculate Gamma term
        // Gamma = (inv * root) % p
        uint64_t gamma_u64 = (uint64_t)inv * fData->rootN[a_factors[i]];
        uint32_t gamma = (uint32_t)(gamma_u64 % p);

        // Centering logic for Knuth-Schroeppel
        // We want the root that minimizes the offset
        if (gamma > (p >> 1)) {
            gamma = p - gamma;
        }

        // B_i = Gamma * (a / p_i)
        // result[i] currently holds (a / p_i)
        result[i].mult_uint32(gamma);
        new_B_values.push_back(result[i]);
    }

    fData->B_values = std::move(new_B_values);
}

/**
 * @brief Used by batch sieving instead of update_a
 * CAUTION: This does NOT update a.
 * The update(s) of a will be computed ON DEVICE.
 */
std::vector<uint32_t> prepareNextBatchIndices(factoringData* fData, int batch_size) {
    int dim = (int)fData->a_factors.size();
    std::vector<uint32_t> batch_indices;
    batch_indices.reserve(batch_size * dim);

    // We simulate the progression of the Hypercube/Sliding Window for K steps.
    // At each step, we record the current indices, then advance the state.

    for (int step = 0; step < batch_size; ++step) {
        // 1. Record current configuration
        // This effectively flattens the 2D structure (Step x Dim) into 1D
        batch_indices.insert(batch_indices.end(), fData->a_factors.begin(), fData->a_factors.end());

        // 2. Advance State (Identical logic to advance_a_factors)
        if (fData->current_a_index + 1 < fData->max_a_index) {
            // --- Case A: Hypercube Traversal (Gray Code) ---

            // Calculate which bit flips to get to the next state
            uint32_t bitToFlip = advanceGray(fData->current_a_index);

            // Update the specific factor index (swap with neighbor in FactorBase)
            // Stride is 1 because factors are packed in 'a_factors',
            // but the values represent indices in 'factorBase'.
            // The logic preserves the property that factors are distinct.
            if (fData->a_factors[bitToFlip] % 2 == 0) {
                fData->a_factors[bitToFlip]++;
            } else {
                fData->a_factors[bitToFlip]--;
            }

            fData->current_a_index++;

        } else {
            // --- Case B: Hypercube Exhausted (Slide Window) ---

            // 1. Reset any odd parities (return to "canonical" base state)
            for (size_t i = 0; i < fData->a_factors.size(); i++) {
                if (fData->a_factors[i] % 2 == 1) {
                    fData->a_factors[i]--;
                }
            }

            // 2. Slide the window of primes
            // Uses the existing 'advance' helper in prime_algorithms.cu
            advance(fData->a_factors, fData->upperHalfStart, 1, 2);  // Move upper half right
            advance(fData->a_factors, fData->lowerHalfStart, -1, 2); // Move lower half left

            // 3. Reset Gray code counter
            fData->current_a_index = 0;
        }

        // Note: We do NOT call recalc_a(fData) here.
        // The Host does not need the actual 'a' or 'B_values' BigInts
        // because the GPU 'generatePolynomialsKernel' will compute them
        // from the indices we just recorded.
    }

    return batch_indices;
}

} // namespace sieve
} // namespace mpqs
