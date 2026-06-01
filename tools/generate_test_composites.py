#!/usr/bin/env python3
"""Generate test composites with specific factor structures for M10 BCD testing.

Categories:
  A — Several coprime prime factors (3-4 primes)
  B — Prime power factors (p^a * q^b, a or b >= 2)
  C — Mixed: coprime factors with one prime power (p^2 * q * r)

All prime factors > 3,000,000 to exceed any FB_bound chosen by autotune
for numbers in the 80-90 bit range.  Product targets: 80-90 bits.

Uses a fixed random seed for reproducibility.
"""

import random
import math

SEED = 42
MIN_PRIME = 3_000_001
rng = random.Random(SEED)


# --- Miller-Rabin primality test ---

def _miller_rabin_witnesses(n, witnesses):
    """Return True if n passes Miller-Rabin for all given witnesses."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Write n-1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in witnesses:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3e24; probabilistic beyond."""
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    return _miller_rabin_witnesses(n, small)


def random_prime(lo, hi):
    """Return a random prime in [lo, hi]."""
    for _ in range(100_000):
        c = rng.randrange(lo, hi + 1) | 1  # ensure odd
        if is_prime(c):
            return c
    raise RuntimeError(f"Failed to find prime in [{lo}, {hi}]")


def bit_length(n):
    return n.bit_length()


# --- Category A: Several coprime prime factors ---

def generate_category_a(count=8):
    """3 coprime primes, product 80-90 bits.
    Each prime ~27-30 bits (range ~134M to ~1B).
    """
    results = []
    lo, hi = 134_000_000, 1_000_000_000  # ~27-30 bits
    attempts = 0
    while len(results) < count and attempts < 10000:
        attempts += 1
        p1 = random_prime(lo, hi)
        p2 = random_prime(lo, hi)
        p3 = random_prime(lo, hi)
        if p1 == p2 or p1 == p3 or p2 == p3:
            continue
        n = p1 * p2 * p3
        if 80 <= bit_length(n) <= 90:
            factors = sorted([p1, p2, p3])
            results.append((n, factors, [1, 1, 1]))
    return results


def generate_category_a_four(count=2):
    """4 coprime primes, product 80-90 bits.
    Each prime ~20-23 bits (range ~3M to ~8M).
    """
    results = []
    lo, hi = 3_000_001, 8_000_000
    attempts = 0
    while len(results) < count and attempts < 10000:
        attempts += 1
        primes = [random_prime(lo, hi) for _ in range(4)]
        if len(set(primes)) < 4:
            continue
        n = math.prod(primes)
        if 80 <= bit_length(n) <= 90:
            factors = sorted(primes)
            results.append((n, factors, [1, 1, 1, 1]))
    return results


# --- Category B: Prime power factors ---

def generate_category_b(count=8):
    """p^2 * q or p^2 * q^2, product 80-90 bits."""
    results = []
    attempts = 0
    while len(results) < count and attempts < 10000:
        attempts += 1
        variant = rng.choice(["p2q", "p2q2"])
        if variant == "p2q":
            # p^2 ~54 bits, q ~27-36 bits
            p = random_prime(134_000_000, 1_000_000_000)   # ~27-30 bits
            q = random_prime(134_000_000, 1_000_000_000)
            if p == q:
                continue
            n = p * p * q
            if 80 <= bit_length(n) <= 90:
                factors = sorted([p, q])
                exps = [2, 1] if factors[0] == p else [1, 2]
                # Reorder exponents to match sorted factors
                if factors[0] == p:
                    exps = [2, 1]
                else:
                    exps = [1, 2]
                results.append((n, factors, exps))
        else:
            # p^2 * q^2: each prime ~20-23 bits
            p = random_prime(3_000_001, 50_000_000)   # ~22-26 bits
            q = random_prime(3_000_001, 50_000_000)
            if p == q:
                continue
            n = p * p * q * q
            if 80 <= bit_length(n) <= 90:
                factors = sorted([p, q])
                results.append((n, factors, [2, 2]))
    return results


# --- Category C: Mixed — p^2 * q * r ---

def generate_category_c(count=5):
    """p^2 * q * r, all primes > 3M, product 80-90 bits."""
    results = []
    # p ~22-25 bits, q,r ~13-20 bits — but all > 3M so ~22 bits min
    # p^2 ~44-50 bits, q*r ~30-46 bits → total 80-90
    lo, hi = 3_000_001, 30_000_000  # ~22-25 bits
    attempts = 0
    while len(results) < count and attempts < 10000:
        attempts += 1
        p = random_prime(lo, hi)
        q = random_prime(lo, hi)
        r = random_prime(lo, hi)
        if len({p, q, r}) < 3:
            continue
        n = p * p * q * r
        if 80 <= bit_length(n) <= 90:
            factors = sorted([p, q, r])
            exps = []
            for f in factors:
                exps.append(2 if f == p else 1)
            results.append((n, factors, exps))
    return results


def format_factorization(factors, exps):
    """Format as '134000009^2 × 500000003 × 700000001'."""
    parts = []
    for f, e in zip(factors, exps):
        if e == 1:
            parts.append(str(f))
        else:
            parts.append(f"{f}^{e}")
    return " × ".join(parts)


def main():
    cat_a3 = generate_category_a(8)
    cat_a4 = generate_category_a_four(2)
    cat_b = generate_category_b(8)
    cat_c = generate_category_c(5)

    lines = []
    lines.append("# Test composites for M10 BCD (coprime refinement) testing")
    lines.append("# Generated by tools/generate_test_composites.py (seed=42)")
    lines.append("# Format: N # = factorization  [bits]")
    lines.append("")

    lines.append("# --- Category A: Several coprime prime factors ---")
    for n, factors, exps in cat_a3 + cat_a4:
        fstr = format_factorization(factors, exps)
        lines.append(f"{n} # = {fstr}  [{bit_length(n)} bits]")

    lines.append("")
    lines.append("# --- Category B: Prime power factors ---")
    for n, factors, exps in cat_b:
        fstr = format_factorization(factors, exps)
        lines.append(f"{n} # = {fstr}  [{bit_length(n)} bits]")

    lines.append("")
    lines.append("# --- Category C: Mixed — p^2 × q × r ---")
    for n, factors, exps in cat_c:
        fstr = format_factorization(factors, exps)
        lines.append(f"{n} # = {fstr}  [{bit_length(n)} bits]")

    output = "\n".join(lines) + "\n"

    # Print to stdout
    print(output)

    # Write to file
    import os
    outpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "multi_factor_candidates.txt")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        f.write(output)
    print(f"Written to {outpath}")


if __name__ == "__main__":
    main()
