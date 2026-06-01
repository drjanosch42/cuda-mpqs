#!/usr/bin/env python3
"""Generate large multi-factor test composites (60-90 digit scale) for GPU pipeline testing.

Categories:
  A — 3 coprime prime factors, 60-75 digits (8 composites)
  B — Prime power factors (p^2*q or p^2*q^2), 60-75 digits (6 composites)
  C — Mixed: p^2*q*r, 70-85 digits (5 composites)
  D — 4 coprime prime factors, 75-90 digits (4 composites)

All prime factors > 10^7 (above any reasonable factor base bound).
Fixed seed (2026) for reproducibility. Pure Python, no external deps.
"""

import random
import math
import os

SEED = 2026
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


def random_prime_digits(min_digits, max_digits):
    """Return a random prime with digit count in [min_digits, max_digits]."""
    lo = 10 ** (min_digits - 1)
    hi = 10 ** max_digits - 1
    for _ in range(200_000):
        c = rng.randrange(lo, hi + 1) | 1
        if is_prime(c):
            return c
    raise RuntimeError(f"Failed to find prime with {min_digits}-{max_digits} digits")


def digit_count(n):
    return len(str(n))


# --- Category A: 3 coprime primes, 60-75 digits ---

def generate_category_a(count=8):
    """N = p1 * p2 * p3, each prime 20-25 digits, product 60-75 digits."""
    results = []
    attempts = 0
    while len(results) < count and attempts < 50000:
        attempts += 1
        p1 = random_prime_digits(20, 25)
        p2 = random_prime_digits(20, 25)
        p3 = random_prime_digits(20, 25)
        if len({p1, p2, p3}) < 3:
            continue
        n = p1 * p2 * p3
        d = digit_count(n)
        if 60 <= d <= 75:
            factors = sorted([p1, p2, p3])
            results.append((n, factors, [1, 1, 1], "A"))
    return results


# --- Category B: Prime power factors, 60-75 digits ---

def generate_category_b(count=6):
    """p1^2 * p2 or p1^2 * p2^2, product 60-75 digits."""
    results = []
    attempts = 0
    # Alternate: 3 of p^2*q, 3 of p^2*q^2
    target_p2q = 3
    target_p2q2 = 3
    got_p2q = 0
    got_p2q2 = 0

    while (got_p2q < target_p2q or got_p2q2 < target_p2q2) and attempts < 100000:
        attempts += 1
        if got_p2q < target_p2q:
            # p1^2 * p2: p1 15-20 digits, p2 20-30 digits
            p1 = random_prime_digits(15, 20)
            p2 = random_prime_digits(20, 30)
            if p1 == p2:
                continue
            n = p1 * p1 * p2
            d = digit_count(n)
            if 60 <= d <= 75:
                factors = sorted([p1, p2])
                exps = [2, 1] if factors[0] == p1 else [1, 2]
                results.append((n, factors, exps, "B"))
                got_p2q += 1
                continue

        if got_p2q2 < target_p2q2:
            # p1^2 * p2^2: p1 and p2 15-18 digits
            p1 = random_prime_digits(15, 18)
            p2 = random_prime_digits(15, 18)
            if p1 == p2:
                continue
            n = p1 * p1 * p2 * p2
            d = digit_count(n)
            if 60 <= d <= 75:
                factors = sorted([p1, p2])
                results.append((n, factors, [2, 2], "B"))
                got_p2q2 += 1

    return results


# --- Category C: Mixed p^2 * q * r, 70-85 digits ---

def generate_category_c(count=5):
    """N = p1^2 * p2 * p3, p1 15-18 digits, p2/p3 18-22 digits, product 70-85 digits."""
    results = []
    attempts = 0
    while len(results) < count and attempts < 100000:
        attempts += 1
        p1 = random_prime_digits(15, 18)
        p2 = random_prime_digits(18, 22)
        p3 = random_prime_digits(18, 22)
        if len({p1, p2, p3}) < 3:
            continue
        n = p1 * p1 * p2 * p3
        d = digit_count(n)
        if 70 <= d <= 85:
            factors = sorted([p1, p2, p3])
            exps = []
            for f in factors:
                exps.append(2 if f == p1 else 1)
            results.append((n, factors, exps, "C"))
    return results


# --- Category D: 4 coprime primes, 75-90 digits ---

def generate_category_d(count=4):
    """N = p1 * p2 * p3 * p4, each 18-23 digits, product 75-90 digits."""
    results = []
    attempts = 0
    while len(results) < count and attempts < 100000:
        attempts += 1
        primes = [random_prime_digits(18, 23) for _ in range(4)]
        if len(set(primes)) < 4:
            continue
        n = math.prod(primes)
        d = digit_count(n)
        if 75 <= d <= 90:
            factors = sorted(primes)
            results.append((n, factors, [1, 1, 1, 1], "D"))
    return results


def format_factorization(factors, exps):
    parts = []
    for f, e in zip(factors, exps):
        if e == 1:
            parts.append(str(f))
        else:
            parts.append(f"{f}^{e}")
    return " \u00d7 ".join(parts)


def main():
    cat_a = generate_category_a(8)
    cat_b = generate_category_b(6)
    cat_c = generate_category_c(5)
    cat_d = generate_category_d(4)

    all_composites = cat_a + cat_b + cat_c + cat_d

    lines = []
    lines.append("# Large multi-factor test composites for GPU pipeline testing")
    lines.append(f"# Generated by tools/generate_test_composites_large.py (seed={SEED})")
    lines.append("# Format: N # = factorization [Category X] [digits]")
    lines.append("")

    lines.append("# --- Category A: 3 coprime prime factors (60-75 digits) ---")
    for n, factors, exps, cat in cat_a:
        fstr = format_factorization(factors, exps)
        lines.append(f"{n} # = {fstr} [Category {cat}] [{digit_count(n)}d]")

    lines.append("")
    lines.append("# --- Category B: Prime power factors (60-75 digits) ---")
    for n, factors, exps, cat in cat_b:
        fstr = format_factorization(factors, exps)
        lines.append(f"{n} # = {fstr} [Category {cat}] [{digit_count(n)}d]")

    lines.append("")
    lines.append("# --- Category C: Mixed p^2 \u00d7 q \u00d7 r (70-85 digits) ---")
    for n, factors, exps, cat in cat_c:
        fstr = format_factorization(factors, exps)
        lines.append(f"{n} # = {fstr} [Category {cat}] [{digit_count(n)}d]")

    lines.append("")
    lines.append("# --- Category D: 4 coprime prime factors (75-90 digits) ---")
    for n, factors, exps, cat in cat_d:
        fstr = format_factorization(factors, exps)
        lines.append(f"{n} # = {fstr} [Category {cat}] [{digit_count(n)}d]")

    output = "\n".join(lines) + "\n"

    # Print to stdout
    print(output)

    # Verify all composites
    print("--- Verification ---")
    for n, factors, exps, cat in all_composites:
        product = 1
        for f, e in zip(factors, exps):
            product *= f ** e
        assert product == n, f"MISMATCH: {n} != product of factors"
        for f in factors:
            assert f > 10**7, f"Factor {f} not > 10^7"
            assert is_prime(f), f"Factor {f} not prime"
        print(f"  OK: {digit_count(n)}d [{cat}] {n}")

    print(f"\nTotal composites: {len(all_composites)}")
    print(f"  Category A: {len(cat_a)}")
    print(f"  Category B: {len(cat_b)}")
    print(f"  Category C: {len(cat_c)}")
    print(f"  Category D: {len(cat_d)}")

    # Write to both output files
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ["non-rsa-candidates.txt", "multi_factor_candidates.txt"]:
        outpath = os.path.join(repo_root, "tests", fname)
        with open(outpath, "w") as f:
            f.write(output)
        print(f"Written to {outpath}")


if __name__ == "__main__":
    main()
