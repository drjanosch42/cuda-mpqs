# Common Module (`src/common/`)

## Overview

Shared primitives for the CUDA-MPQS pipeline: 512-bit unsigned arithmetic (`uint512`), Montgomery modular arithmetic, SoA relation containers with device/host transfer, math utilities, and structured logging.

Static library `mpqs_common`. Separable CUDA compilation ON. Links `cudampqs_build_flags`. No external dependencies beyond CUDA and Thrust.

## Files

| File | Purpose |
|------|---------|
| `uint512.cuh` | 512-bit unsigned integer (16x`uint32_t`, little-endian). Full arithmetic, division, modular ops, shifts, sqrt. All methods `__host__ __device__`. |
| `montgomery.cuh` | Montgomery form context (`mpqs::math::Montgomery`): CIOS multiplication, squaring, exponentiation. R = 2^512. |
| `math_utils.cuh` | `modpow`, `mod_inverse`, `gcd`, `legendre_symbol`, `sqrt_mod_3mod4`, `crt_2`, `calculateQ_unsigned`, `calculate_sqrt_of_QX`. |
| `uint128_helper.cuh` | Safe 64-bit modular arithmetic (`add_mod`, `sub_mod`, `mul_mod`, `pow_mod`) via `__int128` intermediates. |
| `mpqs_structures.h` | AoS `Relation` struct (`mpqs::structures`): polynomial coefficients (a, b, x), factor list (max 64 entries), large prime cofactor. |
| `mpqs_soa.h` | SoA containers (`mpqs::structures`): `HostRelationBatch` (host), `RelationBatch` (device), `RelationBatchView` (kernel-passable). |
| `mpqs_soa.cu` | SoA batch operations: resize, append, validation kernel, D-to-H/H-to-D transfer, counter management. |
| `logger/hpc_logger.h` | `HPCLogger` singleton + `LogMessage` proxy. Severity levels, stage tracking, macros. |
| `logger/hpc_logger.cpp` | Logger implementation: thread-safe writes, console/file output, CSV mode. |
| `relation_io.h` | Binary I/O interface (`mpqs::io`): v1 (`HostRelationBatch` + projected LP) and v2 (full smooths + raw partials + `V2Metadata`) serialization. Auto-detection via `detect_and_deserialize`. |
| `relation_io.cpp` | Implementation of v1/v2 serialize/deserialize with `"MPQS_SOA\0"` and `"MPQS_V2\0"` magic headers. |
| `cuda_check.h` | `CUDA_CHECK(call)` / `CUDA_CHECK_FATAL(call)` macros: log error via `HPCLogger` at `LOG_ERROR_CRITICAL` then throw `std::runtime_error`. |

## Key Data Structures

### uint512

16x`uint32_t` limbs, little-endian (`limbs[0]` = LSB). All methods `__host__ __device__`. Namespace: `mpqs`.

Free helpers in `mpqs` namespace: `clz32(uint32_t)`, `ctz32(uint32_t)` — portable CLZ/CTZ using CUDA intrinsics or `__builtin_clz`/`__builtin_ctz`.

| Category | Methods |
|----------|---------|
| Constructors | `uint512()`, `uint512(uint32_t)`, `uint512(uint64_t)`, `uint512(unsigned __int128)`, `uint512(const char*)`, `uint512(const uint32_t(&)[16])` |
| Factory | `static max_value()` — returns 2^512 - 1 |
| Arithmetic | `add`, `sub`, `mult`, `div`, `mod`, `div_mod_core(divisor, remainder_out*)` |
| Small-type arithmetic | `add_uint32(uint32_t)`, `mult_uint32(uint32_t)`, `mul_uint64_inplace(uint64_t)` → returns overflow carry, `div_uint32_inplace(uint32_t)` → returns remainder, `div_uint64_inplace(uint64_t)` → returns remainder, `div_uint32_const(uint32_t)` → non-destructive quotient, `mod_uint32(uint32_t)` → const read-only remainder |
| Modular | `add_mod`, `sub_mod`, `double_mod`, `mul_mod`, `negate_mod_inplace`, `additive_inverse_mod_n` |
| Specialized | `mul_add_mod_signed(int64_t x, b, N)` — computes `(a*x + b) mod N` for polynomial evaluation |
| Bit ops | `lshift`, `rshift`, `msb()`, `countr_zero()`, `msb_is_set()` |
| Queries | `is_zero()`, `is_one()`, `fits_in_128()`, `to_uint128()` |
| Number theory | `sqrt()` — Newton-Raphson integer square root |
| Signed helpers | `abs_twos_complement(int8_t& sign)` — interprets as two's complement, returns magnitude and sign |
| Conversion | `to_string()` (decimal, host only), `to_hex_string()` (host only), `print()` (device-compatible) |
| Operators | Full set: `+`, `-`, `*`, `/`, `%`, comparisons. Mixed-type overloads for `uint32_t` and `uint64_t`. |

### Relation (AoS)

Namespace: `mpqs::structures`.

```cpp
struct Relation {
    uint64_t relation_index;
    uint512 a, b;                            // Polynomial coefficients
    int64_t x;                               // Sieve offset
    uint8_t sign_of_Q;                       // 1 = positive, 255 = negative (two's complement -1)
    uint32_t val_2_exp;                      // Power of 2 in factorization
    struct Factor { uint32_t p_index; uint8_t count; };
    uint32_t num_factors;
    Factor factors[64];                      // {fb_index, multiplicity} pairs
    unsigned __int128 large_prime_remainder; // 0=error, 1=full, >1=LP cofactor
};
```

Semantics: `(ax+b)^2 - N = sign * 2^val_2_exp * prod(p_i^count_i) * large_prime_remainder`.

### HostRelationBatch (SoA, host-side)

Namespace: `mpqs::structures`.

| Field | Type | Description |
|-------|------|-------------|
| `sqrt_Q` | `vector<uint512>` | \|ax+b\| for each relation |
| `signs` | `vector<uint8_t>` | Sign of Q(x): 1 = positive, 255 = negative |
| `val_2_exps` | `vector<int32_t>` | Power of 2 |
| `large_primes` | `vector<unsigned __int128>` | Large prime cofactor (1 if full relation) |
| `factor_offsets` | `vector<uint64_t>` | CSR row pointers, size n+1 |
| `factor_indices` | `vector<uint32_t>` | Factor base indices (CSR values) |
| `factor_counts` | `vector<uint8_t>` | Multiplicities (CSR values) |
| `num_relations` | `size_t` | Fill count |
| `num_factors` | `size_t` | Fill count |

Methods: `resize(n_rels, n_factors)` — grow-only; `clear()` — clears all vectors and resets counts.

### RelationBatch (SoA, device-side)

Namespace: `mpqs::structures`. Device-resident SoA with atomic dual-counters and pinned host mirrors. Non-copyable, move-only.

| Method | Description |
|--------|-------------|
| `initiate(device_id)` | Set device, allocate pinned counters and device atomics |
| `resize(n_rels, n_factors)` | Non-destructive grow; preserves existing data |
| `clear(stream)` | Reset counters to zero (no deallocation) |
| `append(other, count, stream)` | Safe append with overflow discard; rebases CSR offsets via kernel |
| `reset_counters(stream)` | Zero atomic counters only (no data zeroing) |
| `moveToHost(dest, stream)` | Download to `HostRelationBatch` |
| `uploadFromHost(host_batch)` | Upload from `HostRelationBatch` (for `LINALG_ONLY` mode) |
| `getCount(stream)` | Sync read of relation count from device atomic |
| `getFactorCount(stream)` | Sync read of factor count from device atomic |
| `setTargetCap(cap)` | Set adaptive convergence relation cap (0 = unlimited); propagated to `RelationBatchView` |
| `requestStats(stream)` / `updateStats(stream)` / `readStats()` | Async stats via pinned memory or tiny kernel |
| `get_view()` | Returns `RelationBatchView` for kernel launch |
| `validate_relations(N, d_fb, fb_size)` | GPU validation kernel: checks `|(ax+b)^2 - N| == product` |
| `validate_host_batch(batch, fb, N)` | Static helper: uploads host batch, validates on GPU, cleans up |
| `getLargePrimesData()` / `getFactorOffsetsData()` | Raw device pointer accessors (for Thrust) |
| `getCapacityRels()` / `getCapacityFactors()` | Allocated capacity (not fill count) |

### RelationBatchView (kernel-passable)

Namespace: `mpqs::structures`. Raw device pointers + atomic counters + bounds. Passed by value to CUDA kernels for lock-free relation accumulation.

| Field | Description |
|-------|-------------|
| `sqrt_Q`, `signs`, `val_2_exps`, `large_primes` | Per-relation arrays |
| `factor_offsets`, `factor_indices`, `factor_counts` | CSR factor arrays |
| `global_count`, `global_factor_idx` | Device atomic counters |
| `max_relations`, `max_factors` | Bounds for overflow checking |
| `target_cap` | Adaptive convergence ceiling (0 = unlimited) |

### Montgomery

Namespace: `mpqs::math`.

| Field/Method | Description |
|-------------|-------------|
| `N` | Modulus (must be odd) |
| `R2` | R^2 mod N, where R = 2^512 |
| `n_prime` | -N^{-1} mod 2^{32} |
| `Montgomery()` | Default constructor (uninitialized) |
| `Montgomery(N)` | Computes `R2 = (2^511 mod N * 2 mod N)^2 mod N` and `n_prime` via Newton's method |
| `transform(a)` | Standard to Montgomery form: `a*R mod N` |
| `reduce(a)` | Montgomery to standard form: `a*R^{-1} mod N` |
| `mul(a, b)` | CIOS Montgomery multiplication: `a*b*R^{-1} mod N` |
| `sqr(a)` | Montgomery squaring (delegates to `mul`) |
| `pow(base_mont, exp)` | Binary exponentiation in Montgomery form; result in Montgomery form |

## Math Utilities (`math_utils.cuh`)

All functions in `mpqs::math`, all `__host__ __device__ inline` unless noted.

| Function | Signature | Description |
|----------|-----------|-------------|
| `modpow` | `(uint512 base, uint512 exp, const uint512& mod)` | Binary exponentiation |
| `modpow` | `(uint512 base, uint32_t exp, const uint512& mod)` | `uint32_t` exponent overload |
| `mod_inverse` | `(const uint512& a, const uint512& m)` | Extended Euclidean; returns 0 if not coprime |
| `mod_inverse` | `(const uint512& a, uint32_t m)` → `uint32_t` | Fast 32-bit modulus overload |
| `gcd` | `(uint512 a, uint512 b)` | Euclidean algorithm |
| `legendre_symbol` | `(uint512 a, const uint512& p)` | Euler criterion: returns 0, 1, or p-1 |
| `sqrt_mod_3mod4` | `(uint512 n, const uint512& p)` | Square root mod p for p ≡ 3 (mod 4); returns 0 if non-residue |
| `crt_2` | `(r1, m1, r2, m2)` | Chinese Remainder Theorem for 2 congruences; assumes gcd(m1,m2)=1 |
| `calculateQ_unsigned` | `(a, b, x, N, Q_out, sign_out)` | `__device__` only. Computes \|(ax+b)^2 - N\| with sign |
| `calculate_sqrt_of_QX` | `(a, b, x, sqrt_out)` | `__host__ __device__`. Computes \|ax+b\| |

## uint128 Helpers (`uint128_helper.cuh`)

All functions in `mpqs::math`, all `__host__ __device__ __forceinline__`. Typedef `uint128_t = unsigned __int128`. Operate on `uint64_t` using `__int128` intermediates to prevent overflow.

| Function | Description |
|----------|-------------|
| `add_mod(a, b, m)` | `(a + b) % m`, overflow-safe |
| `sub_mod(a, b, m)` | `(a - b) % m`, underflow-safe |
| `mul_mod(a, b, m)` | `(a * b) % m`, essential for Hensel lifting where `p^2 > 2^32` |
| `pow_mod(base, exp, m)` | Binary exponentiation with `mul_mod` |

## Logging (`HPCLogger`)

Singleton logger with thread-safe writes, console + file output, optional CSV mode.

**Severity levels:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `LOG_RESULT` | -4 | Factorization result only (`--mute` mode) |
| `LOG_ERROR_CRITICAL` (`LOG_ERROR`) | -3 | Fatal errors, unrecoverable CUDA failures |
| `LOG_WARNING` (`LOG_ERROR_MAJOR`) | -2 | Warnings, non-fatal errors; `LOG_ERROR_MAJOR` is a backward-compat alias |
| `LOG_INFO` | -1 | Stage transitions, key milestones (default console threshold) |
| `LOG_STATS` | 0 | Statistics summaries, buffer telemetry, throughput (`--verbose`) |
| `LOG_DEBUG_1` through `LOG_DEBUG_3` | 1--3 | Increasing verbosity: per-batch / per-kernel / developer trace |

**Stage IDs:** 0 (init), 500 (tuning), 550 (autotune), 1000 (sieve), 1500 (sieving), 2000 (postproc), 3000 (multiprimes), 4000 (matrix), 5000--8800 (Block Wiedemann), 9000 (sqrt).

**Macros:**

| Macro | Usage |
|-------|-------|
| `LOG(level)` | `LOG(LOG_INFO) << "message";` |
| `LOG_IF(level, cond)` | Conditional logging |
| `LOG_SET_STAGE(id)` | Set thread-local stage ID |
| `LOG_INCREMENT_STAGE(n)` | Increment stage ID |

**Configuration** (`LogConfig`): `enable_cout`, `enable_file`, `enable_date`, `enable_time`, `enable_stage`, `enable_rank`, `csv_format`, severity thresholds (`min_severity_cout`, `min_severity_file`), MPI rank, `file_path`.

**Thread context** (`ThreadContext`, thread-local `g_log_context`): `algorithm_stage_id` (int), `thread_name` (string). Set via `LOG_SET_STAGE`.

**Helper:** `FormatDuration(double ms)` -- converts milliseconds to `"Hh Mm Ss ms"` string.
