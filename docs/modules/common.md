# Common Module (`src/common/`)

## Overview

Shared primitives for the CUDA-MPQS pipeline: 512-bit unsigned arithmetic (`uint512`) plus its RSA-155 wide-intermediate helpers, Montgomery modular arithmetic, SoA relation containers with device/host transfer, binary relation I/O, the mid-sieve checkpoint artifact, the canonical relation dedup hash, math utilities, and structured logging.

Static library `mpqs_common`. Separable CUDA compilation ON. Links `cudampqs_build_flags`. No external dependencies beyond CUDA and Thrust.

## Files

| File | Purpose |
|------|---------|
| `uint512.cuh` | 512-bit unsigned integer (16x`uint32_t`, little-endian). Full arithmetic, division, modular ops, shifts, sqrt. Free wide-intermediate helpers `abs_square_minus_N` / `isqrt_2N` for N ≥ 2^511 (RSA-155). All methods `__host__ __device__`. |
| `montgomery.cuh` | Montgomery form context (`mpqs::math::Montgomery`): CIOS multiplication, squaring, exponentiation. R = 2^512. |
| `math_utils.cuh` | `modpow`, `mod_inverse`, `gcd`, `legendre_symbol`, `sqrt_mod_3mod4`, `crt_2`, `calculateQ_unsigned`, `calculate_sqrt_of_QX` (3- and 5-arg forms). |
| `uint128_helper.cuh` | Safe 64-bit modular arithmetic (`add_mod`, `sub_mod`, `mul_mod`, `pow_mod`) via `__int128` intermediates. |
| `mpqs_structures.h` | AoS `Relation` struct (`mpqs::structures`): polynomial coefficients (a, b, x), factor list (max 64 entries), large prime cofactor, `char_bits`; `Hash128` functor for `unsigned __int128` map keys. |
| `mpqs_soa.h` | SoA containers (`mpqs::structures`): `HostRelationBatch` (host), `RelationBatch` (device), `RelationBatchView` (kernel-passable). |
| `mpqs_soa.cu` | SoA batch operations: resize, append, validation kernel, D-to-H/H-to-D transfer (incl. managed-memory path), counter management. |
| `relation_hash.h` | `mpqs::computeRelationHash()` — SINGLE SOURCE OF TRUTH for the 64-bit relation dedup hash, byte-for-byte identical to the GPU `compute_relation_hashes_soa`; shared by the cluster `RelationAccumulator` and the solo checkpoint dedup. |
| `logger/hpc_logger.h` | `HPCLogger` singleton + `LogMessage` proxy. Severity levels, multi-sink `SinkConfig`, stage/module/submodule context, macros. |
| `logger/hpc_logger.cpp` | Logger implementation: thread-safe writes, console/file/error-file sinks, CSV mode. |
| `relation_io.h` | Binary I/O interface (`mpqs::io`): v1 (`HostRelationBatch` + projected LP) and v2 (full smooths + raw partials + `V2Metadata`) serialization. Auto-detection via `detect_and_deserialize`. v2 optionally carries branch character bits (`FLAG_HAS_CHAR_BITS`). |
| `relation_io.cpp` | Implementation of v1/v2 serialize/deserialize with `"MPQS_SOA\0"` and `"MPQS_V2\0"` magic headers. |
| `sieve_checkpoint.h` / `.cpp` | Mid-sieve checkpoint artifact (`mpqs::ckpt`): atomic, crash-safe `sieve.ckpt` = verbatim `serialize_v2` payload + progress trailer (`CheckpointTrailer`) + optional cluster block (`CheckpointClusterBlock`) + fixed 29-byte EOF footer (`MPQS_CKFT` magic as completeness sentinel). `dedupRelationsInPlace()` shrinks the file via `computeRelationHash`. |
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
| Small-type arithmetic | `add_uint32(uint32_t)`, `mult_uint32(uint32_t)`, `mul_uint64_inplace(uint64_t)` → returns overflow carry, `div_uint32_inplace(uint32_t)` → returns remainder, `div_uint64_inplace(uint64_t)` → returns remainder, `div_uint32_const(uint32_t)` → non-destructive quotient, `mod_uint32(uint32_t)` → const read-only remainder, `mod_uint64(uint64_t)` → const read-only 64-bit remainder (limb-by-limb `__int128`; used by branch char-bit capture to compute `|ax+b| mod q`) |
| Modular | `add_mod`, `sub_mod`, `double_mod`, `mul_mod`, `negate_mod_inplace`, `additive_inverse_mod_n` — `add_mod` is the overflow-safe `(X+Y) mod N` used by the sqrt stage before `gcd(X+Y, N)` (a plain 512-bit add would wrap for N ≥ 2^511) |
| Specialized | `mul_add_mod_signed(int64_t x, b, N)` — computes `(a*x + b) mod N` for polynomial evaluation |
| Bit ops | `lshift`, `rshift`, `msb()`, `countr_zero()`, `msb_is_set()` |
| Queries | `is_zero()`, `is_one()`, `fits_in_128()`, `to_uint128()` |
| Number theory | `sqrt()` — Newton-Raphson integer square root, exact `floor(sqrt(n))` (see below) |
| Signed helpers | `abs_twos_complement(int8_t& sign)` — interprets as two's complement, returns magnitude and sign |
| Conversion | `to_string()` (decimal, host only), `to_hex_string()` (host only), `print()` (device-compatible) |
| Operators | Full set: `+`, `-`, `*`, `/`, `%`, comparisons. Mixed-type overloads for `uint32_t` and `uint64_t`. |

#### `uint512::sqrt()` — seed-from-above fix (v1.0.4c)

`sqrt()` seeds Newton's iteration from **above** the root: `x0 = 2^(floor(msb/2)+1) ≥ sqrt(n)` (`uint512.cuh:739-763`). Newton's map `x = (x + n/x)/2` is non-increasing once `x ≥ sqrt(n)` (AM-GM), so the "stop when it stops decreasing" guard correctly yields the exact `floor(sqrt(n))` and `x + n/x` never overflows. (The old seed `2^floor(msb/2)` sat *below* the root: the first step ascended, tripped the guard on iteration 0, and returned the un-refined seed — making `a_target = sqrt(2N)/M` 0.5–0.9× optimal and costing up to ~11% smooth yield at RSA-140. Correctness was never affected.)

#### Wide-intermediate helpers for N ≥ 2^511 (RSA-155)

`uint512` wraps silently at 2^512. Three SIQS transients have an *intermediate* that exceeds 2^512 only when N ≥ 2^511, even though the final result fits. Two free `__host__ __device__` helpers in `uint512.cuh` (plus `add_mod` above) carry the intermediate wider and are **strict no-ops for N < 2^511** (bit-for-bit identical results for RSA-150 and below):

| Helper | Purpose |
|--------|---------|
| `abs_square_minus_N(s, N, sign&)` (`uint512.cuh:992`) | `Q = \|s² − N\|` with sign, where s = \|ax+b\|. Forms s² in a 1024-bit (32-limb) accumulator — the square reaches ~2N ~ 2^513 — compares against zero-extended N, and returns the low 512 bits (the true magnitude satisfies \|Q\| < N < 2^512). Used by the postprocessing trial-division core. |
| `isqrt_2N(N)` (`uint512.cuh:1066`) | `floor(sqrt(2N))` for a_target computation. For N < 2^511: exact `(N<<1).sqrt()`. For N ≥ 2^511 (where `N<<1` would drop bit 511): `sqrt(2)·sqrt(N)` via exact `N.sqrt()` scaled by a Q32 fixed-point `sqrt(2)` (~2^-32 relative error — immaterial for a sizing target). |

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
    uint32_t char_bits = 0;                  // Branch-fixed r-bit char vector (0 unless --char_mode branch)
};
```

Semantics: `(ax+b)^2 - N = sign * 2^val_2_exp * prod(p_i^count_i) * large_prime_remainder`.

`mpqs_structures.h` also provides `mpqs::Hash128`, a hash functor for `unsigned __int128` keys in `std::unordered_map/set`.

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
| `char_bits` | `vector<uint32_t>` | Branch-fixed per-relation character vector (Stage 4); 0 in `norm`/`none` mode. |
| `num_relations` | `size_t` | Fill count |
| `num_factors` | `size_t` | Fill count |

Methods: `resize(n_rels, n_factors)` — grow-only; `clear()` — clears all vectors and resets counts.

### RelationBatch (SoA, device-side)

Namespace: `mpqs::structures`. Device-resident SoA with atomic dual-counters and pinned host mirrors. Non-copyable and non-movable (raw CUDA pointers require explicit lifecycle management). On Jetson / unified-memory devices the arrays are allocated via `cudaMallocManaged` (`useManagedMemory()`), with `cudaMemAdvise` when `concurrentManagedAccess` is supported.

| Method | Description |
|--------|-------------|
| `initiate(device_id)` | Set device, allocate pinned counters and device atomics |
| `resize(n_rels, n_factors)` | Non-destructive grow; preserves existing data |
| `clear(stream)` | Reset counters to zero (no deallocation) |
| `append(other, count, stream)` | Safe append with overflow discard; rebases CSR offsets via kernel |
| `reset_counters(stream)` | Zero atomic counters only (no data zeroing) |
| `moveToHost(dest, stream)` | Download to `HostRelationBatch` |
| `moveRangeToHost(dest, offset, count, stream)` | Download a sub-range; two-phase (flat arrays → sync for CSR offsets → factors), re-bases `factor_offsets` to 0 |
| `syncCounters(stream)` | Sync only the 16-byte atomic counters to host pinned memory (managed-memory `moveToHost` fast path) |
| `uploadFromHost(host_batch)` | Upload from `HostRelationBatch` (for `LINALG_ONLY` mode) |
| `getCount(stream)` | Sync read of relation count from device atomic |
| `getFactorCount(stream)` | Sync read of factor count from device atomic |
| `setTargetCap(cap)` | Set adaptive convergence relation cap (0 = unlimited); propagated to `RelationBatchView` |
| `requestStats(stream)` / `updateStats(stream)` / `readStats()` | Async stats via pinned memory or tiny kernel |
| `get_view()` | Returns `RelationBatchView` for kernel launch |
| `validate_relations(N, d_fb, fb_size)` | GPU validation kernel: checks `|(ax+b)^2 - N| == product` |
| `validate_host_batch(batch, fb, N)` | Static helper: uploads host batch, validates on GPU, cleans up |
| `getLargePrimesData()` / `getFactorOffsetsData()` | Raw device pointer accessors (for Thrust) |
| `getDeviceCountPtr()` | Device pointer to the relation-count atomic (telemetry kernels) |
| `getCapacityRels()` / `getCapacityFactors()` | Allocated capacity (not fill count) |

### RelationBatchView (kernel-passable)

Namespace: `mpqs::structures`. Raw device pointers + atomic counters + bounds. Passed by value to CUDA kernels for lock-free relation accumulation.

| Field | Description |
|-------|-------------|
| `sqrt_Q`, `signs`, `val_2_exps`, `large_primes` | Per-relation arrays |
| `char_bits` | Branch-fixed per-relation character vector (Stage 4); 0 in `norm`/`none` mode |
| `factor_offsets`, `factor_indices`, `factor_counts` | CSR factor arrays |
| `global_count`, `global_factor_idx` | Device atomic counters (`uint64_t*`) |
| `max_relations`, `max_factors` | Bounds for overflow checking |
| `target_cap` | Adaptive convergence ceiling (0 = unlimited) |

### Relation Dedup Hash (`relation_hash.h`)

`mpqs::computeRelationHash(batch, i)` computes the canonical 64-bit dedup hash
`(len << 48) | (exp_xor << 32) | body_xor`, where `body_xor` folds
`factor_indices·0x9e3779b9`, the sign (encoding-agnostic "negative iff `signs[i] != 1`"),
and `2^val_2_exp`. It is byte-for-byte identical to the GPU
`compute_relation_hashes_soa` kernel and is the ONE definition shared by the cluster
`RelationAccumulator` and the solo checkpoint dedup — divergent copies are forbidden
(a divergent copy is a silent dedup bug). `char_bits` is deliberately excluded
(deterministic function of `(ax+b)`; cannot change dedup identity).

### Relation I/O v2 Character Extension

The v2 format (`relation_io.h`) carries the branch-fixed character data behind a forward-tolerant
flag bit, `FLAG_HAS_CHAR_BITS` (0x4). When set, the file appends — at the **end** of the
fixed-position metadata block — the aux-prime metadata (`uint32_t r`, then `r` aux primes `q_s` and
`r` Tonelli roots `t_s`) and the per-relation `char_bits` vectors (after the smooth/partial batch
records). Old `.v2` files (no char flag) parse byte-for-byte unchanged, loading with empty
`char_bits` and `V2Metadata::has_char_bits == false`. Consumers (e.g. the orchestrator under
`--char_mode branch`) check `has_char_bits` to refuse char-less input. See [matrix.md](matrix.md)
for how the vectors are used.

### Sieve Checkpoint (`sieve_checkpoint.h` / `.cpp`, namespace `mpqs::ckpt`)

The mid-sieve fault-tolerance artifact `sieve.ckpt`:
`[ serialize_v2 payload (verbatim) ][ progress trailer ][ optional cluster block ][ fixed EOF footer ]`.
- `CheckpointTrailer`: `global_a_index` (solo cursor, u64), target/loaded counts, `lp1_bound`, `sieve_bound`, N, `elapsed_sieve_sec`, `cluster_section_present`.
- `CheckpointClusterBlock` (coordinator only): `completed_prefix_cursor` + per-node `initial_high_water[]`.
- The 29-byte EOF footer (`MPQS_CKFT` magic + `{trailer_offset, trailer_len, schema_version}`) is the **completeness sentinel** — a torn write never has it, so a partial checkpoint is never loadable; the trailer is located by seeking from the footer (variable-size once the cluster block is present).
- Written atomically (unlink stale tmp → write → fsync → rotate `.prev` → rename → dir-fsync); `dedupRelationsInPlace()` shrinks the payload on a scratch host copy via the shared `computeRelationHash`.
- `deserialize_v2` reads section-by-section without checking EOF, so the trailer + footer are invisible to ordinary `relations.v2` consumers — the checkpoint reuses the tested serializer verbatim.

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
| `calculateQ_unsigned` | `(a, b, x, N, Q_out, sign_out)` | `__device__` only. Computes \|(ax+b)^2 - N\| with sign. Contract: `-2^511 < (ax+b)^2 - N < 2^511` (the postprocessing hot path uses `mpqs::abs_square_minus_N` instead, which lifts this contract for RSA-155) |
| `calculate_sqrt_of_QX` | `(a, b, x, sqrt_out, sign_axb_out)` | `__host__ __device__`. Computes \|ax+b\| **and** the sign of the signed `(ax+b)` — the 5-arg form needed by the branch-fixed character capture |
| `calculate_sqrt_of_QX` | `(a, b, x, sqrt_out)` | 3-arg legacy form; discards the sign |

## uint128 Helpers (`uint128_helper.cuh`)

All functions in `mpqs::math`, all `__host__ __device__ __forceinline__`. Typedef `uint128_t = unsigned __int128`. Operate on `uint64_t` using `__int128` intermediates to prevent overflow.

| Function | Description |
|----------|-------------|
| `add_mod(a, b, m)` | `(a + b) % m`, overflow-safe |
| `sub_mod(a, b, m)` | `(a - b) % m`, underflow-safe |
| `mul_mod(a, b, m)` | `(a * b) % m`, essential for Hensel lifting where `p^2 > 2^32` |
| `pow_mod(base, exp, m)` | Binary exponentiation with `mul_mod` |

## Logging (`HPCLogger`)

Singleton logger with thread-safe writes and **multiple independently-filtered sinks** (console, file, error-file), each with its own severity window, prefix toggles, optional line wrapping and CSV mode.

**Severity levels:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `LOG_RESULT` | -4 | Factorization result only (`--mute` mode) |
| `LOG_ERROR_CRITICAL` (`LOG_ERROR`) | -3 | Fatal errors, unrecoverable CUDA failures |
| `LOG_WARNING` (`LOG_ERROR_MAJOR`) | -2 | Warnings, non-fatal errors; `LOG_ERROR_MAJOR` is a backward-compat alias |
| `LOG_INFO` | -1 | Stage transitions, key milestones (default console threshold) |
| `LOG_STATS` | 0 | Statistics summaries, buffer telemetry, throughput (`--verbose`) |
| `LOG_DEBUG_1` through `LOG_DEBUG_3` | 1--3 | Increasing verbosity: per-batch / per-kernel / developer trace |

**Stage IDs:** 0 (init), 500 (tuning), 550 (autotune), 1000 (sieve), 1500 (sieving), 2000 (postproc), 3000 (multiprimes), 4000 (matrix), 5000--8800 (Block Wiedemann: init 5000, self-verification 5400, autotune 5700, stages 6000/7000/8000, postprocessing 8800), 9000 (sqrt).

**Macros:**

| Macro | Usage |
|-------|-------|
| `LOG(level)` | `LOG(LOG_INFO) << "message";` (gated on `ShouldLog` — zero formatting cost when filtered) |
| `LOG_IF(level, cond)` | Conditional logging |
| `LOG_SET_STAGE(id[, name])` | Set thread-local stage ID (variadic: optional human-readable stage name) |
| `LOG_INCREMENT_STAGE(n)` | Increment stage ID |
| `LOG_SET_MODULE(mod)` / `LOG_SET_SUBMODULE(sub)` | Set thread-local module / submodule context |
| `LOG_SCOPED_MODULE(mod)` | RAII (`LogModuleGuard`): set module for the current scope, restore on exit |

**Configuration** (`LogConfig`): `std::vector<SinkConfig> sinks` (empty = default console sink at `LOG_INFO`) + `mpi_rank`. Each `SinkConfig` carries: `type` (`CONSOLE`/`FILE`/`ERROR_FILE`), `path`, a severity window (`min_severity`/`max_severity` — a message passes if `max_severity ≤ urgency ≤ min_severity`), prefix toggles (`show_date`, `show_time`, `show_rank`, `show_stage`, `show_module`, `show_submodule`, `show_level`), `wrap_width`, `csv_format`.

**Thread context** (`ThreadContext`, thread-local `g_log_context`): `stage_id` (+ compat alias `algorithm_stage_id`), `stage_name`, `module`, `submodule`.

**Helper:** `FormatDuration(double ms)` -- converts milliseconds to `"Hh Mm Ss ms"` string.
