# Changelog

All notable changes to cuda-mpqs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-06-09
### Fixed
- Matrix preprocessing (CPU `--matrix_mode preprocess`) no longer silently returns
  only trivial congruences (X ≡ ±Y, no factor). Three independent defects fixed:
  - **Higher-weight merge** column elimination is now all-or-nothing: the pivot row
    was previously deleted even when a fill-in skip left the column un-eliminated,
    making the reduction non-kernel-preserving. (`src/matrix/merge_filter.cpp`)
  - **Truncation** is now size-gated — skipped when the reduced matrix is already
    Block-Wiedemann-tractable, avoiding the lightest-row selection that confined the
    kernel to the trivial subspace. (`--truncation_min_rows`, default 5,000,000)
  - **High-LP partial handling**: raw single-large-prime relations are no longer
    materialized as 2-cycle matrix rows at high large-prime fraction (which captured
    the genus character and forced every dependency trivial); gated by the
    combined-smooth LP fraction. (`--preprocess_lp_materialize_max`, default 0.45)

### Added
- `--truncation_min_rows` and `--preprocess_lp_materialize_max` — the facet-2 and
  facet-3 preprocessing fix controls (above).
- `--merge_max_weight` and `--force_preprocess` — DIAGNOSTIC flags for the
  preprocessing investigation; default-inert (no effect on normal runs).
- Diagnostic tooling under `tools/preprocessing_analysis/` (relation / genus
  verifiers used to characterise and validate the fixes).

### Notes
- Preprocessing at high LP fraction is now **correct** but remains **≤ legacy** in
  yield; prefer legacy / keep the large-prime bound below the cliff at high LP. The
  default `--matrix_mode` AUTO → legacy is unchanged. Validated: RSA-100 factored
  (235 s); RSA-100 / 94-digit / RSA-110 preprocess all factor.

## [1.0.1] - 2026-06-05
### Added
- `--char_mode {norm,branch,none}` — selectable quadratic character-column symbol:
  `branch` = correct branch-fixed field-element character (fixed Tonelli root),
  `norm` = legacy genus-blind NORM symbol, `none` = zero character columns.
- 64-bit number-theory primitives (Tonelli–Shanks, Jacobi, deterministic
  Miller–Rabin primality) for large-prime-bound auxiliary primes.
- `--sqrt_diagnostic`: per-solution reporting of the nontrivial-GCD rate per
  Block-Wiedemann solution (debug level; capture with `--log_file`).
- `relation_validator` tool — deterministic relation + large-prime validation
  (composite / out-of-range large-prime detection).

### Changed
- Default `--char_mode` is now **`none`** (character columns off by default).
  Validation found character columns confer no demonstrable factoring benefit at
  reachable scales — inert where the quadratic sieve is unobstructed (RSA-100/110),
  insufficient where it is (the 94-digit 2-cycle cliff).
- `--matrix_mode` AUTO now resolves to **legacy** for normal runs; preprocessing
  engages only via explicit `--matrix_mode preprocess` or `--matrix_only`. The
  automatic LP-fraction → preprocess switch was removed because preprocessing
  degrades the obstructed high-LP regime.
- Sieve telemetry: the per-LP-call "Combined N full relations in this batch."
  line is demoted from `--verbose` (STATS) to `--debug` (DEBUG_1). `--verbose`
  now shows only the consolidated ETA / throughput / buffer-fill telemetry during
  sieving, with the detached technical large-prime combination counts at `--debug`.

### Deprecated
- `--lp_preprocess_threshold` (and alias `--lp_matrix_threshold`) — now inert;
  AUTO no longer auto-selects preprocess from LP fraction.

### Fixed
- GPU sieve no longer drops relations whose candidate falls exactly on a
  sieving-block boundary: the backward (factor-recording) pass now scans the
  same position range as the forward sieve, so boundary candidates get their
  full prime factorization recorded instead of being silently discarded.

## [1.0.0] - 2026-06-01
### Added
- Initial public release.
- Full SIQS/MPQS pipeline: parameter tuning, optional autotuning, GPU sieving,
  sparse GF(2) matrix construction, Block Wiedemann linear algebra, and square
  root refinement.
- Single-node and distributed cluster execution across heterogeneous multi-node
  setups (explicitly tested with more than two nodes).
- Autotuning: 4-stage joint (F, L) optimizer with persistent history.
- GPU preprocessing: packed sparse GF(2) matrix with compact-merge cycles.
- Large-prime variant: single large prime via GPU slab hash table.
- Supports NVIDIA Turing, Ampere, Hopper, and Blackwell GPU architectures,
  including the A100 (Ampere) and H100 (Hopper) data-center accelerators.
- Supports the Jetson Orin Nano Super 8 GB embedded platform.
