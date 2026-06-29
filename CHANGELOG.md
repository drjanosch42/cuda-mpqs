# Changelog

All notable changes to cuda-mpqs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] - 2026-06-29
### Added
- `--matrix_lp1_bound <L>` (suffix-aware K/M/B/T): a large-prime-magnitude
  down-filter for `--matrix_only` replay runs. When set, it drops every
  large-prime-combined relation whose large prime exceeds `L` (and the matching
  partial relations) right after the relation file is loaded, exactly reproducing
  the relation set a sieve run at large-prime bound `L` would have yielded — with
  no re-sieve. Pure smooth relations are never dropped, and the large-prime
  fraction is reported before and after the filter. This turns a lower effective
  large-prime bound into a matrix-stage replay parameter: it can be lowered
  iteratively to drop below the high-large-prime square-root cliff without
  recomputing the sieve. Matrix-replay only — no effect on the sieve stage or its
  wall-clock. Validated on stored RSA-110 relations. Default 0 (disabled).
- Periodic, atomic, crash-safe mid-sieve checkpointing and resume
  (`--checkpoint_interval`, `--checkpoint_batches`, `--checkpoint_dir`,
  `--resume`). A killed or wall-clock-expired sieve resumes from the last
  checkpoint instead of restarting from zero. Default-off; the solo and
  cluster-coordinator paths are supported (workers are stateless).

### Changed
- Sieve and autotune memory accounting hardened: VRAM-fraction global-bucket
  sizing with headroom, 64-bit bucket arithmetic, and an autotune
  total-footprint OOM guard, so memory-infeasible kernel configurations are
  rejected before launch instead of failing at allocation time.

### Fixed
- Sieve memory-bandwidth regression: the per-prime `B_values` are moved out of
  the streamed per-prime struct into a separate device array, shrinking the
  struct the sieve kernels read on every polynomial evaluation and restoring
  (and improving on) steady-state sieve throughput.
- `uint512::sqrt()` returned the un-refined Newton seed, which made the SIQS
  `a_target` too small (up to ~2x) and reduced smooth-relation yield; it now
  seeds from above the root and returns an exact `floor(sqrt(n))`. (Correctness
  was never affected — `a_target` is only a target.)
- Hypercube dimensions above 16 no longer silently drop factor-base a-primes,
  which previously produced zero relations at larger sieve intervals.
- Out-of-bounds guard added in `advanceRoots` for the autotune `polyBlockSize=1`
  probe; compute-sanitizer clean.

## [1.0.3] - 2026-06-22
cuda-mpqs 1.0.3 consolidates the multi-node cluster correctness and performance
work that landed since 1.0.2. It is validated end-to-end by an RSA-130
factorization on a multi-node, multi-GPU cluster that ran clean and well clear
of the high large-prime 2-cycle square-root cliff, with cross-node work overlap
eliminated and the coordinator GPU kept at full duty for the whole run.

### Added
- `--cluster_pool_oversize <float>` (coordinator only, default 1.0): an
  over-provisioning multiplier for the on-demand a-value overflow pool so it
  cannot run dry before the relation target is met. Overflow windows are drawn
  only on demand and the run stops at the relation cap, so over-sizing is
  essentially free.

### Changed
- The cluster coordinator now self-assigns overflow chunks, so its local GPU
  keeps sieving as one more consumer of the overflow pool instead of idling
  after its initial a-range. Engaged only with remote workers present; solo and
  worker paths are unchanged.

### Fixed
- Cross-node overflow-chunk range overlap and aliasing: an initial-range
  completion could split a live overflow chunk and make two GPUs sieve the same
  a-range. Reclaim is now gated on the coordinator's authoritative per-worker
  tracker, with a dispatch-time overlap invariant.
- Chunk-assignment delivery and receive races that could permanently strand a
  worker. Assignment now routes through one send-checked path, the worker I/O
  thread is the sole socket reader, idle workers re-request work, and the
  coordinator proactively re-feeds workers holding no chunk.
- Overflow-pool sizing: the on-demand a-value pool could drain at a fraction of
  the relation target; it is now sized from the relation target at a
  conservative yield floor and clamped to the a-factor walk's capacity.
- Duplicate-partial corruption on multi-node runs: byte-identical duplicate
  partials made cross-node large-prime combines square a relation against its
  own duplicate, yielding trivial square roots. Fixed by correcting per-worker
  a-range accounting under CUDA-graph replay, bounding the coordinator's local
  sieve, and skipping large-prime matches between partials that share the same
  sqrt_Q.

## [1.0.2] - 2026-06-09
### Fixed
- Matrix preprocessing (CPU `--matrix_mode preprocess`) no longer silently
  returns only trivial congruences (X ≡ ±Y, no factor). Three defects were
  fixed: higher-weight merge column elimination is now all-or-nothing
  (kernel-preserving); truncation is size-gated and skipped when the reduced
  matrix is already Block-Wiedemann-tractable; and raw single-large-prime
  relations are no longer materialized as 2-cycle rows at high large-prime
  fraction.

### Added
- `--truncation_min_rows` (default 5,000,000) and
  `--preprocess_lp_materialize_max` (default 0.45) — controls for the truncation
  and high-large-prime preprocessing fixes above.
- `--merge_max_weight` and `--force_preprocess` — default-inert diagnostic flags.

### Notes
- Preprocessing at high large-prime fraction is now correct but remains ≤ legacy
  in yield; prefer legacy, or keep the large-prime bound below the cliff. The
  default `--matrix_mode` AUTO → legacy is unchanged.

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
