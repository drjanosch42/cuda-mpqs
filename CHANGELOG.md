# Changelog

All notable changes to cuda-mpqs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] - 2026-07-17
This release adds 512-bit factoring capability and the sieve, linear-algebra,
and cluster fixes that made it possible. It is validated end-to-end by two
milestone factorizations: RSA-140 (463-bit) on 2026-06-29 in about 104 GPU-hours
on a 16× NVIDIA H100 cluster, and RSA-155 (512-bit) on 2026-07-14 in 700.64
GPU-hours total — a 689.74 GPU-hour distributed sieve on 64× NVIDIA H100 GPUs
(16 nodes) plus 10.90 GPU-hours of single-H100 linear algebra — yielding two
product-verified 78-digit prime factors.

### Added
- **Wide sieve-accumulator path for 512-bit inputs** (`--wide_accum
  {auto|u8sat|u16}`, default `auto`). Numbers around RSA-150/155 scale
  accumulate per-position sieve logarithms that exceed the 8-bit accumulator the
  narrow path uses, so full smooth relations were silently rejected. The wide
  path carries the accumulator without wrapping and is auto-selected for
  inputs of roughly 150 digits and larger; smaller inputs keep the exact narrow
  (8-bit) path, byte-for-byte unchanged. Two wide variants are available: a
  16-bit accumulator (`u16`) and a saturating-8-bit accumulator (`u8sat`) that
  restores a wider sieving interval by clamping at 255 instead of wrapping.
  `auto` selects `u8sat` only when a config-time gate proves it bit-identical to
  the 16-bit path's candidate selection, and falls back to `u16` otherwise.
- **`--bucket_size_factor <F>`** (default 0 = legacy sizing): sets the
  large-prime bucket capacity to `F` times the sieving-block size. At large
  factor-base bounds the legacy capacity silently overflowed, dropping
  large-prime hits and preferentially demoting full smooths; raising this factor
  removes the discard. Default 0 reproduces the legacy capacity exactly, so all
  sieve paths remain byte-identical to prior behavior when the flag is unset.
- **Bucket-overflow telemetry** (wide path): a read-only host-side report of the
  overflowed-bucket fraction, bucket capacity, and peak fill at the periodic
  stats cadence, making the previously-invisible large-prime discard observable.
  No change to sieving behavior; the narrow production path takes no extra work.
- **`--matrix_max_rows <N>`** (default 0 = off): caps the relation batch to the
  first `N` rows before matrix construction, preserving row/relation/factor-base
  alignment. An optional performance and memory lever for 512-bit-scale linear
  algebra; default-off is byte-identical to prior behavior.
- **Floor-safe wide-path autotuning**: the parameter optimizer now tunes the
  wide sieve geometry against a survivors-per-second objective, and only adopts a
  tuned configuration when it beats the safe default, otherwise keeping the
  default. The narrow path is unaffected.
- **Meta-sieve tuning knobs**: SM-count-aware sizing of the SCATTER kernel grid
  (replacing a fixed grid that under-occupied large GPUs), plus
  `--sieve_gather_block_dim <N>` and `--sieve_meta_cycle_cap <N>` (both
  default-off) for occupancy and bucket-write-window experiments. Result-invariant.
- **Cluster coordinator LP-occupancy telemetry** (`[Cluster] LPocc:`): one
  plain-integer log line at the existing large-prime stats cadence that
  partitions the coordinator's matching-thread wall time across its stages
  (receive, deserialize, buffer, accumulate, match, control, idle) plus a
  worst-single-call timer and live table occupancy. Coordinator-path only; solo
  runs are unaffected; overhead is negligible.
- **Block Wiedemann stage-boundary checkpointing** (experimental —
  `--bw_max_solutions <N>`, `--bw_checkpoint_dir <path>`, `--bw_resume`): save
  Krylov, lingen, and solution artifacts at stage boundaries and resume from the
  last completed stage. `--bw_max_solutions` bounds how many candidate columns
  are converted to solutions. Resumes are stage-granular, not mid-stage. **Not
  yet validated end-to-end**; supporting checkpoint I/O and an integrity tag were
  added to the linear-algebra submodule, and results are unchanged when
  checkpointing is off.

### Changed
- **`src/linalg` (block-wiedemann) submodule updated to v1.0.1**
  (SpMM-autotuner delta-encoding overflow fix — see Fixed below), plus the
  experimental stage-boundary checkpoint I/O for Block Wiedemann, documented
  in the submodule's own `[Unreleased]` changelog section.

### Fixed
- **RSA-155 (512-bit) unblock — three >512-bit overflow transients** that
  corrupted moduli with `N >= 2^511`, now carried in a wider intermediate and
  reduced back to 512 bits. A strict no-op for `N < 2^511`, so RSA-150 and every
  smaller size are bit-for-bit unaffected. The three sites: `(ax+b)^2 - N` in
  postprocessing (the hard blocker — the square reaches about 2N and previously
  truncated, corrupting every relation); `a_target = sqrt(2N)/M` polynomial
  sizing; and `X+Y` before the final gcd in the square-root step, now reduced
  modulo N first. The relation-validation kernel was widened to match.
- **32-bit overflow in the device meta-sieve bucket write index**: the
  bucket-address expression evaluated in 32 bits and wrapped for sieve
  geometries with more than 2^32 bucket entries — the geometry selected on GPUs
  larger than 40 GB — so high buckets aliased into low memory (all in-bounds, no
  fault) and the sieve silently produced zero candidates. Fixed by performing the
  multiply in 64 bits. Inert below 2^32 entries.
- **Cluster work-assignment frame exceeded the 64 MB transport cap at large
  factor-base bounds**: the assignment message serialized the full factor base,
  so at large bounds it was rejected and workers failed to start. The message now
  ships a compact FNV-1a hash of the factor base and each worker regenerates the
  factor base locally from the authoritative `(N, F)` and verifies the hash
  before sieving, making the payload size independent of the bound. The transport
  protocol version was bumped so a stale binary fails cleanly rather than
  mis-parsing.
- **SpMM-autotuner illegal-memory crash in the linear-algebra stage**
  (submodule 1.0.0 → 1.0.1) that blocked 512-bit linear algebra: when the matrix
  is padded to square with more than 65535 columns, the delta-encoded escape
  stream accumulated its length in 32 bits and, at 512-bit scale, wrapped and
  under-allocated its buffer, corrupting the CUDA context. The length is now
  summed in 64 bits and the encoding is skipped for oversized slices, falling
  back to a warp-based format with the context intact.
- **False-positive interval-coverage guard**: the narrow-path coverage check
  compared a single launch's coverage against the full interval and emitted a
  spurious critical error for every legitimate multi-batch geometry. It now
  checks total coverage across all batches. Diagnostic-only; no sieve arithmetic
  changed.
- **Wide-path default kernel geometry** restored a large sieve-throughput
  recovery at 512-bit scale by setting sensible wide-path defaults (big-prime
  start and polynomial-count limits); the narrow path is unchanged.

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
- `--sqrt_diagnostic`: log solution-diversity statistics (distinct Block-Wiedemann
  solutions by hash). The per-solution nontrivial-GCD-rate report added in this
  release is unconditional (independent of this flag), at debug level; capture
  either with `--debug --log_file`.
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
