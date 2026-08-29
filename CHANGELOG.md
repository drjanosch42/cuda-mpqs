# Changelog

All notable changes to cuda-mpqs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Documentation only. **No product source changed** — `src/`, the linear-algebra
submodule and the build system are untouched.

### Documented
- `README.md` gained a performance refresh and a full table of the factorization
  records, from a cross-architecture RSA-100 benchmark of 1.0.6 over five devices
  and eight configurations, every run product-verified.
- **A measured limitation of the multiprocessor-aligned launch geometry added in
  1.0.6: it does not transfer to H100.** On a 108-multiprocessor A100 the
  technique is worth −17.4 % of the RSA-100 sieve wall; on a 132-multiprocessor
  H100 the aligned configuration measures **+2.5 % total and +3.6 % sieve wall**,
  i.e. a loss, for the structural reason recorded in the 1.0.6 entry. H100 keeps
  its power-of-two configuration.

## [1.0.6] - 2026-08-25
This release extends CUDA-graph replay from the sieving kernels to the whole
per-batch pipeline, gives the narrow batch sieve opt-in control over its launch
geometry and its shared-memory footprint, and fixes six defects present in 1.0.5
and every earlier release — three of them on the single-node GPU large-prime
path, one of which silently admitted guaranteed-trivial rows into roughly 2 % of
every such matrix. The linear-algebra submodule keeps its independent version, 1.0.1.

Every geometry knob added here is default-off and reachable only by pinning
`--params`; with no such flag the binary reproduces 1.0.5 behavior on every path.
`src/sieve/kernel.cu` is byte-identical to 1.0.5 — all of the geometry work is
host-side configuration and validation.

### Added
- **`--cuda_graph_capture {sieve|postproc|full}`** (default: `full` in
  single-node runs, `postproc` in cluster runs): selects how much of the sieve
  pipeline is captured into the CUDA graph replayed by `--cuda_graph_unroll`.
  `sieve` captures only the sieving kernels, reproducing 1.0.5 behavior exactly
  and serving as the rollback path; `postproc` additionally captures batch trial
  division; `full` additionally captures the GPU large-prime pipeline and is
  available in single-node runs only — requesting it in a cluster run is reduced
  to `postproc` with a warning. The flag has no effect when
  `--cuda_graph_unroll 0` disables graph capture.
- **`--cuda_graph_lp_stride <K>`** (default 1): in-graph large-prime cadence —
  one dispatch every `K`-th captured batch. `1` dispatches per batch, matching
  what `--cuda_graph_unroll 0` has always done; `0` or any value at or above the
  unroll factor collapses to a single dispatch per replay, which is the 1.0.5
  graph cadence. Only meaningful when the large-prime pipeline is actually
  captured (single-node runs at capture scope `full`).
- **Multiprocessor-aligned sieve launch geometry — opt-in via `--params`.** The
  narrow **batch** sieve path accepts non-power-of-two
  `{num_polysPerSieveCall, metaGridDim, sasGridDim, num_polyBlocksPerThreadBlock}`
  tuples, so the scattering and gathering grids can be aligned to the device's
  multiprocessor count and run exact waves. Previously the grids were restricted
  to powers of two, which on a 108-multiprocessor A100 left 44 of 108
  multiprocessors idle for the kernel that is about a third of RSA-100 GPU time.
  Measured on that device at RSA-100, `--params 864,8,8,8,108,1024,864,1024`
  sieves **17.4 % faster** than the previous recommended tuple and 10.0 % faster
  than the best power-of-two tuple, at the same relations per polynomial, and
  with 6.2 % less energy above idle. On an RTX 5070 Ti (70 multiprocessors),
  `--params 560,8,8,8,70,1024,280,1024 --cuda_graph_unroll 0` factors RSA-100 in
  76.12 s total / 75.90 s core at the established operating point, 5.8 % under
  the previous record.

  Such a tuple is derived from one device's multiprocessor count and is an
  operator-pinned value, not a portable constant: it must be recomputed, never
  copied, for another GPU. No multiprocessor count appears anywhere in the
  source, and the autotuner neither proposes, projects nor auto-applies such a
  tuple — non-power-of-two geometries are reachable only by pinning `--params`.
  The technique is also **not universal**: on a 132-multiprocessor H100 the
  aligned rung measures 2.5 % *slower*, because the sieve caps the polynomials
  per call at `2^(hypercube dimension − 1)` and alignment there forces that count
  down by 48 %, nearly doubling the batch count. Check that cap before deriving
  an aligned tuple for a new device.

  The relaxation is guarded rather than merely permitted. Both validators (the
  sieve controller's and the autotuner's) now express the exact-partition
  arithmetic directly — `num_polysPerSieveCall` must be an exact multiple of
  `metaGridDim × polyBlockSize`, and `sasGridDim` must divide it with a
  power-of-two quotient and stay at or above the large-prime path's minimum — and
  reject a violating tuple before any sieving begins, naming the clause it
  failed. The configuration loader no longer floors the derived
  `num_polyBlocksPerThreadBlock` to make a tuple fit; a non-dividing tuple is an
  error rather than a silently different geometry. A launch-occupancy preflight
  additionally rejects a non-power-of-two geometry whose blocks would not fit one
  per multiprocessor at their shared-memory footprint. The legacy
  (`--sieve_batch_size 0`) and wide (uint16/u8sat) paths keep the mandatory
  power-of-two rules and reject such tuples loudly. Every previously valid
  configuration is bit-identical.
- **`--sieve_block_size <N>`** (default 0 = off, narrow batch path only):
  overrides the derived sieving block size, which is otherwise
  `min(M, largest power of two ≤ ¾ · opt-in shared memory per block)`. Admissible
  values are 0 or a power of two at or above 256 — a power of two because the
  gathering kernel masks positions with `blockSize − 1`; the upper bounds (at
  most `M`, and the shared-memory sum) are enforced by the existing downstream
  checks.
- **`--sieve_big_prime_start <N>`** (default 0 = off, narrow batch path only):
  overrides the large-prime transition index, which is otherwise coupled
  unconditionally to `sieving block size / 32`. Admissible values are 0 or
  greater than 32, since 32 is the mid-prime start index and at or below it the
  mid-prime range inverts and a whole factor-base band would be dropped silently.

  Together these two cut the gathering kernel's shared-memory footprint enough to
  make it co-resident (two blocks per multiprocessor) at production sieve
  intervals, which is worth about 15 % of the sieve wall in isolation. Whether
  that pays depends on the resulting number of meta-sieve cycles — the ratio of
  the interval count to the active blocks per cycle — because the scattering
  kernel walks the whole factor base once per cycle and pays its per-prime setup
  again each time. At one cycle the halved geometry is a net win on A100
  (measured 9.9 % of the sieve wall, 8.7 % end to end at RSA-100); at two cycles
  it is a net loss. It remains an experimental configuration rather than a
  recommended default: it costs about 5 % more energy above idle at neutral board
  energy, and it has not reproduced as a win elsewhere — the equivalent geometry
  on an RTX 5070 Ti is about 2.3 % slower than that card's own best tuple.
  **A halved transition index requires `--bucket_size_factor 1.0`**, without
  which the legacy bucket capacity reaches 97 % fill and large-prime hits are
  discarded.

  Both overrides require a pinned `--params` (without it they would be silently
  inert), and both are rejected — loudly, before any sieving — on the legacy
  path, on the wide accumulator path, and in combination with any autotuning
  flag. Nothing is ever silently floored, clamped or downgraded.
- **`--sieve_bucket_overflow_stats`** (default off): reports bucket occupancy and
  overflow percentage on the narrow accumulator path, which previously reported
  only on the wide path. The device-side overflow flag was always width-agnostic;
  only the host reader was restricted, and for a reason — reading it costs an
  asynchronous copy plus a synchronization on the sieving stream at the roughly
  five-second statistics cadence, which the narrow production pipeline otherwise
  never pays. The measured cost at production geometry is at most 0.09 % of the
  sieve wall, so the switch is opt-in only because it is a diagnostic, not
  because it is expensive.

### Changed
- **CUDA graph capture now covers post-processing (and, in single-node runs,
  large-prime matching), not just the sieving kernels.** Previously the graph
  held only the sieve kernels of N batches, all writing one accumulation buffer,
  with trial division, large-prime matching and a host synchronization running
  strictly after each replay — so none of that work could overlap the sieve. The
  graph now holds N double-buffered iterations of the ordinary batch loop,
  letting each batch's post-processing overlap its successor's sieve, and the
  per-replay host synchronization is removed for single-node runs. Cluster runs
  keep a per-replay synchronization, which the relation-extraction path requires
  for exact counters, and never capture large-prime processing.
- **In single-node runs at capture scope `full`, the last captured batch's
  post-processing is deferred into the next replay**, where it runs concurrently
  with that replay's first sieve instead of as an exposed tail at the end of the
  replay. One batch of candidates is post-processed after the graph loop
  finishes. This applies only when the large-prime pipeline is captured and the
  unroll factor is even; every other configuration keeps the undeferred body.
- **Large-prime processing inside the graph runs once per batch** rather than
  once per replay, which keeps each dispatch's work proportional to one batch.
  `--lp_interval` consequently governs the non-graph sieve path and
  `--cuda_graph_unroll 0`; the graph path processes every batch, or every `K`-th
  batch with `--cuda_graph_lp_stride K`. (The 1.0.5 graph path likewise ignored
  `--lp_interval`, processing once per replay.)
- **A finer large-prime cadence raises the large-prime fraction slightly**,
  because partials are matched sooner. At settings that already sit near the
  high-large-prime square-root cliff this can be the difference between a run
  that yields factors and one whose dependencies are all trivial; the documented
  110-digit command at a 1e12 large-prime bound is such a setting, and it behaves
  the same way with `--cuda_graph_unroll 0`. Use a lower large-prime bound, or
  `--cuda_graph_lp_stride` set to the unroll factor, in that regime.
- **Large-prime telemetry and the yield-prediction step counter are now
  maintained on the device**, so they stay correct inside a captured graph, where
  a host scalar passed by value would be frozen at capture time.
- **The sieve stage's cumulative large-prime relation counter is now assigned
  from the pipeline's own cumulative total** rather than accumulated per drain,
  which keeps it correct when several large-prime dispatches occur per graph
  replay.
- **The sieve-stage summary's `Sieved full` and `LP combined` lines are now
  counted from the final relation set** instead of being derived from the
  cumulative (pre-deduplication) large-prime counter, so they add up to the
  reported deduplicated total and agree with the reported large-prime fraction.
- Capturing trial division per batch substantially reduces accumulation-buffer
  pressure — measured peak occupancy falls by roughly the unroll factor — so
  fewer candidates are dropped on a full buffer and a run may produce slightly
  more relations per batch than before at the same settings.
- **Autotune history files are not portable back to 1.0.5.** A history file
  written by 1.0.6 can carry a non-power-of-two tuple that 1.0.5's power-of-two
  check rejects on load — a loud abort, not silent corruption. When downgrading,
  delete the autotune history file in the working directory or run with
  `--autotune_no_history`.

### Fixed
- **The narrow batch sieve could silently sieve only half its interval.** The
  production batch entry point never advances the sieve's start index — the
  per-batch offsetting that the legacy, debug and autotune loops perform does not
  happen there — so the count of sieving-block batches is a fiction on that path.
  The coverage check nevertheless multiplied coverage by that count, and
  therefore accepted a configuration covering only `[-M, 0)`: about half the
  yield, with no error and no warning, and it can even present as a *higher*
  relation count, because a smaller region suffers less per-block relation
  truncation — a signature that reads like a win. Coverage is now checked as
  `intervals × block size ≥ 2M` in 64-bit arithmetic on the narrow batch path,
  and a violating configuration aborts loudly naming the required interval count.
  The check is an inequality, so the autotuner's legitimate over-coverage stays
  legal. The defect is present in 1.0.5 and every earlier release; it stayed
  latent because every shipped narrow tuple happens to satisfy the equality
  exactly. Configurations on smaller-shared-memory devices (Turing-class, Jetson)
  that are half-sieving today will now fail loudly instead; the escape is to
  raise the interval count, which is `--params` field 2.
- **The GPU large-prime table's directory spin-lock could spin forever.** The
  wait loop re-read the directory word through an ordinary load, which is served
  from a multiprocessor's own L1 cache; that cache is not coherent between
  multiprocessors, and the lock holder's release is published at L2, so a waiting
  thread could keep observing its own stale copy of the locked value indefinitely
  while the lock had in fact been free for a long time. The whole run then made
  no further progress, with the GPU pinned at full utilization but drawing
  idle-level power. The spin now re-reads through an L2-serviced atomic, the
  idiom the rest of the module already uses, at a cost of one uncontended atomic
  per inserted witness. The defect is present in 1.0.5 and every earlier release,
  on every launch geometry, in every single-node large-prime run; it became
  frequently reproducible only once multiprocessor-aligned geometry changed the
  concurrency pattern enough to make same-bucket collisions likely. Cluster runs
  were never exposed: workers sieve without large primes and the coordinator
  matches on the host.
- **The standard (non-graph) batch sieve no longer re-sieves its final polynomial
  batch when the host stalls.** `prepareSievingBatch()` staged each batch's
  hypercube factor indices through a single reused pinned host buffer with no
  completion gate: the host copy into that buffer is immediate while the
  host-to-device copy that consumes it is deferred to GPU time, so with the host
  running several batches ahead of the device every queued transfer read
  whichever index set was written most recently. In steady state this only
  relabelled which batch sieved which polynomials, but whenever the host stopped
  preparing batches — at the end of the sieve loop, or at a checkpoint quiesce —
  every transfer still queued read the same final index set and that batch was
  sieved once per queued transfer, re-emitting its candidates and its large
  primes verbatim. The staging buffer is now double-buffered with a per-slot
  completion event that gates the host write, the same discipline the CUDA-graph
  path already used. At RSA-100 this removes about 9 % of all collected relations
  as pure waste (duplicates removed at deduplication fall from 25,673 to 889).
  Present in 1.0.5 and every earlier release. Runs with `--cuda_graph_unroll 0`
  legitimately report lower raw relation counts than they did under 1.0.5, since
  the wasted batches are gone.
- **The GPU large-prime combiner no longer merges a partial relation with
  itself.** When a partial is presented to the table more than once — a duplicate
  emission from the sieve, or a repeated range at the end of a cluster worker's
  assignment — it matches the witness copy inserted by its own first
  presentation, because matched witnesses are not purged at production
  factor-base sizes. Merging that pair produced a perfect square, a valid
  congruence that can only ever yield the trivial `X ≡ ±Y` dependency, and
  because each such row is unique it survived deduplication into the matrix as
  ballast that dilutes the per-solution success rate: roughly **2 % of every
  single-node large-prime matrix** built by 1.0.5 and earlier.
  `global_combine_kernel` now skips a combine whose two constituents share the
  same `sqrt_Q`, mirroring the guard the host-side cluster matcher has carried
  since 1.0.3a. At RSA-100 this removes roughly 7–8 rows per sieve batch in every
  mode, and the measured per-solution nontrivial-factor rate rises from 48.2 % to
  52.2 %. Cluster runs were never exposed.
- **Single-node large-prime relation accounting was losing roughly a quarter of
  its count.** The cumulative figure was accumulated on the host from a mapped
  counter read before the device had published it, which under an overlapped
  pipeline sampled the sequence rather than summing it; the reported cumulative
  could fall below the number of large-prime relations actually present in the
  final relation set, which is impossible for a pre-deduplication total. The
  count is now maintained on the device at the point where large-prime relations
  are committed, so it is counted exactly once each. Relation output was never
  affected — this was a reporting error only.
- **A legacy (non-batch) run could read an uninitialized batch size.** Both
  configuration loaders zeroed only the first of the four per-stage batch-size
  fields, leaving the other three indeterminate whenever the run never called the
  batch-size setter — that is, in every legacy run. Nothing read them in 1.0.5,
  so the defect was latent there; the batch-mode predicate introduced by this
  release does, and a stale non-zero value would let a legacy run accept an
  aligned tuple its gathering kernel cannot execute. Both loaders now zero all
  four, mirroring what the setter does for zero.

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
