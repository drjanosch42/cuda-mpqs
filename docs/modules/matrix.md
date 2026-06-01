# Matrix Module (`src/matrix/`)

Converts SoA relation data to a sparse GF(2) CSR matrix for the Block Wiedemann solver. Two preprocessing pipelines: **V1** (binary CSR + merge tree, M2-M8) for cluster/replay modes, and **V2** (packed 24+8 bit entries + GPU batch merges + compact-merge cycles, M9v2/M10-M12) for solo mode.

## Files

| File | Purpose |
|------|---------|
| `matrix_constructor.h` | `HostMatrixCSR` struct, `MatrixConstructor` class, free function declarations |
| `matrix_constructor.cu` | CUDA kernels, chunked CSR construction pipeline, validation and conversion helpers |
| `expanded_matrix.h` | `ExpandedMatrixBuilder` — CPU (F+2+L)-column expanded CSR construction |
| `expanded_matrix.cpp` | Expanded matrix builder implementation (LP column assignment, smooth/partial rows) |
| `merge_filter.h` | `MergeFilterPipeline` — CPU singleton removal + weight-2/higher-weight merges |
| `merge_filter.cpp` | Iterative singleton, weight-2, Markowitz merge implementations |
| `merge_tree.h` | `MergeTree` — records merge history for sqrt reconstruction (V1 pipeline) |
| `merge_tree.cpp` | Merge tree construction and `expandKernelVector()` |
| `matrix_truncation.h` | `truncateMatrix()` — CPU row truncation (select k sparsest rows) |
| `matrix_truncation.cpp` | Truncation implementation via `std::nth_element` partial sort |
| `character_columns.h` | `CharacterColumnComputer`, `CharacterColumns`, `computeProductCharacterColumns()` |
| `character_columns.cpp` | Quadratic character column computation (standard + product) |
| `gpu_singleton.cuh` | `gpuRemoveSingletons()` — binary CSR GPU singleton removal (cluster mode) |
| `gpu_singleton.cu` | 5-kernel singleton pipeline (S1-S5) + Thrust prefix sums |
| `gpu_char_cols.cuh` | `gpuComputeCharacterColumns()` — GPU standard character columns (CC1 kernel) |
| `gpu_char_cols.cu` | Jacobi symbol kernel for 32 auxiliary prime Legendre symbols |
| `preprocess.h` | `PreprocessResult`, `PreprocessResultV2`, `preprocessMatrix()`, `gpuPreprocessMatrix_packed()`, `selectKernelVectorRows()` |
| `preprocess.cpp` | Backend dispatch, V1 pipeline driver, V2 packed pipeline driver |
| `device_packed_csr.cuh` | `PackedEntry`, `DevicePackedCSR`, `DevicePackedView`, packed entry accessors |
| `device_packed_csr.cu` | RAII implementation, `isJetsonDevice()`, `downloadToHost()` |
| `gpu_packed_expanded.cuh` | `gpuBuildPackedMatrix()` — device-resident packed CSR construction (M9a) |
| `gpu_packed_expanded.cu` | E1-E4 kernels + Thrust LP column assignment + host driver |
| `gpu_singleton_packed.cuh` | `gpuRemoveSingletons_packed()` — packed CSR singleton removal (M9b) |
| `gpu_singleton_packed.cu` | Adapted S1-S5 kernels for packed entries + metadata compaction |
| `gpu_truncation_packed.cuh` | `gpuTruncate_packed()` — packed CSR row truncation (M9c) |
| `gpu_truncation_packed.cu` | Weight-sorted row selection + entry/metadata compaction |
| `gpu_batch_merge.cuh` | `gpuBatchMerge()`, `MergeCandidate`, `MontgomeryContext`, `DeviceMergeWorkspace`, `BatchMergeResult` (M9e); dual `d_col_weight`/`d_gf2_col_weight` (M11a) |
| `gpu_batch_merge.cu` | `execute_merges_kernel`, GF(2)-aware merge planner, workspace RAII |
| `gpu_compact_packed.cuh` | `gpuCompactPackedCSR()` (M10a), `gpuCompactMergeCycles()` (M10b), `CompactResult`, `CompactMergeResult` |
| `gpu_compact_packed.cu` | 4 compaction kernels (K1–K4) + K4-GF2 GF(2) column-weight kernels, host drivers |
| `gpu_gf2_extract.cuh` | `gpuExtractGF2()` — odd-exponent filter from packed merged CSR (M9f) |
| `gpu_gf2_extract.cu` | Two-pass GF(2) extraction kernels (count + write) |
| `gpu_product_char_packed.cuh` | `gpuProductCharCols_packed()` — Jacobi-only product char cols (M9f) |
| `gpu_product_char_packed.cu` | Jacobi kernel on pre-merged `sqrt_Q` (no Montgomery recomputation) |
| `matrix_utils.h` | Small host helpers (`fmtNum()` — thousands-separator formatting for logs) |
| `CMakeLists.txt` | Builds `mpqs_matrix` static library |

## `HostMatrixCSR`

```cpp
struct HostMatrixCSR {
    std::vector<uint32_t> row_offsets;  // size = n_rows + 1; row_offsets[i+1] - row_offsets[i] = Hamming weight of row i
    std::vector<uint32_t> col_indices;  // size = total_nnz; column indices sorted strictly increasing per row
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;                // factor_base_size + 2
};
```

## Column Mapping (GF(2) semantics)

| Source field | Condition | Column index |
|---|---|---|
| `signs[i]` | `!= 1` (negative) | 0 |
| `val_2_exps[i]` | odd (`& 1`) | 1 |
| `factor_indices[k]` (FB index `j`) | `factor_counts[k] & 1` | `j + 2` |

Each entry records the parity of the prime exponent. In GF(2), duplicate entries cancel (XOR), so column indices within a row must be strictly increasing — no duplicates allowed.

## `MatrixConstructor`

Manages one CUDA stream (`stream_compute_`) and three lazily-allocated device buffers (`d_nnz_per_row`, `d_row_offsets`, `d_col_chunk`), reused across calls with growth-factor reallocation.

```
static const size_t MATRIX_GENERATION_CHUNK_SIZE = 4194304;  // 4M rows per chunk
```

### Construction Pipeline (`constructFromSoA`)

1. **Allocate metadata buffers** — `d_nnz_per_row` and `d_row_offsets`, each of size `num_relations + 1`. Reallocated with 1.2× growth factor (minimum `num_relations + 1`) when capacity is exceeded.

2. **Compute row weights** — `calc_row_weights_soa_kernel` (256 threads/block, on `stream_compute_`): each thread counts the odd-exponent primes for one relation (sign column + factor-2 column + each odd factor-base prime).

3. **Exclusive scan** — Zero-initializes `d_nnz_per_row[num_relations]` (sentinel), then Thrust `exclusive_scan` over `d_nnz_per_row[0..num_relations]` → `d_row_offsets` on `stream_compute_`, giving CSR row pointers.

4. **Download offsets** — Async `cudaMemcpy` of `d_row_offsets` to `out_csr.row_offsets` on `stream_compute_`, then `cudaStreamSynchronize`. Reads `total_nnz = row_offsets[num_relations]`.

5. **Chunked column generation** — Iterates over rows in blocks of up to 4M:
   - Reallocates `d_col_chunk` with 1.5× growth factor if needed.
   - `fill_matrix_soa_chunk_kernel` (on `stream_compute_`): each thread writes the column indices for one row into the chunk buffer, using local offset `d_row_offsets[global_idx] - d_row_offsets[chunk_start_row]`.
   - Async `cudaMemcpy` of chunk to the correct position in `out_csr.col_indices` (same `stream_compute_`).

6. **Final sync** — `cudaStreamSynchronize(stream_compute_)`.

> **Note:** Caller must set `out_csr.n_cols = fb_size + 2` before validation; `constructFromSoA` only populates row structure and column indices.

## CUDA Kernels

### `calc_row_weights_soa_kernel`

```
inputs:  factor_offsets, signs, val_2_exps, factor_counts, num_relations
output:  d_nnz_per_row[i] = Hamming weight of row i
```

One thread per relation. Iterates over `factor_offsets[i]..factor_offsets[i+1]` counting `factor_counts[k] & 1`.

### `fill_matrix_soa_chunk_kernel`

```
inputs:  factor_offsets, factor_indices, factor_counts, signs, val_2_exps,
         chunk_start_row, chunk_num_rows, d_row_offsets
output:  d_cols_out[...] = sorted column indices for rows [chunk_start_row, chunk_start_row+chunk_num_rows)
```

One thread per row in the chunk. Columns are written in fixed order: 0 (sign), 1 (prime 2), then `factor_indices[k] + 2` for each odd-exponent factor-base prime. This order is strictly increasing because factor base indices are stored in ascending order in the SoA.

## Free Functions

| Function | Signature | Description |
|---|---|---|
| `ValidateHostMatrixCSR` | `(const HostMatrixCSR&) -> bool` | Three checks: (1) `row_offsets` size = `n_rows+1` and `row_offsets[0]=0` and last offset = `col_indices.size()`; (2) monotonicity (`start ≤ end` for each row); (3) column indices in bounds and strictly increasing within each row. |
| `ConvertFromCSR` | `(const HostMatrixCSR&) -> HostMatrix` | Converts CSR to `HostMatrix` (jagged `vector<vector<uint32_t>>`) for the Block Wiedemann solver. |

## Dependencies

- **Links:** `mpqs_postproc`, `mpqs_common`, `cudampqs_build_flags`
- **Headers:** `mpqs_soa.h` (`RelationBatchView`), `linalg/cuda_spmm/include/common.h` (`HostMatrix`), Thrust
- **Include paths:** `src/linalg/include` (for `lingen/types.h`), `src/linalg/cuda_spmm/include` (for `common.h` defining `HostMatrix`)

---

## V2 Packed GPU Preprocessing Pipeline (M9v2)

The V2 pipeline carries **packed 1-partial entries** through all preprocessing stages, computing `sqrt_Q` products incrementally via Montgomery multiplication during merges. The merge tree is eliminated entirely. The GF(2) matrix for BW is extracted at the end by an odd-exponent filter.

**Activation conditions** (gate in `orchestrator.cpp:1185`, `use_packed_pipeline`):
- `cluster_mode == SOLO` (workers/coordinator use the CPU V1 pipeline)
- GPU backend selected (`--matrix_backend gpu` or `auto`)
- **Either** a live sieve postprocessor exists (device-resident smooth + LP witness batches),
  **or** `MATRIX_ONLY` mode is active (host relations uploaded device-side for the packed path)

`LINALG_ONLY` and cluster mode always use the CPU V1 pipeline.

**Pipeline (current, M9v2 + M10–M12 — chained by `gpuPreprocessMatrix_packed()` in `preprocess.cpp`):**

```
M9a:  gpuBuildPackedMatrix()          — packed (F+2+L)-column CSR on device
M9b:  gpuRemoveSingletons_packed()    — in-place singleton removal
M10b: gpuCompactMergeCycles()         — merge → compact → merge … (M11a GF(2)-aware planner;
        ├ gpuBatchMerge()               compact_cycles==0 falls back to a single gpuBatchMerge pass)
        └ gpuCompactPackedCSR()       — fresh contiguous CSR per cycle (4 compaction kernels)
M9f:  gpuExtractGF2()                 — odd-exponent filter -> binary CSR for BW
M11b: MergeFilterPipeline::removeSingletons()
                                      — post-merge GF(2) singleton removal on host CSR
M9c-post: truncateMatrix()            — M12-S1 coverage-greedy, char-col-aware row truncation
M9f:  gpuProductCharCols_packed()     — Jacobi-only product character columns (appended last)
```

The standalone `gpuTruncate_packed()` packed-pre-merge truncation (M9c) was removed from the
driver in M11c-S1 (it preferentially dropped LP-combined rows and broke LP column pairs);
truncation now runs once, post-merge, on the clean GF(2) CSR.

### Packed Entry Format

Each CSR entry stores a column index and exponent in a single `uint32_t`:

```cpp
using PackedEntry = uint32_t;

uint32_t packed_col(PackedEntry e) { return e >> 8; }       // 24-bit column [0, 16M)
uint8_t  packed_exp(PackedEntry e) { return e & 0xFF; }     // 8-bit exponent [0, 255]
PackedEntry make_packed(uint32_t col, uint8_t exp) { return (col << 8) | exp; }
```

Same 4 bytes/entry as binary CSR, but retains full exponents throughout all merge stages. Column range [0, 2^24) = 16M suffices up to RSA-200. All accessors are `__host__ __device__`.

### `DevicePackedCSR`

RAII owner of device-resident packed CSR buffers + per-row 1-partial metadata.

```cpp
struct DevicePackedCSR {
    uint32_t*    d_row_offsets;  // [n_rows + 1]
    PackedEntry* d_entries;      // [nnz] — (col_index << 8) | exponent

    uint512*     d_sqrt_Q;       // [n_rows] — Montgomery product of constituent sqrt_Q
    uint8_t*     d_signs;        // [n_rows] — product of signs
    int32_t*     d_val_2_exps;   // [n_rows] — sum of val_2 exponents

    uint32_t n_rows, n_cols, nnz;
    bool use_managed;            // true on Jetson (cudaMallocManaged)
};
```

`DevicePackedView` is the non-owning, kernel-passable POD subset (same fields, no RAII).

### M9a: Device-Resident Packed CSR Construction

`gpuBuildPackedMatrix()` builds the expanded (F+2+L)-column packed CSR directly on device from the sieve stage's persistent `RelationBatch` (smooths) and LP witness batch (partials). No host round-trip.

**Column encoding in packed entries:**
- Column 0 (sign): exponent = 1 if sign is negative
- Column 1 (prime 2): exponent = `val_2_exp`
- Columns [2, fb_size+1] (FB primes): exponent = `factor_counts[k]`
- Columns [fb_size+2, ...] (LP): exponent = 1 (smooth rows with LP-combined origins get exponent 2 for the L^2 factor)

**LP column assignment:** Deterministic sort-based ascending LP value order via `thrust::sort` + `thrust::unique` + `thrust::lower_bound`. This is a column permutation relative to the CPU `ExpandedMatrixBuilder`'s insertion-order assignment.

**CUDA kernels:**

| Kernel | Grid | Description |
|--------|------|-------------|
| E1: `extract_lp_values_kernel` | ceil(n_partial/256) | Extract LP from `__int128` to `uint64_t` for Thrust sort |
| E2: `count_row_nnz_packed_kernel` | ceil(n_total/256) | Count ALL nonzero-exponent entries per row (not just odd-exponent) |
| E3: `fill_packed_csr_kernel` | ceil(n_total/256) | Write `make_packed(col, exp)` entries in sorted column order |
| E4: `copy_row_metadata_kernel` | ceil(n_total/256) | Copy `sqrt_Q`, `sign`, `val_2_exp` from relation batches |

### M9b: Packed Singleton Removal

`gpuRemoveSingletons_packed()` adapts the 5-kernel binary singleton pipeline for packed entries. Column indices extracted via `entry >> 8`. Per-row metadata (`sqrt_Q`, `signs`, `val_2_exps`) compacted in parallel. Input `DevicePackedCSR` is consumed (freed on return).

Column remapping preserves exponents:
```cpp
new_entry = make_packed(col_map[packed_col(old_entry)], packed_exp(old_entry))
```

Returns `PackedSingletonResult` with reduced `DevicePackedCSR` (device-resident), host `row_map` and `col_map`.

### M9c: Packed Truncation (legacy — not in current driver)

`gpuTruncate_packed()` selects the k = ceil(`truncation_factor` * `n_cols`) sparsest rows. Entries copied verbatim (no column remapping -- only rows removed). Per-row metadata compacted in parallel. Input consumed.

Algorithm: compute row weights from CSR offsets, `thrust::sort_by_key` ascending, mark first k as alive, prefix-sum scatter, compact CSR + metadata.

> **Removed from the V2 driver in M11c-S1.** Pre-merge packed truncation preferentially removed LP-combined rows (which carry more columns), breaking LP column pairs and collapsing ~99.98% of columns on high-LP data. The driver now truncates *once*, post-merge, on the host GF(2) CSR via the coverage-greedy `truncateMatrix()` (M9c-post / M12-S1, below). The `gpu_truncation_packed.*` files remain in-tree but are unused by `gpuPreprocessMatrix_packed()`.

### M9e: Batch-Planned GPU Merge Execution

The core innovation. CPU plans merge schedules; GPU executes non-conflicting merges in parallel.

**Architecture: CPU planner + GPU executor**

```
CPU:  scan col_weight[] -> identify weight-2/Markowitz candidates
      upload MergeCandidate[] to device
GPU:  one thread per candidate:
      atomicCAS(row_locks[r1,r2]) -> claim ownership
      two-pointer packed merge (exponent addition)
      Montgomery sqrt_Q multiply
      atomic_reserve_dual -> workspace slot
      write merged row + update row_ptr + col_weight
      release locks
```

**Merge semantics:** For packed entries, matching columns get `exp_sum = e1 + e2` (exponent addition), unlike binary CSR where matching columns cancel (XOR). The merged row is strictly a union of columns with summed exponents. The GF(2) equivalence is recovered by the odd-exponent filter in M9f: `exp_sum` even implies both original exponents had the same parity, hence XOR cancellation. Proof: see audit v4 Section 10.5.

**M11c — sign-encoding fix.** The merge kernel (and cluster CPU LP matching) originally assumed a
`{1, 255}` sign encoding, but the relation data uses `{0, 1}`. The merged sign is now computed with
encoding-agnostic XOR boolean logic (`signs[r] = signs[r1] ^ signs[r2]` at the boolean level), and
the cluster CPU LP path was switched from XOR-on-the-wrong-encoding to a multiply. This eliminated a
53% `HalveExponents` validity-check failure rate downstream in sqrt. M11c-S2 also hardened the
merge: defensive `cudaMemset` of `d_ws_signs`/`d_ws_val_2_exps`/`d_ws_sqrt_Q`, and a
`__threadfence()` before merge-lock release.

**`MergeCandidate`**

```cpp
struct MergeCandidate {
    uint32_t r1;         // first row (survives as merge result)
    uint32_t r2;         // second row (consumed, marked DEAD)
    uint32_t pivot_col;  // column being eliminated
};
```

**`MontgomeryContext`**

Kernel-passable POD subset of the `Montgomery` class (~136 bytes). Provides `mul()`, `transform()`, `reduce()` as `__device__` methods. Constructed on host via `makeMontgomeryContext()`.

```cpp
struct MontgomeryContext {
    uint512  N;        // modulus
    uint512  R2;       // R^2 mod N (for transform)
    uint32_t n_prime;  // -N^(-1) mod 2^32 (CIOS reduction)
};
```

**`DeviceMergeWorkspace`**

Append-only workspace for merged rows. Original CSR is immutable; merged rows live here.

```cpp
struct DeviceMergeWorkspace {
    PackedEntry* d_ws_entries;      // flat array of merged row entries
    uint32_t*    d_ws_row_starts;   // [max_merged_rows] start offset per merged row
    uint32_t*    d_ws_row_lengths;  // [max_merged_rows] length per merged row
    uint512*     d_ws_sqrt_Q;       // [max_merged_rows]
    uint8_t*     d_ws_signs;        // [max_merged_rows]
    int32_t*     d_ws_val_2_exps;   // [max_merged_rows]
    uint64_t*    d_dual_counter;    // packed [row_count | entry_count] bump allocator
    uint32_t*    d_row_ptr;         // [n_total_rows] logical->physical indirection
    uint32_t*    d_row_locks;       // [n_total_rows] atomicCAS ownership
    uint32_t*    d_col_weight;      // [n_cols] structural weight (all entries), maintained across rounds
    uint32_t*    d_gf2_col_weight;  // [n_cols] GF(2) weight (odd-exponent entries only) — M11a planning
    uint32_t*    d_abort_count;     // per-launch abort counter
};
```

**M11a — dual GF(2)/structural column weights.** Merge *planning* (candidate selection for
weight-2 and Markowitz merges) ranks columns by `d_gf2_col_weight` (odd-exponent entries only),
because a column that is GF(2)-trivial cannot pivot a kernel vector even if it carries many
even-exponent entries. Merge *execution* and compaction remain parity-agnostic and update the
structural `d_col_weight`. The inverted-index builders that feed the planner filter by
odd-exponent parity. `BatchMergeResult::h_gf2_col_weight` exposes the final GF(2) weights to the
driver for the GF(2) dimension estimate (`M11a:` log line) and the M12-S2 floor.

**Row indirection (`d_row_ptr`):**
- MSB=0: original CSR row index (read from `DevicePackedCSR`)
- MSB=1: workspace row index (`index & 0x7FFFFFFF`, read from `d_ws_*`)
- `0xFFFFFFFF` (`ROW_DEAD`): row consumed by merge

**Conflict resolution:** Lazy `atomicCAS` on `d_row_locks`. If either row lock fails, the candidate aborts and retries next round. Expected abort rate <5% at RSA-110 (240K row references across 305K rows).

**Round structure (RSA-110 estimates):**
- Weight-2 phase: 5-8 rounds, ~120K total merges
- Higher-weight (Markowitz) phase: 3-5 rounds, ~35K total merges
- CPU planning cost: O(n_cols) scan per round, <1ms

### M10: Compact-Merge Cycles

A single `gpuBatchMerge()` pass saturates: as merges accumulate, the workspace fills, scattered
indirection (`d_row_ptr`) grows stale, and the abort rate climbs — leaving merge opportunities on
the table. `gpuCompactMergeCycles()` interleaves merge passes with **compaction** to restart from a
clean, contiguous state each cycle:

```
loop until convergence / budget / GF(2) floor:
    gpuBatchMerge()         — one batch-merge pass on the current contiguous CSR
    gpuCompactPackedCSR()   — collapse scattered CSR into a fresh DevicePackedCSR
final gpuBatchMerge()       — one last pass, no trailing compaction
```

**`gpuCompactPackedCSR()`** (`gpu_compact_packed.cuh:67`) takes the post-merge `DevicePackedCSR`
plus the `BatchMergeResult` (workspace + `h_row_ptr`) and produces a fresh contiguous
`DevicePackedCSR` with: only alive rows (`d_row_ptr[r] != ROW_DEAD`), workspace rows resolved
inline, columns remapped to drop empty columns, and per-row metadata (`sqrt_Q`, `signs`,
`val_2_exps`) preserved. It returns a `CompactResult` carrying per-cycle `row_map` /`col_map`
(for cumulative composition) and `gf2_n_cols` (alive GF(2) column count, for the M12-S2 floor).

**4 compaction kernels** (`gpu_compact_packed.cu`):

| Kernel | Description |
|--------|-------------|
| K1: `mark_alive_kernel` | Build `d_alive_mask` from `d_row_ptr` (skip `ROW_DEAD`) |
| K2: `compute_new_lengths_kernel` | Resolve per-row lengths via original/workspace indirection |
| K3: `compact_entries_metadata_kernel` | Fused entry copy (with column remap) + metadata copy |
| K4: `compute_col_weight_alive_kernel` | Accurate column weights from alive rows only |

A parallel K4-GF2 kernel (`compute_gf2_col_weight_alive_kernel`) recomputes GF(2) column weights
(odd-exponent parity guard) so the next cycle's planner and the M12-S2 floor see accurate GF(2)
diversity.

**`gpuCompactMergeCycles()`** (`gpu_compact_packed.cuh:119`) drives the loop. Termination criteria:

1. **Convergence** — `cycle_merges < 0.02 × alive_rows` (less than 2% of rows merged this cycle).
2. **Budget exhausted** — `cycle >= max_cycles` (the `--compact_cycles` value, default 5).
3. **GF(2) column-diversity floor (M12-S2)** — stop when post-compaction GF(2) columns drop below
   `max(gf2_min_floor, gf2_floor_factor × initial_gf2_cols)`.

Per-cycle merge budget: `computeCycleBudget() = n_rows − ceil(truncation_factor × n_cols)`. The
driver composes the per-cycle `row_map`/`col_map` into a `CompactMergeResult` with
`cumulative_row_map` (final compacted row → post-singleton relation index) and `cumulative_col_map`
(final compacted col → post-singleton column index), used downstream for sqrt traceability and
`singleton_col_map` reconstruction. `compact_cycles == 0` bypasses this driver entirely and runs a
single `gpuBatchMerge()` pass (pre-M10 behavior).

### M9f: GF(2) Extraction

`gpuExtractGF2()` produces the binary CSR for BW by filtering packed entries with odd exponents.

Two-pass GPU algorithm:
1. Count odd-exponent entries per row (`entry & 1`)
2. Prefix sum over counts
3. Write column indices (`entry >> 8`) for odd-exponent entries

For rows that went through merges, entries are read via `d_row_ptr` indirection (original CSR or workspace). The resulting GF(2) CSR is **identical** to the binary CSR that would result from performing all merges as GF(2) XOR operations.

Returns `GF2ExtractionResult` with `HostMatrixCSR` + `row_map` (gf2_row -> merged_row index).

### M11b: Post-Merge GF(2) Singleton Removal

Merging can create new GF(2) singletons (weight-1 columns) that the pre-merge packed pass (M9b)
could not see. The driver runs the host `MergeFilterPipeline::removeSingletons()` directly on the
GF(2) CSR from M9f. When rows are removed, the singleton `row_map` is **composed** with the GF(2)
`row_map` (`composed[i] = gf2.row_map[sr.row_map[i]]`) so the final reduced rows still trace back to
merged-row indices for sqrt. A zero-dimension guard prevents CUDA crashes on degenerate matrices.

### M9c-post / M12-S1: Coverage-Greedy Truncation

`truncateMatrix()` (`matrix_truncation.h`) runs on the clean host GF(2) CSR, *before* product
character columns are appended. M12-S1 replaced the old sparsest-row bias with a **coverage-greedy**
selector and a **char-col-aware** target:

```
target_rows = max(n_cols + n_extra_cols + k_excess, <coverage minimum>)
```

The driver calls it with `n_extra_cols = 32` (the product char columns about to be appended) and
`k_excess = truncation_excess` (CLI `--matrix_truncation_excess`, default 200), so the
post-augmentation matrix is overdetermined by a controlled excess rather than by a hardcoded
factor. `truncation_factor > 0` is retained only as an on/off switch; the actual target is
excess-based. The selector greedily keeps rows that cover still-uncovered columns until the column
coverage threshold is met, then fills to `target_rows`; a rare relaxation fallback guarantees no
column is left with zero coverage.

### M9f: Product Character Columns (Packed)

`gpuProductCharCols_packed()` computes 32 quadratic character columns from pre-merged `sqrt_Q` values. No Montgomery recomputation needed -- `sqrt_Q` products were computed incrementally during M9e merges.

Per row r, evaluates: `jacobi((sqrt_Q[r]^2 - N) mod q, q)` for 32 auxiliary primes q.

Structurally identical to the standard `char_col_kernel` (CC1), reading from merged `sqrt_Q` instead of original relation `sqrt_Q`. Cost: ~5ms at RSA-110 (155K rows x 32 Jacobi evaluations).

### `PreprocessResultV2`

Unified output of the V2 pipeline. Replaces `PreprocessResult` + `MergeTree` with direct merged 1-partial data.

```cpp
struct PreprocessResultV2 {
    HostMatrixCSR reduced;                        // GF(2) CSR for BW (odd-exponent filter)
    std::vector<uint32_t> row_map;                // reduced_row -> merged_row index

    // Merged 1-partial data (replaces merge tree)
    std::vector<uint512>  merged_sqrt_Q;           // [n_merged_rows]
    std::vector<uint8_t>  merged_signs;            // [n_merged_rows]
    std::vector<int32_t>  merged_val_2_exps;       // [n_merged_rows]
    std::vector<uint32_t> merged_factor_offsets;    // [n_merged_rows + 1] CSR into factors
    std::vector<uint32_t> merged_factor_indices;    // [nnz_merged] column indices
    std::vector<uint8_t>  merged_factor_exponents;  // [nnz_merged] full exponents

    uint32_t fb_size;                             // columns [2..fb_size+1] = FB
    std::vector<uint32_t> singleton_col_map;      // post-singleton col -> expanded col

    // Diagnostics
    uint32_t singletons_removed, singleton_iterations;
    uint32_t w2_merges, hw_merges, gf2_nnz;
};
```

### `selectKernelVectorRows()`

Replaces `expandKernelVector()` for V2. For each set bit i in the BW solution, `row_map[i]` gives the merged row index. No merge tree traversal needed.

```cpp
std::vector<uint32_t> selectKernelVectorRows(
    const std::vector<uint64_t>& packed_bits,
    uint32_t num_reduced_rows,
    const std::vector<uint32_t>& row_map);
```

### `gpuPreprocessMatrix_packed()`

Top-level V2 pipeline driver (`preprocess.h:153`). Chains
M9a -> M9b -> M10b (compact-merge cycles) -> M9f (GF(2) extract) -> M11b (post-merge GF(2)
singleton) -> M9c-post (M12-S1 truncation) -> M9f (product char cols).

```cpp
PreprocessResultV2 gpuPreprocessMatrix_packed(
    const structures::RelationBatchView& smooth_view, uint64_t n_smooth,
    const structures::RelationBatchView& partial_view, uint64_t n_partial,
    uint32_t fb_size,
    const uint512& N,                   // for MontgomeryContext
    const std::vector<uint32_t>& fb,    // for character column aux prime selection
    uint32_t k_max = 10,                // max column weight for higher-weight merges
    uint32_t max_weight = 200,          // fill-in limit for higher-weight merges
    double   truncation_factor = 1.05,  // > 0 enables M9c-post truncation, 0 disables
    uint32_t compact_cycles = 5,        // M10 max compact-merge cycles; 0 = single pass
    uint32_t truncation_excess = 200,   // M12-S1 excess rows over (n_cols + 32 char cols)
    double   gf2_floor_factor = 0.5,    // M12-S2 floor as fraction of initial GF(2) col count
    uint32_t gf2_min_floor = 8192);     // M12-S2 absolute GF(2) col floor
```

> The legacy `k_max=10, max_weight=200, truncation_factor=1.05` defaults still seed the merge
> planner, but `truncation_factor` now acts as an on/off switch (the real truncation target is the
> excess-based M12-S1 formula). The four trailing parameters (`compact_cycles`, `truncation_excess`,
> `gf2_floor_factor`, `gf2_min_floor`) drive the M10 compact-merge loop and the M12-S2 GF(2) floor.

### Sqrt Consumption

The sqrt stage reads merged 1-partials directly:
- **X computation:** `X = product(merged_sqrt_Q[selected]) mod N` -- the `sqrt_Q` values in `PreprocessResultV2` are already the products of all constituent leaf relations.
- **Y computation:** `Y = product(p^(e_merged/2))` from `merged_factor_exponents[]`. Exponents are summed across all leaves, guaranteed even by the GF(2) kernel vector constraint. Halve and exponentiate.

No merge tree expansion. No LP correction. LP-combined smooth rows carry exponent 2 for the L^2 factor in their packed entries, and the `large_primes[i] = 1` override in the combined batch ensures the sqrt pipeline does not double-count.

### LP-Combined Smooth Row Handling

LP-combined relations (two raw partials matched on the same large prime L) have `Q = Q1 * Q2`, which includes the factor L^2. In the expanded matrix:
- The L column gets exponent 2 (even), so it vanishes in GF(2) -- correct behavior.
- The packed representation stores `make_packed(lp_col, 2)`, preserving the L^2 for sqrt Y computation.

This replaces the M5 `ApplyLPCorrection` mechanism from the V1 pipeline.

### Memory Budget (RSA-110, RTX 5070 Ti)

| Component | Size | Notes |
|---|---|---|
| `DevicePackedCSR` (expanded) | ~71 MB | 325K rows, ~10M NNZ + 23 MB metadata |
| `DeviceMergeWorkspace` | ~44 MB | ~155K merges, ~8M new entries |
| Column weights + row locks + row_ptr | ~3.5 MB | 280K cols + 305K rows |
| **Total GPU merge phase** | **~119 MB** | <1% of RTX 5070 Ti VRAM |

### Mode Selection

The preprocessing path is chosen in `orchestrator.cpp` from `--matrix_mode`, `--matrix_backend`,
the measured LP fraction vs. `--lp_preprocess_threshold`, and the execution mode.

| Mode | Pipeline | Reason |
|---|---|---|
| Solo, GPU backend, preprocess | V2 (packed + M10–M12) | Device-resident data, full GPU acceleration |
| Solo, CPU backend | V1 (binary CSR + merge tree) | Explicit `--matrix_backend cpu` |
| Cluster | V1 (binary CSR + merge tree) | Relations arrive on host via TCP |
| `LINALG_ONLY` | V1 (binary CSR + merge tree) | Relations loaded from disk to host |
| `MATRIX_ONLY` (`--matrix_only`) | Load v2 relations → Matrix → BW → Sqrt | Replay device-saved v2 relations; can drive V2 packed path |

> **`MATRIX_ONLY` mode** (`ExecutionMode::MATRIX_ONLY`, `orchestrator.h:63`): loads device-format
> `relations.v2` and runs Matrix → BW → Sqrt without sieving. Combined with `--partial_subsample` /
> `--smooth_subsample` it is the standard harness for matrix-preprocessing experiments against
> stored relation sets (`mpqs_work/*.v2`).

### CLI

Flag spellings and defaults below are the single source of truth as parsed in
`tests/cuda-mpqs.cpp` (matrix block ~lines 220–233, `--matrix_only` at ~line 375) and mapped to
`MPQSConfig` fields in `include/orchestrator.h:98–126`.

| Flag | Default | Description |
|---|---|---|
| `--matrix_backend <cpu\|gpu\|auto>` | `cpu` | Preprocessing backend. `auto` = GPU if available + >10K rows |
| `--matrix_mode <legacy\|preprocess>` | auto | `legacy` = projected FB+2 columns; `preprocess` = expanded + merges |
| `--truncation_factor <float>` | 1.05 | Truncation enable flag (>0 enabled, 0 disabled). Actual target is excess-based — see `--matrix_truncation_excess` |
| `--matrix_truncation_excess <N>` | 200 | M12-S1 excess rows above `(n_cols + n_extra_cols)` after truncation |
| `--matrix_gf2_floor_factor <float>` | 0.5 | M12-S2: stop compact-merge cycles when GF(2) cols fall below `factor × initial_gf2_cols` [0.0–1.0] |
| `--matrix_gf2_min_floor <N>` | 8192 | M12-S2 absolute minimum GF(2) column floor |
| `--compact_cycles <N>` | 5 | M10 max compact-merge cycles (GPU backend); 0 = single pass (pre-M10 behavior) |
| `--lp_preprocess_threshold <float>` | 0.55 | LP fraction above which preprocess mode auto-activates |
| `--lp_matrix_threshold <float>` | — | **DEPRECATED** alias for `--lp_preprocess_threshold` (backwards compatibility) |
| `--matrix_only` | off | Load v2 relations, run matrix preprocessing + BW + sqrt (no sieving) |
| `--partial_subsample <float>` | 1.0 | `matrix_only` experiments: fraction of partials/LP-combined to retain [0.0–1.0] |
| `--smooth_subsample <float>` | 1.0 | `matrix_only` experiments: fraction of pure smooths to retain (LP-combined always kept) [0.0–1.0] |
