// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_singleton_packed.cu
//
// GPU singleton removal for packed CSR matrices (M9b).
// Adapts the 5-kernel binary singleton pipeline (gpu_singleton.cu) to operate on
// packed entries ((col_index << 8) | exponent). Adds kernel PS6 for per-row metadata
// compaction (sqrt_Q, signs, val_2_exps).
//
// Column index extraction: entry >> 8 (packed_col()).
// Column remapping:        make_packed(col_map[packed_col(e)], packed_exp(e)).
// Metadata compaction:     gather via d_row_map — identical semantics to entry compaction.
//
// The input DevicePackedCSR is consumed: device buffers are freed after compaction.
// No H→D upload is required (DevicePackedCSR is already device-resident from M9a).

#include "gpu_singleton_packed.cuh"
#include "cuda_check.h"
#include "hpc_logger.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace mpqs {
namespace matrix {

// ============================================================================
// Scratch buffer RAII struct
// ============================================================================

/// All GPU working buffers for packed singleton removal.
/// Does NOT own the input d_row_offsets and d_entries (those live in DevicePackedCSR).
/// On Jetson (use_managed == true), all allocations use cudaMallocManaged.
/// d_removed_count uses cudaMallocHost (pinned, zero-copy) on RTX.
struct PackedSingletonBuffers {
    uint32_t*    d_col_weight      = nullptr;  ///< [n_cols]     atomic column weights
    uint8_t*     d_row_alive       = nullptr;  ///< [n_rows]     1 = alive, 0 = removed
    uint32_t*    d_removed_count   = nullptr;  ///< [1]          pinned (RTX) or managed (Jetson)
    uint32_t*    d_row_prefix      = nullptr;  ///< [n_rows]     exclusive scan of d_row_alive
    uint32_t*    d_col_prefix      = nullptr;  ///< [n_cols]     exclusive scan of (col_weight > 0)
    uint32_t*    d_old_to_new_col  = nullptr;  ///< [n_cols]     old_col -> new_col (UINT32_MAX if dead)
    uint32_t*    d_row_map         = nullptr;  ///< [n_rows]     new_row -> original_row
    uint32_t*    d_new_row_offsets = nullptr;  ///< [n_rows + 1] compacted row offsets
    PackedEntry* d_new_entries     = nullptr;  ///< [nnz]        compacted packed entries (upper bound)
    uint32_t*    d_new_row_nnz     = nullptr;  ///< [n_rows]     per-new-row NNZ (temp for scan)

    bool use_managed = false;

    void alloc(uint32_t n_rows, uint32_t n_cols, uint32_t nnz, bool jetson);
    ~PackedSingletonBuffers();
};

void PackedSingletonBuffers::alloc(
        uint32_t n_rows, uint32_t n_cols, uint32_t nnz, bool jetson) {
    use_managed = jetson;

    // Allocate at least 1 element to avoid zero-size alloc edge cases.
    const uint32_t safe_rows = (n_rows > 0) ? n_rows : 1u;
    const uint32_t safe_cols = (n_cols > 0) ? n_cols : 1u;
    const uint32_t safe_nnz  = (nnz  > 0)  ? nnz    : 1u;

    auto device_malloc = [&](void** ptr, size_t bytes) {
        if (use_managed) {
            CUDA_CHECK(cudaMallocManaged(ptr, bytes));
        } else {
            CUDA_CHECK(cudaMalloc(ptr, bytes));
        }
    };

    device_malloc((void**)&d_col_weight,      safe_cols * sizeof(uint32_t));
    device_malloc((void**)&d_row_alive,       safe_rows * sizeof(uint8_t));
    device_malloc((void**)&d_row_prefix,      safe_rows * sizeof(uint32_t));
    device_malloc((void**)&d_col_prefix,      safe_cols * sizeof(uint32_t));
    device_malloc((void**)&d_old_to_new_col,  safe_cols * sizeof(uint32_t));
    device_malloc((void**)&d_row_map,         safe_rows * sizeof(uint32_t));
    device_malloc((void**)&d_new_row_offsets, (safe_rows + 1) * sizeof(uint32_t));
    device_malloc((void**)&d_new_entries,     safe_nnz  * sizeof(PackedEntry));
    device_malloc((void**)&d_new_row_nnz,     safe_rows * sizeof(uint32_t));

    // d_removed_count: pinned host memory for zero-copy on RTX; managed on Jetson.
    if (use_managed) {
        CUDA_CHECK(cudaMallocManaged((void**)&d_removed_count, sizeof(uint32_t)));
    } else {
        CUDA_CHECK(cudaMallocHost((void**)&d_removed_count, sizeof(uint32_t)));
    }
}

PackedSingletonBuffers::~PackedSingletonBuffers() {
    auto free_device = [](void* ptr) { if (ptr) cudaFree(ptr); };

    free_device(d_col_weight);
    free_device(d_row_alive);
    free_device(d_row_prefix);
    free_device(d_col_prefix);
    free_device(d_old_to_new_col);
    free_device(d_row_map);
    free_device(d_new_row_offsets);
    free_device(d_new_entries);
    free_device(d_new_row_nnz);

    if (d_removed_count) {
        if (use_managed) cudaFree(d_removed_count);
        else             cudaFreeHost(d_removed_count);
    }
}

// ============================================================================
// Thrust functors
// ============================================================================

/// Cast uint8_t alive flag to uint32_t for prefix scan accumulation.
/// Uses Packed prefix to avoid ODR collision with gpu_singleton.cu's AliveToUint32.
struct PackedAliveToUint32 {
    __host__ __device__ uint32_t operator()(uint8_t v) const {
        return static_cast<uint32_t>(v);
    }
};

/// Map column weight to 0/1 alive indicator for prefix scan.
struct PackedColAlive {
    __host__ __device__ uint32_t operator()(uint32_t w) const {
        return w > 0u ? 1u : 0u;
    }
};

/// Given original row index r_old, return its NNZ from d_row_offsets.
struct PackedRowNNZFunctor {
    const uint32_t* d_row_offsets;
    __host__ __device__ uint32_t operator()(uint32_t r_old) const {
        return d_row_offsets[r_old + 1] - d_row_offsets[r_old];
    }
};

// ============================================================================
// Kernel PS1: compute_col_weights_packed_kernel
// ============================================================================

/// One thread per row. For each alive row, atomically increments the weight of
/// every column referenced by a packed entry (column index = entry >> 8).
/// Called once before the fixpoint loop.
__global__ __launch_bounds__(256)
void compute_col_weights_packed_kernel(
    const uint32_t*    __restrict__ d_row_offsets,
    const PackedEntry* __restrict__ d_entries,
    const uint8_t*     __restrict__ d_row_alive,
    uint32_t* __restrict__ d_col_weight,
    uint32_t n_rows)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !d_row_alive[r]) return;

    const uint32_t begin = d_row_offsets[r];
    const uint32_t end   = d_row_offsets[r + 1];
    for (uint32_t j = begin; j < end; ++j) {
        atomicAdd(&d_col_weight[d_entries[j] >> 8], 1u);
    }
}

// ============================================================================
// Kernel PS2: singleton_pass_packed_kernel
// ============================================================================

/// One thread per row.  Critical fixpoint kernel — run iteratively until convergence:
///   1. Skip dead rows.
///   2. Scan columns of this row; check if any d_col_weight[packed_col(entry)] == 1.
///   3. If a singleton column is found: mark row dead, decrement all its column weights,
///      increment d_removed_count.
///
/// Correctness: concurrent atomicSub from dying rows may cause additional rows to die
/// within the same iteration (GPU parallel removal). The singleton-free fixpoint is the
/// same regardless of removal order.
__global__ __launch_bounds__(256)
void singleton_pass_packed_kernel(
    const uint32_t*    __restrict__ d_row_offsets,
    const PackedEntry* __restrict__ d_entries,
    uint8_t*  __restrict__ d_row_alive,
    uint32_t* __restrict__ d_col_weight,
    uint32_t* d_removed_count,
    uint32_t n_rows)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !d_row_alive[r]) return;

    const uint32_t begin = d_row_offsets[r];
    const uint32_t end   = d_row_offsets[r + 1];

    bool has_singleton = false;
    for (uint32_t j = begin; j < end; ++j) {
        if (d_col_weight[packed_col(d_entries[j])] == 1u) {
            has_singleton = true;
            break;
        }
    }

    if (has_singleton) {
        d_row_alive[r] = 0u;
        for (uint32_t j = begin; j < end; ++j) {
            atomicSub(&d_col_weight[packed_col(d_entries[j])], 1u);
        }
        atomicAdd(d_removed_count, 1u);
    }
}

// ============================================================================
// Kernel PS3: build_row_map_packed_kernel
// ============================================================================

/// One thread per row.  For each alive row r, writes r into
/// d_row_map[d_row_prefix[r]], building the new_row -> orig_row map.
/// d_row_prefix must be the exclusive scan of d_row_alive (uint8_t -> uint32_t).
__global__ __launch_bounds__(256)
void build_row_map_packed_kernel(
    const uint8_t*  __restrict__ d_row_alive,
    const uint32_t* __restrict__ d_row_prefix,
    uint32_t* __restrict__ d_row_map,
    uint32_t n_rows)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    if (d_row_alive[r]) {
        d_row_map[d_row_prefix[r]] = r;
    }
}

// ============================================================================
// Kernel PS4: build_col_map_packed_kernel
// ============================================================================

/// One thread per column.  Writes the old->new column remapping:
///   d_old_to_new_col[c] = d_col_prefix[c]   if d_col_weight[c] > 0 (alive)
///   d_old_to_new_col[c] = UINT32_MAX          otherwise (dead column)
/// d_col_prefix must be the exclusive scan of (col_weight[c] > 0 ? 1 : 0).
__global__ __launch_bounds__(256)
void build_col_map_packed_kernel(
    const uint32_t* __restrict__ d_col_weight,
    const uint32_t* __restrict__ d_col_prefix,
    uint32_t* __restrict__ d_old_to_new_col,
    uint32_t n_cols)
{
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cols) return;
    d_old_to_new_col[c] = (d_col_weight[c] > 0u) ? d_col_prefix[c] : UINT32_MAX;
}

// ============================================================================
// Kernel PS5: compact_packed_csr_kernel
// ============================================================================

/// One thread per new row r_new.  Copies surviving packed entries from the original CSR
/// to the compacted CSR, remapping the column-index portion via d_old_to_new_col while
/// preserving the 8-bit exponent:
///   new_entry = make_packed(col_map[packed_col(old_entry)], packed_exp(old_entry))
///
/// Column order within each row is preserved: col_map is monotone for alive columns
/// (since it is a prefix-sum prefix), and LP column is appended last.
/// d_new_row_offsets must be filled before launch.
__global__ __launch_bounds__(256)
void compact_packed_csr_kernel(
    const uint32_t*    __restrict__ d_old_row_offsets,
    const PackedEntry* __restrict__ d_old_entries,
    const uint32_t*    __restrict__ d_row_map,
    const uint32_t*    __restrict__ d_new_row_offsets,
    const uint32_t*    __restrict__ d_old_to_new_col,
    PackedEntry* __restrict__ d_new_entries,
    uint32_t new_n_rows)
{
    const uint32_t r_new = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_new >= new_n_rows) return;

    const uint32_t r_old     = d_row_map[r_new];
    uint32_t       write_pos = d_new_row_offsets[r_new];
    const uint32_t begin     = d_old_row_offsets[r_old];
    const uint32_t end       = d_old_row_offsets[r_old + 1];

    for (uint32_t j = begin; j < end; ++j) {
        const PackedEntry old_e   = d_old_entries[j];
        const uint32_t    new_col = d_old_to_new_col[packed_col(old_e)];
        // All surviving rows reference only alive columns; new_col != UINT32_MAX here.
        d_new_entries[write_pos++] = make_packed(new_col, packed_exp(old_e));
    }
}

// ============================================================================
// Kernel PS6: compact_metadata_packed_kernel
// ============================================================================

/// One thread per new row i.  Gathers per-row metadata (sqrt_Q, signs, val_2_exps)
/// from the original row d_row_map[i] into the compacted result arrays.
/// Must be launched after d_row_map is populated (after PS3).
__global__ __launch_bounds__(256)
void compact_metadata_packed_kernel(
    const uint32_t* __restrict__ d_row_map,
    const uint512*  __restrict__ d_old_sqrt_Q,
    const uint8_t*  __restrict__ d_old_signs,
    const int32_t*  __restrict__ d_old_val_2_exps,
    uint512* __restrict__ d_new_sqrt_Q,
    uint8_t* __restrict__ d_new_signs,
    int32_t* __restrict__ d_new_val_2_exps,
    uint32_t new_n_rows)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= new_n_rows) return;

    const uint32_t old_r  = d_row_map[i];
    d_new_sqrt_Q[i]       = d_old_sqrt_Q[old_r];
    d_new_signs[i]        = d_old_signs[old_r];
    d_new_val_2_exps[i]   = d_old_val_2_exps[old_r];
}

// ============================================================================
// Host driver
// ============================================================================

PackedSingletonResult gpuRemoveSingletons_packed(DevicePackedCSR& device_csr) {
    LOG_SET_MODULE("Matrix");

    const uint32_t n_rows     = device_csr.n_rows;
    const uint32_t n_cols     = device_csr.n_cols;
    const uint32_t nnz        = device_csr.nnz;
    const bool     jetson     = isJetsonDevice();

    // -------------------------------------------------------------------------
    // Early exit for degenerate inputs.
    // -------------------------------------------------------------------------
    if (n_rows == 0 || n_cols == 0) {
        LOG(LOG_INFO) << "GPU packed singleton removal: empty input (n_rows="
                      << n_rows << ", n_cols=" << n_cols << "), returning as-is.";
        PackedSingletonResult empty;
        // Transfer ownership of the (empty) device buffers to the result.
        empty.reduced = std::move(device_csr);
        for (uint32_t r = 0; r < n_rows; ++r) empty.row_map.push_back(r);
        for (uint32_t c = 0; c < n_cols; ++c) empty.col_map.push_back(c);
        return empty;
    }

    if (jetson) {
        LOG(LOG_INFO) << "GPU packed singleton removal: using cudaMallocManaged (Jetson).";
    }

    // -------------------------------------------------------------------------
    // Allocate scratch buffers.
    // -------------------------------------------------------------------------
    PackedSingletonBuffers buf;
    buf.alloc(n_rows, n_cols, nnz, jetson);

    // Initialize: all rows alive (1), all column weights zero.
    CUDA_CHECK(cudaMemset(buf.d_row_alive,  1u, n_rows * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(buf.d_col_weight, 0u, n_cols * sizeof(uint32_t)));

    constexpr uint32_t kBlock    = 256;
    const uint32_t     grid_rows = (n_rows + kBlock - 1) / kBlock;
    const uint32_t     grid_cols = (n_cols + kBlock - 1) / kBlock;

    // -------------------------------------------------------------------------
    // Phase 1: compute initial column weights.
    // -------------------------------------------------------------------------
    compute_col_weights_packed_kernel<<<grid_rows, kBlock>>>(
        device_csr.d_row_offsets, device_csr.d_entries, buf.d_row_alive,
        buf.d_col_weight, n_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------------------
    // Phase 2: fixpoint loop — iterate until no singletons remain.
    // -------------------------------------------------------------------------
    constexpr uint32_t kMaxIter  = 100;
    uint32_t           total_removed = 0;
    uint32_t           iteration     = 0;

    while (iteration < kMaxIter) {
        // Zero the per-iteration counter.  Prior cudaDeviceSynchronize ensures the
        // previous kernel has finished and the host write is visible to the device.
        *buf.d_removed_count = 0u;
        CUDA_CHECK(cudaDeviceSynchronize());

        singleton_pass_packed_kernel<<<grid_rows, kBlock>>>(
            device_csr.d_row_offsets, device_csr.d_entries,
            buf.d_row_alive, buf.d_col_weight,
            buf.d_removed_count, n_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        const uint32_t removed = *buf.d_removed_count;
        total_removed += removed;
        ++iteration;

        LOG(LOG_INFO) << "GPU packed singleton pass " << iteration
                      << ": removed " << removed << " rows, "
                      << (n_rows - total_removed) << " remaining.";

        if (removed == 0) break;
    }

    if (iteration >= kMaxIter) {
        LOG(LOG_WARNING) << "GPU packed singleton removal did not converge in "
                         << kMaxIter << " iterations.";
    }

    const uint32_t new_n_rows = n_rows - total_removed;

    // -------------------------------------------------------------------------
    // Phase 3: prefix sums for compaction.
    // -------------------------------------------------------------------------

    // Row prefix: exclusive scan of d_row_alive (uint8_t -> uint32_t).
    // d_row_prefix[r] = number of alive rows with index < r (= new row index for r).
    thrust::transform_exclusive_scan(
        thrust::device_ptr<uint8_t>(buf.d_row_alive),
        thrust::device_ptr<uint8_t>(buf.d_row_alive) + n_rows,
        thrust::device_ptr<uint32_t>(buf.d_row_prefix),
        PackedAliveToUint32{}, 0u, thrust::plus<uint32_t>{}
    );
    CUDA_CHECK(cudaGetLastError());

    // Col prefix: exclusive scan of (col_weight[c] > 0 ? 1 : 0).
    // d_col_prefix[c] = new column index for old column c (if alive).
    thrust::transform_exclusive_scan(
        thrust::device_ptr<uint32_t>(buf.d_col_weight),
        thrust::device_ptr<uint32_t>(buf.d_col_weight) + n_cols,
        thrust::device_ptr<uint32_t>(buf.d_col_prefix),
        PackedColAlive{}, 0u, thrust::plus<uint32_t>{}
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute new_n_cols from last prefix value + last alive indicator.
    // Tiny 2×uint32_t D→H copies; Thrust default-stream scans are complete by now.
    uint32_t new_n_cols = 0;
    {
        uint32_t h_last_col_prefix = 0;
        uint32_t h_last_col_weight = 0;
        CUDA_CHECK(cudaMemcpy(&h_last_col_prefix,
                              buf.d_col_prefix + (n_cols - 1),
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_last_col_weight,
                              buf.d_col_weight + (n_cols - 1),
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        new_n_cols = h_last_col_prefix + (h_last_col_weight > 0u ? 1u : 0u);
    }

    // -------------------------------------------------------------------------
    // Phase 4: build row and column index maps.
    // -------------------------------------------------------------------------
    build_row_map_packed_kernel<<<grid_rows, kBlock>>>(
        buf.d_row_alive, buf.d_row_prefix, buf.d_row_map, n_rows);
    CUDA_CHECK(cudaGetLastError());

    build_col_map_packed_kernel<<<grid_cols, kBlock>>>(
        buf.d_col_weight, buf.d_col_prefix, buf.d_old_to_new_col, n_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------------------
    // Phase 5: build d_new_row_offsets and compact packed entries.
    // -------------------------------------------------------------------------
    uint32_t new_nnz = 0;

    if (new_n_rows > 0) {
        // d_new_row_nnz[r_new] = NNZ of original row d_row_map[r_new].
        // (All entries in surviving rows reference alive columns — no filtering needed.)
        thrust::transform(
            thrust::device_ptr<uint32_t>(buf.d_row_map),
            thrust::device_ptr<uint32_t>(buf.d_row_map) + new_n_rows,
            thrust::device_ptr<uint32_t>(buf.d_new_row_nnz),
            PackedRowNNZFunctor{device_csr.d_row_offsets}
        );
        CUDA_CHECK(cudaGetLastError());

        // Exclusive scan -> d_new_row_offsets[0..new_n_rows-1].
        thrust::exclusive_scan(
            thrust::device_ptr<uint32_t>(buf.d_new_row_nnz),
            thrust::device_ptr<uint32_t>(buf.d_new_row_nnz) + new_n_rows,
            thrust::device_ptr<uint32_t>(buf.d_new_row_offsets),
            0u
        );
        CUDA_CHECK(cudaGetLastError());

        // Set d_new_row_offsets[new_n_rows] = total new NNZ (tiny D→H copies).
        uint32_t h_last_offset = 0;
        uint32_t h_last_nnz    = 0;
        CUDA_CHECK(cudaMemcpy(&h_last_offset,
                              buf.d_new_row_offsets + (new_n_rows - 1),
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_last_nnz,
                              buf.d_new_row_nnz + (new_n_rows - 1),
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        new_nnz = h_last_offset + h_last_nnz;
        CUDA_CHECK(cudaMemcpy(buf.d_new_row_offsets + new_n_rows,
                              &new_nnz,
                              sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Write compacted packed entries with remapped column indices.
        const uint32_t grid_new_rows = (new_n_rows + kBlock - 1) / kBlock;
        compact_packed_csr_kernel<<<grid_new_rows, kBlock>>>(
            device_csr.d_row_offsets, device_csr.d_entries,
            buf.d_row_map, buf.d_new_row_offsets, buf.d_old_to_new_col,
            buf.d_new_entries, new_n_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // -------------------------------------------------------------------------
    // Allocate result DevicePackedCSR and copy compacted data.
    // -------------------------------------------------------------------------
    PackedSingletonResult result;
    result.reduced.alloc(new_n_rows, new_n_cols, new_nnz, jetson);

    if (new_n_rows > 0) {
        // D→D copies for CSR structure (fast, ~10 MB).
        CUDA_CHECK(cudaMemcpy(result.reduced.d_row_offsets,
                              buf.d_new_row_offsets,
                              (new_n_rows + 1) * sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice));
        if (new_nnz > 0) {
            CUDA_CHECK(cudaMemcpy(result.reduced.d_entries,
                                  buf.d_new_entries,
                                  new_nnz * sizeof(PackedEntry),
                                  cudaMemcpyDeviceToDevice));
        }

        // PS6: compact per-row metadata (sqrt_Q, signs, val_2_exps).
        const uint32_t grid_new_rows = (new_n_rows + kBlock - 1) / kBlock;
        compact_metadata_packed_kernel<<<grid_new_rows, kBlock>>>(
            buf.d_row_map,
            device_csr.d_sqrt_Q, device_csr.d_signs, device_csr.d_val_2_exps,
            result.reduced.d_sqrt_Q, result.reduced.d_signs, result.reduced.d_val_2_exps,
            new_n_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // No surviving rows: ensure d_row_offsets[0] = 0.
        const uint32_t zero = 0u;
        CUDA_CHECK(cudaMemcpy(result.reduced.d_row_offsets, &zero,
                              sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // -------------------------------------------------------------------------
    // Download row_map and col_map to host.
    // -------------------------------------------------------------------------
    result.row_map.resize(new_n_rows);
    if (new_n_rows > 0) {
        CUDA_CHECK(cudaMemcpy(result.row_map.data(),
                              buf.d_row_map,
                              new_n_rows * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    }

    // Invert d_old_to_new_col on host to produce new_col -> old_col map.
    {
        std::vector<uint32_t> h_old_to_new(n_cols);
        CUDA_CHECK(cudaMemcpy(h_old_to_new.data(),
                              buf.d_old_to_new_col,
                              n_cols * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        result.col_map.resize(new_n_cols);
        for (uint32_t c = 0; c < n_cols; ++c) {
            if (h_old_to_new[c] != UINT32_MAX) {
                result.col_map[h_old_to_new[c]] = c;
            }
        }
    }

    result.iterations   = iteration;
    result.rows_removed = total_removed;
    result.cols_removed = n_cols - new_n_cols;

    // -------------------------------------------------------------------------
    // Consume input: free all 5 device arrays and null out pointers.
    // The DevicePackedCSR destructor checks for null before freeing — safe.
    // -------------------------------------------------------------------------
    if (device_csr.d_row_offsets) {
        CUDA_CHECK(cudaFree(device_csr.d_row_offsets));
        device_csr.d_row_offsets = nullptr;
    }
    if (device_csr.d_entries) {
        CUDA_CHECK(cudaFree(device_csr.d_entries));
        device_csr.d_entries = nullptr;
    }
    if (device_csr.d_sqrt_Q) {
        CUDA_CHECK(cudaFree(device_csr.d_sqrt_Q));
        device_csr.d_sqrt_Q = nullptr;
    }
    if (device_csr.d_signs) {
        CUDA_CHECK(cudaFree(device_csr.d_signs));
        device_csr.d_signs = nullptr;
    }
    if (device_csr.d_val_2_exps) {
        CUDA_CHECK(cudaFree(device_csr.d_val_2_exps));
        device_csr.d_val_2_exps = nullptr;
    }
    device_csr.n_rows = 0;
    device_csr.n_cols = 0;
    device_csr.nnz    = 0;

    LOG(LOG_INFO) << "GPU packed singleton removal complete: "
                  << n_rows  << " -> " << new_n_rows  << " rows ("
                  << total_removed         << " removed), "
                  << n_cols  << " -> " << new_n_cols  << " cols ("
                  << (n_cols - new_n_cols) << " removed), "
                  << iteration << " iterations.";

    return result;
}

} // namespace matrix
} // namespace mpqs
