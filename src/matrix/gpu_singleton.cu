// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_singleton.cu
//
// GPU singleton removal: iterative atomic fixpoint over a sparse GF(2) matrix.
// Five kernels + host driver.  Drop-in replacement for
// MergeFilterPipeline::removeSingletons().
//
// Algorithm:
//   Phase 1  — compute initial column weights (atomic counting).
//   Phase 2  — fixpoint: mark rows dead if any column has weight == 1;
//              decrement weights; repeat until no change.
//   Phase 3  — prefix sums for row/column compaction.
//   Phase 4  — build row/column index maps.
//   Phase 5  — compact CSR: write surviving rows with remapped column indices.
//   Phase 6  — download to host, assemble SingletonResult.

#include "gpu_singleton.cuh"
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
// Device buffer RAII struct
// ============================================================================

/// All GPU working buffers for singleton removal.
/// Constructed with alloc(); destroyed automatically via ~GPUSingletonBuffers().
/// On Jetson (use_managed == true), all buffers use cudaMallocManaged.
/// On RTX, regular cudaMalloc is used except d_removed_count (cudaMallocHost).
struct GPUSingletonBuffers {
    uint32_t* d_row_offsets    = nullptr;  ///< [n_rows + 1] — original CSR row offsets
    uint32_t* d_col_indices    = nullptr;  ///< [nnz]         — original CSR col indices
    uint32_t* d_col_weight     = nullptr;  ///< [n_cols]       — atomic column weights
    uint8_t*  d_row_alive      = nullptr;  ///< [n_rows]       — 1 = alive, 0 = removed
    uint32_t* d_removed_count  = nullptr;  ///< [1]  pinned host (RTX) or managed (Jetson)

    // Compaction buffers
    uint32_t* d_row_prefix     = nullptr;  ///< [n_rows + 1] exclusive scan of d_row_alive
    uint32_t* d_col_prefix     = nullptr;  ///< [n_cols + 1] exclusive scan of (col_weight > 0)
    uint32_t* d_old_to_new_col = nullptr;  ///< [n_cols]     old col index -> new col index
    uint32_t* d_row_map        = nullptr;  ///< [n_rows]     new row -> original row index
    uint32_t* d_new_row_offsets = nullptr; ///< [n_rows + 1] compacted row offsets
    uint32_t* d_new_col_indices = nullptr; ///< [nnz]        compacted col indices (upper bound)
    uint32_t* d_new_row_nnz    = nullptr;  ///< [n_rows]     per-new-row NNZ (temp)

    bool use_managed = false;

    void alloc(uint32_t n_rows, uint32_t n_cols, uint32_t nnz, bool jetson);
    ~GPUSingletonBuffers();
};

void GPUSingletonBuffers::alloc(uint32_t n_rows, uint32_t n_cols, uint32_t nnz, bool jetson) {
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

    device_malloc((void**)&d_row_offsets,    (safe_rows + 1) * sizeof(uint32_t));
    device_malloc((void**)&d_col_indices,     safe_nnz  * sizeof(uint32_t));
    device_malloc((void**)&d_col_weight,      safe_cols * sizeof(uint32_t));
    device_malloc((void**)&d_row_alive,       safe_rows * sizeof(uint8_t));
    device_malloc((void**)&d_row_prefix,     (safe_rows + 1) * sizeof(uint32_t));
    device_malloc((void**)&d_col_prefix,     (safe_cols + 1) * sizeof(uint32_t));
    device_malloc((void**)&d_old_to_new_col,  safe_cols * sizeof(uint32_t));
    device_malloc((void**)&d_row_map,         safe_rows * sizeof(uint32_t));
    device_malloc((void**)&d_new_row_offsets,(safe_rows + 1) * sizeof(uint32_t));
    device_malloc((void**)&d_new_col_indices, safe_nnz  * sizeof(uint32_t));
    device_malloc((void**)&d_new_row_nnz,     safe_rows * sizeof(uint32_t));

    // d_removed_count: pinned host memory for zero-copy on RTX; managed on Jetson.
    // cudaMallocHost gives a pinned pointer accessible from both host and device
    // via UVA on modern CUDA systems.
    if (use_managed) {
        CUDA_CHECK(cudaMallocManaged((void**)&d_removed_count, sizeof(uint32_t)));
    } else {
        CUDA_CHECK(cudaMallocHost((void**)&d_removed_count, sizeof(uint32_t)));
    }
}

GPUSingletonBuffers::~GPUSingletonBuffers() {
    // cudaFree handles both cudaMalloc and cudaMallocManaged pointers.
    auto free_device = [](void* ptr) { if (ptr) cudaFree(ptr); };

    free_device(d_row_offsets);
    free_device(d_col_indices);
    free_device(d_col_weight);
    free_device(d_row_alive);
    free_device(d_row_prefix);
    free_device(d_col_prefix);
    free_device(d_old_to_new_col);
    free_device(d_row_map);
    free_device(d_new_row_offsets);
    free_device(d_new_col_indices);
    free_device(d_new_row_nnz);

    if (d_removed_count) {
        if (use_managed) cudaFree(d_removed_count);
        else             cudaFreeHost(d_removed_count);
    }
}

// ============================================================================
// Thrust functors (no --expt-extended-lambda required)
// ============================================================================

/// Cast uint8_t alive flag to uint32_t for prefix scan accumulation.
struct AliveToUint32 {
    __host__ __device__ uint32_t operator()(uint8_t v) const {
        return static_cast<uint32_t>(v);
    }
};

/// Map column weight to 0/1 alive indicator for prefix scan.
struct ColAlive {
    __host__ __device__ uint32_t operator()(uint32_t w) const {
        return w > 0u ? 1u : 0u;
    }
};

/// Given original row index r_old, return its NNZ from the original row_offsets.
struct RowNNZFunctor {
    const uint32_t* d_row_offsets;
    __host__ __device__ uint32_t operator()(uint32_t r_old) const {
        return d_row_offsets[r_old + 1] - d_row_offsets[r_old];
    }
};

// ============================================================================
// Kernel 1: compute_col_weights_kernel
// ============================================================================

/// One thread per row. For each alive row, atomically increments the weight
/// of every column it contains.  Called once before the fixpoint loop.
__global__ __launch_bounds__(256)
void compute_col_weights_kernel(
    const uint32_t* __restrict__ d_row_offsets,
    const uint32_t* __restrict__ d_col_indices,
    const uint8_t*  __restrict__ d_row_alive,
    uint32_t* __restrict__ d_col_weight,
    uint32_t n_rows)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !d_row_alive[r]) return;

    const uint32_t begin = d_row_offsets[r];
    const uint32_t end   = d_row_offsets[r + 1];
    for (uint32_t j = begin; j < end; ++j) {
        atomicAdd(&d_col_weight[d_col_indices[j]], 1u);
    }
}

// ============================================================================
// Kernel 2: singleton_pass_kernel
// ============================================================================

/// One thread per row.  The critical fixpoint kernel — run iteratively:
///   1. Skip dead rows.
///   2. Scan this row's columns; set has_singleton if any weight == 1.
///   3. If has_singleton: mark row dead, atomicSub all its column weights,
///      atomicAdd d_removed_count.
///
/// Correctness note: concurrent atomicSub from dying rows may cause a column
/// weight to drop from 2 to 1 mid-iteration, causing an additional row to die
/// in the same iteration (vs. the next iteration on CPU).  The fixpoint is the
/// same because the singleton-free submatrix is unique regardless of removal order.
__global__ __launch_bounds__(256)
void singleton_pass_kernel(
    const uint32_t* __restrict__ d_row_offsets,
    const uint32_t* __restrict__ d_col_indices,
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
        if (d_col_weight[d_col_indices[j]] == 1u) {
            has_singleton = true;
            break;
        }
    }

    if (has_singleton) {
        d_row_alive[r] = 0u;
        for (uint32_t j = begin; j < end; ++j) {
            atomicSub(&d_col_weight[d_col_indices[j]], 1u);
        }
        atomicAdd(d_removed_count, 1u);
    }
}

// ============================================================================
// Kernel 3: build_row_map_kernel
// ============================================================================

/// One thread per row.  For each alive row r, writes r into
/// d_row_map[d_row_prefix[r]], building the new_row -> orig_row map.
/// d_row_prefix must be the exclusive scan of d_row_alive.
__global__ __launch_bounds__(256)
void build_row_map_kernel(
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
// Kernel 4: build_col_map_kernel
// ============================================================================

/// One thread per column.  Writes the old->new column remapping:
///   d_old_to_new_col[c] = d_col_prefix[c]  if d_col_weight[c] > 0 (alive)
///   d_old_to_new_col[c] = UINT32_MAX        otherwise
/// d_col_prefix must be the exclusive scan of (col_weight[c] > 0 ? 1 : 0).
__global__ __launch_bounds__(256)
void build_col_map_kernel(
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
// Kernel 5: compact_csr_kernel
// ============================================================================

/// One thread per new row r_new.  Copies the surviving column indices from the
/// original CSR into the compacted CSR, remapping via d_old_to_new_col.
/// d_new_row_offsets must already be filled before this kernel launches.
__global__ __launch_bounds__(256)
void compact_csr_kernel(
    const uint32_t* __restrict__ d_row_offsets,
    const uint32_t* __restrict__ d_col_indices,
    const uint32_t* __restrict__ d_row_map,
    const uint32_t* __restrict__ d_new_row_offsets,
    const uint32_t* __restrict__ d_old_to_new_col,
    uint32_t* __restrict__ d_new_col_indices,
    uint32_t new_n_rows)
{
    const uint32_t r_new = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_new >= new_n_rows) return;

    const uint32_t r_old     = d_row_map[r_new];
    uint32_t       write_pos = d_new_row_offsets[r_new];
    const uint32_t begin     = d_row_offsets[r_old];
    const uint32_t end       = d_row_offsets[r_old + 1];

    for (uint32_t j = begin; j < end; ++j) {
        // All alive rows reference only alive columns; UINT32_MAX never used here.
        d_new_col_indices[write_pos++] = d_old_to_new_col[d_col_indices[j]];
    }
}

// ============================================================================
// Host driver
// ============================================================================

SingletonResult gpuRemoveSingletons(const HostMatrixCSR& input) {
    LOG_SET_MODULE("Matrix");

    const uint32_t n_rows = input.n_rows;
    const uint32_t n_cols = input.n_cols;
    const uint32_t nnz    = static_cast<uint32_t>(input.col_indices.size());

    // Early exit for degenerate inputs.
    if (n_rows == 0 || n_cols == 0) {
        LOG(LOG_INFO) << "GPU singleton removal: empty input, returning as-is.";
        SingletonResult empty;
        empty.reduced      = input;
        empty.iterations   = 0;
        empty.rows_removed = 0;
        empty.cols_removed = 0;
        for (uint32_t r = 0; r < n_rows; ++r) empty.row_map.push_back(r);
        for (uint32_t c = 0; c < n_cols; ++c) empty.col_map.push_back(c);
        return empty;
    }

    // Detect Jetson: SM 8.7 with unified addressing, or VRAM < 12 GB.
    // Matches the detection pattern in gpu_char_cols.cu.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const bool jetson = (prop.major == 8 && prop.minor == 7) ||
                        (prop.unifiedAddressing &&
                         prop.totalGlobalMem < 12ULL * 1024 * 1024 * 1024);

    if (jetson) {
        LOG(LOG_INFO) << "GPU singleton removal: using cudaMallocManaged (Jetson/integrated GPU: SM 8.7 or VRAM < 12 GB).";
    }

    // Allocate all device buffers.
    GPUSingletonBuffers buf;
    buf.alloc(n_rows, n_cols, nnz, jetson);

    // Upload the expanded CSR to device.
    CUDA_CHECK(cudaMemcpy(buf.d_row_offsets, input.row_offsets.data(),
                          (n_rows + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (nnz > 0) {
        CUDA_CHECK(cudaMemcpy(buf.d_col_indices, input.col_indices.data(),
                              nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Initialize: all rows alive (1), all column weights zero.
    CUDA_CHECK(cudaMemset(buf.d_row_alive,  1u, n_rows * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(buf.d_col_weight, 0u, n_cols * sizeof(uint32_t)));

    // -------------------------------------------------------------------------
    // Phase 1: compute initial column weights
    // -------------------------------------------------------------------------
    constexpr uint32_t kBlock    = 256;
    const uint32_t     grid_rows = (n_rows + kBlock - 1) / kBlock;
    const uint32_t     grid_cols = (n_cols + kBlock - 1) / kBlock;

    compute_col_weights_kernel<<<grid_rows, kBlock>>>(
        buf.d_row_offsets, buf.d_col_indices, buf.d_row_alive,
        buf.d_col_weight, n_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------------------
    // Phase 2: fixpoint loop — iterate until no singletons remain
    // -------------------------------------------------------------------------
    constexpr uint32_t kMaxIter = 100;
    uint32_t total_removed = 0;
    uint32_t iteration     = 0;

    while (iteration < kMaxIter) {
        // Zero the per-iteration counter.  The prior cudaDeviceSynchronize
        // guarantees the previous kernel has finished and the host write
        // is ordered before the next kernel launch (CUDA API memory ordering).
        *buf.d_removed_count = 0u;
        CUDA_CHECK(cudaDeviceSynchronize());  // flush host write; wait for prior kernel

        singleton_pass_kernel<<<grid_rows, kBlock>>>(
            buf.d_row_offsets, buf.d_col_indices,
            buf.d_row_alive, buf.d_col_weight,
            buf.d_removed_count, n_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());  // wait; make d_removed_count visible to host

        const uint32_t removed = *buf.d_removed_count;
        total_removed += removed;
        ++iteration;

        LOG(LOG_INFO) << "GPU singleton pass " << iteration
                      << ": removed " << removed << " rows, "
                      << (n_rows - total_removed) << " remaining.";

        if (removed == 0) break;
    }

    if (iteration >= kMaxIter) {
        LOG(LOG_WARNING) << "GPU singleton removal did not converge in "
                         << kMaxIter << " iterations.";
    }

    const uint32_t new_n_rows = n_rows - total_removed;

    // -------------------------------------------------------------------------
    // Phase 3: prefix sums for compaction
    // -------------------------------------------------------------------------

    // Row prefix: exclusive scan of d_row_alive (uint8_t -> uint32_t).
    // d_row_prefix[r] = number of alive rows with index < r.
    thrust::transform_exclusive_scan(
        thrust::device_ptr<uint8_t>(buf.d_row_alive),
        thrust::device_ptr<uint8_t>(buf.d_row_alive) + n_rows,
        thrust::device_ptr<uint32_t>(buf.d_row_prefix),
        AliveToUint32{}, 0u, thrust::plus<uint32_t>{}
    );
    CUDA_CHECK(cudaGetLastError());

    // Col prefix: exclusive scan of (col_weight[c] > 0 ? 1 : 0).
    // d_col_prefix[c] = new column index for old column c (if alive).
    thrust::transform_exclusive_scan(
        thrust::device_ptr<uint32_t>(buf.d_col_weight),
        thrust::device_ptr<uint32_t>(buf.d_col_weight) + n_cols,
        thrust::device_ptr<uint32_t>(buf.d_col_prefix),
        ColAlive{}, 0u, thrust::plus<uint32_t>{}
    );
    CUDA_CHECK(cudaGetLastError());

    // Determine new_n_cols from last prefix value + last alive indicator.
    // Both copies are tiny (2 x uint32_t); the Thrust scans are complete by now
    // because cudaMemcpy on the default stream follows Thrust's default-stream kernels.
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
    // Phase 4: build row and column maps
    // -------------------------------------------------------------------------
    build_row_map_kernel<<<grid_rows, kBlock>>>(
        buf.d_row_alive, buf.d_row_prefix, buf.d_row_map, n_rows);
    CUDA_CHECK(cudaGetLastError());

    build_col_map_kernel<<<grid_cols, kBlock>>>(
        buf.d_col_weight, buf.d_col_prefix, buf.d_old_to_new_col, n_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------------------
    // Phase 5: build d_new_row_offsets and compact col indices
    // -------------------------------------------------------------------------
    uint32_t new_nnz = 0;

    if (new_n_rows > 0) {
        // d_new_row_nnz[r_new] = NNZ of the original row d_row_map[r_new].
        thrust::transform(
            thrust::device_ptr<uint32_t>(buf.d_row_map),
            thrust::device_ptr<uint32_t>(buf.d_row_map) + new_n_rows,
            thrust::device_ptr<uint32_t>(buf.d_new_row_nnz),
            RowNNZFunctor{buf.d_row_offsets}
        );
        CUDA_CHECK(cudaGetLastError());

        // d_new_row_offsets[0..new_n_rows-1] = exclusive scan of d_new_row_nnz.
        thrust::exclusive_scan(
            thrust::device_ptr<uint32_t>(buf.d_new_row_nnz),
            thrust::device_ptr<uint32_t>(buf.d_new_row_nnz) + new_n_rows,
            thrust::device_ptr<uint32_t>(buf.d_new_row_offsets),
            0u
        );
        CUDA_CHECK(cudaGetLastError());

        // Set d_new_row_offsets[new_n_rows] = total new NNZ.
        // Read the last scan value and last NNZ count from device to host.
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

        // Write compacted column indices.
        const uint32_t grid_new_rows = (new_n_rows + kBlock - 1) / kBlock;
        compact_csr_kernel<<<grid_new_rows, kBlock>>>(
            buf.d_row_offsets, buf.d_col_indices,
            buf.d_row_map, buf.d_new_row_offsets, buf.d_old_to_new_col,
            buf.d_new_col_indices, new_n_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // -------------------------------------------------------------------------
    // Phase 6: download to host and assemble SingletonResult
    // -------------------------------------------------------------------------
    SingletonResult result;
    result.reduced.n_rows = new_n_rows;
    result.reduced.n_cols = new_n_cols;

    // row_offsets: new_n_rows + 1 entries.
    result.reduced.row_offsets.resize(new_n_rows + 1);
    if (new_n_rows > 0) {
        CUDA_CHECK(cudaMemcpy(result.reduced.row_offsets.data(),
                              buf.d_new_row_offsets,
                              (new_n_rows + 1) * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    } else {
        result.reduced.row_offsets[0] = 0u;
    }

    // col_indices: new_nnz entries.
    result.reduced.col_indices.resize(new_nnz);
    if (new_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(result.reduced.col_indices.data(),
                              buf.d_new_col_indices,
                              new_nnz * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    }

    // row_map: new_row -> original_row (for merge tree leaf assignment in M3/M4).
    result.row_map.resize(new_n_rows);
    if (new_n_rows > 0) {
        CUDA_CHECK(cudaMemcpy(result.row_map.data(),
                              buf.d_row_map,
                              new_n_rows * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    }

    // col_map: new_col -> old_col (inverse of d_old_to_new_col).
    // Download d_old_to_new_col and invert on host (O(n_cols), done once).
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

    // Validate output CSR before returning.
    if (!ValidateHostMatrixCSR(result.reduced)) {
        LOG(LOG_ERROR) << "GPU singleton removal: ValidateHostMatrixCSR failed!";
    }

    LOG(LOG_INFO) << "GPU singleton removal complete: "
                  << n_rows  << " -> " << new_n_rows  << " rows ("
                  << total_removed         << " removed), "
                  << n_cols  << " -> " << new_n_cols  << " cols ("
                  << (n_cols - new_n_cols) << " removed), "
                  << iteration << " iterations.";

    return result;
}

} // namespace matrix
} // namespace mpqs
