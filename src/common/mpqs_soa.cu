// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "mpqs_soa.h"
#include "cuda_check.h"
#include <cstdio>
#include <cstring>   // std::memcpy
#include <algorithm> // std::swap
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace mpqs {
namespace structures {

// --- HostRelationBatch Implementation ---

void HostRelationBatch::resize(size_t n_rels, size_t n_factors) {
    // NOTE: Counters are set to the requested capacity, not clamped to old count.
    // This is intentional — callers (e.g. moveToHost) always pass the exact fill
    // count and then overwrite the data, so min(old, new) semantics are unnecessary.
    num_relations = n_rels;
    num_factors = n_factors;
    
    if (sqrt_Q.size() < n_rels) sqrt_Q.resize(n_rels);
    if (signs.size() < n_rels) signs.resize(n_rels);
    if (val_2_exps.size() < n_rels) val_2_exps.resize(n_rels);
    if (large_primes.size() < n_rels) large_primes.resize(n_rels);
    
    if (factor_offsets.size() < n_rels + 1) factor_offsets.resize(n_rels + 1);
    if (factor_indices.size() < n_factors) factor_indices.resize(n_factors);
    if (factor_counts.size() < n_factors) factor_counts.resize(n_factors);  
}

void HostRelationBatch::clear() {
    num_relations = 0; num_factors = 0;
    sqrt_Q.clear(); signs.clear(); val_2_exps.clear(); large_primes.clear();
    factor_offsets.clear(); factor_indices.clear(); factor_counts.clear();
}

// --- RelationBatch Implementation ---

RelationBatch::~RelationBatch() {
    // We shouldn't throw in destructor, but we should free if possible
    // Note: If device is reset, these frees might fail, which is acceptable on shutdown.
    if (d_sqrt_Q) cudaFree(d_sqrt_Q);
    if (d_signs)  cudaFree(d_signs);
    if (d_val_2_exps) cudaFree(d_val_2_exps);
    if (d_large_primes) cudaFree(d_large_primes);
    
    if (d_factor_offsets) cudaFree(d_factor_offsets);
    if (d_factor_indices) cudaFree(d_factor_indices);
    if (d_factor_counts)  cudaFree(d_factor_counts);

    if (d_global_count_ptr) cudaFree(d_global_count_ptr);
    if (d_global_factor_ptr) cudaFree(d_global_factor_ptr);
    
    if (h_pinned_counters) cudaFreeHost(h_pinned_counters);
}  

void RelationBatch::initiate(int device_id)
{
    this->device_id = device_id;
    CUDA_CHECK(cudaSetDevice(device_id));

    // Detect unified memory hardware (Jetson Orin SM 8.7, or low-VRAM unified addressing)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    use_managed_memory_ = (prop.major == 8 && prop.minor == 7) ||
                          (prop.unifiedAddressing &&
                           prop.totalGlobalMem < 12ULL * 1024 * 1024 * 1024);
    // Environment override: MPQS_NO_MANAGED=1 forces cudaMalloc even on Jetson
    const char* no_managed = std::getenv("MPQS_NO_MANAGED");
    if (no_managed && std::string(no_managed) == "1") {
        use_managed_memory_ = false;
    }
    // cudaMemAdvise hints require concurrentManagedAccess (discrete GPUs only).
    // Integrated GPUs (Jetson) share physical memory — hints are unnecessary and unsupported.
    use_mem_advise_ = prop.concurrentManagedAccess != 0;

    if (use_managed_memory_) {
        LOG(LOG_INFO) << "[RelationBatch] Using cudaMallocManaged (unified memory detected)";
    }

    // Allocate pinned memory for 2 uint64_t values
    // [0] = Relations Count, [1] = Factors Count
    CUDA_CHECK(cudaMallocHost(&h_pinned_counters, 2 * sizeof(uint64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_pinned_counters, h_pinned_counters, 0));

    // Allocate Device Atomic Counters
    CUDA_CHECK(cudaMalloc(&d_global_count_ptr, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_global_factor_ptr, sizeof(uint64_t)));

    // reset counters
    this->clear();
}
  
// -----------------------------------------------------------------------------
// RelationBatch: Clear
// -----------------------------------------------------------------------------
void RelationBatch::clear(cudaStream_t stream) {
    // Initialize counters to 0
  
    if (d_global_count_ptr) CUDA_CHECK(cudaMemsetAsync(d_global_count_ptr, 0, sizeof(uint64_t), stream))
;
    if (d_global_factor_ptr) CUDA_CHECK(cudaMemsetAsync(d_global_factor_ptr, 0, sizeof(uint64_t), stream));
    // Reset offset[0] to 0 for consistency
    if (d_factor_offsets) CUDA_CHECK(cudaMemsetAsync(d_factor_offsets, 0, sizeof(uint64_t), stream));

    if(h_pinned_counters) {
        h_pinned_counters[0] = 0;
	h_pinned_counters[1] = 0;
    }
}

// -----------------------------------------------------------------------------
// RelationBatch: Non-Destructive Resize
// -----------------------------------------------------------------------------
void RelationBatch::resize(size_t num_rels_needed, size_t num_factors_needed) {
    CUDA_CHECK(cudaSetDevice(device_id));

    // 1. Policy: Apply a growth factor to avoid frequent reallocations
    //    but respect the user's explicit request if it's a shrink or specific sizing.
    //    Here we strictly ensure we meet the "needed" size.
    size_t new_cap_rels = std::max(cap_rels, num_rels_needed);
    size_t new_cap_factors = std::max(cap_factors, num_factors_needed);
    
    // If no change needed, return
    if (new_cap_rels == cap_rels && new_cap_factors == cap_factors) return;

    LOG(LOG_DEBUG_1) << "[RelationBatch] Resizing: Rels " << cap_rels << "->" << new_cap_rels 
                     << ", Factors " << cap_factors << "->" << new_cap_factors;

    // 2. Allocate New Buffers
    mpqs::uint512* new_sqrt_Q = nullptr;
    uint8_t* new_signs = nullptr;
    int32_t* new_val_2_exps = nullptr;
    unsigned __int128* new_large_primes = nullptr;
    uint64_t* new_factor_offsets = nullptr;
    uint32_t* new_factor_indices = nullptr;
    uint8_t* new_factor_counts = nullptr;

    // Helper for allocation: managed memory on Jetson, device memory on discrete GPU
    auto safe_alloc = [&](void** ptr, size_t size) {
        if (size > 0) {
            if (use_managed_memory_) {
                CUDA_CHECK(cudaMallocManaged(ptr, size));
                if (use_mem_advise_) {
#if CUDART_VERSION >= 13000
                    cudaMemLocation loc = {};
                    loc.type = cudaMemLocationTypeDevice;
                    loc.id = device_id;
                    CUDA_CHECK(cudaMemAdvise(*ptr, size, cudaMemAdviseSetPreferredLocation, loc));
                    CUDA_CHECK(cudaMemAdvise(*ptr, size, cudaMemAdviseSetAccessedBy, loc));
#else
                    CUDA_CHECK(cudaMemAdvise(*ptr, size, cudaMemAdviseSetPreferredLocation, device_id));
                    CUDA_CHECK(cudaMemAdvise(*ptr, size, cudaMemAdviseSetAccessedBy, device_id));
#endif
                }
            } else {
                CUDA_CHECK(cudaMalloc(ptr, size));
            }
        }
    };

    safe_alloc((void**)&new_sqrt_Q,       new_cap_rels * sizeof(mpqs::uint512));
    safe_alloc((void**)&new_signs,        new_cap_rels * sizeof(uint8_t));
    safe_alloc((void**)&new_val_2_exps,   new_cap_rels * sizeof(int32_t));
    safe_alloc((void**)&new_large_primes, new_cap_rels * sizeof(unsigned __int128));
    safe_alloc((void**)&new_factor_offsets, (new_cap_rels + 1) * sizeof(uint64_t));

    safe_alloc((void**)&new_factor_indices, new_cap_factors * sizeof(uint32_t));
    safe_alloc((void**)&new_factor_counts,  new_cap_factors * sizeof(uint8_t));

    // 3. Preserve Data (Copy Old -> New)
    // We rely on the atomic counters to know how much valid data exists.
    // However, since resize might be called from host thread async, 
    // we do a sync read or assume the caller has synchronized. 
    // For safety, we copy everything we *might* have.
    
    // Note: We do NOT re-allocate the counters (d_global_count_ptr). 
    // They persist and keep the current count.
    
    // We need to know current fill to copy relevant data.
    // Since resize is a heavy op, a sync is acceptable to get exact counts.
    uint64_t current_rels = 0;
    uint64_t current_factors = 0;
    
    if (d_global_count_ptr) {
        CUDA_CHECK(cudaMemcpy(&current_rels, d_global_count_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }
    if (d_global_factor_ptr) {
        CUDA_CHECK(cudaMemcpy(&current_factors, d_global_factor_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    // Clamp copy size to minimum of (old_capacity, new_capacity, current_count)
    // If shrinking, we truncate.
    size_t copy_rels = std::min({cap_rels, new_cap_rels, (size_t)current_rels});
    size_t copy_factors = std::min({cap_factors, new_cap_factors, (size_t)current_factors});

    auto safe_copy = [&](void* dst, void* src, size_t size) {
        if (dst && src && size > 0) CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    };

    if (copy_rels > 0) {
        safe_copy(new_sqrt_Q, d_sqrt_Q, copy_rels * sizeof(mpqs::uint512));
        safe_copy(new_signs, d_signs, copy_rels * sizeof(uint8_t));
        safe_copy(new_val_2_exps, d_val_2_exps, copy_rels * sizeof(int32_t));
        safe_copy(new_large_primes, d_large_primes, copy_rels * sizeof(unsigned __int128));
        // Offsets: copy N+1.
        safe_copy(new_factor_offsets, d_factor_offsets, (copy_rels + 1) * sizeof(uint64_t));
    } else {
        // Initialize offsets[0] = 0 if this is a fresh start and we have capacity
        if (new_factor_offsets) CUDA_CHECK(cudaMemset(new_factor_offsets, 0, sizeof(uint64_t)));
    }

    if (copy_factors > 0) {
        safe_copy(new_factor_indices, d_factor_indices, copy_factors * sizeof(uint32_t));
        safe_copy(new_factor_counts, d_factor_counts, copy_factors * sizeof(uint8_t));
    }

    // 4. Free Old Buffers
    if (d_sqrt_Q) cudaFree(d_sqrt_Q);
    if (d_signs) cudaFree(d_signs);
    if (d_val_2_exps) cudaFree(d_val_2_exps);
    if (d_large_primes) cudaFree(d_large_primes);
    if (d_factor_offsets) cudaFree(d_factor_offsets);
    if (d_factor_indices) cudaFree(d_factor_indices);
    if (d_factor_counts) cudaFree(d_factor_counts);

    // 5. Update State
    d_sqrt_Q = new_sqrt_Q;
    d_signs = new_signs;
    d_val_2_exps = new_val_2_exps;
    d_large_primes = new_large_primes;
    d_factor_offsets = new_factor_offsets;
    d_factor_indices = new_factor_indices;
    d_factor_counts = new_factor_counts;

    cap_rels = new_cap_rels;
    cap_factors = new_cap_factors;

    // Ensure counter pointers exist (create if first time)
    if (!d_global_count_ptr) {
        CUDA_CHECK(cudaMalloc(&d_global_count_ptr, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_global_count_ptr, 0, sizeof(uint64_t)));
    }
    if (!d_global_factor_ptr) {
        CUDA_CHECK(cudaMalloc(&d_global_factor_ptr, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_global_factor_ptr, 0, sizeof(uint64_t)));
    }
}

// -----------------------------------------------------------------------------
// Append Logic Kernel (Device to Device)
// -----------------------------------------------------------------------------

namespace kernels {

// Simple rebase:
__global__ void rebase_kernel(
    uint64_t* dst,
    const uint64_t* src,
    uint64_t base,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
      // src[idx+1] because src[0] is 0, we want the ends of the relations
      dst[idx] = src[idx + 1] + base;
    }
};
 
} // namespace kernels

// -----------------------------------------------------------------------------
// RelationBatch: Safe Append (Discard on Overflow)
// -----------------------------------------------------------------------------
void RelationBatch::append(
    RelationBatch& other, 
    uint64_t other_count_rels, 
    cudaStream_t stream
) {
    if (other_count_rels == 0) return;

    // 1. Get Current Counts (Host)
    // We use pinned memory to fetch atomic counters asynchronously, 
    // but we must wait for them to decide the cut-off.
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_counters, d_global_count_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_counters+1, d_global_factor_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    uint64_t start_rels = h_pinned_counters[0];
    uint64_t start_factors = h_pinned_counters[1];
    
    // 2. Calculate Available Space
    uint64_t avail_rels = (cap_rels > start_rels) ? (cap_rels - start_rels) : 0;
    uint64_t avail_factors = (cap_factors > start_factors) ? (cap_factors - start_factors) : 0;

    // 3. Determine Candidate Count (Relation Limit)
    uint64_t candidate_rels = std::min(other_count_rels, avail_rels);
    
    if (candidate_rels == 0) {
        LOG(LOG_WARNING) << "[RelationBatch] Append DISCARDED ALL: Relation buffer full.";
        return;
    }

    // 4. Determine Actual Count (Factor Limit)
    // We need to know how many factors 'candidate_rels' would consume.
    // 'other.d_factor_offsets' is inclusive of the previous relation, 
    // so offset at index [candidate_rels] is exactly the factor count for first 'candidate_rels'.
    
    uint64_t factors_needed = 0;
    CUDA_CHECK(cudaMemcpyAsync(&factors_needed, other.d_factor_offsets + candidate_rels, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t actual_rels_to_add = candidate_rels;
    uint64_t actual_factors_to_add = factors_needed;

    // If factors don't fit, we must reduce relations further
    if (factors_needed > avail_factors) {
        // Use binary search (upper_bound) on 'other.d_factor_offsets' to find 
        // the largest index 'k' such that offsets[k] <= avail_factors.
        
        // Thrust policies on stream
        auto policy = thrust::cuda::par.on(stream);
        
        // offsets range to search: [0, candidate_rels]
        // We look for the first element that is GREATER than avail_factors.
        // The element *before* that is the last one that fits.
        
        // Note: d_factor_offsets is uint64_t*
        uint64_t* offsets_begin = other.d_factor_offsets;
        uint64_t* offsets_end = offsets_begin + candidate_rels + 1; // +1 because offsets size is N+1
        
        auto iter = thrust::upper_bound(policy, offsets_begin, offsets_end, avail_factors);
        
        // Index of the first element > avail_factors
        uint64_t idx = iter - offsets_begin; 
        
        // So actual_rels is idx - 1. 
        // (Because offsets[k] is total factors for k relations. If offsets[k] <= avail, k relations fit.)
        if (idx > 0) {
            actual_rels_to_add = idx - 1;
        } else {
            actual_rels_to_add = 0;
        }

        // Re-read the exact factor count for the new reduced relation count
        if (actual_rels_to_add > 0) {
            CUDA_CHECK(cudaMemcpyAsync(&actual_factors_to_add, other.d_factor_offsets + actual_rels_to_add, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            actual_factors_to_add = 0;
        }
    }

    // 5. Warnings and Execution
    if (actual_rels_to_add < other_count_rels) {
        LOG(LOG_WARNING) << "[RelationBatch] Append DISCARDING data. Buffer Full. Added " 
                         << actual_rels_to_add << "/" << other_count_rels << " relations.";
    }

    if (actual_rels_to_add == 0) return;

    // 6. Deep Copy Data
    auto copy = [&](void* dst, const void* src, size_t size) {
        if (size > 0) CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
    };

    copy(d_sqrt_Q + start_rels, other.d_sqrt_Q, actual_rels_to_add * sizeof(mpqs::uint512));
    copy(d_signs + start_rels,  other.d_signs,  actual_rels_to_add * sizeof(uint8_t));
    copy(d_val_2_exps + start_rels, other.d_val_2_exps, actual_rels_to_add * sizeof(int32_t));
    copy(d_large_primes + start_rels, other.d_large_primes, actual_rels_to_add * sizeof(unsigned __int128));
    
    // Copy Factors
    copy(d_factor_indices + start_factors, other.d_factor_indices, actual_factors_to_add * sizeof(uint32_t));
    copy(d_factor_counts + start_factors, other.d_factor_counts, actual_factors_to_add * sizeof(uint8_t));

    // 7. Offset Rebase Kernel
    // We copy offsets 1..N (from other) to start_rels+1 .. start_rels+N (in this)
    // And add 'start_factors' to each.
    // offsets[0] is always 0 relative to the batch start, but in the global array it must match.
    // Wait, the standard CSR offset array:
    // Global Offset[k] = Offset[k-1] + count.
    // We are appending.
    // dst[start_rels] is already valid (end of previous).
    // dst[start_rels + i] = dst[start_rels] + src[i] (where src[i] is cumulative from 0).
    
    // We launch a kernel to copy and add 'start_factors' to other.offsets
    // src range: [1, actual_rels_to_add] (inclusive)
    // dst starts at: start_rels + 1
    
    uint32_t num_offsets_to_copy = (uint32_t)actual_rels_to_add;
    uint32_t threads = 256;
    uint32_t blocks = (num_offsets_to_copy + threads - 1) / threads;

    kernels::rebase_kernel<<<blocks, threads, 0, stream>>>(
        d_factor_offsets + start_rels + 1, 
        other.d_factor_offsets, 
        start_factors, 
        num_offsets_to_copy
    );

    // 8. Update Atomic Counters
    h_pinned_counters[0] = start_rels + actual_rels_to_add;
    h_pinned_counters[1] = start_factors + actual_factors_to_add;
    CUDA_CHECK(cudaMemcpyAsync(d_global_count_ptr, &h_pinned_counters[0], sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_global_factor_ptr, &h_pinned_counters[1], sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
}

void RelationBatch::reset_counters(cudaStream_t stream) {
    // We do NOT memset the data arrays here. That is the massive perf gain.
    CUDA_CHECK(cudaMemsetAsync(d_global_count_ptr, 0, sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_global_factor_ptr, 0, sizeof(uint64_t), stream));
}

RelationBatchView RelationBatch::get_view() {
    RelationBatchView v;
    v.sqrt_Q         = d_sqrt_Q;
    v.signs          = d_signs;
    v.val_2_exps     = d_val_2_exps;
    v.large_primes   = d_large_primes;
    v.factor_offsets = d_factor_offsets;
    v.factor_indices = d_factor_indices;
    v.factor_counts  = d_factor_counts;
    
    v.global_count      = d_global_count_ptr;
    v.global_factor_idx = d_global_factor_ptr;

    v.max_relations = (uint32_t)cap_rels;
    v.max_factors   = (uint64_t)cap_factors;

    v.target_cap = target_cap_;
    return v;
}

// -----------------------------------------------------------------------------
// Sync Counters (16 bytes D→H for managed-memory moveToHost path)
// -----------------------------------------------------------------------------

void RelationBatch::syncCounters(cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_counters, d_global_count_ptr,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_counters + 1, d_global_factor_ptr,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
}

// -----------------------------------------------------------------------------
// Move to Host
// -----------------------------------------------------------------------------

void RelationBatch::moveToHost(HostRelationBatch& dest, cudaStream_t stream) {
    // Get counts
    syncCounters(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t n_rels = h_pinned_counters[0];
    uint64_t n_factors = h_pinned_counters[1];

    dest.resize(n_rels, n_factors);
    if (n_rels == 0) return;

    if (use_managed_memory_) {
        // Must sync ALL device streams before CPU reads managed memory.
        // On Jetson (concurrentManagedAccess=0), LP writes may be pending on
        // lp_stream while pp_stream is already synced — stale reads otherwise.
        CUDA_CHECK(cudaDeviceSynchronize());
        std::memcpy(dest.sqrt_Q.data(),       d_sqrt_Q,       n_rels * sizeof(mpqs::uint512));
        std::memcpy(dest.signs.data(),        d_signs,        n_rels * sizeof(uint8_t));
        std::memcpy(dest.val_2_exps.data(),   d_val_2_exps,   n_rels * sizeof(int32_t));
        std::memcpy(dest.large_primes.data(), d_large_primes, n_rels * sizeof(unsigned __int128));
        std::memcpy(dest.factor_offsets.data(), d_factor_offsets, (n_rels + 1) * sizeof(uint64_t));
        if (n_factors > 0) {
            std::memcpy(dest.factor_indices.data(), d_factor_indices, n_factors * sizeof(uint32_t));
            std::memcpy(dest.factor_counts.data(),  d_factor_counts,  n_factors * sizeof(uint8_t));
        }
    } else {
        // Discrete GPU path: original cudaMemcpy D→H (UNCHANGED)
        auto copyToHost = [&](void* dst, const void* src, size_t size) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
        };

        copyToHost(dest.sqrt_Q.data(),       d_sqrt_Q,       n_rels * sizeof(mpqs::uint512));
        copyToHost(dest.signs.data(),        d_signs,        n_rels * sizeof(uint8_t));
        copyToHost(dest.val_2_exps.data(),   d_val_2_exps,   n_rels * sizeof(int32_t));
        copyToHost(dest.large_primes.data(), d_large_primes, n_rels * sizeof(unsigned __int128));
        copyToHost(dest.factor_offsets.data(), d_factor_offsets, (n_rels + 1) * sizeof(uint64_t));

        if (n_factors > 0) {
            copyToHost(dest.factor_indices.data(), d_factor_indices, n_factors * sizeof(uint32_t));
            copyToHost(dest.factor_counts.data(),  d_factor_counts,  n_factors * sizeof(uint8_t));
        }
    }

    dest.num_relations = n_rels;
    dest.num_factors = n_factors;
}

// -----------------------------------------------------------------------------
// Move Range to Host (sub-range D2H for cluster extraction)
// -----------------------------------------------------------------------------

void RelationBatch::moveRangeToHost(HostRelationBatch& dest, uint64_t offset, uint64_t count,
                                     cudaStream_t stream) {
    if (count == 0) { dest.num_relations = 0; dest.num_factors = 0; return; }

    // --- Phase 1: Flat SoA arrays + CSR offsets (async) ---
    dest.resize(count, count * 32);  // Conservative factor estimate; Phase 2 may not use all

    auto d2h = [&](void* dst, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
    };

    d2h(dest.sqrt_Q.data(),       d_sqrt_Q + offset,       count * sizeof(mpqs::uint512));
    d2h(dest.signs.data(),        d_signs + offset,        count * sizeof(uint8_t));
    d2h(dest.val_2_exps.data(),   d_val_2_exps + offset,   count * sizeof(int32_t));
    d2h(dest.large_primes.data(), d_large_primes + offset, count * sizeof(unsigned __int128));
    d2h(dest.factor_offsets.data(), d_factor_offsets + offset, (count + 1) * sizeof(uint64_t));

    // --- Sync: read CSR offsets on CPU ---
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t factor_start = dest.factor_offsets[0];
    uint64_t factor_end   = dest.factor_offsets[count];
    uint64_t n_factors    = factor_end - factor_start;

    // --- Phase 2: CSR factor data (async) ---
    if (n_factors > 0) {
        if (dest.factor_indices.size() < n_factors) dest.factor_indices.resize(n_factors);
        if (dest.factor_counts.size() < n_factors) dest.factor_counts.resize(n_factors);

        d2h(dest.factor_indices.data(), d_factor_indices + factor_start, n_factors * sizeof(uint32_t));
        d2h(dest.factor_counts.data(),  d_factor_counts + factor_start,  n_factors * sizeof(uint8_t));
    }

    // --- Re-base factor_offsets to start at 0 ---
    for (uint64_t i = 0; i <= count; ++i) {
        dest.factor_offsets[i] -= factor_start;
    }

    dest.num_relations = count;
    dest.num_factors   = n_factors;
}

uint64_t RelationBatch::getCount(cudaStream_t stream) const {
    if (!d_global_count_ptr) return 0;
    // We can use the pinned buffer to avoid a full hard sync if we want,
    // but the getter implies "current state".
    // For safety, let's read the device atomic.
    uint64_t val = 0;
    CUDA_CHECK(cudaMemcpyAsync(&val, d_global_count_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return val;
}

uint64_t RelationBatch::getFactorCount(cudaStream_t stream) const {
    if (!d_global_factor_ptr) return 0;
    uint64_t val = 0;
    CUDA_CHECK(cudaMemcpyAsync(&val, d_global_factor_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return val;
}

void RelationBatch::requestStats(cudaStream_t stream) {
    if (!d_global_count_ptr || !h_pinned_counters) return;
    // Async copy to pinned memory
    CUDA_CHECK(cudaMemcpyAsync(&h_pinned_counters[0], d_global_count_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_pinned_counters[1], d_global_factor_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
}

namespace kernels {

// -----------------------------------------------------------------------------
// Asynchronous Host-Pinned Memory Counter update
// -----------------------------------------------------------------------------
__global__ void publish_stats_kernel(
    uint64_t* internal_device_counter_0,
    uint64_t* internal_device_counter_1,
    volatile uint64_t* mapped_host_counter
) {
    // Only one thread does the PCI-e write
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (internal_device_counter_0
	    && internal_device_counter_1
	    && mapped_host_counter) {
            // Write to host memory. 
            // __threadfence_system() ensures the write is visible to the host 
            // before the kernel is considered "complete" by stream semantics.
	    *mapped_host_counter = *internal_device_counter_0;
	    *(mapped_host_counter+1) = *internal_device_counter_1;
            __threadfence_system(); 
        }
    }
}
  
// -----------------------------------------------------------------------------
// Square a uint512: separate __noinline__ function to prevent NVCC 13.0
// from miscompiling the inlined mult at -O3 on SM 12.0.
__device__ __noinline__ mpqs::uint512 square_uint512(const mpqs::uint512& x) {
    mpqs::uint512 result = x;
    result.mult(x);
    return result;
}

// Validation Kernel
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 1) void debug_validate_soa_kernel(
    RelationBatchView view,
    uint32_t num_rels,
    mpqs::uint512 N,
    const uint32_t* __restrict__ fb_primes,
    uint32_t* failure_count,
    RelationBatch::ValidationInfo* debug_out,
    uint32_t max_debug_entries
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rels) return;

    // Detect LP-combined relations: sqrt_Q was computed as
    // sqrt_Q_1 * sqrt_Q_2 mod N, so exact |sqrt_Q² - N| overflows
    // uint512 (up to N² ≈ 2^596) and the identity is modular, not exact.
    unsigned __int128 lp = view.large_primes[i];
    bool is_combined = (lp > 1);

    // ================================================================
    // PATH A: Single relations — exact uint512 validation
    // PATH B: LP-combined relations — modular validation (mod N)
    // ================================================================

    // 1. Compute LHS = sqrt_Q² (or sqrt_Q² mod N for combined)
    const mpqs::uint512 sqrt_Q_val = view.sqrt_Q[i];
    mpqs::uint512 diff;
    bool is_negative_residue = false;

    if (is_combined) {
        // Modular path: LHS = sqrt_Q² mod N
        diff = sqrt_Q_val;
        diff.mul_mod(sqrt_Q_val, N);
    } else {
        // Exact path: LHS = |sqrt_Q² - N|
        mpqs::uint512 val_sq = square_uint512(sqrt_Q_val);
        if (val_sq < N) {
            is_negative_residue = true;
            diff = N;
            diff.sub(val_sq);
        } else {
            diff = val_sq;
            diff.sub(N);
        }
    }

    // 2. Check Sign Consistency (single relations only)
    // Combined relations merge signs multiplicatively; the modular
    // check subsumes sign correctness, so we skip the sign test.
    uint8_t s = view.signs[i];
    bool sign_ok = true;
    uint8_t expected_s = 0; // 0 = N/A (combined relations)
    if (!is_combined) {
        expected_s = is_negative_residue ? 255 : 1;
        if (s != expected_s) sign_ok = false;
    }

    // 3. Compute RHS: Product of factors
    mpqs::uint512 prod((uint32_t)1);

    // A. Factor 2
    int32_t v2 = view.val_2_exps[i];
    if (v2 > 0) prod.lshift(v2);

    // B. Large Prime contribution.
    // Single partial: Q = sign * 2^v2 * LP * ∏fb^e       → multiply LP once.
    // Combined pair:  Q1*Q2 = sign * 2^v2 * LP² * ∏fb^e  → multiply LP twice.
    if (lp > 1) {
        mpqs::uint512 lp_val;
        lp_val.limbs[0] = (uint32_t)lp;
        lp_val.limbs[1] = (uint32_t)(lp >> 32);
        lp_val.limbs[2] = (uint32_t)(lp >> 64);
        lp_val.limbs[3] = (uint32_t)(lp >> 96);
        prod.mult(lp_val);
        if (is_combined) prod.mult(lp_val);  // LP² for combined relations
    }

    // C. Factor Base from CSR
    uint64_t start = view.factor_offsets[i];
    uint64_t end = view.factor_offsets[i+1];

    if (end < start || (end - start) > 400) {
         uint32_t idx = atomicAdd(failure_count, 1);
         if (idx < max_debug_entries) {
             debug_out[idx].rel_idx = i;
             debug_out[idx].error_type = 3; // Bad CSR
         }
         return;
    }

    for (uint64_t k = start; k < end; ++k) {
        uint32_t fb_idx = view.factor_indices[k];
        uint8_t  cnt    = view.factor_counts[k];
        uint32_t p      = fb_primes[fb_idx];

        for(uint8_t c = 0; c < cnt; ++c) {
            prod.mult_uint32(p);
        }
    }

    // 4. Validate
    bool prod_ok;
    if (is_combined) {
        // Modular path: reduce RHS mod N, compare with LHS (already mod N).
        // For combined relations, Q_1*Q_2 = sign * prod * LP, and
        // sqrt_Q² ≡ Q_1*Q_2 (mod N). The sign is absorbed: if the product
        // of signs is negative, then prod ≡ N - sqrt_Q² (mod N).
        //
        // Check both: prod mod N == diff, or (N - prod mod N) == diff.
        mpqs::uint512 prod_mod = prod;
        prod_mod.mod(N);

        if (prod_mod == diff) {
            prod_ok = true;
        } else {
            // Try with negation (sign = -1 combined)
            mpqs::uint512 neg_prod = N;
            neg_prod.sub(prod_mod);
            prod_ok = (neg_prod == diff);
        }
    } else {
        // Exact path: |sqrt_Q² - N| == product
        prod_ok = (diff == prod);
    }

    if (!sign_ok || !prod_ok) {
        uint32_t idx = atomicAdd(failure_count, 1);
        if (idx < max_debug_entries) {
            debug_out[idx].rel_idx = i;
            debug_out[idx].error_type = (!sign_ok) ? 1 : 2;
            debug_out[idx].calculated_diff = diff;
            debug_out[idx].calculated_prod = prod;
            debug_out[idx].sign_stored = s;
            debug_out[idx].sign_expected = expected_s;
        }
    }
}
 
} // namespace kernels

void RelationBatch::updateStats(cudaStream_t stream) {
    if (!d_global_count_ptr || !h_pinned_counters) return;
    // Async copy to pinned memory
    // It queues a tiny job into the stream.
    // It does NOT block the CPU.
    
    // Publish Relation and Factor counts
    kernels::publish_stats_kernel<<<1, 1, 0, stream>>>(
        d_global_count_ptr,                  // Fast device memory
        d_global_factor_ptr,                 // Fast device memory
        h_pinned_counters                    // Mapped host memory
    );
}

std::pair<uint64_t, uint64_t> RelationBatch::readStats() const {
    if (!h_pinned_counters) return {0, 0};
    // Read directly from pinned memory (CPU visible)
    return { h_pinned_counters[0], h_pinned_counters[1] };
}

void RelationBatch::validate_relations(const mpqs::uint512& N, const uint32_t* d_factor_base, size_t fb_size) {
    if (!d_sqrt_Q) return;

    uint64_t count = this->getCount(0); // Using stream 0 for debug sync
    if (count == 0) return;

    // Count LP-combined relations (lp > 1) for diagnostic
    {
        std::vector<unsigned __int128> h_lp(count);
        CUDA_CHECK(cudaMemcpy(h_lp.data(), d_large_primes, count * sizeof(unsigned __int128), cudaMemcpyDeviceToHost));
        uint64_t combined_count = 0;
        for (uint64_t i = 0; i < count; i++) {
            if (h_lp[i] > 1) combined_count++;
        }
        LOG(LOG_INFO) << "[RelationBatch] LP-combined (lp>1): " << combined_count << " / " << count;
    }

    LOG(LOG_INFO) << "[RelationBatch] Validating " << count << " relations on GPU...";

    // Alloc Debug Buffers
    uint32_t* d_fail_count;
    ValidationInfo* d_debug_out;
    uint32_t max_debug = 10;
    
    CUDA_CHECK(cudaMalloc(&d_fail_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_fail_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_debug_out, max_debug * sizeof(ValidationInfo)));

    // Launch
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    
    kernels::debug_validate_soa_kernel<<<blocks, threads>>>(
        this->get_view(),
        (uint32_t)count,
        N,
        d_factor_base,
        d_fail_count,
        d_debug_out,
        max_debug
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read Results
    uint32_t fail_count = 0;
    CUDA_CHECK(cudaMemcpy(&fail_count, d_fail_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    std::vector<ValidationInfo> failures(std::min(fail_count, max_debug));
    if (!failures.empty()) {
        CUDA_CHECK(cudaMemcpy(failures.data(), d_debug_out, failures.size() * sizeof(ValidationInfo), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_fail_count));
    CUDA_CHECK(cudaFree(d_debug_out));

    if (fail_count == 0) {
        LOG(LOG_INFO) << "[RelationBatch] OK. All " << count << " relations valid.";
    } else {
        LOG(LOG_ERROR_CRITICAL) << "[RelationBatch] VALIDATION FAILED! " << fail_count << " / " << count 
                                << " relations are corrupt (" << (100.0*fail_count/count) << "%).";
        
        // Detailed Dump
        RelationBatchView v = this->get_view(); // pointers are device pointers
        // To print factor indices, we need to copy a small chunk of indices/counts from device to host
        // We do this lazily inside the loop.

        for (const auto& f : failures) {
            LOG(LOG_ERROR_MAJOR) << "--- Failure Rel #" << f.rel_idx << " ---";
            if (f.error_type == 1) {
                LOG(LOG_ERROR_MAJOR) << " Type: SIGN MISMATCH. Stored: " << (int)f.sign_stored 
                                     << " Expected: " << (int)f.sign_expected;
            } else if (f.error_type == 2) {
                LOG(LOG_ERROR_MAJOR) << " Type: PRODUCT MISMATCH";
                LOG(LOG_ERROR_MAJOR) << " LHS |Q| : " << f.calculated_diff.to_string();
                LOG(LOG_ERROR_MAJOR) << " RHS Prod: " << f.calculated_prod.to_string();
            } else {
                LOG(LOG_ERROR_MAJOR) << " Type: BAD CSR OFFSETS";
            }

            // Fetch CSR info for this relation to print factors
            uint64_t offsets[2];
            CUDA_CHECK(cudaMemcpy(offsets, v.factor_offsets + f.rel_idx, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            
            uint64_t len = offsets[1] - offsets[0];
            LOG(LOG_ERROR_MAJOR) << " Factor Count: " << len;
            if (len > 0 && len < 100) {
                std::vector<uint32_t> idxs(len);
                CUDA_CHECK(cudaMemcpy(idxs.data(), v.factor_indices + offsets[0], len * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                std::stringstream ss;
                ss << " Indices: ";
                for(auto x : idxs) ss << x << " ";
                LOG(LOG_ERROR_MAJOR) << ss.str();
            }
        }
    }
}

void RelationBatch::validate_host_batch(const HostRelationBatch& host_batch, const std::vector<uint32_t>& host_fb, const mpqs::uint512& N)
{
    if (host_batch.num_relations == 0) return;

    // 1. Create Temp Device Batch on the currently active device
    int current_device = 0;
    CUDA_CHECK(cudaGetDevice(&current_device));
    RelationBatch temp_batch;
    temp_batch.initiate(current_device);
    temp_batch.resize(host_batch.num_relations, host_batch.num_factors);

    // 2. Upload Data manually to the pointers exposed by get_view()
    // Note: get_view returns device pointers.
    RelationBatchView v = temp_batch.get_view();
    
    auto copyToDev = [](void* dst, const void* src, size_t sz) {
        if(sz) CUDA_CHECK(cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice));
    };

    copyToDev(v.sqrt_Q,       host_batch.sqrt_Q.data(),       host_batch.num_relations * sizeof(mpqs::uint512));
    copyToDev(v.signs,        host_batch.signs.data(),        host_batch.num_relations * sizeof(uint8_t));
    copyToDev(v.val_2_exps,   host_batch.val_2_exps.data(),   host_batch.num_relations * sizeof(int32_t));
    copyToDev(v.large_primes, host_batch.large_primes.data(), host_batch.num_relations * sizeof(unsigned __int128));
    copyToDev(v.factor_offsets, host_batch.factor_offsets.data(), (host_batch.num_relations + 1) * sizeof(uint64_t));

    if (host_batch.num_factors > 0) {
        copyToDev(v.factor_indices, host_batch.factor_indices.data(), host_batch.num_factors * sizeof(uint32_t));
        copyToDev(v.factor_counts,  host_batch.factor_counts.data(),  host_batch.num_factors * sizeof(uint8_t));
    }

    // Set atomic counters
    uint64_t cnts[2] = { (uint64_t)host_batch.num_relations, (uint64_t)host_batch.num_factors };
    CUDA_CHECK(cudaMemcpy(v.global_count, &cnts[0], sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v.global_factor_idx, &cnts[1], sizeof(uint64_t), cudaMemcpyHostToDevice));

    // 3. Upload Factor Base
    uint32_t* d_fb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fb, host_fb.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_fb, host_fb.data(), host_fb.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 4. Run Validation
    temp_batch.validate_relations(N, d_fb, host_fb.size());

    // 5. Cleanup
    CUDA_CHECK(cudaFree(d_fb));
    // temp_batch destructor cleans up relations
}

// -----------------------------------------------------------------------------
// Upload from Host (for LINALG_ONLY mode)
// -----------------------------------------------------------------------------
void RelationBatch::uploadFromHost(const HostRelationBatch& host_batch) {
    if (host_batch.num_relations == 0) return;

    CUDA_CHECK(cudaSetDevice(device_id));

    // 1. Resize to fit (uses managed or device alloc depending on use_managed_memory_)
    resize(host_batch.num_relations, host_batch.num_factors);

    size_t nr = host_batch.num_relations;
    size_t nf = host_batch.num_factors;

    if (use_managed_memory_) {
        // Unified memory path: CPU-to-CPU memcpy into managed pointers
        std::memcpy(d_sqrt_Q,        host_batch.sqrt_Q.data(),        nr * sizeof(mpqs::uint512));
        std::memcpy(d_signs,         host_batch.signs.data(),         nr * sizeof(uint8_t));
        std::memcpy(d_val_2_exps,    host_batch.val_2_exps.data(),    nr * sizeof(int32_t));
        std::memcpy(d_large_primes,  host_batch.large_primes.data(),  nr * sizeof(unsigned __int128));
        std::memcpy(d_factor_offsets, host_batch.factor_offsets.data(), (nr + 1) * sizeof(uint64_t));
        if (nf > 0) {
            std::memcpy(d_factor_indices, host_batch.factor_indices.data(), nf * sizeof(uint32_t));
            std::memcpy(d_factor_counts,  host_batch.factor_counts.data(),  nf * sizeof(uint8_t));
        }
    } else {
        // Discrete GPU path: original cudaMemcpy H→D (UNCHANGED)
        auto copyToDev = [](void* dst, const void* src, size_t sz) {
            if (sz) CUDA_CHECK(cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice));
        };

        copyToDev(d_sqrt_Q,        host_batch.sqrt_Q.data(),        nr * sizeof(mpqs::uint512));
        copyToDev(d_signs,         host_batch.signs.data(),         nr * sizeof(uint8_t));
        copyToDev(d_val_2_exps,    host_batch.val_2_exps.data(),    nr * sizeof(int32_t));
        copyToDev(d_large_primes,  host_batch.large_primes.data(),  nr * sizeof(unsigned __int128));
        copyToDev(d_factor_offsets, host_batch.factor_offsets.data(), (nr + 1) * sizeof(uint64_t));
        if (nf > 0) {
            copyToDev(d_factor_indices, host_batch.factor_indices.data(), nf * sizeof(uint32_t));
            copyToDev(d_factor_counts,  host_batch.factor_counts.data(),  nf * sizeof(uint8_t));
        }
    }

    // 3. Set atomic counters (always device memory, not managed)
    uint64_t nr64 = nr;
    uint64_t nf64 = nf;
    CUDA_CHECK(cudaMemcpy(d_global_count_ptr, &nr64, sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_global_factor_ptr, &nf64, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // 4. Update pinned counters
    h_pinned_counters[0] = nr64;
    h_pinned_counters[1] = nf64;
}

} // namespace structures
} // namespace mpqs
