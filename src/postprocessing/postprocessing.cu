// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski

#include "postprocessing.h"
#include "../largeprimes/largeprime.h" // SLPPinnedStats (full definition for prediction kernel)
#include "logger/hpc_logger.h"
#include "cuda_check.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace mpqs {
namespace postprocessing {

namespace kernels {
 
// -----------------------------------------------------------------------------
// Compaction & Expansion Kernel
// -----------------------------------------------------------------------------
__global__ void expandAndAccumulateKernel(
    const mpqs::sieve::candidateRelation* __restrict__ input,
    uint32_t input_len,
    mpqs::uint512 current_a,
    const uint32_t* __restrict__ dev_current_a_factors,
    uint32_t num_a_factors,
    int32_t start_index,
    mpqs::sieve::DenseCandidate* __restrict__ output,
    uint32_t* __restrict__ counter,
    uint32_t max_capacity
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= input_len) return;

    // 1. Load from Global -> Local Registers
    // This creates a mutable copy 'in_rel' private to this thread.
    mpqs::sieve::candidateRelation in_rel = input[tid];

    if (in_rel.num_factors > 0) {
        uint32_t n_sieve = in_rel.num_factors;
        if (n_sieve > 32) return; // Overflow guard — before slot reservation

        uint32_t pos = atomicAdd(counter, 1);
        if (pos < max_capacity) {

	    // We first sort in_rel.factors in increasing order
	    // Since this array is statistically almost sorted,
	    // We chose a sorting strategy which takes this into
	    // account.
            // --- Sort Step (In-Place on Registers) ---
            // Adaptive: Runs in O(N) if already sorted.
            // No extra memory buffer needed.

            for (int i = 1; i < n_sieve; ++i) {
                uint32_t key = in_rel.factors[i];
                int j = i - 1;

                // Only enters this loop if an inversion is found
                while (j >= 0 && in_rel.factors[j] > key) {
                    in_rel.factors[j + 1] = in_rel.factors[j];
                    j--;
                }
                in_rel.factors[j + 1] = key;
            }

            // --- Output Generation ---
            output[pos].a = current_a;
            output[pos].b = in_rel.b;
            output[pos].true_x = (int32_t)in_rel.sieve_offset;

	    // --- Merge Step (Global Merge Sort) ---
            // Merge Lists into a single strictly increasing sequence:
            // List A: Factors of 'a' (dev_current_a_factors)
            // List B: Factors from Sieve Hint (in_rel.factors)
            
            uint8_t count = 0;
            uint32_t idx_a = 0;
            uint32_t idx_sieve = 0;

            // Merge Loop
            while (idx_a < num_a_factors && idx_sieve < in_rel.num_factors && count < 48) {
                // Load values (val_a from Global, val_sieve from Register)
                uint32_t val_a = dev_current_a_factors[idx_a];
                uint32_t val_sieve = in_rel.factors[idx_sieve];

                if (val_a < val_sieve) {
                    output[pos].factor_indices[count++] = val_a;
                    idx_a++;
                } else {
                    output[pos].factor_indices[count++] = val_sieve;
                    idx_sieve++;
                }
            }

            // Flush remaining factors of 'a'
            while (idx_a < num_a_factors && count < 48) {
                output[pos].factor_indices[count++] = dev_current_a_factors[idx_a++];
            }

            // Flush remaining sieve factors
            while (idx_sieve < in_rel.num_factors && count < 48) {
                output[pos].factor_indices[count++] = in_rel.factors[idx_sieve++];
            }

            output[pos].num_factors = count;	    
        }
    }
}

// -----------------------------------------------------------------------------
// SOA Batched Factorization Kernel
// -----------------------------------------------------------------------------

// --- Single atomic dual counter primitives ---
// Packed dual counter encoding:
// Let (R,F) be 32-bit counters: R = #relations, F = #factors.
// pack(R,F) = (uint64_t(R) << 32) | uint64_t(F).
// Unpack: R = x >> 32, F = x & 0xFFFFFFFF.
__device__ __forceinline__ uint64_t pack_rf(uint32_t R, uint32_t F) {
    return (uint64_t(R) << 32) | uint64_t(F);
}
__device__ __forceinline__ uint32_t unpack_r(uint64_t x) { return uint32_t(x >> 32); }
__device__ __forceinline__ uint32_t unpack_f(uint64_t x) { return uint32_t(x & 0xFFFFFFFFULL); }

__global__ void commit_dual_counter_kernel(
    const uint64_t* __restrict__ dual,
    uint64_t* __restrict__ out_rel_count,
    uint64_t* __restrict__ out_factor_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint64_t x = *dual;
        *out_rel_count    = (uint64_t)unpack_r(x);
        *out_factor_count = (uint64_t)unpack_f(x);
    }
}

/// @brief Inverse of commit: packs global counts back into a dual counter.
__global__ void sync_dual_counter_from_batch_kernel(
    uint64_t* __restrict__ dual,
    const uint64_t* __restrict__ rel_count,
    const uint64_t* __restrict__ factor_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dual = pack_rf((uint32_t)*rel_count, (uint32_t)*factor_count);
    }
}

/**
 * @brief Warp-Aggregated Append to SoA.
 * 
 * Writes relations to the RelationBatchView buffers. 
 * Uses Warp-level primitives to aggregate atomic increments, reducing contention 
 * on the global counters.
 */
__device__ __forceinline__ 
void append_to_soa(
    bool predicate,                               
    const uint32_t* __restrict__ my_factors,      
    const uint8_t*  __restrict__ my_counts,       
    int num_factors,                              
    const mpqs::structures::Relation& meta_data,  
    const mpqs::uint512& sqrt_Q_val,              
    mpqs::structures::RelationBatchView view,     
    unsigned long long* __restrict__ dual_counter 
) {
    // 0. Safety Checks
    if (view.global_count == nullptr) return;    
    if (dual_counter == nullptr) return;          

    // 1. Get Mask of currently active threads (handles partial warps correctly)
    uint32_t warp_mask = __activemask();

    // 2. Ballot: Which active threads actually have data?
    uint32_t active_mask = __ballot_sync(warp_mask, predicate);
    
    // If no one in the warp has data, exit
    if (active_mask == 0) return;

    // 3. Intra-Warp Scan for Relations
    int lane_id = threadIdx.x & 31;
    int my_rel_rank = __popc(active_mask & ((1U << lane_id) - 1));
    int total_rels_warp = __popc(active_mask);

    // 4. Intra-Warp Scan for Factors (Inclusive Prefix Sum)
    uint32_t val = predicate ? (uint32_t)num_factors : 0u;

    // Standard Kogge-Stone scan
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) {
        uint32_t n = __shfl_up_sync(warp_mask, val, i);
        if (lane_id >= i) val += n;
    }

    // Retrieve total from the highest active lane, not lane 31.
    // __clz counts leading zeros. For a full warp, clz=0 -> pos=31. 
    // For partial warp (e.g. 0-10 active), mask is 0x7FF, clz=21 -> pos=10.
    int highest_active_lane = 31 - __clz(warp_mask);
    uint32_t total_factors_warp = __shfl_sync(warp_mask, val, highest_active_lane);

    // Calculate exclusive offset for this thread
    uint32_t my_factor_offset = val - (predicate ? (uint32_t)num_factors : 0u);
    
    // 5. Leader Allocation (Lowest active lane in mask allocates)
    uint64_t base_rel_idx = 0;
    uint64_t base_factor_idx = 0;

    int leader_lane = __ffs(active_mask) - 1; 
    
    if (lane_id == leader_lane) {
        uint32_t dR = (uint32_t)total_rels_warp;
        uint32_t dF = (uint32_t)total_factors_warp;

        // Atomic CAS Loop on packed dual counter
        unsigned long long old = atomicAdd(dual_counter, 0ULL);
        while (true) {
            uint32_t R = (uint32_t)(old >> 32);
            uint32_t F = (uint32_t)(old & 0xFFFFFFFFULL);

            // Cap check: skip append if persistent batch has reached ceiling
            if (view.target_cap > 0 && R >= view.target_cap) {
                base_rel_idx = UINT64_MAX;  // sentinel: skip
                break;
            }

            // Capacity check: prevent counter inflation past buffer capacity
            if (R + dR > view.max_relations) {
                base_rel_idx = UINT64_MAX;  // sentinel: skip
                break;
            }

            unsigned long long desired =
                pack_rf(R + dR, F + dF); // Using helper or manual shift

            unsigned long long prev = atomicCAS(dual_counter, old, desired);
            if (prev == old) {
                base_rel_idx    = (uint64_t)R;
                base_factor_idx = (uint64_t)F;
                break;
            }
            old = prev;
        }
    }

    // 6. Broadcast Bases
    base_rel_idx    = __shfl_sync(warp_mask, base_rel_idx, leader_lane);
    if (base_rel_idx == UINT64_MAX) return;  // Capped — discard warp's relations
    base_factor_idx = __shfl_sync(warp_mask, base_factor_idx, leader_lane);

    // 7. Write Data (Only threads with predicate=true)
    if (!predicate) return;

    uint64_t my_global_rel_idx = base_rel_idx + my_rel_rank;
    uint64_t my_global_factor_start = base_factor_idx + my_factor_offset;

    // Check against View capacities, drop relation if necessary
    if (my_global_rel_idx >= view.max_relations) return;
    if ((my_global_factor_start + num_factors) > view.max_factors) return;

    view.sqrt_Q[my_global_rel_idx]       = sqrt_Q_val;
    view.signs[my_global_rel_idx]        = meta_data.sign_of_Q;
    view.val_2_exps[my_global_rel_idx]   = meta_data.val_2_exp;
    view.large_primes[my_global_rel_idx] = meta_data.large_prime_remainder;

    // Write CSR Offsets
    // This defines the range [start, start + count) for this relation
    view.factor_offsets[my_global_rel_idx] = my_global_factor_start;
    view.factor_offsets[my_global_rel_idx + 1] = my_global_factor_start + num_factors;
    
    // Write Factors
    for (int i = 0; i < num_factors; ++i) {
        view.factor_indices[my_global_factor_start + i] = my_factors[i];
        view.factor_counts[my_global_factor_start + i]  = my_counts[i];
    }
}

// -----------------------------------------------------------------------------
// Shared trial-division body for batched factorization kernels.
// -----------------------------------------------------------------------------
// Body is identical between batchedFactorizationKernel (host count) and
// batchedBatchFactorizationKernelSoA (device-pointer count). Only the count
// guard differs; the kernels delegate the rest to this inline function.
__device__ __forceinline__ void processCandidate(
    mpqs::sieve::DenseCandidate* __restrict__ input,
    uint32_t tid,
    mpqs::sieve::devicePointers data,
    PostProcConfig cfg,
    mpqs::uint512 N_val,
    mpqs::structures::RelationBatchView full_view,
    mpqs::structures::RelationBatchView partial_view,
    unsigned long long* __restrict__ full_dual_counter,
    unsigned long long* __restrict__ partial_dual_counter
) {
    // Load candidate from dense buffer
    mpqs::sieve::DenseCandidate c = input[tid];

    // --- 1. Compute Sqrt(Q) and Q ---
    mpqs::uint512 sqrt_Q;
    mpqs::math::calculate_sqrt_of_QX(c.a, c.b, c.true_x, sqrt_Q);

    mpqs::uint512 Q = sqrt_Q;
    int8_t sign;

    Q.mult(Q); // Q = (ax+b)^2

    if (Q < N_val) {
        sign = -1;
        mpqs::uint512 tmp = N_val;
        tmp.sub(Q);
        Q = tmp;
    } else {
        sign = 1;
        Q.sub(N_val);
    }

    // --- 2. Trial Division Setup ---
    uint32_t local_factors[64];
    uint8_t  local_counts[64];
    int num_factors_found = 0;

    int32_t val_2_exp = (int32_t) Q.countr_zero();
    if (val_2_exp > 0) Q.rshift(val_2_exp);

    // --- 3. Factorization Loop ---
    for (uint32_t k = 0; k < c.num_factors; ++k) {
        uint32_t fb_idx = c.factor_indices[k];
        uint32_t p = data.dev_factorBase[fb_idx];

        if (p > 1) {
            uint8_t cnt = 0;
            while (Q.mod_uint32(p) == 0) {
                Q.div_uint32_inplace(p);
                cnt++;
            }

            if (cnt > 0 && num_factors_found < 64) {
                local_factors[num_factors_found] = fb_idx;
                local_counts[num_factors_found]  = cnt;
                num_factors_found++;
            }
        }
    }

    // --- 4. Classification ---
    bool is_one = Q.is_one();
    unsigned __int128 remainder = 0;
    bool is_valid_partial = false;

    if (!is_one) {
        bool fits_128 = true;
        #pragma unroll
        for(int k=4; k<16; ++k) {
            if (Q.limbs[k] != 0) { fits_128 = false; break; }
        }

        if (fits_128) {
            unsigned __int128 lo = Q.limbs[0] | ((unsigned __int128)Q.limbs[1] << 32);
            unsigned __int128 hi = Q.limbs[2] | ((unsigned __int128)Q.limbs[3] << 32);
            remainder = lo | (hi << 64);

            if (remainder < cfg.lp1_bound) {
                is_valid_partial = true;
            }
        }
    }

    // --- 5. Prepare Transport Data ---
    mpqs::structures::Relation meta;
    meta.sign_of_Q = sign;
    meta.val_2_exp = val_2_exp;
    meta.large_prime_remainder = is_one ? 1 : remainder;

    // --- 6. Interface Calls (Dual Append) ---
    append_to_soa(
        is_one,
        local_factors, local_counts, num_factors_found,
        meta, sqrt_Q,
        full_view,
        full_dual_counter
    );

    append_to_soa(
        is_valid_partial,
        local_factors, local_counts, num_factors_found,
        meta, sqrt_Q,
        partial_view,
        partial_dual_counter
    );
}

__global__ void batchedFactorizationKernel(
    mpqs::sieve::DenseCandidate* __restrict__ input,
    uint32_t count,
    mpqs::sieve::devicePointers data,
    PostProcConfig cfg,
    mpqs::uint512 N_val,
    mpqs::structures::RelationBatchView full_view,
    mpqs::structures::RelationBatchView partial_view,
    unsigned long long* __restrict__ full_dual_counter,
    unsigned long long* __restrict__ partial_dual_counter
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    processCandidate(input, tid, data, cfg, N_val,
                     full_view, partial_view,
                     full_dual_counter, partial_dual_counter);
}

// -----------------------------------------------------------------------------
// Batch-Sieve SoA Factorization Kernel
// -----------------------------------------------------------------------------
// Identical to batchedFactorizationKernel but reads count from a device pointer
// *d_count (zero CPU/GPU sync) instead of a host parameter. This enables the
// batch sieve loop to launch factorization without any host-device sync.
__global__ void batchedBatchFactorizationKernelSoA(
    mpqs::sieve::DenseCandidate* __restrict__ input,
    uint32_t* d_count,
    mpqs::sieve::devicePointers data,
    PostProcConfig cfg,
    mpqs::uint512 N_val,
    mpqs::structures::RelationBatchView full_view,
    mpqs::structures::RelationBatchView partial_view,
    unsigned long long* __restrict__ full_dual_counter,
    unsigned long long* __restrict__ partial_dual_counter
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *d_count) return;

    processCandidate(input, tid, data, cfg, N_val,
                     full_view, partial_view,
                     full_dual_counter, partial_dual_counter);
}

// Forces the counter to stay within bounds.
// We may run this immediately after expandAndAccumulate.
__global__ void clampCounterKernel(uint32_t* counter, uint32_t max_capacity) {
    if (*counter > max_capacity) {
        *counter = max_capacity;
    }
}

__global__ void find_first_bad_offset(
    const uint64_t* __restrict__ off,
    uint64_t nrels,
    uint64_t nfacts,
    uint32_t* __restrict__ bad_i   // init to 0xFFFFFFFF
) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrels) return;
    uint64_t a = off[i];
    uint64_t b = off[i+1];
    bool bad = (b < a) || (a > nfacts) || (b > nfacts) || (b - a > 128); // 128 just a sanity upper bound
    if (bad) atomicMin(bad_i, (uint32_t)i);
} 

////////// DUPLICATE REMOVAL LOGIC //////////

// -----------------------------------------------------------------------------
// Hashing Kernel (SoA Adapted)
// -----------------------------------------------------------------------------

/**
 * @brief Computes the deduplication hash (SoA version).
 * * Hash Composition (64-bit):
 * [63..48] Length (num_factors)
 * [47..32] XOR Sum of Factor Exponents (counts)
 * [31.. 0] XOR Sum of (Factor Index * Magic) ^ SignLogic
 */
__global__ void compute_relation_hashes_soa(
    mpqs::structures::RelationBatchView view,
    uint32_t num_relations,
    uint64_t* __restrict__ out_hashes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_relations) return;

    // --- Part 1: Length (16 bits) ---
    uint64_t start = view.factor_offsets[idx];
    uint64_t end   = view.factor_offsets[idx+1];
    uint16_t len   = (uint16_t)(end - start);

    // --- Part 2 & 3: Iterate Factors ---
    uint16_t exp_xor = 0;
    uint32_t body_xor = 0;
    const uint32_t MAGIC = 0x9e3779b9; // Standard hashing constant

    for (uint64_t k = start; k < end; ++k) {
        // XOR of Exponents/Counts
        exp_xor ^= (uint16_t)view.factor_counts[k];
        
        // XOR of Indices * Magic
        body_xor ^= (view.factor_indices[k] * MAGIC);
    }

    // --- Part 3 Continued: Sign Logic ---
    // Logic: xor sign * (1 << exponent of 2) into the dedup hash so that
    // two relations with identical factor sets but opposite Q-signs hash
    // distinctly.
    // Encoding (canonical, see audit Appendix A): view.signs[idx] is uint8_t
    // with {1 = positive Q, 0xFF = negative Q}. Use the M11c encoding-agnostic
    // "negative iff != 1" extraction (matches expanded_matrix.cpp:20,
    // gpu_batch_merge.cu:319, sqrt_step.cu:393).
    int32_t sign_val = (view.signs[idx] != 1u) ? -1 : 1;
    
    // Safety mask for shift (0x1F) to prevent UB
    int32_t shift_val = sign_val * (1 << (view.val_2_exps[idx] & 0x1F));
    
    body_xor ^= (uint32_t)shift_val;

    // --- Pack ---
    out_hashes[idx] = ((uint64_t)len << 48) | ((uint64_t)exp_xor << 32) | (uint64_t)body_xor;
}

// -----------------------------------------------------------------------------
// Reorder / Compact Kernel
// -----------------------------------------------------------------------------

/**
 * @brief Gathers relations from the old batch into the new batch based on survivor indices.
 * Copies the scalar values (primes, roots) and the vector values (factors).
 */
__global__ void gather_soa_relations_kernel(
    mpqs::structures::RelationBatchView src,
    mpqs::structures::RelationBatchView dest,
    const uint32_t* __restrict__ survivor_indices, // Indices in 'src' that survived
    uint32_t num_survivors,
    const uint64_t* __restrict__ dest_offsets // Offsets for 'dest' factors (computed via scan)
) {
    uint32_t new_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_idx >= num_survivors) return;

    uint32_t old_idx = survivor_indices[new_idx];

    // 1. Copy Scalars
    dest.large_primes[new_idx] = src.large_primes[old_idx];
    dest.sqrt_Q[new_idx]       = src.sqrt_Q[old_idx];
    dest.signs[new_idx]        = src.signs[old_idx];
    dest.val_2_exps[new_idx]   = src.val_2_exps[old_idx];

    // 2. Copy Factors
    // Src Range
    uint64_t src_start = src.factor_offsets[old_idx];
    uint64_t src_end   = src.factor_offsets[old_idx+1];
    uint64_t len       = src_end - src_start;

    // Dest Range
    // Note: dest_offsets is length num_survivors+1. 
    // dest_offsets[new_idx] is where this relation writes to.
    uint64_t dest_start = dest.factor_offsets[new_idx]; 
    
    // Copy loop (CSR segment copy)
    for (uint64_t k = 0; k < len; ++k) {
        dest.factor_indices[dest_start + k] = src.factor_indices[src_start + k];
        dest.factor_counts[dest_start + k]  = src.factor_counts[src_start + k];
    }
}

/**
 * @brief Computes the length of factors for survivor relations to prep exclusive_scan.
 */
__global__ void compute_new_lengths_kernel(
    const uint64_t* __restrict__ old_offsets,
    const uint32_t* __restrict__ survivor_indices,
    uint32_t num_survivors,
    uint32_t* __restrict__ out_lengths
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_survivors) return;
    
    uint32_t old_idx = survivor_indices[idx];
    out_lengths[idx] = (uint32_t)(old_offsets[old_idx+1] - old_offsets[old_idx]);
}

// -----------------------------------------------------------------------------
// Yield Prediction Kernel (adaptive batch convergence)
// -----------------------------------------------------------------------------

/**
 * @brief Single-thread kernel estimating whether enough relations have been
 *        collected to terminate sieving.
 *
 * Reads the persistent batch's device-side relation counter and, when LP is
 * active, the LP stats to predict additional yield.  Writes the result to
 * mapped pinned memory so the host can poll without synchronization.
 *
 * Termination condition:  effective_R >= target + target/20   (5 % margin)
 *
 * @param result        Mapped pinned PredictionResult (host-visible).
 * @param global_count  Device pointer to persistent batch relation counter.
 * @param target        Required number of relations (FB size + extra).
 * @param total_steps   Total sieve batches processed so far.
 * @param lp_stats      Device pointer to SLP pinned stats (nullable).
 */
__global__ void yield_prediction_kernel(
    mpqs::postprocessing::PredictionResult* __restrict__ result,
    const uint64_t* __restrict__ global_count,
    uint32_t target,
    uint32_t total_steps,
    const mpqs::lp::SLPPinnedStats* __restrict__ lp_stats
) {
    // Current relation count (lower 32 bits — count never exceeds 2^32)
    uint32_t R = static_cast<uint32_t>(*global_count);

    // Yield rate: relations per sieve step
    float lambda = (total_steps > 0) ? static_cast<float>(R) / static_cast<float>(total_steps) : 0.0f;

    // LP match rate and predicted LP yield
    float mu = 0.0f;
    if (lp_stats) {
        uint64_t lp_full = lp_stats->total_full_relations;
        uint64_t witnesses = lp_stats->total_witnesses;
        uint64_t denom = lp_full + witnesses;
        mu = (denom > 0) ? static_cast<float>(lp_full) / static_cast<float>(denom) : 0.0f;
    }

    uint32_t effective_R = R;  // LP full relations are already appended to persistent batch

    // 5% dropout margin: target_with_margin = target + target/20
    uint32_t target_with_margin = target + target / 20;

    // Write result to mapped pinned memory
    result->effective_R    = effective_R;
    result->yield_rate     = lambda;
    result->lp_match_rate  = mu;
    result->should_terminate = (effective_R >= target_with_margin) ? 1u : 0u;
}

/**
 * @brief Publishes buffer fill levels to mapped pinned memory.
 * Single-thread kernel -- ~3 global loads + 6 stores + 1 fence. Sub-microsecond.
 */
__global__ void publish_buffer_fill_kernel(
    const uint64_t* __restrict__ persistent_rel_count,
    const uint64_t* __restrict__ partial_rel_count,    // nullptr if LP disabled
    const uint32_t* __restrict__ accum_counter,
    BufferFillSnapshot* __restrict__ fill_snap
) {
    fill_snap->accum_fill      = *accum_counter;
    fill_snap->persistent_fill = *persistent_rel_count;
    fill_snap->partial_fill    = partial_rel_count ? *partial_rel_count : 0;
    __threadfence_system();
    fill_snap->generation++;
}

} // namespace kernels

// -----------------------------------------------------------------------------
// Controller Implementation
// -----------------------------------------------------------------------------

DevicePostProcessingController::DevicePostProcessingController() 
    : d_accumulation_buffer(nullptr), d_processing_buffer(nullptr),
      processing_active(false), accumulation_count(0)
{
    // Initiate takes care of everything
}

DevicePostProcessingController::~DevicePostProcessingController() {
    clearBuffers();
    clearPersistentBuffer();
    cudaEventDestroy(reset_event);
    CUDA_CHECK(cudaStreamDestroy(proc_stream)); // Careful: Large Prime may run on the same stream!
}

void DevicePostProcessingController::clearBuffers() {
    // We do not free d_persistent_relations; this is handled in clearPersistentBuffer();
    // We do not free d_current_accumulation_counter,
    // or d_current_processing_counter,
    // for they point to the same address as d_counter_A or B.
    if (d_counter_A) { CUDA_CHECK(cudaFree(d_counter_A)); d_counter_A = nullptr; }
    if (d_counter_B) { CUDA_CHECK(cudaFree(d_counter_B)); d_counter_B = nullptr; }
    if (d_accumulation_buffer) { CUDA_CHECK(cudaFree(d_accumulation_buffer)); d_accumulation_buffer = nullptr; }
    if (d_processing_buffer)  { CUDA_CHECK(cudaFree(d_processing_buffer)); d_processing_buffer = nullptr; }
    if (d_full_dual_counter) { cudaFree(d_full_dual_counter); d_full_dual_counter = nullptr; }
    if (d_partial_dual_counter) { cudaFree(d_partial_dual_counter); d_partial_dual_counter = nullptr; }
    if (d_persistent_dual_counter) { cudaFree(d_persistent_dual_counter); d_persistent_dual_counter = nullptr; }

    if (d_full_batch) d_full_batch.reset();
    if (d_partial_batch) d_partial_batch.reset();

    // Prediction result cleanup (mapped pinned memory)
    if (h_prediction_result) {
        CUDA_CHECK(cudaFreeHost((void*)h_prediction_result));
        h_prediction_result = nullptr;
        d_prediction_result = nullptr;
    }

    // Buffer fill telemetry cleanup (mapped pinned memory)
    if (h_buffer_fill_) {
        cudaFreeHost(h_buffer_fill_);
        h_buffer_fill_ = nullptr;
        d_buffer_fill_ = nullptr;
    }

    // Pinned LP partial count
    if (h_pinned_partial_count) {
        CUDA_CHECK(cudaFreeHost((void*)h_pinned_partial_count));
        h_pinned_partial_count = nullptr;
    }

    // Batch-sieve DoubleBuffer cleanup
    if (h_pinned_accumulation_counter) {
        CUDA_CHECK(cudaFreeHost((void*)h_pinned_accumulation_counter));
        h_pinned_accumulation_counter = nullptr;

        // DoubleBuffer events are co-allocated with pinned counter
        CUDA_CHECK(cudaEventDestroy(buffers[0].safe_to_write_event));
        CUDA_CHECK(cudaEventDestroy(buffers[0].safe_to_read_event));
        CUDA_CHECK(cudaEventDestroy(buffers[1].safe_to_write_event));
        CUDA_CHECK(cudaEventDestroy(buffers[1].safe_to_read_event));
    }
}

void DevicePostProcessingController::clearPersistentBuffer() {
    if (d_persistent_batch) d_persistent_batch.reset();
    if (h_pinned_persistent_count) {
        CUDA_CHECK(cudaFreeHost((void*)h_pinned_persistent_count));
        h_pinned_persistent_count = nullptr;
    }
}

void DevicePostProcessingController::initiate(
    const mpqs::sieve::factoringData& f_data, 
    const mpqs::sieve::devicePointers& common_ptrs, 
    const PostProcConfig& cfg
) {
    LOG_SET_MODULE("PostProc");
    this->config = cfg;
    this->N = f_data.N;
    this->dev_common = common_ptrs;

    size_t dense_size = config.accumulate_buffer_size * sizeof(mpqs::sieve::DenseCandidate);

    LOG(LOG_DEBUG_1) << "===== Initialization =====";
    LOG(LOG_DEBUG_2) << "GPU: " << cfg.device_id;
    CUDA_CHECK(cudaSetDevice(cfg.device_id));
    LOG(LOG_DEBUG_1) << "Device memory allocation:";
    LOG(LOG_DEBUG_1) << " Double buffers: " << (2*dense_size)/(1024*1024) << " MB (combined)";
    LOG(LOG_DEBUG_1) << " Accumulate Buffer Batch size: " << config.accumulate_buffer_size << ",  Purge thres: " << config.accumulate_batch_purge_threshold;

    LOG(LOG_DEBUG_2) << "Creating proc CUDA stream.";
    CUDA_CHECK(cudaStreamCreate(&proc_stream));
    // Create event with DisableTiming flag for better performance
    CUDA_CHECK(cudaEventCreateWithFlags(&reset_event, cudaEventDisableTiming));    

    // Allocate Accumulation Double Buffers (Manual CUDA)
    
    LOG(LOG_DEBUG_2) << "Allocating accumulation double buffers.";
    // Allocate TWO counters
    CUDA_CHECK(cudaMalloc(&d_counter_A, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_counter_B, sizeof(uint32_t)));
    // Initialize pointers
    d_current_accumulation_counter = d_counter_A;
    d_current_processing_counter   = d_counter_B;
    // Zero both
    CUDA_CHECK(cudaMemset(d_counter_A, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_counter_B, 0, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d_accumulation_buffer, dense_size));
    CUDA_CHECK(cudaMalloc(&d_processing_buffer, dense_size));

    // --- Initialize SoA Batches ---
    LOG(LOG_DEBUG_2) << "Allocating SoA Batch Buffers.";
    // 0. Atomic dual counter
    CUDA_CHECK(cudaMalloc(&d_full_dual_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_full_dual_counter, 0, sizeof(unsigned long long)));  
    // 1. Full Relations Batch (Temporary buffer for this processing step)
    d_full_batch = std::make_unique<mpqs::structures::RelationBatch>();
    d_full_batch->initiate(config.device_id);
    // Pre-allocate capacity to avoid reallocation during kernel execution
    // Capacity = accumulate_buffer_size
    // Note: Average factors is ~20, max 64. Allocate conservatively for factors.
    size_t est_factors = config.accumulate_buffer_size * 32;
    LOG(LOG_DEBUG_2) << " Allocating full relations batch factors buffer: " << (est_factors*(sizeof(uint32_t)+sizeof(uint8_t)))/(1024*1024) << " MB";
    d_full_batch->resize(config.accumulate_buffer_size, est_factors);
    LOG(LOG_DEBUG_1) << " Full relations batch factors buffer: " << (est_factors*(sizeof(uint32_t)+sizeof(uint8_t)))/(1024*1024) << " MB";

    // 2. Persistent Batch
    if(config.persistent_device_buffer_size) {
        d_persistent_batch = std::make_unique<mpqs::structures::RelationBatch>();
	d_persistent_batch->initiate(config.device_id);
        size_t p_est_factors = config.persistent_device_buffer_size * 32;
        d_persistent_batch->resize(config.persistent_device_buffer_size, p_est_factors);
	LOG(LOG_DEBUG_1) << " Persistent batch factors buffer: " << (p_est_factors*(sizeof(uint32_t)+sizeof(uint8_t)))/(1024*1024) << " MB";
    }

    // 3. Partial Batch
    if (config.lp1_bound > 0) {
        uint32_t p_size = (config.partial_buffer_size > 0) ? config.partial_buffer_size : config.accumulate_buffer_size;
        d_partial_batch = std::make_unique<mpqs::structures::RelationBatch>();
	d_partial_batch->initiate(config.device_id);
        size_t partial_est_factors = p_size * 32;
        d_partial_batch->resize(p_size, partial_est_factors);
	LOG(LOG_DEBUG_1) << " Partial relations factors buffer: " << (partial_est_factors*(sizeof(uint32_t)+sizeof(uint8_t)))/(1024*1024) << " MB";
        LOG(LOG_DEBUG_1) << "Large Prime Variation ENABLED. Bound: " << config.lp1_bound;
        LOG(LOG_DEBUG_2) << "Partial Batch Buffer: " << p_size << " entries.";
	// dual atomic counters
	CUDA_CHECK(cudaMalloc(&d_partial_dual_counter, sizeof(unsigned long long)));
	CUDA_CHECK(cudaMemset(d_partial_dual_counter, 0, sizeof(unsigned long long)));
    } else {
        d_partial_dual_counter = nullptr;
    }

    // -------------------------------------------------------------------------
    // BATCH PIPELINE: Initialize DoubleBuffering & Host-Side Polling
    // -------------------------------------------------------------------------
    buffers[0].d_data = d_accumulation_buffer;
    buffers[0].d_counter = d_counter_A;
    buffers[0].capacity = config.accumulate_buffer_size;
    CUDA_CHECK(cudaEventCreateWithFlags(&buffers[0].safe_to_write_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&buffers[0].safe_to_read_event, cudaEventDisableTiming));

    buffers[1].d_data = d_processing_buffer;
    buffers[1].d_counter = d_counter_B;
    buffers[1].capacity = config.accumulate_buffer_size;
    CUDA_CHECK(cudaEventCreateWithFlags(&buffers[1].safe_to_write_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&buffers[1].safe_to_read_event, cudaEventDisableTiming));

    CUDA_CHECK(cudaMallocHost((void**)&h_pinned_accumulation_counter, sizeof(uint32_t)));
    *h_pinned_accumulation_counter = 0;

    // Pinned partial count for LP input hint (zero-sync readback)
    if (d_partial_batch) {
        CUDA_CHECK(cudaMallocHost((void**)&h_pinned_partial_count, sizeof(uint32_t)));
        *h_pinned_partial_count = 0;
    }

    // Persistent dual counter for batch-sieve direct-to-persistent writes
    if (config.persistent_device_buffer_size) {
        CUDA_CHECK(cudaMalloc(&d_persistent_dual_counter, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_persistent_dual_counter, 0, sizeof(unsigned long long)));

        CUDA_CHECK(cudaMallocHost((void**)&h_pinned_persistent_count, sizeof(uint32_t)));
        *h_pinned_persistent_count = 0;
    }

    // --- Prediction result (mapped pinned memory for zero-sync host readback) ---
    {
        CUDA_CHECK(cudaHostAlloc(
            (void**)&h_prediction_result,
            sizeof(PredictionResult),
            cudaHostAllocMapped
        ));
        memset((void*)h_prediction_result, 0, sizeof(PredictionResult));
        CUDA_CHECK(cudaHostGetDevicePointer(
            (void**)&d_prediction_result,
            (void*)h_prediction_result,
            0
        ));
        LOG(LOG_DEBUG_2) << "Prediction result: mapped pinned memory allocated.";
    }

    // --- Buffer fill telemetry (mapped pinned memory for zero-sync host readback) ---
    {
        CUDA_CHECK(cudaHostAlloc(
            (void**)&h_buffer_fill_,
            sizeof(BufferFillSnapshot),
            cudaHostAllocMapped
        ));
        memset((void*)h_buffer_fill_, 0, sizeof(BufferFillSnapshot));
        CUDA_CHECK(cudaHostGetDevicePointer(
            (void**)&d_buffer_fill_,
            (void*)h_buffer_fill_,
            0
        ));
        // Set capacities once (constant for the lifetime of the pipeline)
        h_buffer_fill_->accum_capacity = config.accumulate_buffer_size;
        h_buffer_fill_->partial_capacity = config.partial_buffer_size;
        h_buffer_fill_->persistent_capacity = config.persistent_device_buffer_size;
        LOG(LOG_DEBUG_2) << "Buffer fill telemetry: mapped pinned memory allocated.";
    }

    // sets *d_current_accumulation_counter and accumulation_count to 0
    resetAccumulation();
}

void DevicePostProcessingController::resetAccumulation() {
    CUDA_CHECK(cudaMemsetAsync(d_current_accumulation_counter, 0, sizeof(uint32_t), proc_stream));
    accumulation_count = 0;
}

void DevicePostProcessingController::swapBuffers() {
    // Swap Buffers
    mpqs::sieve::DenseCandidate* temp = d_accumulation_buffer;
    d_accumulation_buffer = d_processing_buffer;
    d_processing_buffer = temp;

    // Swap Counters
    uint32_t* temp_count = d_current_accumulation_counter;
    d_current_accumulation_counter = d_current_processing_counter;
    d_current_processing_counter = temp_count;    
}

bool DevicePostProcessingController::accumulate(
    const mpqs::sieve::candidateRelation* raw_input, 
    uint32_t raw_size, 
    mpqs::uint512 current_a,
    const uint32_t* dev_a_factors_ptr,
    uint32_t num_a_factors,
    int32_t start_index,
    cudaStream_t stream
) {
    is_flushed = false;
    int threads = 256;
    int blocks = (raw_size + threads - 1) / threads;

    // [CRITICAL] Wait for the counter reset from the other stream
    // Ensure 'reset_event' was created in initiate()
    cudaStreamWaitEvent(stream, reset_event, 0);
    
    // Note: Ensure max_capacity passed here matches the buffer size
    // Launch on the Siever's stream (serialization guarantee)
    kernels::expandAndAccumulateKernel<<<blocks, threads, 0, stream>>>(
        raw_input,
        raw_size,
        current_a,
        dev_a_factors_ptr,
        num_a_factors,
	start_index,
        d_accumulation_buffer,
        d_current_accumulation_counter,
        config.accumulate_buffer_size
    );

    // This ensures the Host never sees a value > accumulate_buffer_size
    kernels::clampCounterKernel<<<1, 1, 0, stream>>>(
        d_current_accumulation_counter, 
        config.accumulate_buffer_size
    );

    // Async check of counter (requires pinned memory or careful sync)
    // For simplicity, we stick to existing logic but note the sync point
    uint32_t current_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&current_count, d_current_accumulation_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); 
    
    accumulation_count = current_count;
    
    if (accumulation_count >= config.accumulate_batch_purge_threshold) {
        LOG(LOG_DEBUG_2) << "[Accumulate] Triggering buffer purge: acc_cnt = " << accumulation_count << " (thres = " << config.accumulate_batch_purge_threshold << ")";
        return true; 
    }
    return false;    
}

// -----------------------------------------------------------------------------
// Get Accumulated Count
// -----------------------------------------------------------------------------

uint32_t DevicePostProcessingController::getAccumulatedCount(cudaStream_t sync_stream) {
    uint32_t current_count = 0;

    // Fallback to proc_stream if 0 is passed (preserves legacy behavior)
    cudaStream_t target_stream = (sync_stream != 0) ? sync_stream : proc_stream;

    CUDA_CHECK(cudaMemcpyAsync(
        &current_count,
        d_current_accumulation_counter,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        target_stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(target_stream));

    accumulation_count = current_count;
    return accumulation_count;
}

// -----------------------------------------------------------------------------
// Process Buffered Candidates (Sort -> Batch Factorize)
// -----------------------------------------------------------------------------

void DevicePostProcessingController::processBufferedCandidates() {
    if(is_flushed) {
        LOG(LOG_DEBUG_2) << "Not processing buffered candidates: flushed.";
        return;
    }

    LOG(LOG_DEBUG_2) << "Processing buffered candidates";
    processing_active = true;

    // 1. Swap pointers (Host side). 
    // d_processing_buffer now points to the filled data.
    // d_accumulation_buffer points to the stale buffer (ready to be overwritten).
    swapBuffers();

    // Capture the count *before* we reset the accumulator logic
    int count = accumulation_count;

    // 2. Reset accumulation logic for the *next* batch
    // This memsets d_current_accumulation_counter to 0 on proc_stream
    resetAccumulation();

    // [CRITICAL SYNC]
    // Record that the reset is done. The 'accumulate' method (on the sieve stream) 
    // waits for this event before writing new data, preventing race conditions.
    cudaEventRecord(reset_event, proc_stream);    
    
    if (count == 0) {
        processing_active = false;
        return;
    }

    // 3. Sort Candidates to improve branch divergence in factorization
    // d_processing_buffer is a raw pointer. We wrap it in device_ptr for Thrust.

    try {
        thrust::device_ptr<mpqs::sieve::DenseCandidate> ptr_start(d_processing_buffer);
        thrust::device_ptr<mpqs::sieve::DenseCandidate> ptr_end(d_processing_buffer + count);
        
        thrust::sort(
            thrust::cuda::par.on(proc_stream), 
            ptr_start, 
            ptr_end, 
            [] __device__ (const mpqs::sieve::DenseCandidate& a, const mpqs::sieve::DenseCandidate& b) {
                // Sorting by 'true_x' correlates with sieve location, 
                // effectively grouping relations found in the same sieve block.
                return a.true_x < b.true_x; 
            }
        );
    } catch (std::exception& e) {
        LOG(LOG_ERROR_CRITICAL) << "Thrust Sort Error: " << e.what();
    }
    
    // 4. Reset Output Batch Counters (Prepare for writing)
    d_full_batch->reset_counters(proc_stream);
    if (d_partial_batch) {
        d_partial_batch->reset_counters(proc_stream);
    }
    
    // 5. Prepare Views for the Kernel
    auto full_view = d_full_batch->get_view();
    
    // Initialize partial_view with braces {} to ensure all pointers are nullptr 
    // if d_partial_batch doesn't exist. This is critical for the kernel safety check.
    mpqs::structures::RelationBatchView partial_view = {}; 
    if (d_partial_batch) {
        partial_view = d_partial_batch->get_view();
    }

    // 6. Launch Factorization Kernel
    
    // Reset dual counters
    CUDA_CHECK(cudaMemsetAsync(d_full_dual_counter, 0, sizeof(unsigned long long), proc_stream));
    if (d_partial_dual_counter) {
        CUDA_CHECK(cudaMemsetAsync(d_partial_dual_counter, 0, sizeof(unsigned long long), proc_stream));
    }

    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    kernels::batchedFactorizationKernel<<<blocks, threads, 0, proc_stream>>>(
        d_processing_buffer,
        count,
        dev_common,
        config,
        N,
        full_view,
        partial_view,
	d_full_dual_counter,
	d_partial_dual_counter	
    );

    kernels::commit_dual_counter_kernel<<<1,1,0,proc_stream>>>(
	(const uint64_t*)d_full_dual_counter,
	full_view.global_count,
	full_view.global_factor_idx
   );

    if (d_partial_dual_counter && partial_view.global_count != nullptr) {
        kernels::commit_dual_counter_kernel<<<1,1,0,proc_stream>>>(
            (const uint64_t*)d_partial_dual_counter,
	    partial_view.global_count,
	    partial_view.global_factor_idx
        );
    }
    // Publish buffer fill telemetry (zero-sync, sub-microsecond)
    if (d_buffer_fill_) {
        kernels::publish_buffer_fill_kernel<<<1, 1, 0, proc_stream>>>(
            d_persistent_batch ? d_persistent_batch->getDeviceCountPtr() : nullptr,
            d_partial_batch    ? d_partial_batch->getDeviceCountPtr()    : nullptr,
            d_current_accumulation_counter,
            d_buffer_fill_
        );
    }

    // Wait for stream to terminate.
    CUDA_CHECK(cudaStreamSynchronize(proc_stream));
}

// -----------------------------------------------------------------------------
// Consolidate to Persistent (Device-to-Device Append)
// -----------------------------------------------------------------------------

void DevicePostProcessingController::consolidateToPersistent() {
    if (!d_persistent_batch || !d_full_batch) return;

    // 1. Get count of new Full relations generated in this batch
    // This reads the atomic counter from d_full_batch (async copy internally)
    const uint64_t new_full_count = d_full_batch->getCount(proc_stream);

    if (new_full_count == 0) return;
    
    LOG(LOG_DEBUG_2) << "Consolidating " << new_full_count << " full relations to persistent storage.";
    
    // 2. Append directly on device
    // Note: Dereference *d_full_batch to pass by reference
    d_persistent_batch->append(*d_full_batch, new_full_count, proc_stream);
        
    // 3. Reset the temporary batch counters for the next run.
    // This effectively "clears" d_full_batch without reallocating memory.
    // (Note: We do NOT clear d_partial_batch here; the LargePrime variant handles that lifecycle)
    d_full_batch->reset_counters(proc_stream);
}

uint32_t DevicePostProcessingController::getPartialCount() {
    // 1. Safety check: Ensure the batch and its counter exist
    if (!d_partial_batch) return 0;

    uint64_t count = d_partial_batch->getCount(proc_stream);

    return count;
}

void DevicePostProcessingController::resetPartialBatch() {
    resetPartialBatch(proc_stream);
}

void DevicePostProcessingController::resetPartialBatch(cudaStream_t stream) {
    if (d_partial_batch) {
        d_partial_batch->reset_counters(stream);
    }
    if (d_partial_dual_counter) {
        CUDA_CHECK(cudaMemsetAsync(d_partial_dual_counter, 0,
                                    sizeof(unsigned long long), stream));
    }
}

void DevicePostProcessingController::resyncPersistentDualCounter() {
    if (!d_persistent_batch || !d_persistent_dual_counter) return;

    auto view = d_persistent_batch->get_view();
    kernels::sync_dual_counter_from_batch_kernel<<<1, 1, 0, proc_stream>>>(
        (uint64_t*)d_persistent_dual_counter,
        view.global_count,
        view.global_factor_idx
    );

    // Also update the pinned telemetry counter
    if (h_pinned_persistent_count && view.global_count) {
        cudaMemcpyAsync(
            (void*)h_pinned_persistent_count,
            view.global_count,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            proc_stream
        );
    }
}

uint64_t DevicePostProcessingController::getPersistentCount() {
    // 1. Safety check: Ensure the batch and its counter exist
    if (!d_persistent_batch) {
        return 0;
    }

    uint64_t count = d_persistent_batch->getCount(proc_stream);

    return count;
}

void DevicePostProcessingController::flush() {
    // 1. Read the current active counter (Async on proc_stream)
    uint32_t final_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &final_count, 
        d_current_accumulation_counter, 
        sizeof(uint32_t), 
        cudaMemcpyDeviceToHost, 
        proc_stream
    ));
    
    // Wait for value
    CUDA_CHECK(cudaStreamSynchronize(proc_stream));

    // 2. Update Host State
    accumulation_count = final_count;

    // 3. Process
    if (accumulation_count > 0) {
        processBufferedCandidates();
        consolidateToPersistent();
    }
}

// -----------------------------------------------------------------------------
// Controller Logic
// -----------------------------------------------------------------------------

void DevicePostProcessingController::deduplicatePersistentBatch() {
    if (!d_persistent_batch) return;

    uint32_t current_size = (uint32_t)d_persistent_batch->getCount(proc_stream);
    LOG(LOG_STATS) << "Deduplicating " << current_size << " relations on GPU...";

    // 1. Allocate Temp Buffers for Hashing & Sorting
    // (We keep using device_vector for local temporary storage as it handles cleanup automatically)
    thrust::device_vector<uint64_t> d_hashes(current_size);
    thrust::device_vector<uint32_t> d_indices(current_size);

    // Initialize indices [0, 1, ..., N-1]
    thrust::sequence(thrust::cuda::par.on(proc_stream), d_indices.begin(), d_indices.end());

    // 2. Compute Hashes
    int blockSize = 256;
    int numBlocks = (current_size + blockSize - 1) / blockSize;

    // Note: get_view() works directly
    kernels::compute_relation_hashes_soa<<<numBlocks, blockSize, 0, proc_stream>>>(
        d_persistent_batch->get_view(),
        current_size,
        thrust::raw_pointer_cast(d_hashes.data())
    );
    CUDA_CHECK(cudaGetLastError());

    // 3. Sort by Hash (Keys=Hashes, Values=Indices)
    try {
        thrust::sort_by_key(
            thrust::cuda::par.on(proc_stream),
            d_hashes.begin(),
            d_hashes.end(),
            d_indices.begin()
        );
    } catch (std::exception& e) {
        LOG(LOG_ERROR_CRITICAL) << "Thrust Sort Error in dedup: " << e.what();
    }

    // 4. Unique by Hash
    auto new_end_pair = thrust::unique_by_key(
        thrust::cuda::par.on(proc_stream),
        d_hashes.begin(),
        d_hashes.end(),
        d_indices.begin()
    );

    uint32_t new_size = (uint32_t)(new_end_pair.first - d_hashes.begin());
    uint32_t duplicates = current_size - new_size;

    if (duplicates == 0) {
        LOG(LOG_DEBUG_1) << "No duplicates found.";
        return;
    }

    LOG(LOG_STATS) << "Removed " << duplicates << " duplicate relations (" << (100.0*(float)duplicates/(float)current_size) << "%). New count: " << new_size;

#ifdef DEBUG_SOA    
    uint64_t nrels = d_persistent_batch->getCount(proc_stream);

    // Read nfacts from device counter pointer (global_factor_idx)
    mpqs::structures::RelationBatchView pview = d_persistent_batch->get_view();

    uint64_t nfacts = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &nfacts,
	pview.global_factor_idx,                 // device pointer -> copy value
	sizeof(uint64_t),
	cudaMemcpyDeviceToHost,
	proc_stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(proc_stream));

    if (nrels == 0) return;

    // device flag: first bad index (uint32_t), init to 0xFFFFFFFF
    uint32_t* d_bad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bad, sizeof(uint32_t)));
    
    const uint32_t BAD_INIT = 0xFFFFFFFFu;
    CUDA_CHECK(cudaMemcpyAsync(d_bad, &BAD_INIT, sizeof(uint32_t), cudaMemcpyHostToDevice, proc_stream));
    CUDA_CHECK(cudaStreamSynchronize(proc_stream));
    
    // launch check kernel on persistent offsets
    const uint64_t* d_off = d_persistent_batch->getFactorOffsetsData(); // device pointer [file:3]
    
    int block = 256;
    int grid  = (int)((nrels + block - 1) / block);
    
    kernels::find_first_bad_offset<<<grid, block, 0, proc_stream>>>(
	d_off,
	nrels,
	nfacts,
	d_bad
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(proc_stream));

    // copy back result
    uint32_t bad_i = 0xFFFFFFFFu;
    CUDA_CHECK(cudaMemcpyAsync(&bad_i, d_bad, sizeof(uint32_t), cudaMemcpyDeviceToHost, proc_stream));
    CUDA_CHECK(cudaStreamSynchronize(proc_stream));

    if (bad_i != 0xFFFFFFFFu) {
        uint64_t off_i = 0, off_ip1 = 0;

	CUDA_CHECK(cudaMemcpyAsync(
            &off_i,
	    d_off + (uint64_t)bad_i,
	    sizeof(uint64_t),
	    cudaMemcpyDeviceToHost,
	    proc_stream
        ));
	CUDA_CHECK(cudaMemcpyAsync(
            &off_ip1,
	    d_off + (uint64_t)bad_i + 1,
	    sizeof(uint64_t),
	    cudaMemcpyDeviceToHost,
	    proc_stream
        ));
	CUDA_CHECK(cudaStreamSynchronize(proc_stream));

        LOG(LOG_ERROR_CRITICAL)
	  << "Bad CSR offsets at i=" << bad_i
	  << " off[i]=" << off_i
	  << " off[i+1]=" << off_ip1
	  << " (nrels=" << nrels << ", nfacts=" << nfacts << ")";
	
	// optional: bail out early to avoid allocating insane buffers
	CUDA_CHECK(cudaFree(d_bad));
	return;
    }

    CUDA_CHECK(cudaFree(d_bad));
#endif

    
    // 5. Compact / Rebuild SoA Batch
    
    // A. Calculate new factor storage requirement
    thrust::device_vector<uint32_t> d_new_lengths(new_size);
    thrust::device_vector<uint64_t> d_new_offsets(new_size + 1);

    numBlocks = (new_size + blockSize - 1) / blockSize;
    
    // Note: getFactorOffsetsData() now returns uint64_t*, no cast needed
    kernels::compute_new_lengths_kernel<<<numBlocks, blockSize, 0, proc_stream>>>(
        d_persistent_batch->getFactorOffsetsData(),
        thrust::raw_pointer_cast(d_indices.data()),
        new_size,
        thrust::raw_pointer_cast(d_new_lengths.data())
    );

    // B. Scan lengths to get offsets
    thrust::exclusive_scan(
        thrust::cuda::par.on(proc_stream),
        d_new_lengths.begin(),
        d_new_lengths.end(),
        d_new_offsets.begin()
    );

    // Get total factors
    uint64_t last_offset;
    uint32_t last_len;
    // Helper to read from device vector pointer arithmetic
    thrust::device_ptr<uint64_t> ptr_off = d_new_offsets.data();
    thrust::device_ptr<uint32_t> ptr_len = d_new_lengths.data();
    
    CUDA_CHECK(cudaMemcpyAsync(&last_offset, thrust::raw_pointer_cast(ptr_off + new_size - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost, proc_stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_len,    thrust::raw_pointer_cast(ptr_len + new_size - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, proc_stream));
    CUDA_CHECK(cudaStreamSynchronize(proc_stream));
    
    uint64_t total_new_factors = last_offset + last_len;

    LOG(LOG_DEBUG_1) << "Total new factors: " << total_new_factors;
    
    // Write the final total to the end of d_new_offsets
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(ptr_off + new_size), &total_new_factors, sizeof(uint64_t), cudaMemcpyHostToDevice, proc_stream));

    // C. Allocate New Batch
    auto d_clean_batch = std::make_unique<mpqs::structures::RelationBatch>();
    d_clean_batch->initiate(config.device_id);
    d_clean_batch->resize(new_size, total_new_factors);
    
    // Copy the new offsets into the batch structure
    // We wrap the raw destination pointer for Thrust
    thrust::device_ptr<uint64_t> dest_offsets_ptr(d_clean_batch->getFactorOffsetsData());
    thrust::copy(
        thrust::cuda::par.on(proc_stream), 
        d_new_offsets.begin(), 
        d_new_offsets.end(), 
        dest_offsets_ptr
    );

    // D. Gather Data
    kernels::gather_soa_relations_kernel<<<numBlocks, blockSize, 0, proc_stream>>>(
        d_persistent_batch->get_view(),
        d_clean_batch->get_view(),
        thrust::raw_pointer_cast(d_indices.data()),
        new_size,
        d_clean_batch->getFactorOffsetsData()
    );
#ifdef DEBUG_SOA    
    CUDA_CHECK(cudaGetLastError());
#endif

    // 6. UPDATE ATOMIC COUNTERS
    // The d_clean_batch was resized but its atomic counters are 0.
    // We must manually update them to reflect the data we just gathered.
    auto view = d_clean_batch->get_view();
    uint64_t val_rels = new_size;
    uint64_t val_facts = total_new_factors;
    
    CUDA_CHECK(cudaMemcpyAsync(view.global_count,      &val_rels,  sizeof(uint64_t), cudaMemcpyHostToDevice, proc_stream));
    CUDA_CHECK(cudaMemcpyAsync(view.global_factor_idx, &val_facts, sizeof(uint64_t), cudaMemcpyHostToDevice, proc_stream));

    CUDA_CHECK(cudaStreamSynchronize(proc_stream));

    // 7. Swap
    d_persistent_batch = std::move(d_clean_batch);
}

#ifdef DEBUG_SOA
void DevicePostProcessingController::debugDumpHead(int n) {
    auto view = d_persistent_batch->get_view();
    std::vector<uint64_t> h_offsets(n + 1);

    // Copy first N offsets
    cudaMemcpy(h_offsets.data(), view.factor_offsets, (n+1)*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    LOG(LOG_DEBUG_1) << "[Debug] First " << n << " relations offsets:";
    for(int i=0; i<n; ++i) {
        uint64_t start = h_offsets[i];
        uint64_t end = h_offsets[i+1];
        LOG(LOG_DEBUG_1) << " Rel " << i << ": [" << start << ", " << end << ") Len=" << (end-start);

        // Sanity Check
        if (end < start) LOG(LOG_ERROR_CRITICAL) << " DATA CORRUPTION: Negative length";
        if (end - start > 64) LOG(LOG_WARNING) << " SUSPICIOUS: Length > 64";
    }
}
#endif

// -----------------------------------------------------------------------------
// Batch-Sieve Factorization (Zero-Sync Path)
// -----------------------------------------------------------------------------

void DevicePostProcessingController::processBatchBufferedCandidates() {
    if (!d_persistent_batch || !d_persistent_dual_counter) {
        LOG(LOG_ERROR_CRITICAL) << "processBatchBufferedCandidates() called without persistent batch";
        return;
    }

    // 1. Get the buffer that was just filled by the Siever
    DoubleBuffer* active_buffer = &buffers[active_accum_idx];

    // 2. Wait for the Siever to finish writing to this buffer
    cudaStreamWaitEvent(proc_stream, active_buffer->safe_to_read_event, 0);

    processing_active = true;

    // 3. Prepare SoA views
    auto persistent_view = d_persistent_batch->get_view();

    // Partial view: populated when LP is enabled, empty otherwise.
    // The kernel's append_to_soa() has a null-pointer guard on view.global_count,
    // so an empty view is a safe no-op.
    mpqs::structures::RelationBatchView partial_view = {};
    if (d_partial_batch) {
        partial_view = d_partial_batch->get_view();
    }

    // Note: Neither d_persistent_dual_counter nor d_partial_dual_counter are reset
    // between calls — both accumulate across all batch processing invocations.
    // commit_dual_counter_kernel does ASSIGNMENT (=), not addition (+=), so the
    // cumulative dual counter value is written directly to global_count each time.

    // 4. Launch SoA factorization kernel (reads count from device pointer)
    int max_threads = 256;
    int max_blocks = (active_buffer->capacity + max_threads - 1) / max_threads;

    kernels::batchedBatchFactorizationKernelSoA<<<max_blocks, max_threads, 0, proc_stream>>>(
        active_buffer->d_data,
        active_buffer->d_counter,   // device pointer — zero CPU/GPU sync
        dev_common,
        config,
        N,
        persistent_view,
        partial_view,
        d_persistent_dual_counter,
        d_partial_dual_counter      // nullptr when LP disabled; kernel handles safely
    );

    // 5. Commit dual counters → update batch internal counters
    kernels::commit_dual_counter_kernel<<<1, 1, 0, proc_stream>>>(
        (const uint64_t*)d_persistent_dual_counter,
        persistent_view.global_count,
        persistent_view.global_factor_idx
    );

    if (d_partial_dual_counter && partial_view.global_count) {
        kernels::commit_dual_counter_kernel<<<1, 1, 0, proc_stream>>>(
            (const uint64_t*)d_partial_dual_counter,
            partial_view.global_count,
            partial_view.global_factor_idx
        );

        // Async copy partial count to pinned memory for LP input hint (zero-sync)
        if (h_pinned_partial_count) {
            cudaMemcpyAsync(
                (void*)h_pinned_partial_count,
                partial_view.global_count,
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                proc_stream
            );
        }
    }

    // 6. Telemetry: async copy counters to pinned memory for CPU polling
    cudaMemcpyAsync(
        (void*)h_pinned_accumulation_counter,
        active_buffer->d_counter,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        proc_stream
    );

    if (h_pinned_persistent_count && persistent_view.global_count) {
        // Copy lower 32 bits of the 64-bit relation count (little-endian)
        cudaMemcpyAsync(
            (void*)h_pinned_persistent_count,
            persistent_view.global_count,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            proc_stream
        );
    }

    // 7. Yield prediction kernel (adaptive convergence)
    if (prediction_target_ > 0 && d_prediction_result) {
        kernels::yield_prediction_kernel<<<1, 1, 0, proc_stream>>>(
            d_prediction_result,
            persistent_view.global_count,
            prediction_target_,
            prediction_total_steps_,
            d_lp_stats_
        );
    }

    // 8. Publish buffer fill telemetry (zero-sync, sub-microsecond)
    if (d_buffer_fill_) {
        kernels::publish_buffer_fill_kernel<<<1, 1, 0, proc_stream>>>(
            d_persistent_batch ? d_persistent_batch->getDeviceCountPtr() : nullptr,
            d_partial_batch    ? d_partial_batch->getDeviceCountPtr()    : nullptr,
            active_buffer->d_counter,
            d_buffer_fill_
        );
    }

    // 9. Reset the Device Counter for the NEXT time the Siever uses this buffer
    cudaMemsetAsync(active_buffer->d_counter, 0, sizeof(uint32_t), proc_stream);

    // 10. Record safe_to_write_event — tells Siever it's safe to overwrite
    cudaEventRecord(active_buffer->safe_to_write_event, proc_stream);

    // 11. Toggle active buffer index
    toggleActiveBuffer();
}

void DevicePostProcessingController::requestStats(cudaStream_t stream) {
    if (d_partial_batch) d_partial_batch->requestStats(stream);
    if (d_persistent_batch) d_persistent_batch->requestStats(stream);
}

void DevicePostProcessingController::updateStats() {
    if (d_partial_batch) { d_partial_batch->updateStats(proc_stream); }
    if (d_persistent_batch) { d_persistent_batch->updateStats(proc_stream); }
}
 
std::pair<uint64_t, uint64_t> DevicePostProcessingController::getPartialStats() const {
    return d_partial_batch ? d_partial_batch->readStats() : std::make_pair(0UL, 0UL);
}

std::pair<uint64_t, uint64_t> DevicePostProcessingController::getPersistentStats() const {
    return d_persistent_batch ? d_persistent_batch->readStats() : std::make_pair(0UL, 0UL);
}

} // namespace postprocessing
} // namespace mpqs
