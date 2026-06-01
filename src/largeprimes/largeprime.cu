// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski

#include "largeprime.h"
#include "cuda_check.h"
#include <cstdio>
#include <stdexcept>
#ifdef LP_DEBUG
#include <set>
#endif

namespace mpqs {
namespace lp {  

// -----------------------------------------------------------------------------
// Core Architecture Constants
// -----------------------------------------------------------------------------

/// @brief 128-bytes (exactly one standard L2 Cache Line). 
/// Essential for triggering vectorized loads (ulonglong4) during Hash Probing.
constexpr uint32_t ROW_WIDTH_ELEMS = 16; 

/// @brief Absolute ceiling on merged factors to prevent structural overflows.
constexpr int MAX_MERGE_FACTORS = 64;

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

namespace kernels {

// -----------------------------------------------------------------------------
// Atomic Primitives & CSR Merging
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint64_t pack_rf(uint32_t R, uint32_t F) {
    return (uint64_t(R) << 32) | uint64_t(F);
}

__device__ __forceinline__ void unpack_rf(uint64_t x, uint32_t& R, uint32_t& F) {
    R = uint32_t(x >> 32);
    F = uint32_t(x & 0xFFFFFFFFULL);
}

/**
 * @brief Merges two sorted arrays of factors into a local buffer.
 * Includes strictly safe bounds checking.
 */
__device__ __forceinline__
uint32_t merge_factors_csr(
    const uint32_t* __restrict__ idx_a, const uint8_t* __restrict__ cnt_a, uint32_t len_a,
    const uint32_t* __restrict__ idx_b, const uint8_t* __restrict__ cnt_b, uint32_t len_b,
    uint32_t* out_idx, uint8_t* out_cnt, uint32_t max_cap
) {
    uint32_t i = 0, j = 0, k = 0;

    while (i < len_a && j < len_b) {
        if (k >= max_cap) break; // Hard stop on overflow

        uint32_t val_a = idx_a[i];
        uint32_t val_b = idx_b[j];

        if (val_a < val_b) {
            out_idx[k] = val_a;
            out_cnt[k] = cnt_a[i];
            i++;
        } else if (val_b < val_a) {
            out_idx[k] = val_b;
            out_cnt[k] = cnt_b[j];
            j++;
        } else {
            // Collision on factor: Sum exponents
            out_idx[k] = val_a;
            out_cnt[k] = cnt_a[i] + cnt_b[j];
            i++; j++;
        }
        k++;
    }

    // Flush remaining
    while (i < len_a && k < max_cap) { out_idx[k] = idx_a[i]; out_cnt[k] = cnt_a[i]; i++; k++; }
    while (j < len_b && k < max_cap) { out_idx[k] = idx_b[j]; out_cnt[k] = cnt_b[j]; j++; k++; }

    return k;
}

/**
 * @brief Atomically reserves a slot for a relation and its factors.
 * Ensures strictly ordered consistency between relation index and factor offset,
 * which is mathematically required for a valid CSR `offsets` array.
 */
__device__ __forceinline__
void atomic_reserve_dual(
    uint64_t* counter,
    uint32_t factors_to_add,
    uint32_t max_rels,
    uint32_t max_factors,
    uint32_t* out_rel_idx,
    uint32_t* out_factor_offset,
    bool* success
) {
    unsigned long long old = atomicAdd((unsigned long long*)counter, 0ULL);
    while(true) {
        uint32_t r, f;
        unpack_rf(old, r, f);
        
        // Bounds Check
        if (r >= max_rels || (f + factors_to_add) > max_factors) {
            *success = false;
            return;
        }
        
        unsigned long long desired = pack_rf(r + 1, f + factors_to_add);
        unsigned long long prev = atomicCAS((unsigned long long*)counter, old, desired);
        
        if (prev == old) {
            *out_rel_idx = r;
            *out_factor_offset = f;
            *success = true;
            return;
        }
        old = prev;
    }
}

// --- Sync Kernel for Dual Counter ---
__global__ void sync_dual_counter_kernel(
    uint64_t* dual_counter,
    uint64_t* global_rel_ptr,
    uint64_t* global_fact_ptr,
    bool pack
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (pack) {
            *dual_counter = pack_rf((uint32_t)*global_rel_ptr, (uint32_t)*global_fact_ptr);
        } else {
            uint32_t r, f;
            unpack_rf(*dual_counter, r, f);
            *global_rel_ptr = r;
            *global_fact_ptr = f;
        }
    }
}

// -----------------------------------------------------------------------------
// Diagnostic: Validate combine kernel inputs (LP match, factor integrity)
// -----------------------------------------------------------------------------

struct CombineDiagEntry {
    uint32_t rel_idx;        // Output relation index
    uint32_t target_idx;     // Global witness index
    uint64_t input_lp;       // LP from input partial
    uint64_t witness_lp;     // LP from witness store
    uint32_t input_flen;     // Factor count from input
    uint32_t witness_flen;   // Factor count from witness
    uint64_t input_fstart;   // Factor offset start from input
    uint64_t witness_fstart; // Factor offset start from witness
    uint8_t error_type;      // 1=LP mismatch, 2=bad input offsets, 3=bad witness offsets
};

/// @brief Diagnostic kernel: validates combine inputs for MATCH_FOUND elements.
/// Writes up to max_entries diagnostic entries for detected anomalies.
__global__ void diagnose_combine_inputs_kernel(
    const SLPStatus* __restrict__ status_flags,
    const uint32_t* __restrict__ target_idx_array,
    mpqs::structures::RelationBatchView input_view,
    mpqs::structures::RelationBatchView global_witness_view,
    const uint64_t* __restrict__ d_count,
    CombineDiagEntry* __restrict__ diag_out,
    uint32_t* __restrict__ diag_count,
    uint32_t max_entries,
    uint32_t* __restrict__ total_match_count
) {
    uint32_t num_items = (uint32_t)*d_count;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    if (status_flags[tid] != SLPStatus::MATCH_FOUND) return;

    atomicAdd(total_match_count, 1);

    uint32_t my_idx = tid;
    uint32_t target_global_idx = target_idx_array[tid];

    uint64_t input_lp = (uint64_t)input_view.large_primes[my_idx];
    uint64_t witness_lp = (uint64_t)global_witness_view.large_primes[target_global_idx];

    uint64_t start_A = input_view.factor_offsets[my_idx];
    uint64_t end_A   = input_view.factor_offsets[my_idx+1];
    uint64_t start_B = global_witness_view.factor_offsets[target_global_idx];
    uint64_t end_B   = global_witness_view.factor_offsets[target_global_idx+1];

    uint8_t error = 0;
    if (input_lp != witness_lp) error = 1;
    else if (end_A < start_A || (end_A - start_A) > 200) error = 2;
    else if (end_B < start_B || (end_B - start_B) > 200) error = 3;

    if (error > 0) {
        uint32_t idx = atomicAdd(diag_count, 1);
        if (idx < max_entries) {
            diag_out[idx].rel_idx = my_idx;
            diag_out[idx].target_idx = target_global_idx;
            diag_out[idx].input_lp = input_lp;
            diag_out[idx].witness_lp = witness_lp;
            diag_out[idx].input_flen = (uint32_t)(end_A - start_A);
            diag_out[idx].witness_flen = (uint32_t)(end_B - start_B);
            diag_out[idx].input_fstart = start_A;
            diag_out[idx].witness_fstart = start_B;
            diag_out[idx].error_type = error;
        }
    }
}

// -----------------------------------------------------------------------------
// Device-side count snapshot (Stage 3 async: avoids host readback of input count)
// -----------------------------------------------------------------------------

/// @brief Copies the device-side atomic counter into a local device variable.
/// Single-thread kernel — latency is irrelevant since it serializes nothing.
__global__ void snapshot_count_kernel(uint64_t* __restrict__ dst,
                                      const uint64_t* __restrict__ src) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dst = *src;
    }
}

// -----------------------------------------------------------------------------
// Stage 3 Directory Fetch & State Packing Kernel
// -----------------------------------------------------------------------------

/**
 * @brief Directory Fetch Kernel: Computes hash/tag from LP values, fetches Directory
 *        state, and packs 64-bit routing keys.
 *
 * Mathematical Packing Model (64-bit Key K):
 * [ Status (4 bits) | RowIdx (28 bits) | Tag (32 bits) ]
 * Bit ranges:
 *   60-63 : Status Flag (SLPStatus)
 *   32-59 : Global RowIdx (Slab Index). Max 2^28 (~268 Million) rows.
 *    0-31 : Hash Tag
 *
 * Hash/Tag derivation from large prime p:
 *   p_shifted = p >> 1   (discard parity bit)
 *   H = p_shifted & ((1 << B) - 1)
 *   T = p_shifted >> B
 */
__global__ void directory_fetch_kernel(
    const unsigned __int128* __restrict__ large_primes,
    uint64_t* __restrict__ keys_out,            // Packed Routing Key
    SLPStatus* __restrict__ status_flags,       // Output State
    const uint64_t* __restrict__ directory,
    const uint64_t* __restrict__ d_count,
    uint32_t hash_bits
) {
    uint32_t num_items = (uint32_t)*d_count;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    // Compute hash and tag directly from the large prime value
    uint64_t p = (uint64_t)large_primes[tid];
    uint64_t p_shifted = p >> 1;
    uint64_t hash_mask = (1ULL << hash_bits) - 1;
    uint64_t hash = p_shifted & hash_mask;
    uint32_t tag  = (uint32_t)(p_shifted >> hash_bits);

    // Fetch directory state
    uint64_t dir_entry = directory[hash];
    uint32_t count = (dir_entry >> 48) & 0x7FFF;
    uint64_t row_idx = dir_entry & 0xFFFFFFFFULL;

    SLPStatus status;
    if (count == 0) {
        status = SLPStatus::NEEDS_ALLOCATION;
    } else {
        status = SLPStatus::NEEDS_PROBE;
    }
    status_flags[tid] = status;

    // Pack: [Status: 4 bits | RowIdx: 28 bits | Tag: 32 bits]
    uint64_t packed_out = ((uint64_t)static_cast<uint8_t>(status) << 60) |
                          ((row_idx & 0xFFFFFFFULL) << 32) |
                          tag;
    keys_out[tid] = packed_out;
}
 
// -----------------------------------------------------------------------------
// Stage 4 Vectorized Hash Probe Kernel
// -----------------------------------------------------------------------------

/**
 * @brief Stage 4 Kernel: Executes Vectorized Probing against the Payload Slabs.
 * 
 * Mathematical Memory Model:
 * Each Row in the Payload Slab contains `ROW_WIDTH_ELEMS` (e.g., 16) 64-bit entries.
 * 16 * 8 bytes = 128 bytes (Exactly 1 L2 Cache Line).
 * 
 * Vectorized Load:
 * NVIDIA hardware's maximum native memory transaction per thread is 128 bits (16 bytes).
 * We cast the row pointer to `ulonglong2` (16 bytes per fetch) to force the compiler 
 * to emit the maximum-width `LDG.E.128` transaction.
 * 
 * Result:
 * If a `Tag` match is found, state transitions to `MATCH_FOUND` and the 
 * Global Witness Index is recorded. Otherwise, it transitions to `NEEDS_APPEND`.
 */
__global__ void probe_hash_table_kernel(
    const uint64_t* __restrict__ keys_in,        // Packed: [Status|RowIdx|Tag]
    SLPStatus* __restrict__ status_flags,        // In/Out State transitions
    uint64_t* __restrict__ payload_slabs,        // Global Slab Matrix (non-const for purge)
    uint32_t* __restrict__ target_idx_array,     // Output match locations
    const uint64_t* __restrict__ d_count,
    uint32_t row_width_elems,
    bool purge_after_match                       // Invalidate matched entry to prevent reuse
) {
    uint32_t num_items = (uint32_t)*d_count;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    // Fetch the current status directly from the packed key
    uint64_t key = keys_in[tid];
    SLPStatus status = static_cast<SLPStatus>(key >> 60);

    // Only threads marked NEEDS_PROBE in Stage 3 execute the heavy memory fetches
    if (status == SLPStatus::NEEDS_PROBE) {

        uint32_t tag = (uint32_t)(key & 0xFFFFFFFFULL);
        uint64_t row_idx = (key >> 32) & 0xFFFFFFFULL;

        // Base pointer to the 128-byte cache line (Row)
        const uint64_t* row_base = &payload_slabs[row_idx * row_width_elems];

        // Use ulonglong2 to trigger the 128-bit hardware maximum load instruction
        const ulonglong2* row_ptr = reinterpret_cast<const ulonglong2*>(row_base);

        // CHUNKS = ROW_WIDTH_ELEMS / 2 (e.g., 16/2 = 8 iterations of 16 bytes)
        constexpr int CHUNKS = ROW_WIDTH_ELEMS / 2;

        bool matched = false;
        uint32_t matched_global_idx = 0xFFFFFFFF;
        uint32_t matched_slot_offset = 0;  // Offset within row for purge

        // Loop unrolling forces the PTX generator to schedule the loads optimally
        #pragma unroll
        for (int i = 0; i < CHUNKS; ++i) {

            // 128-bit Read-Only Cache Load
            ulonglong2 chunk = __ldg(&row_ptr[i]);

            // Compare Tags (Upper 32 bits of each 64-bit entry)
            if ((uint32_t)(chunk.x >> 32) == tag) {
                matched = true; matched_global_idx = (uint32_t)chunk.x;
                matched_slot_offset = 2 * i;
                break;
            }
            if ((uint32_t)(chunk.y >> 32) == tag) {
                matched = true; matched_global_idx = (uint32_t)chunk.y;
                matched_slot_offset = 2 * i + 1;
                break;
            }
        }

        if (matched) {
            if (purge_after_match) {
                // Atomically claim the witness — only one thread succeeds.
                // Zeroing the entry prevents future probes from matching it.
                uint64_t expected = ((uint64_t)tag << 32) | matched_global_idx;
                uint64_t* slot = &payload_slabs[row_idx * row_width_elems + matched_slot_offset];
                uint64_t old = atomicCAS((unsigned long long*)slot,
                                         (unsigned long long)expected,
                                         0ULL);
                if (old == expected) {
                    status = SLPStatus::MATCH_FOUND;
                    target_idx_array[tid] = matched_global_idx;
                } else {
                    // Another thread purged first — fall through as new witness
                    status = SLPStatus::NEEDS_APPEND;
                }
            } else {
                status = SLPStatus::MATCH_FOUND;
                target_idx_array[tid] = matched_global_idx;
            }
        } else {
            status = SLPStatus::NEEDS_APPEND;
        }
    }
    status_flags[tid] = status;
}

// -----------------------------------------------------------------------------
// Stage 5 Global Commit Kernels
// -----------------------------------------------------------------------------

/**
 * @brief Stage 5A Kernel: Assembles Full Relations from Global Hash Matches.
 * 
 * Logic:
 * For every element marked MATCH_FOUND, we retrieve the corresponding Global Witness
 * from the persistent SoA structure and merge it with our local 1-partial.
 */
__global__ void global_combine_kernel(
    const SLPStatus* __restrict__ status_flags,
    const uint32_t* __restrict__ target_idx_array,
    mpqs::structures::RelationBatchView input_view,
    mpqs::structures::RelationBatchView global_witness_view,
    mpqs::structures::RelationBatchView output_view,
    uint64_t* __restrict__ output_dual_counter,
    const uint64_t* __restrict__ d_count,
    mpqs::uint512 N,
    uint64_t* __restrict__ output_overflow_counter
) {
    uint32_t num_items = (uint32_t)*d_count;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    if (status_flags[tid] == SLPStatus::MATCH_FOUND) {
        uint32_t my_idx = tid;
        uint32_t target_global_idx = target_idx_array[tid];

        // Guard: verify LP values actually match (tag collision → different LPs).
        // Tag collisions occur when two LPs hash to the same bucket and have
        // identical truncated tags (common for small LPs where tag = LP >> (B+1) = 0).
        uint64_t lp_input   = (uint64_t)input_view.large_primes[my_idx];
        uint64_t lp_witness = (uint64_t)global_witness_view.large_primes[target_global_idx];
        if (lp_input != lp_witness) return;  // Tag collision, not a true match — skip

        // 1. Algebra Merge
        mpqs::uint512 Q_res = input_view.sqrt_Q[my_idx];
        Q_res.mul_mod(global_witness_view.sqrt_Q[target_global_idx], N);

        // 2. CSR Factor Merge
        uint32_t m_idx[MAX_MERGE_FACTORS];
        uint8_t  m_cnt[MAX_MERGE_FACTORS];
        
        uint64_t start_A = input_view.factor_offsets[my_idx];
        uint32_t len_A   = (uint32_t)(input_view.factor_offsets[my_idx+1] - start_A);
        uint64_t start_B = global_witness_view.factor_offsets[target_global_idx];
        uint32_t len_B   = (uint32_t)(global_witness_view.factor_offsets[target_global_idx+1] - start_B);
        
        uint32_t m_len = merge_factors_csr(
            input_view.factor_indices + start_A, input_view.factor_counts + start_A, len_A,
            global_witness_view.factor_indices + start_B, global_witness_view.factor_counts + start_B, len_B,
            m_idx, m_cnt, MAX_MERGE_FACTORS
        );

        // 3. Output Commitment
        uint32_t pos, f_pos;
        bool success;
        atomic_reserve_dual(output_dual_counter, m_len, output_view.max_relations, (uint32_t)output_view.max_factors, &pos, &f_pos, &success);

        if (success) {
            output_view.sqrt_Q[pos] = Q_res;
            // Sign combination: encoding-agnostic. sign=1 means positive; anything else means negative.
            { bool neg_a = (input_view.signs[my_idx] != 1u), neg_b = (global_witness_view.signs[target_global_idx] != 1u);
              output_view.signs[pos] = (neg_a ^ neg_b) ? static_cast<uint8_t>(0xFF) : static_cast<uint8_t>(1); }
            output_view.val_2_exps[pos] = input_view.val_2_exps[my_idx] + global_witness_view.val_2_exps[target_global_idx];
            output_view.large_primes[pos] = input_view.large_primes[my_idx]; // Original LP stored for reference

            output_view.factor_offsets[pos] = f_pos;
            output_view.factor_offsets[pos+1] = f_pos + m_len;

            for(uint32_t k=0; k<m_len; ++k) {
                output_view.factor_indices[f_pos + k] = m_idx[k];
                output_view.factor_counts[f_pos + k]  = m_cnt[k];
            }
        } else {
            atomicAdd((unsigned long long*)output_overflow_counter, 1ULL);
        }
    }
}

/**
 * @brief Stage 5B Kernel: Atomically Appends unique 1-partials to the Global Hash Table.
 * 
 * Logic (The Spin-Lock):
 * 1. Reserve a slot in the Global Witness SoA structure.
 * 2. Write SoA Payload.
 * 3. Atomically acquire the Directory Bucket Spin-Lock (Bit 63).
 * 4. Write Payload to Slab Row.
 * 5. Release Lock and Increment Count.
 */
__global__ void global_append_kernel(
    const SLPStatus* __restrict__ status_flags,
    const uint64_t* __restrict__ keys_in,         // Packed Key: [Status|RowIdx|Tag]
    mpqs::structures::RelationBatchView input_view,
    mpqs::structures::RelationBatchView global_witness_view,
    uint64_t* __restrict__ directory,
    uint64_t* __restrict__ payload_slabs,
    uint32_t* __restrict__ global_row_allocator,
    uint64_t* __restrict__ witness_dual_counter,
    const uint64_t* __restrict__ d_count,
    uint32_t hash_bits,
    uint32_t row_width_elems,
    uint32_t max_slab_rows,
    uint64_t* __restrict__ slab_overflow_counter,
    uint64_t* __restrict__ witness_overflow_counter
) {
    uint32_t num_items = (uint32_t)*d_count;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    static_assert(false, "global_append_kernel requires SM 7.0+ (independent thread scheduling) for correctness of the atomicCAS spin-lock.");
#endif
    if (tid >= num_items) return;

    SLPStatus status = status_flags[tid];

    if (status == SLPStatus::NEEDS_ALLOCATION || status == SLPStatus::NEEDS_APPEND) {

        uint32_t my_idx = tid;
        uint64_t packed_key = keys_in[tid];
        uint32_t tag = (uint32_t)(packed_key & 0xFFFFFFFFULL);

        // Recover Hash via reverse math (Tag << hash_bits + Hash == p >> 1)
        uint64_t p_shifted = (uint64_t)input_view.large_primes[my_idx] >> 1;
        uint32_t hash = (uint32_t)(p_shifted & ((1ULL << hash_bits) - 1));

        // 1. SoA Reservation
        uint64_t start_A = input_view.factor_offsets[my_idx];
        uint32_t len_A   = (uint32_t)(input_view.factor_offsets[my_idx+1] - start_A);

        uint32_t global_idx, f_pos;
        bool success;
        atomic_reserve_dual(witness_dual_counter, len_A, global_witness_view.max_relations, (uint32_t)global_witness_view.max_factors, &global_idx, &f_pos, &success);

        if (!success) {
            atomicAdd((unsigned long long*)witness_overflow_counter, 1ULL);
            return; // Structural Capacity Exceeded
        }

        // 2. Write SoA Data
        global_witness_view.sqrt_Q[global_idx] = input_view.sqrt_Q[my_idx];
        global_witness_view.signs[global_idx]  = input_view.signs[my_idx];
        global_witness_view.val_2_exps[global_idx] = input_view.val_2_exps[my_idx];
        global_witness_view.large_primes[global_idx] = input_view.large_primes[my_idx];

        global_witness_view.factor_offsets[global_idx]   = f_pos;
        global_witness_view.factor_offsets[global_idx+1] = f_pos + len_A;
        
        for(uint32_t k=0; k<len_A; ++k) {
            global_witness_view.factor_indices[f_pos + k] = input_view.factor_indices[start_A + k];
            global_witness_view.factor_counts[f_pos + k]  = input_view.factor_counts[start_A + k];
        }

        // 3. Atomically Insert into Hash Table Slabs
        uint64_t lock_mask = 1ULL << 63;
        uint64_t payload = ((uint64_t)tag << 32) | global_idx;

        bool spin = true;
        while (spin) {
            uint64_t current_dir = directory[hash];
            
            if (current_dir & lock_mask) {
                // Bucket is locked by another warp/thread. Spin.
                continue; 
            }

            // Attempt to acquire lock
            uint64_t locked_dir = current_dir | lock_mask;
            uint64_t old = atomicCAS((unsigned long long*)&directory[hash], 
                                     (unsigned long long)current_dir, 
                                     (unsigned long long)locked_dir);

            if (old == current_dir) {
                // --- CRITICAL SECTION ACQUIRED ---
                uint32_t count = (uint32_t)((current_dir >> 48) & 0x7FFF); // 15-bit count
                uint64_t row   = current_dir & 0xFFFFFFFFFFFFULL;          // 48-bit row

                if (count == 0) {
                    // Allocator returns monotonic increment
                    row = atomicAdd(global_row_allocator, 1);
                    if (row >= max_slab_rows) {
                        // Slab rows exhausted — release lock, skip write
                        atomicAdd((unsigned long long*)slab_overflow_counter, 1ULL);
                        atomicExch((unsigned long long*)&directory[hash],
                                   (unsigned long long)(current_dir & ~lock_mask));
                        return;
                    }
                }

                if (count < row_width_elems) {
                    // Append payload
                    payload_slabs[row * row_width_elems + count] = payload;
                    count++;
                } else {
                    atomicAdd((unsigned long long*)slab_overflow_counter, 1ULL);
                }

                // Memory Fence: Guarantee Payload visibility before lock release
                __threadfence(); 

                // Release Lock & Update State
                uint64_t new_dir = ((uint64_t)count << 48) | row; 
                atomicExch((unsigned long long*)&directory[hash], (unsigned long long)new_dir);
                
                spin = false;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 4: Device-side SoA Append (replaces host-side RelationBatch::append)
// -----------------------------------------------------------------------------

/// @brief Device-side SoA append: copies relations from src (LP output) to dst (persistent).
/// MUST be launched as <<<1, 256>>> (single block). Thread 0 reads counters and broadcasts
/// via shared memory; all threads copy data via grid-stride loop.
/// If capacity is insufficient, appends 0 relations (conservative — no partial append).
/// @param dst_view   Persistent batch view (destination)
/// @param src_view   LP output batch view (source)
/// @param src_count  Device pointer to source relation count
/// @param src_factor_count Device pointer to source factor count
/// @param dst_count  Device pointer to destination relation count
/// @param dst_factor_count Device pointer to destination factor count
/// @param appended_count_pinned  Mapped pinned memory: write actual appended count for host telemetry
__global__ void device_append_kernel(
    mpqs::structures::RelationBatchView dst_view,
    mpqs::structures::RelationBatchView src_view,
    const uint64_t* __restrict__ src_count,
    const uint64_t* __restrict__ src_factor_count,
    uint64_t* __restrict__ dst_count,
    uint64_t* __restrict__ dst_factor_count,
    uint64_t* __restrict__ appended_count_pinned
) {
    __shared__ uint32_t s_num_rels;
    __shared__ uint32_t s_num_factors;
    __shared__ uint32_t s_dst_base_rel;
    __shared__ uint64_t s_dst_base_factor;
    __shared__ bool s_fits;

    if (threadIdx.x == 0) {
        uint32_t src_rels = (uint32_t)*src_count;
        uint32_t src_facts = (uint32_t)*src_factor_count;

        if (src_rels == 0) {
            s_num_rels = 0;
            s_num_factors = 0;
            s_fits = false;
            if (appended_count_pinned) *appended_count_pinned = 0;
        } else {
            uint32_t dst_current_rels = (uint32_t)*dst_count;
            uint64_t dst_current_facts = *dst_factor_count;

            uint32_t remaining_rels = dst_view.max_relations - dst_current_rels;
            uint64_t remaining_facts = dst_view.max_factors - dst_current_facts;

            if (src_rels > remaining_rels || src_facts > remaining_facts) {
                s_num_rels = 0;
                s_num_factors = 0;
                s_fits = false;
                if (appended_count_pinned) *appended_count_pinned = 0;
            } else {
                s_num_rels = src_rels;
                s_num_factors = src_facts;
                s_dst_base_rel = dst_current_rels;
                s_dst_base_factor = dst_current_facts;
                s_fits = true;

                // Update destination counters
                *dst_count = dst_current_rels + src_rels;
                *dst_factor_count = dst_current_facts + src_facts;

                if (appended_count_pinned) *appended_count_pinned = src_rels;
            }
        }
    }
    __syncthreads();

    if (!s_fits || s_num_rels == 0) return;

    uint32_t num_rels = s_num_rels;
    uint32_t num_factors = s_num_factors;
    uint32_t dst_base_rel = s_dst_base_rel;
    uint64_t dst_base_factor = s_dst_base_factor;

    // Grid-stride copy of per-relation arrays
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;

    for (uint32_t i = tid; i < num_rels; i += stride) {
        dst_view.sqrt_Q[dst_base_rel + i] = src_view.sqrt_Q[i];
        dst_view.signs[dst_base_rel + i] = src_view.signs[i];
        dst_view.val_2_exps[dst_base_rel + i] = src_view.val_2_exps[i];
        dst_view.large_primes[dst_base_rel + i] = src_view.large_primes[i];

        // Rebase factor offsets: src offsets are batch-relative, dst needs absolute offset
        dst_view.factor_offsets[dst_base_rel + i] = src_view.factor_offsets[i] + dst_base_factor;
    }

    // Sentinel: factor_offsets[dst_base_rel + num_rels] marks end of last relation's factors
    if (tid == 0) {
        dst_view.factor_offsets[dst_base_rel + num_rels] =
            src_view.factor_offsets[num_rels] + dst_base_factor;
    }

    // Copy factor arrays (indices + counts)
    for (uint32_t i = tid; i < num_factors; i += stride) {
        dst_view.factor_indices[dst_base_factor + i] = src_view.factor_indices[i];
        dst_view.factor_counts[dst_base_factor + i] = src_view.factor_counts[i];
    }
}

/**
 * @brief Telemetry Kernel: Non-blocking computation of Hash Table fill states.
 * DESIGN: Expected to be launched with EXACTLY 1 Block.
 * Uses a grid-stride loop and shared memory atomic reductions to safely compute stats,
 * then commits them to pinned host memory via a strict generation ticket.
 */
__global__ void update_telemetry_kernel(
    const uint64_t* __restrict__ directory,
    uint32_t num_buckets,
    uint32_t row_width_elems,
    uint64_t current_partials_count,
    uint64_t last_batch_full,
    uint64_t total_full_relations,
    const uint64_t* __restrict__ witness_dual_counter,
    uint64_t prev_witness_count,
    SLPPinnedStats* __restrict__ pinned_stats,
    const uint64_t* __restrict__ slab_overflow_count,
    const uint64_t* __restrict__ witness_overflow_count,
    const uint64_t* __restrict__ output_overflow_count
) {
    // Shared memory for the single block's reduction
    __shared__ uint32_t empty_count_smem;
    __shared__ uint32_t full_count_smem;

    if (threadIdx.x == 0) {
        empty_count_smem = 0;
        full_count_smem = 0;
    }
    __syncthreads();

    // Since gridDim.x is strictly 1, stride is just blockDim.x
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;

    uint32_t local_empty = 0;
    uint32_t local_full = 0;

    // Grid-stride loop over all Hash Buckets
    for (uint32_t i = tid; i < num_buckets; i += stride) {
        uint32_t count = (directory[i] >> 48) & 0x7FFF;
        if (count == 0) local_empty++;
        else if (count >= row_width_elems) local_full++;
    }

    // Safely reduce local counts into shared memory
    atomicAdd(&empty_count_smem, local_empty);
    atomicAdd(&full_count_smem, local_full);
    __syncthreads();

    // The single Master thread commits everything to pinned memory
    if (threadIdx.x == 0) {
        // Read witness count directly from device-side dual counter (upper 32 bits = relation count)
        uint32_t total_witnesses, _f;
        unpack_rf(*witness_dual_counter, total_witnesses, _f);

        pinned_stats->new_partials_buffer_fill  = current_partials_count;
        pinned_stats->total_witnesses           = total_witnesses;
        pinned_stats->total_full_relations      = total_full_relations;
        pinned_stats->last_batch_full_relations = last_batch_full;
        pinned_stats->last_batch_new_witnesses  = total_witnesses - prev_witness_count;

        pinned_stats->empty_hash_buckets = empty_count_smem;
        pinned_stats->full_hash_buckets  = full_count_smem;

        pinned_stats->slab_overflow_count    = *slab_overflow_count;
        pinned_stats->witness_overflow_count = *witness_overflow_count;
        pinned_stats->output_overflow_count  = *output_overflow_count;

        // Mathematical Memory Fence: Guarantee Host sees all fields BEFORE iteration updates
        __threadfence_system();

        pinned_stats->total_iterations += 1;
    }
}

} // namespace kernels
 
// -----------------------------------------------------------------------------
// Constructor & Destructor
// -----------------------------------------------------------------------------

LargePrimeVariant::LargePrimeVariant(const cudaStream_t stream)
  : lp_stream(stream)
{
}

LargePrimeVariant::LargePrimeVariant() : lp_stream(0) {}

LargePrimeVariant::~LargePrimeVariant() {
    clearBuffers();
}


// -----------------------------------------------------------------------------
// Lock-Free Memory Allocation
// -----------------------------------------------------------------------------

void LargePrimeVariant::clearBuffers() {
    // 1. SoA Batches
    d_global_witnesses.reset();
    d_output_batch.reset();

    // 2. Hash Table Core
    if (d_directory) {
        cudaFree(d_directory);
        d_directory = nullptr;
    }
    if (d_payload_slabs) {
        cudaFree(d_payload_slabs);
        d_payload_slabs = nullptr;
    }
    if (d_global_row_allocator) {
        cudaFree(d_global_row_allocator);
        d_global_row_allocator = nullptr;
    }

    // 3. Telemetry
    if (h_pinned_stats) {
        cudaFreeHost(h_pinned_stats);
        h_pinned_stats = nullptr;
        d_pinned_stats = nullptr;
    }

    // 3a. Pinned LP combined counters
    if (h_pinned_lp_combined_count) {
        cudaFreeHost(h_pinned_lp_combined_count);
        h_pinned_lp_combined_count = nullptr;
        d_pinned_lp_combined_count = nullptr;
    }

    // 3a2. Pinned appended count (Stage 4)
    if (h_pinned_appended_count) {
        cudaFreeHost(h_pinned_appended_count);
        h_pinned_appended_count = nullptr;
        d_pinned_appended_count = nullptr;
    }

    // 3b. CUDA Events (Stage 2/3)
    if (partials_ready_event_) { cudaEventDestroy(partials_ready_event_); partials_ready_event_ = nullptr; }
    if (lp_done_event_) { cudaEventDestroy(lp_done_event_); lp_done_event_ = nullptr; }
    if (count_snapshot_event_) { cudaEventDestroy(count_snapshot_event_); count_snapshot_event_ = nullptr; }

    // 3b2. Device-side input count (Stage 3)
    if (d_lp_input_count_) { cudaFree(d_lp_input_count_); d_lp_input_count_ = nullptr; }

    // 3c. Overflow counters
    if (d_slab_overflow_count) { cudaFree(d_slab_overflow_count); d_slab_overflow_count = nullptr; }
    if (d_witness_overflow_count) { cudaFree(d_witness_overflow_count); d_witness_overflow_count = nullptr; }
    if (d_output_overflow_count) { cudaFree(d_output_overflow_count); d_output_overflow_count = nullptr; }

    // Pipeline buffers cleanup (no CUDA_CHECK — may be called from destructor)
    if (d_routing_keys) cudaFree(d_routing_keys);
    d_routing_keys = nullptr;
    pipeline_capacity_ = 0;

    // Cleanup for the status flags (no CUDA_CHECK — destructor path)
    if (d_status_flags)         cudaFree(d_status_flags);
    if (d_output_dual_counter)  cudaFree(d_output_dual_counter);
    if (d_witness_dual_counter) cudaFree(d_witness_dual_counter);
    d_status_flags = nullptr;
    d_output_dual_counter = nullptr;
    d_witness_dual_counter = nullptr;

    // Cleanup for Match Tracking Buffers (no CUDA_CHECK — destructor path)
    if (d_target_idx_array)    cudaFree(d_target_idx_array);
    d_target_idx_array = nullptr;
}

// -----------------------------------------------------------------------------
// Pipeline Buffer Memory Management
// -----------------------------------------------------------------------------

void LargePrimeVariant::resizePipelineBuffers(size_t num_items) {
    // Stage 4: pipeline buffers are pre-allocated in initiate() to max_pipeline_capacity_.
    // Runtime resize is no longer needed. Log a warning if capacity is exceeded (should not happen).
    if (num_items > max_pipeline_capacity_) {
        LOG(LOG_ERROR_CRITICAL) << "Pipeline capacity exceeded: " << num_items
                                << " > " << max_pipeline_capacity_
                                << ". Kernel-level overflow counters will prevent buffer corruption,"
                                << " but excess relations will be silently dropped.";
    }
}

void LargePrimeVariant::initiate(
    const LargePrimeConfig& cfg,
    const mpqs::uint512& modulus
) {
    // Guard against double initialization — cleanup existing state
    if (d_directory) {
        clearBuffers();
    }

    this->config = cfg;
    this->N = modulus;

    // --- 1. Device Setup & Diagnostics ---
    int device_id = (config.device_id != 0) ? config.device_id : 0;
    if (config.device_id == 0) cudaGetDevice(&device_id);
    CUDA_CHECK(cudaSetDevice(device_id));
    this->config.device_id = device_id;

    LOG_SET_MODULE("LargePrime");
    LOG(LOG_DEBUG_1) << "Initializing 2-Stage Slab Hash Table SLP Pipeline on GPU " << device_id;

    // --- 2. Mathematical Boundary Assertions ---
    // Ensure the hash key constraints won't result in truncation.
    // Tag(p) must fit in 32 bits: p < 2^(32 + hash_bits + 1)
    uint32_t safe_hash_limit = 64 - 33; 
    if (config.hash_bits > safe_hash_limit) {
        LOG(LOG_ERROR_CRITICAL) << "Hash bits (" << config.hash_bits 
                                << ") exceeds bounds for 32-bit tag encoding.";
        throw std::invalid_argument("hash_bits too large for mathematically safe tagging.");
    }
    
    // --- 3. Directory Allocation ---
    uint32_t num_buckets = 1 << config.hash_bits;
    size_t dir_size_bytes = num_buckets * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&d_directory, dir_size_bytes));
    // Crucial: 0 implies Lock = 0, Count = 0, Row = 0
    CUDA_CHECK(cudaMemsetAsync(d_directory, 0, dir_size_bytes, lp_stream));

    LOG(LOG_DEBUG_2) << "Allocated Directory: " << num_buckets 
                     << " buckets (" << (dir_size_bytes / (1024.0 * 1024.0)) << " MB)";

    // --- 4. Payload Slab Allocation ---
    // Each bucket maps to at most one row, so rows are capped at 2^hash_bits.
    uint64_t max_rows = std::min(config.max_witness_capacity, (uint64_t)(1u << config.hash_bits));
    max_slab_rows_ = (uint32_t)max_rows;
    size_t slab_size_bytes = max_rows * ROW_WIDTH_ELEMS * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&d_payload_slabs, slab_size_bytes));
    // Initialize all slab entries with sentinel tag 0xFFFFFFFF (upper 32 bits).
    // The probe kernel scans the full row without bounds-checking the directory count,
    // so uninitialized entries with tag=0 would falsely match any LP with tag=0
    // (i.e., LP < 2^(hash_bits+1)). Filling with 0xFF sets tag=0xFFFFFFFF,
    // which no real LP can match.
    CUDA_CHECK(cudaMemsetAsync(d_payload_slabs, 0xFF, slab_size_bytes, lp_stream));

    LOG(LOG_DEBUG_2) << "Allocated Payload Slabs: " << max_rows 
                     << " rows (" << (slab_size_bytes / (1024.0 * 1024.0)) << " MB)";

    // --- 5. Row Allocator ---
    CUDA_CHECK(cudaMalloc(&d_global_row_allocator, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_global_row_allocator, 0, sizeof(uint32_t), lp_stream));

    // --- 6. Telemetry Allocation (Pinned Host Memory) ---
    CUDA_CHECK(cudaHostAlloc((void**)&h_pinned_stats, sizeof(SLPPinnedStats), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_pinned_stats, h_pinned_stats, 0));
    memset((void*)h_pinned_stats, 0, sizeof(SLPPinnedStats));

    // --- 6a. Pinned counters for LP combined output (zero-sync readback) ---
    CUDA_CHECK(cudaHostAlloc((void**)&h_pinned_lp_combined_count, 2 * sizeof(uint64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_pinned_lp_combined_count, h_pinned_lp_combined_count, 0));
    memset(h_pinned_lp_combined_count, 0, 2 * sizeof(uint64_t));

    // --- 6b. CUDA Events for async signaling (Stage 2/3) ---
    CUDA_CHECK(cudaEventCreateWithFlags(&partials_ready_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&lp_done_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&count_snapshot_event_, cudaEventDisableTiming));

    // --- 6b2. Device-side input count for async path (Stage 3) ---
    CUDA_CHECK(cudaMalloc(&d_lp_input_count_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_lp_input_count_, 0, sizeof(uint64_t)));

    // --- 6c. Overflow Counters (device atomics, cumulative, never reset) ---
    CUDA_CHECK(cudaMalloc(&d_slab_overflow_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_slab_overflow_count, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_witness_overflow_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_witness_overflow_count, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_overflow_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_output_overflow_count, 0, sizeof(uint64_t)));

    // --- 7. Global Witness SoA Batch (Append Only) ---
    size_t est_factors = config.max_witness_capacity * 64; 
    
    d_global_witnesses = std::make_unique<mpqs::structures::RelationBatch>();
    d_global_witnesses->initiate(device_id);
    d_global_witnesses->resize(config.max_witness_capacity, est_factors);
    d_global_witnesses->reset_counters(lp_stream);

    // --- 8. Output Batch ---
    size_t out_est_factors = config.max_combined_output * 64;
    d_output_batch = std::make_unique<mpqs::structures::RelationBatch>();
    d_output_batch->initiate(device_id);
    d_output_batch->resize(config.max_combined_output, out_est_factors);
    d_output_batch->reset_counters(lp_stream);

    // --- 9. Pre-allocate pipeline buffers to max capacity (Stage 4: no runtime resize) ---
    max_pipeline_capacity_ = std::max((size_t)65536, (size_t)config.max_witness_capacity);
    CUDA_CHECK(cudaMalloc(&d_routing_keys,     max_pipeline_capacity_ * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_status_flags,     max_pipeline_capacity_ * sizeof(SLPStatus)));
    CUDA_CHECK(cudaMalloc(&d_target_idx_array, max_pipeline_capacity_ * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_output_dual_counter, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_witness_dual_counter, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_witness_dual_counter, 0, sizeof(uint64_t)));
    pipeline_capacity_ = max_pipeline_capacity_;

    LOG(LOG_DEBUG_2) << "Pre-allocated pipeline buffers: " << max_pipeline_capacity_ << " capacity.";

    // --- 10. Pinned counter for device_append_kernel telemetry (Stage 4) ---
    CUDA_CHECK(cudaHostAlloc((void**)&h_pinned_appended_count, sizeof(uint64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_pinned_appended_count, h_pinned_appended_count, 0));
    *h_pinned_appended_count = 0;

    // Sync pipeline allocation to guarantee visibility before any Sieve dispatches.
    CUDA_CHECK(cudaStreamSynchronize(lp_stream));
    LOG(LOG_DEBUG_1) << "SLP Memory Architecture Initialization Complete.";
}

// -----------------------------------------------------------------------------
// Accessors (Non-Blocking Telemetry)
// -----------------------------------------------------------------------------

void LargePrimeVariant::requestStats() {
    // Left empty/no-op. The new pipeline maintains stats asynchronously.
    // Method kept for API compatibility.
}

void LargePrimeVariant::updateStats() {
    if (!d_pinned_stats || !d_directory || !d_global_witnesses) return;

    uint32_t num_buckets = 1 << config.hash_bits;
    uint32_t threads = 1024;
    uint32_t blocks = 1;

    // Read output count from our pinned counter
    uint64_t current_output_count = h_pinned_lp_combined_count ? h_pinned_lp_combined_count[0] : 0;

    // Track cumulative full relations across all LP batches
    cumulative_full_relations_ += current_output_count;

    // Dispatch telemetry kernel — reads witness count directly from d_witness_dual_counter
    // on the GPU, avoiding the host roundtrip that was always returning 0 (the RelationBatch
    // built-in counter is never updated by global_append_kernel which uses the dual counter).
    kernels::update_telemetry_kernel<<<blocks, threads, 0, lp_stream>>>(
        d_directory,
        num_buckets,
        ROW_WIDTH_ELEMS,
        0,                          // partials buffer fill (optional)
        current_output_count,       // Full relations formed in this batch
        cumulative_full_relations_, // Cumulative LP full relations
        d_witness_dual_counter,     // GPU reads witness count directly
        last_witness_count_,        // Previous snapshot for delta computation
        d_pinned_stats,
        d_slab_overflow_count,
        d_witness_overflow_count,
        d_output_overflow_count
    );

    // Update last_witness_count_ from pinned stats written by the PREVIOUS kernel.
    // Safe: updateStats() is only called after the prior LP dispatch completes (event wait).
    // On first call, h_pinned_stats->total_witnesses is 0 (matching last_witness_count_ init).
    last_witness_count_ = h_pinned_stats->total_witnesses;

    // NO cudaStreamSynchronize here — telemetry uses __threadfence_system.
    // Host reads telemetry via getTelemetry() with generation ticket polling.
    // Overflow warnings are deferred to the next time the host reads telemetry.
}

std::pair<uint64_t, uint64_t> LargePrimeVariant::getWitnessStats() const {
    if (d_global_witnesses) {
        return d_global_witnesses->readStats();
    }
    return {0, 0};
}

size_t LargePrimeVariant::getWitnessCapacityRels() const {
    return d_global_witnesses ? d_global_witnesses->getCapacityRels() : 0;
}

size_t LargePrimeVariant::getWitnessCapacityFactors() const {
    return d_global_witnesses ? d_global_witnesses->getCapacityFactors() : 0;
}

// -----------------------------------------------------------------------------
// Core Execution Pipeline
// -----------------------------------------------------------------------------

// =========================================================================
// STAGE 1: Directory Fetch (hash/tag computation + bucket classification)
// STAGE 2: Vectorized Probe & Classify
// STAGE 3: Global Commit (combine matches + append new witnesses)
// =========================================================================
 
void LargePrimeVariant::processAndCommit(
    mpqs::structures::RelationBatch* input_partials,
    mpqs::structures::RelationBatch* persistent_storage,
    uint32_t input_count_hint
) {
    if (!input_partials || !persistent_storage) return;

    // Wait for postprocessor to signal that the partial batch is populated
    if (partials_ready_event_) {
        CUDA_CHECK(cudaStreamWaitEvent(lp_stream, partials_ready_event_, 0));
    }

    // Use pinned counter hint from orchestrator (zero-sync) with capacity fallback
    uint32_t input_cap = (uint32_t)input_partials->getCapacityRels();
    uint32_t input_count = (input_count_hint > 0) ? input_count_hint : (uint32_t)input_partials->getCount(lp_stream);
    if (input_count > input_cap) {
        LOG(LOG_WARNING) << "input_count=" << input_count
                         << " exceeds capacity=" << input_cap << " — clamping";
        input_count = input_cap;
    }
    LOG(LOG_STATS) << "processAndCommit: input_count=" << input_count;
    if (input_count == 0) return; // Nothing to process

    auto input_view = input_partials->get_view();

#ifdef LP_DEBUG
    lp_call_counter_++;
    {
        uint32_t dump_count = std::min(input_count, (uint32_t)100);
        std::vector<unsigned __int128> h_lp(dump_count);
        CUDA_CHECK(cudaMemcpyAsync(h_lp.data(), input_view.large_primes,
                                    dump_count * sizeof(unsigned __int128),
                                    cudaMemcpyDeviceToHost, lp_stream));
        CUDA_CHECK(cudaStreamSynchronize(lp_stream));

        // Count unique values
        std::set<unsigned __int128> unique_set(h_lp.begin(), h_lp.end());
        uint32_t zero_count = 0;
        for (auto& v : h_lp) if (v == 0) zero_count++;

        LOG(LOG_STATS) << "=== LP Call #" << lp_call_counter_
                      << " | input_count=" << input_count
                      << " | unique(first " << dump_count << ")=" << unique_set.size()
                      << " | zeros=" << zero_count;

        // Dump first 20 values in hex
        for (uint32_t i = 0; i < std::min(dump_count, (uint32_t)20); i++) {
            uint64_t hi = (uint64_t)(h_lp[i] >> 64);
            uint64_t lo = (uint64_t)(h_lp[i]);
            LOG(LOG_STATS) << "  lp[" << i << "] = 0x"
                          << std::hex << hi << "_" << lo << std::dec;
        }

        if (unique_set.size() == 1) {
            LOG(LOG_ERROR_MAJOR) << "ANOMALY: All " << dump_count
                           << " sampled LP values are IDENTICAL!";
        }
    }
#endif

    // =========================================================================
    // Pipeline: Directory Fetch → Probe → Commit
    // (Sort and intra-batch dedup stages removed — same-batch LP collisions
    //  become deferred matches with negligible impact.)
    // =========================================================================

    resizePipelineBuffers(input_count);

    // Write host-side count to device variable for kernel consumption
    uint64_t input_count_64 = input_count;
    CUDA_CHECK(cudaMemcpyAsync(d_lp_input_count_, &input_count_64, sizeof(uint64_t),
                                cudaMemcpyHostToDevice, lp_stream));

    uint32_t block_size = 256;
    uint32_t grid_size = (input_count + block_size - 1) / block_size;

    // =========================================================================
    // STAGE 1: Directory Fetch (compute hash/tag inline, classify buckets)
    // =========================================================================

    kernels::directory_fetch_kernel<<<grid_size, block_size, 0, lp_stream>>>(
        input_view.large_primes,  // In: raw LP values
        d_routing_keys,           // Out: [Status|RowIdx|Tag]
        d_status_flags,           // Out: NEEDS_ALLOCATION or NEEDS_PROBE
        d_directory,
        d_lp_input_count_,
        config.hash_bits
    );

    // =========================================================================
    // STAGE 2: Vectorized Probe & Classify
    // =========================================================================

    kernels::probe_hash_table_kernel<<<grid_size, block_size, 0, lp_stream>>>(
        d_routing_keys,        // Packed: Status|RowIdx|Tag
        d_status_flags,        // Output status (MATCH_FOUND / NEEDS_APPEND)
        d_payload_slabs,       // Global Hash Table Payloads
        d_target_idx_array,    // Output Global Index (if matched)
        d_lp_input_count_,
        ROW_WIDTH_ELEMS,
        config.purge_after_match
    );

    // Post-Condition of Stage 2:
    // - Every element belongs to one of 3 terminal states:
    //   1. MATCH_FOUND (Global Hash Match -> Yields Full Relation)
    //   2. NEEDS_APPEND (Hash Collision -> Needs Append to existing Row)
    //   3. NEEDS_ALLOCATION (Empty Bucket -> Needs New Row + Append)

    // =========================================================================
    // STAGE 3: Global Commit
    // =========================================================================

    auto output_view = d_output_batch->get_view();
    kernels::sync_dual_counter_kernel<<<1, 1, 0, lp_stream>>>(
        d_output_dual_counter,
        output_view.global_count,
        output_view.global_factor_idx,
        true // Pack
    );

    auto global_witness_view = d_global_witnesses->get_view();
    kernels::sync_dual_counter_kernel<<<1, 1, 0, lp_stream>>>(
        d_witness_dual_counter,
        global_witness_view.global_count,
        global_witness_view.global_factor_idx,
        true // Pack
    );

    // 3A: Full Relation Generator (identity mapping: tid == original index)
    kernels::global_combine_kernel<<<grid_size, block_size, 0, lp_stream>>>(
        d_status_flags,
        d_target_idx_array,
        input_view,
        global_witness_view,
        output_view,
        d_output_dual_counter,
        d_lp_input_count_,
        this->N,
        d_output_overflow_count
    );

    // --- Diagnostic: validate combine inputs (env-gated: MPQS_LP_DIAG=1) ---
    if (std::getenv("MPQS_LP_DIAG") && std::string(std::getenv("MPQS_LP_DIAG")) == "1") {
        constexpr uint32_t MAX_DIAG = 20;
        kernels::CombineDiagEntry* d_diag;
        uint32_t* d_diag_count;
        uint32_t* d_match_count;
        CUDA_CHECK(cudaMalloc(&d_diag, MAX_DIAG * sizeof(kernels::CombineDiagEntry)));
        CUDA_CHECK(cudaMalloc(&d_diag_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_match_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(d_diag_count, 0, sizeof(uint32_t), lp_stream));
        CUDA_CHECK(cudaMemsetAsync(d_match_count, 0, sizeof(uint32_t), lp_stream));

        kernels::diagnose_combine_inputs_kernel<<<grid_size, block_size, 0, lp_stream>>>(
            d_status_flags, d_target_idx_array,
            input_view, global_witness_view,
            d_lp_input_count_,
            d_diag, d_diag_count, MAX_DIAG, d_match_count
        );
        CUDA_CHECK(cudaStreamSynchronize(lp_stream));

        uint32_t h_diag_count = 0, h_match_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_diag_count, d_diag_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_match_count, d_match_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        LOG(LOG_STATS) << "Matches: " << h_match_count << ", Anomalies: " << h_diag_count;
        if (h_diag_count > 0) {
            std::vector<kernels::CombineDiagEntry> h_diag(std::min(h_diag_count, MAX_DIAG));
            CUDA_CHECK(cudaMemcpy(h_diag.data(), d_diag, h_diag.size() * sizeof(kernels::CombineDiagEntry), cudaMemcpyDeviceToHost));
            for (auto& e : h_diag) {
                LOG(LOG_ERROR_MAJOR) << "rel=" << e.rel_idx
                    << " tgt=" << e.target_idx
                    << " err=" << (int)e.error_type
                    << " inp_lp=" << e.input_lp << " wit_lp=" << e.witness_lp
                    << " inp_flen=" << e.input_flen << " wit_flen=" << e.witness_flen
                    << " inp_fstart=" << e.input_fstart << " wit_fstart=" << e.witness_fstart;
            }
        }
        CUDA_CHECK(cudaFree(d_diag));
        CUDA_CHECK(cudaFree(d_diag_count));
        CUDA_CHECK(cudaFree(d_match_count));
    }

    // 3B: Global Hash Table Appender (identity mapping: tid == original index)
    kernels::global_append_kernel<<<grid_size, block_size, 0, lp_stream>>>(
        d_status_flags,
        d_routing_keys,      // Packed Key containing Tag
        input_view,
        global_witness_view,
        d_directory,
        d_payload_slabs,
        d_global_row_allocator,
        d_witness_dual_counter,
        d_lp_input_count_,
        config.hash_bits,
        ROW_WIDTH_ELEMS,
        max_slab_rows_,
        d_slab_overflow_count,
        d_witness_overflow_count
    );

    // 3C. Unpack and Sync Dual Counters back to SoA Batch
    kernels::sync_dual_counter_kernel<<<1, 1, 0, lp_stream>>>(
        d_output_dual_counter, 
        output_view.global_count, 
        output_view.global_factor_idx, 
        false // Unpack
    );

    kernels::sync_dual_counter_kernel<<<1, 1, 0, lp_stream>>>(
        d_witness_dual_counter, 
        global_witness_view.global_count, 
        global_witness_view.global_factor_idx, 
        false // Unpack
    );

    // --- Async D→H copy of LP combined output count (pinned, no sync) ---
    auto output_view_final = d_output_batch->get_view();
    CUDA_CHECK(cudaMemcpyAsync(
        d_pinned_lp_combined_count,     // dst: device alias of pinned memory
        output_view_final.global_count,  // src: device atomic
        sizeof(uint64_t),
        cudaMemcpyDeviceToDevice,        // device-to-device (mapped pinned appears as device mem)
        lp_stream
    ));
    // Also copy factor count for capacity checks
    CUDA_CHECK(cudaMemcpyAsync(
        d_pinned_lp_combined_count + 1,
        output_view_final.global_factor_idx,
        sizeof(uint64_t),
        cudaMemcpyDeviceToDevice,
        lp_stream
    ));

    // Update telemetry (rewritten to use pinned/device-side state)
    this->updateStats();

    // Record LP completion event (replaces cudaStreamSynchronize)
    CUDA_CHECK(cudaEventRecord(lp_done_event_, lp_stream));

    // Host must wait for LP to finish before reading pinned counters and calling append()
    // This sync will be removed in Stage 3 (async overlap) / Stage 4 (device_append_kernel).
    CUDA_CHECK(cudaEventSynchronize(lp_done_event_));

    // Read from pinned memory (no additional sync needed)
    uint64_t found_full_count = h_pinned_lp_combined_count[0];
    if (found_full_count > 0) {
        LOG(LOG_STATS) << "Combined " << found_full_count << " full relations in this batch.";
        persistent_storage->append(*d_output_batch, found_full_count, lp_stream);
    }
    
    // Clear Output Batch for the next run (Counters reset)
    d_output_batch->reset_counters(lp_stream);

#ifdef LP_DEBUG
    {
        uint64_t combined_count = d_output_batch->getCount(lp_stream);
        // Note: combined_count should be 0 here since we just reset, so check found_full_count instead
        if (found_full_count > input_count / 2) {
            LOG(LOG_ERROR_MAJOR) << "ANOMALY: combined_count=" << found_full_count
                           << " > input_count/2=" << (input_count / 2)
                           << " on LP call #" << lp_call_counter_
                           << ". This indicates degenerate LP data!";
        }
    }
#endif

}

// -----------------------------------------------------------------------------
// Async Execution Pipeline (Stage 3: Sieve/LP Concurrent Overlap)
// -----------------------------------------------------------------------------

void LargePrimeVariant::processAndCommitAsync(
    mpqs::structures::RelationBatch* input_partials,
    mpqs::structures::RelationBatch* persistent_storage
) {
    if (!input_partials || !persistent_storage) return;

    // Save persistent pointer for deferred append (orchestrator responsibility)
    pending_persistent_ = persistent_storage;

    // Wait for postprocessor to signal that the partial batch is populated
    if (partials_ready_event_) {
        CUDA_CHECK(cudaStreamWaitEvent(lp_stream, partials_ready_event_, 0));
    }

    // Snapshot the device-side partial count into d_lp_input_count_ (no host readback)
    uint64_t* d_src_count = input_partials->getDeviceCountPtr();
    kernels::snapshot_count_kernel<<<1, 1, 0, lp_stream>>>(d_lp_input_count_, d_src_count);
    CUDA_CHECK(cudaEventRecord(count_snapshot_event_, lp_stream));

    // Grid sized to capacity — kernels self-limit via d_lp_input_count_
    uint32_t input_cap = (uint32_t)input_partials->getCapacityRels();
    resizePipelineBuffers(input_cap);

    uint32_t block_size = 256;
    uint32_t grid_size = (input_cap + block_size - 1) / block_size;

    LOG(LOG_DEBUG_1) << "processAndCommitAsync: grid_cap=" << input_cap;

    auto input_view = input_partials->get_view();

    // STAGE 1: Directory Fetch
    kernels::directory_fetch_kernel<<<grid_size, block_size, 0, lp_stream>>>(
        input_view.large_primes,
        d_routing_keys,
        d_status_flags,
        d_directory,
        d_lp_input_count_,
        config.hash_bits
    );

    // STAGE 2: Vectorized Probe
    kernels::probe_hash_table_kernel<<<grid_size, block_size, 0, lp_stream>>>(
        d_routing_keys,
        d_status_flags,
        d_payload_slabs,
        d_target_idx_array,
        d_lp_input_count_,
        ROW_WIDTH_ELEMS,
        config.purge_after_match
    );

    // STAGE 3: Global Commit
    auto output_view = d_output_batch->get_view();
    kernels::sync_dual_counter_kernel<<<1, 1, 0, lp_stream>>>(
        d_output_dual_counter, output_view.global_count, output_view.global_factor_idx, true);

    auto global_witness_view = d_global_witnesses->get_view();
    kernels::sync_dual_counter_kernel<<<1, 1, 0, lp_stream>>>(
        d_witness_dual_counter, global_witness_view.global_count, global_witness_view.global_factor_idx, true);

    kernels::global_combine_kernel<<<grid_size, block_size, 0, lp_stream>>>(
        d_status_flags, d_target_idx_array,
        input_view, global_witness_view, output_view,
        d_output_dual_counter, d_lp_input_count_, this->N, d_output_overflow_count);

    // --- Diagnostic: validate combine inputs (async path, sync-gated) ---
    {
        static bool diag_checked = false;
        static bool diag_enabled = false;
        if (!diag_checked) {
            const char* env = std::getenv("MPQS_LP_DIAG");
            diag_enabled = (env && std::string(env) == "1");
            diag_checked = true;
        }
        if (diag_enabled) {
            constexpr uint32_t MAX_DIAG = 20;
            kernels::CombineDiagEntry* d_diag;
            uint32_t* d_diag_count;
            uint32_t* d_match_count;
            CUDA_CHECK(cudaMalloc(&d_diag, MAX_DIAG * sizeof(kernels::CombineDiagEntry)));
            CUDA_CHECK(cudaMalloc(&d_diag_count, sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_match_count, sizeof(uint32_t)));
            CUDA_CHECK(cudaMemsetAsync(d_diag_count, 0, sizeof(uint32_t), lp_stream));
            CUDA_CHECK(cudaMemsetAsync(d_match_count, 0, sizeof(uint32_t), lp_stream));

            kernels::diagnose_combine_inputs_kernel<<<grid_size, block_size, 0, lp_stream>>>(
                d_status_flags, d_target_idx_array,
                input_view, global_witness_view,
                d_lp_input_count_,
                d_diag, d_diag_count, MAX_DIAG, d_match_count
            );
            CUDA_CHECK(cudaStreamSynchronize(lp_stream));

            uint32_t h_diag_count = 0, h_match_count = 0;
            CUDA_CHECK(cudaMemcpy(&h_diag_count, d_diag_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_match_count, d_match_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            LOG(LOG_STATS) << "Matches: " << h_match_count << ", Anomalies: " << h_diag_count;
            if (h_diag_count > 0) {
                std::vector<kernels::CombineDiagEntry> h_diag(std::min(h_diag_count, MAX_DIAG));
                CUDA_CHECK(cudaMemcpy(h_diag.data(), d_diag, h_diag.size() * sizeof(kernels::CombineDiagEntry), cudaMemcpyDeviceToHost));
                for (auto& e : h_diag) {
                    LOG(LOG_ERROR_MAJOR) << "rel=" << e.rel_idx
                        << " tgt=" << e.target_idx
                        << " err=" << (int)e.error_type
                        << " inp_lp=" << e.input_lp << " wit_lp=" << e.witness_lp
                        << " inp_flen=" << e.input_flen << " wit_flen=" << e.witness_flen
                        << " inp_fstart=" << e.input_fstart << " wit_fstart=" << e.witness_fstart;
                }
            }
            CUDA_CHECK(cudaFree(d_diag));
            CUDA_CHECK(cudaFree(d_diag_count));
            CUDA_CHECK(cudaFree(d_match_count));
        }
    }

    kernels::global_append_kernel<<<grid_size, block_size, 0, lp_stream>>>(
        d_status_flags, d_routing_keys,
        input_view, global_witness_view,
        d_directory, d_payload_slabs, d_global_row_allocator,
        d_witness_dual_counter, d_lp_input_count_, config.hash_bits,
        ROW_WIDTH_ELEMS, max_slab_rows_,
        d_slab_overflow_count, d_witness_overflow_count);

    // Unpack dual counters
    kernels::sync_dual_counter_kernel<<<1, 1, 0, lp_stream>>>(
        d_output_dual_counter, output_view.global_count, output_view.global_factor_idx, false);
    kernels::sync_dual_counter_kernel<<<1, 1, 0, lp_stream>>>(
        d_witness_dual_counter, global_witness_view.global_count, global_witness_view.global_factor_idx, false);

    // Async D→H copy of LP combined output count (pinned, no sync)
    CUDA_CHECK(cudaMemcpyAsync(
        d_pinned_lp_combined_count, output_view.global_count,
        sizeof(uint64_t), cudaMemcpyDeviceToDevice, lp_stream));
    CUDA_CHECK(cudaMemcpyAsync(
        d_pinned_lp_combined_count + 1, output_view.global_factor_idx,
        sizeof(uint64_t), cudaMemcpyDeviceToDevice, lp_stream));

    // Telemetry update (enqueued on lp_stream, no sync)
    this->updateStats();

    // Record LP completion — no CPU block
    CUDA_CHECK(cudaEventRecord(lp_done_event_, lp_stream));
}

bool LargePrimeVariant::isComplete() const {
    if (!lp_done_event_) return true;
    cudaError_t result = cudaEventQuery(lp_done_event_);
    return (result == cudaSuccess);
}

// -----------------------------------------------------------------------------
// Stage 4: Device-Side Append (GPU kernel replaces host-side append())
// -----------------------------------------------------------------------------

void LargePrimeVariant::launchDeviceAppend(
    mpqs::structures::RelationBatch* persistent,
    cudaStream_t stream
) {
    if (!persistent || !d_output_batch) return;

    auto output_view = d_output_batch->get_view();
    auto persistent_view = persistent->get_view();

    // Single-block launch: shared memory broadcast requires all threads in same block.
    // 256 threads with grid-stride loop handles typical LP output (<1000 relations).
    kernels::device_append_kernel<<<1, 256, 0, stream>>>(
        persistent_view,
        output_view,
        output_view.global_count,        // src count (device)
        output_view.global_factor_idx,   // src factor count (device)
        persistent_view.global_count,    // dst count (device)
        persistent_view.global_factor_idx, // dst factor count (device)
        d_pinned_appended_count          // mapped pinned (telemetry)
    );

    // Reset output batch on the same stream (ordered after append completes)
    d_output_batch->reset_counters(stream);
    cudaMemsetAsync(d_output_batch->get_view().factor_offsets, 0, sizeof(uint64_t), stream);
}

void LargePrimeVariant::moveWitnessesToHost(
    mpqs::structures::HostRelationBatch& dest, cudaStream_t stream) {
    if (!d_global_witnesses || !d_witness_dual_counter) return;

    // d_global_count_ptr may lag if the final sync_dual_counter_kernel(pack=false)
    // completed on lp_stream after any prior cudaDeviceSynchronize snapshot.
    // Force-sync the authoritative d_witness_dual_counter → global_count now,
    // on the download stream, so moveToHost reads the correct count.
    auto global_witness_view = d_global_witnesses->get_view();
    kernels::sync_dual_counter_kernel<<<1, 1, 0, stream>>>(
        d_witness_dual_counter,
        global_witness_view.global_count,
        global_witness_view.global_factor_idx,
        false  // unpack: d_witness_dual_counter → (global_count, global_factor_idx)
    );

    d_global_witnesses->moveToHost(dest, stream);
    LOG(LOG_INFO) << "moveWitnessesToHost: downloaded " << dest.num_relations << " witnesses.";
}

} // namespace lp
} // namespace mpqs
