// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_batch_merge.cu
//
// Batch-planned GPU merge execution (M9e).
// CPU plans merge candidates from column weight arrays; GPU executes non-conflicting
// merges in parallel via atomicCAS row ownership + atomic_reserve_dual workspace
// allocation. Two-pointer packed merge with exponent addition; Montgomery sqrt_Q
// multiplication maintains per-row 1-partial metadata.
//
// Key design: original DevicePackedCSR is immutable. Merged rows are appended to
// DeviceMergeWorkspace via bump allocator. d_row_ptr tracks indirection (MSB=0
// → original, MSB=1 → workspace, 0xFFFFFFFF → dead).

#include "gpu_batch_merge.cuh"
#include "cuda_check.h"
#include "hpc_logger.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <set>
#include <vector>

namespace mpqs {
namespace matrix {

// =========================================================================
// Constants
// =========================================================================

/// Maximum entries in the per-thread local merge buffer.
/// At average row weight ~33, max merged row ≈ 66 entries × 4 bytes = 264 bytes.
/// Rows exceeding this abort and retry in a later round.
static constexpr uint32_t kMaxLocalBufEntries = 128;

/// Maximum rounds before forced termination (safety valve).
static constexpr uint32_t kMaxMergeRounds = 200;

// =========================================================================
// Device helpers: pack_rf, unpack_rf, atomic_reserve_dual
// =========================================================================
// Copied from src/largeprimes/largeprime.cu to avoid coupling.

static __device__ __forceinline__
uint64_t pack_rf(uint32_t rows, uint32_t factors) {
    return (static_cast<uint64_t>(rows) << 32) | static_cast<uint64_t>(factors);
}

static __device__ __forceinline__
void unpack_rf(uint64_t packed, uint32_t& rows, uint32_t& factors) {
    rows    = static_cast<uint32_t>(packed >> 32);
    factors = static_cast<uint32_t>(packed);
}

/// Atomically reserve a workspace row slot and entry block.
/// CAS loop on packed (row_count, entry_count) dual counter.
static __device__ __forceinline__
void atomic_reserve_dual(
    uint64_t* counter,
    uint32_t entries_to_add,
    uint32_t max_rows,
    uint32_t max_entries,
    uint32_t* out_row_slot,
    uint32_t* out_entry_offset,
    bool* success)
{
    unsigned long long old = atomicAdd((unsigned long long*)counter, 0ULL);
    while (true) {
        uint32_t r, e;
        unpack_rf(old, r, e);
        if (r >= max_rows || (e + entries_to_add) > max_entries) {
            *success = false;
            return;
        }
        unsigned long long desired = pack_rf(r + 1, e + entries_to_add);
        unsigned long long prev = atomicCAS((unsigned long long*)counter, old, desired);
        if (prev == old) {
            *out_row_slot = r;
            *out_entry_offset = e;
            *success = true;
            return;
        }
        old = prev;
    }
}

// =========================================================================
// Device helper: resolve_row
// =========================================================================

/// A pointer + length view of a row's packed entries.
struct RowSlice {
    const PackedEntry* data;
    uint32_t length;
};

/// Resolve a logical row index to its physical location via row_ptr encoding.
/// MSB=0: original CSR. MSB=1: workspace. ROW_DEAD: should not be called.
static __device__ __forceinline__
RowSlice resolve_row(
    uint32_t ptr_val,
    const uint32_t* orig_row_offsets,
    const PackedEntry* orig_entries,
    const uint32_t* ws_row_starts,
    const uint32_t* ws_row_lengths,
    const PackedEntry* ws_entries)
{
    RowSlice s;
    if (ptr_val & ROW_WS_BIT) {
        uint32_t ws_idx = ptr_val & 0x7FFFFFFFu;
        s.data   = ws_entries + ws_row_starts[ws_idx];
        s.length = ws_row_lengths[ws_idx];
    } else {
        s.data   = orig_entries + orig_row_offsets[ptr_val];
        s.length = orig_row_offsets[ptr_val + 1] - orig_row_offsets[ptr_val];
    }
    return s;
}

/// Resolve sqrt_Q for a row via row_ptr encoding.
static __device__ __forceinline__
uint512 resolve_sqrt_Q(
    uint32_t ptr_val,
    const uint512* orig_sqrt_Q,
    const uint512* ws_sqrt_Q)
{
    if (ptr_val & ROW_WS_BIT) {
        return ws_sqrt_Q[ptr_val & 0x7FFFFFFFu];
    }
    return orig_sqrt_Q[ptr_val];
}

/// Resolve sign for a row via row_ptr encoding.
static __device__ __forceinline__
uint8_t resolve_sign(
    uint32_t ptr_val,
    const uint8_t* orig_signs,
    const uint8_t* ws_signs)
{
    if (ptr_val & ROW_WS_BIT) {
        return ws_signs[ptr_val & 0x7FFFFFFFu];
    }
    return orig_signs[ptr_val];
}

/// Resolve val_2_exp for a row via row_ptr encoding.
static __device__ __forceinline__
int32_t resolve_val_2_exp(
    uint32_t ptr_val,
    const int32_t* orig_val_2_exps,
    const int32_t* ws_val_2_exps)
{
    if (ptr_val & ROW_WS_BIT) {
        return ws_val_2_exps[ptr_val & 0x7FFFFFFFu];
    }
    return orig_val_2_exps[ptr_val];
}

// =========================================================================
// MontgomeryContext::mul() — CIOS device implementation
// =========================================================================
// Verbatim copy of Montgomery::mul() from src/common/montgomery.cuh.

__device__ uint512 MontgomeryContext::mul(const uint512& a, const uint512& b) const {
    uint512 T;
    uint32_t extra = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t a_i = a.limbs[i];
        uint32_t u = (T.limbs[0] + a_i * b.limbs[0]) * n_prime;

        unsigned __int128 carry = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned __int128 sum = (unsigned __int128)T.limbs[j]
                                  + (unsigned __int128)a_i * b.limbs[j]
                                  + (unsigned __int128)u * N.limbs[j]
                                  + carry;
            if (j > 0) T.limbs[j-1] = (uint32_t)sum;
            carry = sum >> 32;
        }
        unsigned __int128 sum_top = (unsigned __int128)extra + carry;
        T.limbs[15] = (uint32_t)sum_top;
        extra = (uint32_t)(sum_top >> 32);
    }

    if (extra != 0 || T >= N) T.sub(N);
    return T;
}

/// Construct MontgomeryContext from existing Montgomery class.
MontgomeryContext makeMontgomeryContext(const math::Montgomery& mont) {
    MontgomeryContext ctx;
    ctx.N       = mont.N;
    ctx.R2      = mont.R2;
    ctx.n_prime = mont.n_prime;
    return ctx;
}

// =========================================================================
// Kernel: execute_merges_kernel
// =========================================================================

/// One thread per MergeCandidate. Claims both rows via atomicCAS, performs
/// two-pointer packed merge with exponent addition, Montgomery sqrt_Q multiply,
/// writes result to workspace via atomic_reserve_dual, updates col_weight.
__global__ __launch_bounds__(256)
void execute_merges_kernel(
    // Merge schedule
    const MergeCandidate* __restrict__ candidates,
    uint32_t n_candidates,
    // Original CSR (immutable)
    const uint32_t*    __restrict__ orig_row_offsets,
    const PackedEntry* __restrict__ orig_entries,
    const uint512*     __restrict__ orig_sqrt_Q,
    const uint8_t*     __restrict__ orig_signs,
    const int32_t*     __restrict__ orig_val_2_exps,
    // Workspace (append-only)
    PackedEntry*       ws_entries,
    uint32_t*          ws_row_starts,
    uint32_t*          ws_row_lengths,
    uint512*           ws_sqrt_Q,
    uint8_t*           ws_signs,
    int32_t*           ws_val_2_exps,
    uint64_t*          ws_dual_counter,
    uint32_t           ws_max_rows,
    uint32_t           ws_max_entries,
    // Row indirection + locks
    uint32_t*          row_ptr,
    uint32_t*          row_locks,
    // Column weights (atomically updated)
    uint32_t*          col_weight,
    uint32_t*          gf2_col_weight,   // GF(2) column weights (odd-exp only)
    // Montgomery context (by value, ~136 bytes)
    MontgomeryContext  mont,
    // Output: abort count
    uint32_t*          d_abort_count)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_candidates) return;

    uint32_t r1 = candidates[tid].r1;
    uint32_t r2 = candidates[tid].r2;

    // === Step 1: Claim rows via atomicCAS ===
    if (atomicCAS(&row_locks[r1], 0u, tid + 1) != 0u) {
        atomicAdd(d_abort_count, 1u);
        return;
    }
    if (atomicCAS(&row_locks[r2], 0u, tid + 1) != 0u) {
        row_locks[r1] = 0u;  // release r1
        atomicAdd(d_abort_count, 1u);
        return;
    }

    // Check that neither row is DEAD (a prior merge in this batch killed it)
    uint32_t ptr1 = row_ptr[r1];
    uint32_t ptr2 = row_ptr[r2];
    if (ptr1 == ROW_DEAD || ptr2 == ROW_DEAD) {
        row_locks[r1] = 0u;
        row_locks[r2] = 0u;
        atomicAdd(d_abort_count, 1u);
        return;
    }

    // === Step 2: Resolve row data via row_ptr ===
    RowSlice s1 = resolve_row(ptr1, orig_row_offsets, orig_entries,
                              ws_row_starts, ws_row_lengths, ws_entries);
    RowSlice s2 = resolve_row(ptr2, orig_row_offsets, orig_entries,
                              ws_row_starts, ws_row_lengths, ws_entries);

    // === Step 3: Two-pointer packed merge into local buffer ===
    PackedEntry local_buf[kMaxLocalBufEntries];
    uint32_t merged_len = 0;

    uint32_t i = 0, j = 0;
    while (i < s1.length && j < s2.length && merged_len < kMaxLocalBufEntries) {
        uint32_t c1 = packed_col(s1.data[i]);
        uint32_t c2 = packed_col(s2.data[j]);
        if (c1 < c2) {
            local_buf[merged_len++] = s1.data[i++];
        } else if (c1 > c2) {
            local_buf[merged_len++] = s2.data[j++];
        } else {
            // Same column: add exponents (packed merge keeps all columns)
            uint32_t exp_sum = packed_exp(s1.data[i]) + packed_exp(s2.data[j]);
            local_buf[merged_len++] = make_packed(c1, static_cast<uint8_t>(exp_sum & 0xFF));
            i++; j++;
        }
    }
    // Copy remaining entries from whichever row has leftovers
    while (i < s1.length && merged_len < kMaxLocalBufEntries)
        local_buf[merged_len++] = s1.data[i++];
    while (j < s2.length && merged_len < kMaxLocalBufEntries)
        local_buf[merged_len++] = s2.data[j++];

    // If merged row exceeds local buffer, abort (rare: avg weight ~33, max ~66)
    if (i < s1.length || j < s2.length) {
        row_locks[r1] = 0u;
        row_locks[r2] = 0u;
        atomicAdd(d_abort_count, 1u);
        return;
    }

    // === Step 4: Montgomery sqrt_Q multiply ===
    uint512 sq1 = resolve_sqrt_Q(ptr1, orig_sqrt_Q, ws_sqrt_Q);
    uint512 sq2 = resolve_sqrt_Q(ptr2, orig_sqrt_Q, ws_sqrt_Q);
    // Multiply: transform to Montgomery domain, multiply, reduce back
    uint512 sq1_mont = mont.transform(sq1);
    uint512 sq2_mont = mont.transform(sq2);
    uint512 merged_sq = mont.reduce(mont.mul(sq1_mont, sq2_mont));

    // === Step 5: Combine sign and val_2_exp ===
    uint8_t sign1 = resolve_sign(ptr1, orig_signs, ws_signs);
    uint8_t sign2 = resolve_sign(ptr2, orig_signs, ws_signs);
    int32_t v2_1  = resolve_val_2_exp(ptr1, orig_val_2_exps, ws_val_2_exps);
    int32_t v2_2  = resolve_val_2_exp(ptr2, orig_val_2_exps, ws_val_2_exps);

    // Sign merge: encoding-agnostic. sign=1 means positive; anything else means negative.
    bool neg1 = (sign1 != 1u), neg2 = (sign2 != 1u);
    uint8_t merged_sign = (neg1 ^ neg2) ? static_cast<uint8_t>(0xFF) : static_cast<uint8_t>(1);
    int32_t merged_v2   = v2_1 + v2_2;

    // === Step 6: Reserve workspace slot via atomic_reserve_dual ===
    uint32_t row_slot, entry_offset;
    bool success;
    atomic_reserve_dual(ws_dual_counter, merged_len,
                        ws_max_rows, ws_max_entries,
                        &row_slot, &entry_offset, &success);
    if (!success) {
        // Workspace full — abort this merge
        row_locks[r1] = 0u;
        row_locks[r2] = 0u;
        atomicAdd(d_abort_count, 1u);
        return;
    }

    // === Step 7: Write merged row to workspace ===
    for (uint32_t k = 0; k < merged_len; k++)
        ws_entries[entry_offset + k] = local_buf[k];
    ws_row_starts[row_slot]  = entry_offset;
    ws_row_lengths[row_slot] = merged_len;
    ws_sqrt_Q[row_slot]      = merged_sq;
    ws_signs[row_slot]       = merged_sign;
    ws_val_2_exps[row_slot]  = merged_v2;

    // === Step 8: Update row_ptr ===
    row_ptr[r1] = ROW_WS_BIT | row_slot;  // r1 now points to workspace
    row_ptr[r2] = ROW_DEAD;                // r2 is consumed

    // === Step 9: Update column weights ===
    // Decrement for all cols in source rows (both removed from matrix)
    for (uint32_t k = 0; k < s1.length; k++)
        atomicSub(&col_weight[packed_col(s1.data[k])], 1u);
    for (uint32_t k = 0; k < s2.length; k++)
        atomicSub(&col_weight[packed_col(s2.data[k])], 1u);
    // Increment for all cols in merged row (added to matrix)
    for (uint32_t k = 0; k < merged_len; k++)
        atomicAdd(&col_weight[packed_col(local_buf[k])], 1u);

    // GF(2) weights (parity-conditional — odd exponents only):
    for (uint32_t k = 0; k < s1.length; k++)
        if (packed_exp(s1.data[k]) & 1)
            atomicSub(&gf2_col_weight[packed_col(s1.data[k])], 1u);
    for (uint32_t k = 0; k < s2.length; k++)
        if (packed_exp(s2.data[k]) & 1)
            atomicSub(&gf2_col_weight[packed_col(s2.data[k])], 1u);
    for (uint32_t k = 0; k < merged_len; k++)
        if (packed_exp(local_buf[k]) & 1)
            atomicAdd(&gf2_col_weight[packed_col(local_buf[k])], 1u);

    // === Step 10: Release locks ===
    __threadfence();  // Ensure workspace writes are globally visible before lock release
    row_locks[r1] = 0u;
    row_locks[r2] = 0u;
}

// =========================================================================
// Kernel: clear_locks_kernel
// =========================================================================

/// Belt-and-suspenders: zero all row locks between merge rounds.
__global__ __launch_bounds__(256)
void clear_locks_kernel(uint32_t* row_locks, uint32_t n_rows)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) row_locks[i] = 0u;
}

// =========================================================================
// Kernel: init_row_ptr_kernel
// =========================================================================

/// Initialize d_row_ptr[i] = i (identity mapping).
__global__ __launch_bounds__(256)
void init_row_ptr_kernel(uint32_t* row_ptr, uint32_t n_rows)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) row_ptr[i] = i;
}

// =========================================================================
// Kernel: compute_col_weights_kernel
// =========================================================================

/// One thread per row. Atomically increments column weights from packed entries.
/// Used to recompute col_weight from scratch (M9b's scratch is freed before return).
__global__ __launch_bounds__(256)
void compute_col_weights_kernel(
    const uint32_t*    __restrict__ d_row_offsets,
    const PackedEntry* __restrict__ d_entries,
    uint32_t* __restrict__          d_col_weight,
    uint32_t n_rows)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;

    const uint32_t begin = d_row_offsets[r];
    const uint32_t end   = d_row_offsets[r + 1];
    for (uint32_t j = begin; j < end; ++j) {
        atomicAdd(&d_col_weight[d_entries[j] >> 8], 1u);
    }
}

// =========================================================================
// Kernel: compute_gf2_col_weights_kernel
// =========================================================================

/// One thread per row. Atomically increments GF(2) column weights for odd-exponent entries.
/// d_gf2_col_weight must be pre-zeroed before launch.
__global__ __launch_bounds__(256)
void compute_gf2_col_weights_kernel(
    const uint32_t*    __restrict__ d_row_offsets,
    const PackedEntry* __restrict__ d_entries,
    uint32_t* __restrict__          d_gf2_col_weight,
    uint32_t n_rows)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const uint32_t begin = d_row_offsets[r];
    const uint32_t end   = d_row_offsets[r + 1];
    for (uint32_t j = begin; j < end; ++j)
        if (packed_exp(d_entries[j]) & 1)  // GF(2): odd exponents only
            atomicAdd(&d_gf2_col_weight[packed_col(d_entries[j])], 1u);
}

// =========================================================================
// DeviceMergeWorkspace RAII implementation
// =========================================================================

void DeviceMergeWorkspace::alloc(
    uint32_t n_rows, uint32_t nc,
    uint32_t est_merges, uint32_t est_entries, bool jetson)
{
    use_managed = jetson;
    n_total_rows = n_rows;
    n_cols = nc;

    // 1.5× safety margin on entries (rows already use n_rows upper bound)
    max_merged_rows   = est_merges;
    max_merged_entries = static_cast<uint32_t>(est_entries * 1.5f);

    // Minimum 1 element to avoid zero-size alloc.
    max_merged_rows   = std::max(max_merged_rows, 1u);
    max_merged_entries = std::max(max_merged_entries, 1u);
    const uint32_t safe_rows = std::max(n_rows, 1u);
    const uint32_t safe_cols = std::max(nc, 1u);

    auto device_malloc = [&](void** ptr, size_t bytes) {
        if (use_managed) {
            CUDA_CHECK(cudaMallocManaged(ptr, bytes));
        } else {
            CUDA_CHECK(cudaMalloc(ptr, bytes));
        }
    };

    // Workspace CSR
    device_malloc(reinterpret_cast<void**>(&d_ws_entries),
                  max_merged_entries * sizeof(PackedEntry));
    device_malloc(reinterpret_cast<void**>(&d_ws_row_starts),
                  max_merged_rows * sizeof(uint32_t));
    device_malloc(reinterpret_cast<void**>(&d_ws_row_lengths),
                  max_merged_rows * sizeof(uint32_t));

    // Per-row metadata
    device_malloc(reinterpret_cast<void**>(&d_ws_sqrt_Q),
                  max_merged_rows * sizeof(uint512));
    CUDA_CHECK(cudaMemset(d_ws_sqrt_Q, 0, max_merged_rows * sizeof(uint512)));
    device_malloc(reinterpret_cast<void**>(&d_ws_signs),
                  max_merged_rows * sizeof(uint8_t));
    CUDA_CHECK(cudaMemset(d_ws_signs, 1, max_merged_rows));  // Default sign = +1 (positive)
    device_malloc(reinterpret_cast<void**>(&d_ws_val_2_exps),
                  max_merged_rows * sizeof(int32_t));
    CUDA_CHECK(cudaMemset(d_ws_val_2_exps, 0, max_merged_rows * sizeof(int32_t)));

    // Dual counter
    device_malloc(reinterpret_cast<void**>(&d_dual_counter), sizeof(uint64_t));
    CUDA_CHECK(cudaMemset(d_dual_counter, 0, sizeof(uint64_t)));

    // Row indirection + locks
    device_malloc(reinterpret_cast<void**>(&d_row_ptr),
                  safe_rows * sizeof(uint32_t));
    device_malloc(reinterpret_cast<void**>(&d_row_locks),
                  safe_rows * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(d_row_locks, 0, safe_rows * sizeof(uint32_t)));

    // Column weights
    device_malloc(reinterpret_cast<void**>(&d_col_weight),
                  safe_cols * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(d_col_weight, 0, safe_cols * sizeof(uint32_t)));

    // GF(2) column weights (odd-exponent only)
    device_malloc(reinterpret_cast<void**>(&d_gf2_col_weight),
                  safe_cols * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(d_gf2_col_weight, 0, safe_cols * sizeof(uint32_t)));

    // Abort count
    device_malloc(reinterpret_cast<void**>(&d_abort_count), sizeof(uint32_t));
}

void DeviceMergeWorkspace::initRowState(uint32_t n_rows, cudaStream_t stream) {
    uint32_t blocks = (n_rows + 255u) / 256u;
    init_row_ptr_kernel<<<blocks, 256, 0, stream>>>(d_row_ptr, n_rows);
    CUDA_CHECK(cudaMemsetAsync(d_row_locks, 0, n_rows * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

DeviceMergeWorkspace::~DeviceMergeWorkspace() {
    // cudaFree handles both cudaMalloc and cudaMallocManaged pointers.
    if (d_ws_entries)     cudaFree(d_ws_entries);
    if (d_ws_row_starts)  cudaFree(d_ws_row_starts);
    if (d_ws_row_lengths) cudaFree(d_ws_row_lengths);
    if (d_ws_sqrt_Q)      cudaFree(d_ws_sqrt_Q);
    if (d_ws_signs)       cudaFree(d_ws_signs);
    if (d_ws_val_2_exps)  cudaFree(d_ws_val_2_exps);
    if (d_dual_counter)   cudaFree(d_dual_counter);
    if (d_row_ptr)        cudaFree(d_row_ptr);
    if (d_row_locks)      cudaFree(d_row_locks);
    if (d_col_weight)     cudaFree(d_col_weight);
    if (d_gf2_col_weight) cudaFree(d_gf2_col_weight);
    if (d_abort_count)    cudaFree(d_abort_count);
}

DeviceMergeWorkspace::DeviceMergeWorkspace(DeviceMergeWorkspace&& other) noexcept
    : d_ws_entries    (other.d_ws_entries),
      d_ws_row_starts (other.d_ws_row_starts),
      d_ws_row_lengths(other.d_ws_row_lengths),
      d_ws_sqrt_Q     (other.d_ws_sqrt_Q),
      d_ws_signs      (other.d_ws_signs),
      d_ws_val_2_exps (other.d_ws_val_2_exps),
      d_dual_counter  (other.d_dual_counter),
      d_row_ptr       (other.d_row_ptr),
      d_row_locks     (other.d_row_locks),
      d_col_weight    (other.d_col_weight),
      d_gf2_col_weight(other.d_gf2_col_weight),
      d_abort_count   (other.d_abort_count),
      max_merged_rows  (other.max_merged_rows),
      max_merged_entries(other.max_merged_entries),
      n_total_rows     (other.n_total_rows),
      n_cols           (other.n_cols),
      use_managed      (other.use_managed)
{
    other.d_ws_entries     = nullptr;
    other.d_ws_row_starts  = nullptr;
    other.d_ws_row_lengths = nullptr;
    other.d_ws_sqrt_Q      = nullptr;
    other.d_ws_signs       = nullptr;
    other.d_ws_val_2_exps  = nullptr;
    other.d_dual_counter   = nullptr;
    other.d_row_ptr        = nullptr;
    other.d_row_locks      = nullptr;
    other.d_col_weight     = nullptr;
    other.d_gf2_col_weight = nullptr;
    other.d_abort_count    = nullptr;
    other.max_merged_rows   = 0;
    other.max_merged_entries = 0;
    other.n_total_rows      = 0;
    other.n_cols            = 0;
}

DeviceMergeWorkspace& DeviceMergeWorkspace::operator=(DeviceMergeWorkspace&& other) noexcept {
    if (this == &other) return *this;
    // Free own buffers
    if (d_ws_entries)     cudaFree(d_ws_entries);
    if (d_ws_row_starts)  cudaFree(d_ws_row_starts);
    if (d_ws_row_lengths) cudaFree(d_ws_row_lengths);
    if (d_ws_sqrt_Q)      cudaFree(d_ws_sqrt_Q);
    if (d_ws_signs)       cudaFree(d_ws_signs);
    if (d_ws_val_2_exps)  cudaFree(d_ws_val_2_exps);
    if (d_dual_counter)   cudaFree(d_dual_counter);
    if (d_row_ptr)        cudaFree(d_row_ptr);
    if (d_row_locks)      cudaFree(d_row_locks);
    if (d_col_weight)     cudaFree(d_col_weight);
    if (d_gf2_col_weight) cudaFree(d_gf2_col_weight);
    if (d_abort_count)    cudaFree(d_abort_count);
    // Transfer
    d_ws_entries     = other.d_ws_entries;
    d_ws_row_starts  = other.d_ws_row_starts;
    d_ws_row_lengths = other.d_ws_row_lengths;
    d_ws_sqrt_Q      = other.d_ws_sqrt_Q;
    d_ws_signs       = other.d_ws_signs;
    d_ws_val_2_exps  = other.d_ws_val_2_exps;
    d_dual_counter   = other.d_dual_counter;
    d_row_ptr        = other.d_row_ptr;
    d_row_locks      = other.d_row_locks;
    d_col_weight     = other.d_col_weight;
    d_gf2_col_weight = other.d_gf2_col_weight;
    d_abort_count    = other.d_abort_count;
    max_merged_rows   = other.max_merged_rows;
    max_merged_entries = other.max_merged_entries;
    n_total_rows      = other.n_total_rows;
    n_cols            = other.n_cols;
    use_managed       = other.use_managed;
    // Null out other
    other.d_ws_entries     = nullptr;
    other.d_ws_row_starts  = nullptr;
    other.d_ws_row_lengths = nullptr;
    other.d_ws_sqrt_Q      = nullptr;
    other.d_ws_signs       = nullptr;
    other.d_ws_val_2_exps  = nullptr;
    other.d_dual_counter   = nullptr;
    other.d_row_ptr        = nullptr;
    other.d_row_locks      = nullptr;
    other.d_col_weight     = nullptr;
    other.d_gf2_col_weight = nullptr;
    other.d_abort_count    = nullptr;
    other.max_merged_rows   = 0;
    other.max_merged_entries = 0;
    other.n_total_rows      = 0;
    other.n_cols            = 0;
    return *this;
}

// =========================================================================
// CPU merge planner: buildInvertedIndex
// =========================================================================

/// Build inverted index col_rows[c] from host-side packed CSR row offsets + entries.
/// Downloads row_offsets and entries from device, iterates rows, records which
/// alive rows contain each column.
static void buildInvertedIndex(
    const std::vector<uint32_t>& h_row_offsets,
    const std::vector<PackedEntry>& h_entries,
    const std::vector<uint32_t>& h_row_ptr,
    uint32_t n_rows,
    uint32_t n_cols,
    std::vector<std::vector<uint32_t>>& col_rows)
{
    col_rows.assign(n_cols, {});
    for (uint32_t r = 0; r < n_rows; ++r) {
        if (h_row_ptr[r] == ROW_DEAD) continue;

        uint32_t ptr_val = h_row_ptr[r];
        // For the inverted index we only handle original rows (MSB=0).
        // Workspace rows are not directly represented in the original CSR.
        // After a merge round, r's row_ptr may point to workspace, so we
        // need the workspace data too. But the simplified per-round rebuild
        // only processes the original CSR structure, which is sufficient
        // because the CPU planner downloads col_weight from device.
        // The inverted index for weight-2 is rebuilt each round from
        // the device-authoritative col_weight anyway.
        if (ptr_val == ROW_DEAD) continue;

        // Get entries for this row from original CSR
        // Note: if ptr_val has WS_BIT, the row's entries are in workspace.
        // For the inverted index, we need to download workspace data too.
        // But the simplification says: we only track which rows are alive
        // per column, not the actual entries. The device col_weight is authoritative.
        // The inverted index maps col → alive rows. We scan original CSR for this.
        if (ptr_val & ROW_WS_BIT) {
            // Skip workspace rows — we cannot read workspace entries from host
            // without downloading them. This is handled in the round loop by
            // re-downloading the full col_weight and rebuilding.
            continue;
        }

        uint32_t begin = h_row_offsets[ptr_val];
        uint32_t end   = h_row_offsets[ptr_val + 1];
        for (uint32_t j = begin; j < end; ++j) {
            uint32_t col = packed_col(h_entries[j]);
            if (col < n_cols && (packed_exp(h_entries[j]) & 1))  // GF(2): odd exponents only
                col_rows[col].push_back(r);
        }
    }
}

/// Build inverted index that also handles workspace rows.
/// Requires downloading workspace row data for rows with WS_BIT.
static void buildInvertedIndexFull(
    const std::vector<uint32_t>& h_row_offsets,
    const std::vector<PackedEntry>& h_entries,
    const std::vector<uint32_t>& h_row_ptr,
    const std::vector<uint32_t>& h_ws_row_starts,
    const std::vector<uint32_t>& h_ws_row_lengths,
    const std::vector<PackedEntry>& h_ws_entries,
    uint32_t n_rows,
    uint32_t n_cols,
    std::vector<std::vector<uint32_t>>& col_rows,
    std::vector<uint32_t>& row_weights)
{
    col_rows.assign(n_cols, {});
    row_weights.assign(n_rows, 0);

    for (uint32_t r = 0; r < n_rows; ++r) {
        uint32_t ptr_val = h_row_ptr[r];
        if (ptr_val == ROW_DEAD) continue;

        if (ptr_val & ROW_WS_BIT) {
            uint32_t ws_idx = ptr_val & 0x7FFFFFFFu;
            uint32_t start = h_ws_row_starts[ws_idx];
            uint32_t len   = h_ws_row_lengths[ws_idx];
            row_weights[r] = len;
            for (uint32_t j = 0; j < len; ++j) {
                PackedEntry e = h_ws_entries[start + j];
                uint32_t col = packed_col(e);
                if (col < n_cols && (packed_exp(e) & 1))  // GF(2): odd exponents only
                    col_rows[col].push_back(r);
            }
        } else {
            uint32_t begin = h_row_offsets[ptr_val];
            uint32_t end   = h_row_offsets[ptr_val + 1];
            row_weights[r] = end - begin;
            for (uint32_t j = begin; j < end; ++j) {
                PackedEntry e = h_entries[j];
                uint32_t col = packed_col(e);
                if (col < n_cols && (packed_exp(e) & 1))  // GF(2): odd exponents only
                    col_rows[col].push_back(r);
            }
        }
    }
}

// =========================================================================
// CPU merge planner: planWeight2Merges
// =========================================================================

MergePlan planWeight2Merges(
    const std::vector<uint32_t>& h_col_weight,
    uint32_t n_cols,
    const std::vector<std::vector<uint32_t>>& col_rows,
    const std::vector<uint32_t>& h_row_ptr)
{
    MergePlan plan;

    // Track which rows are already committed in this batch to avoid
    // double-scheduling (reduces abort rate).
    std::vector<bool> row_used(h_row_ptr.size(), false);

    for (uint32_t c = 0; c < n_cols; ++c) {
        if (h_col_weight[c] != 2) continue;

        // Find the two alive rows containing column c.
        uint32_t r1 = UINT32_MAX, r2 = UINT32_MAX;
        for (uint32_t r : col_rows[c]) {
            if (h_row_ptr[r] == ROW_DEAD) continue;
            if (r1 == UINT32_MAX) r1 = r;
            else { r2 = r; break; }
        }
        if (r1 == UINT32_MAX || r2 == UINT32_MAX) continue;

        // Skip if either row is already in this batch (reduce contention)
        if (row_used[r1] || row_used[r2]) continue;

        plan.candidates.push_back({r1, r2, c});
        row_used[r1] = true;
        row_used[r2] = true;
    }

    return plan;
}

// =========================================================================
// CPU merge planner: planHigherWeightMerges
// =========================================================================

MergePlan planHigherWeightMerges(
    const std::vector<uint32_t>& h_col_weight,
    uint32_t n_cols,
    uint32_t k_max,
    uint32_t max_fill_in,
    const std::vector<std::vector<uint32_t>>& col_rows,
    const std::vector<uint32_t>& h_row_ptr,
    const std::vector<uint32_t>& row_weights)
{
    MergePlan plan;

    // Collect eligible columns and sort by ascending weight (Markowitz).
    std::vector<std::pair<uint32_t, uint32_t>> eligible; // (weight, col)
    for (uint32_t c = 0; c < n_cols; ++c) {
        uint32_t w = h_col_weight[c];
        if (w >= 3 && w <= k_max) {
            eligible.push_back({w, c});
        }
    }
    std::sort(eligible.begin(), eligible.end());

    // Track which rows are already committed as r2 (pivot) to avoid
    // conflicting claims on the same pivot from different columns.
    std::vector<bool> row_used(h_row_ptr.size(), false);

    for (auto& [w, c] : eligible) {
        // Re-check weight (may have changed from prior columns in this batch)
        if (h_col_weight[c] < 3 || h_col_weight[c] > k_max) continue;

        // Gather alive rows containing column c.
        std::vector<uint32_t> containing;
        for (uint32_t r : col_rows[c]) {
            if (h_row_ptr[r] != ROW_DEAD) {
                containing.push_back(r);
            }
        }
        if (containing.size() < 2) continue;

        // Choose lightest row as pivot (Markowitz heuristic).
        uint32_t pivot_idx = 0;
        for (size_t i = 1; i < containing.size(); ++i) {
            if (row_weights[containing[i]] < row_weights[containing[pivot_idx]]) {
                pivot_idx = static_cast<uint32_t>(i);
            }
        }
        uint32_t pivot = containing[pivot_idx];

        // Skip if pivot is already committed elsewhere
        if (row_used[pivot]) continue;

        // Emit (other_row, pivot, c) for each non-pivot row.
        // other_row survives (r1), pivot is consumed (r2).
        bool any_emitted = false;
        for (size_t i = 0; i < containing.size(); ++i) {
            if (i == pivot_idx) continue;
            uint32_t r = containing[i];
            if (row_used[r]) continue;

            // Fill-in check: conservative upper bound
            uint32_t est_merged = row_weights[r] + row_weights[pivot];
            if (est_merged > max_fill_in) continue;

            plan.candidates.push_back({r, pivot, c});
            row_used[r] = true;
            any_emitted = true;
        }

        if (any_emitted) {
            row_used[pivot] = true;
        }
    }

    return plan;
}

// =========================================================================
// Host round-loop driver: gpuBatchMerge
// =========================================================================

BatchMergeResult gpuBatchMerge(
    DevicePackedCSR& csr,
    const MontgomeryContext& mont,
    uint32_t k_max,
    uint32_t max_weight,
    uint32_t max_total_merges)
{
    LOG_SET_MODULE("BatchMerge");

    const bool jetson = isJetsonDevice();
    const uint32_t n_rows = csr.n_rows;
    const uint32_t n_cols = csr.n_cols;

    LOG(LOG_INFO) << "Starting GPU batch merge: " << n_rows << " rows x "
                  << n_cols << " cols, k_max=" << k_max
                  << ", max_weight=" << max_weight << ".";

    // --- 1. Estimate workspace size ---
    // Each merge allocates one workspace row. The algorithm can perform up to
    // n_rows merges (theoretical max: every row participates as a merge victim).
    // At high LP fractions, observed merge count ≈ n_cols (all excess eliminated).
    // Use n_rows as the workspace row budget — this is the true upper bound.
    uint32_t est_merges = std::max(n_rows, 1024u);

    // Estimate avg row weight from NNZ/n_rows. Each merged row ≈ 2× avg weight.
    uint32_t avg_weight = (n_rows > 0) ? (csr.nnz / n_rows) : 32;
    uint32_t est_entries = est_merges * avg_weight * 2;

    LOG(LOG_INFO) << "Workspace estimate: " << est_merges << " merges, "
                  << est_entries << " entries (avg_weight=" << avg_weight << ").";

    // --- 2. Allocate workspace ---
    DeviceMergeWorkspace ws;
    ws.alloc(n_rows, n_cols, est_merges, est_entries, jetson);

    // --- 3. Initialize row state ---
    ws.initRowState(n_rows);

    // --- 4. Compute column weights from scratch ---
    // M9b's scratch buffers (including d_col_weight) are freed before returning.
    CUDA_CHECK(cudaMemset(ws.d_col_weight, 0, n_cols * sizeof(uint32_t)));
    {
        uint32_t blocks = (n_rows + 255u) / 256u;
        compute_col_weights_kernel<<<blocks, 256>>>(
            csr.d_row_offsets, csr.d_entries, ws.d_col_weight, n_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- 4b. Compute GF(2) column weights (odd-exponent entries only) ---
    CUDA_CHECK(cudaMemset(ws.d_gf2_col_weight, 0, n_cols * sizeof(uint32_t)));
    {
        uint32_t blocks = (n_rows + 255u) / 256u;
        compute_gf2_col_weights_kernel<<<blocks, 256>>>(
            csr.d_row_offsets, csr.d_entries, ws.d_gf2_col_weight, n_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- 5. Download host copies ---
    std::vector<uint32_t> h_col_weight(n_cols);
    std::vector<uint32_t> h_row_ptr(n_rows);
    std::vector<uint32_t> h_row_offsets(n_rows + 1);
    std::vector<PackedEntry> h_entries(csr.nnz);

    std::vector<uint32_t> h_gf2_col_weight(n_cols);

    CUDA_CHECK(cudaMemcpy(h_col_weight.data(), ws.d_col_weight,
                           n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gf2_col_weight.data(), ws.d_gf2_col_weight,
                           n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row_ptr.data(), ws.d_row_ptr,
                           n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row_offsets.data(), csr.d_row_offsets,
                           (n_rows + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_entries.data(), csr.d_entries,
                           csr.nnz * sizeof(PackedEntry), cudaMemcpyDeviceToHost));

    // Diagnostic: log first few non-zero GF(2) column weights
    {
        uint32_t shown = 0;
        for (uint32_t c = 0; c < n_cols && shown < 5; ++c) {
            if (h_gf2_col_weight[c] > 0) {
                LOG(LOG_DEBUG_1) << "  [S2 diag] gf2_col_weight[" << c << "] = "
                               << h_gf2_col_weight[c];
                ++shown;
            }
        }
    }

    // --- 6. Build initial inverted index ---
    std::vector<std::vector<uint32_t>> col_rows;
    buildInvertedIndex(h_row_offsets, h_entries, h_row_ptr,
                       n_rows, n_cols, col_rows);

    // =====================================================================
    // Weight-2 merge phase
    // =====================================================================
    uint32_t w2_total = 0, w2_rounds = 0;
    uint32_t total_aborts = 0;
    uint32_t cumulative_merges = 0;

    if (max_total_merges > 0) {
        LOG(LOG_INFO) << "Merge budget: " << max_total_merges << " total merges.";
    }

    while (w2_rounds < kMaxMergeRounds) {
        if (max_total_merges > 0 && cumulative_merges >= max_total_merges) {
            LOG(LOG_INFO) << "W2: merge budget exhausted (" << cumulative_merges << ").";
            break;
        }
        MergePlan plan = planWeight2Merges(h_gf2_col_weight, n_cols, col_rows, h_row_ptr);
        if (plan.candidates.empty()) break;

        // Upload candidates to device
        MergeCandidate* d_candidates = nullptr;
        size_t cand_bytes = plan.candidates.size() * sizeof(MergeCandidate);
        CUDA_CHECK(cudaMalloc(&d_candidates, cand_bytes));
        CUDA_CHECK(cudaMemcpy(d_candidates, plan.candidates.data(),
                               cand_bytes, cudaMemcpyHostToDevice));

        // Reset abort counter
        CUDA_CHECK(cudaMemset(ws.d_abort_count, 0, sizeof(uint32_t)));

        // Launch merge kernel
        uint32_t n = static_cast<uint32_t>(plan.candidates.size());
        uint32_t blocks = (n + 255u) / 256u;
        execute_merges_kernel<<<blocks, 256>>>(
            d_candidates, n,
            csr.d_row_offsets, csr.d_entries,
            csr.d_sqrt_Q, csr.d_signs, csr.d_val_2_exps,
            ws.d_ws_entries, ws.d_ws_row_starts, ws.d_ws_row_lengths,
            ws.d_ws_sqrt_Q, ws.d_ws_signs, ws.d_ws_val_2_exps,
            ws.d_dual_counter, ws.max_merged_rows, ws.max_merged_entries,
            ws.d_row_ptr, ws.d_row_locks,
            ws.d_col_weight, ws.d_gf2_col_weight, mont, ws.d_abort_count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Clear locks (belt-and-suspenders)
        clear_locks_kernel<<<(n_rows + 255u) / 256u, 256>>>(ws.d_row_locks, n_rows);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download results
        uint32_t h_abort_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_abort_count, ws.d_abort_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_col_weight.data(), ws.d_col_weight,
                               n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_gf2_col_weight.data(), ws.d_gf2_col_weight,
                               n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_row_ptr.data(), ws.d_row_ptr,
                               n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint32_t succeeded = n - h_abort_count;
        w2_total += succeeded;
        cumulative_merges += succeeded;
        total_aborts += h_abort_count;

        LOG(LOG_INFO) << "W2 round " << w2_rounds << ": " << n << " candidates, "
                      << succeeded << " succeeded, " << h_abort_count << " aborted ("
                      << (n > 0 ? (100.0 * h_abort_count / n) : 0.0) << "%).";

        if (h_abort_count > n / 10) {
            LOG(LOG_WARNING) << "W2 abort rate > 10% in round " << w2_rounds << ".";
        }

        // Rebuild inverted index from fresh data
        buildInvertedIndex(h_row_offsets, h_entries, h_row_ptr,
                           n_rows, n_cols, col_rows);

        CUDA_CHECK(cudaFree(d_candidates));
        w2_rounds++;
    }

    LOG(LOG_INFO) << "Weight-2 phase complete: " << w2_total << " merges in "
                  << w2_rounds << " rounds.";
    {
        uint32_t gf2_n_cols = 0;
        for (uint32_t c = 0; c < n_cols; ++c)
            if (h_gf2_col_weight[c] > 0) ++gf2_n_cols;
        LOG(LOG_INFO) << "  W2 done: GF(2) cols=" << gf2_n_cols
                      << " / " << n_cols << " packed cols.";
    }

    // =====================================================================
    // Higher-weight merge phase
    // =====================================================================
    uint32_t hw_total = 0, hw_rounds = 0;

    // Compute effective max fill-in (same logic as CPU merge_filter)
    uint64_t total_nnz = 0;
    uint32_t alive_rows = 0;
    for (uint32_t r = 0; r < n_rows; ++r) {
        if (h_row_ptr[r] != ROW_DEAD) alive_rows++;
    }
    // Use the CSR NNZ and alive row count for a rough average
    total_nnz = csr.nnz;  // original NNZ as approximation
    double avg_initial = (alive_rows > 0)
        ? static_cast<double>(total_nnz) / alive_rows : 32.0;
    uint32_t effective_max = std::max(max_weight,
                                       static_cast<uint32_t>(2.0 * avg_initial));

    // For higher-weight merges, we need workspace row data on host.
    // Download workspace row metadata for the inverted index.
    // Read dual counter to know how many workspace rows exist.
    uint64_t h_dual_packed = 0;
    CUDA_CHECK(cudaMemcpy(&h_dual_packed, ws.d_dual_counter,
                           sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint32_t ws_row_count = static_cast<uint32_t>(h_dual_packed >> 32);
    uint32_t ws_entry_count = static_cast<uint32_t>(h_dual_packed);

    std::vector<uint32_t> h_ws_row_starts(ws_row_count);
    std::vector<uint32_t> h_ws_row_lengths(ws_row_count);
    std::vector<PackedEntry> h_ws_entries(ws_entry_count);

    if (ws_row_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_ws_row_starts.data(), ws.d_ws_row_starts,
                               ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ws_row_lengths.data(), ws.d_ws_row_lengths,
                               ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    if (ws_entry_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_ws_entries.data(), ws.d_ws_entries,
                               ws_entry_count * sizeof(PackedEntry), cudaMemcpyDeviceToHost));
    }

    // Build full inverted index (including workspace rows) + row weights
    std::vector<uint32_t> row_weights;
    buildInvertedIndexFull(h_row_offsets, h_entries, h_row_ptr,
                           h_ws_row_starts, h_ws_row_lengths, h_ws_entries,
                           n_rows, n_cols, col_rows, row_weights);

    while (hw_rounds < kMaxMergeRounds) {
        if (max_total_merges > 0 && cumulative_merges >= max_total_merges) {
            LOG(LOG_INFO) << "HW: merge budget exhausted (" << cumulative_merges << ").";
            break;
        }
        MergePlan plan = planHigherWeightMerges(
            h_gf2_col_weight, n_cols, k_max, effective_max,
            col_rows, h_row_ptr, row_weights);
        if (plan.candidates.empty()) break;

        // Upload candidates
        MergeCandidate* d_candidates = nullptr;
        size_t cand_bytes = plan.candidates.size() * sizeof(MergeCandidate);
        CUDA_CHECK(cudaMalloc(&d_candidates, cand_bytes));
        CUDA_CHECK(cudaMemcpy(d_candidates, plan.candidates.data(),
                               cand_bytes, cudaMemcpyHostToDevice));

        // Reset abort counter
        CUDA_CHECK(cudaMemset(ws.d_abort_count, 0, sizeof(uint32_t)));

        // Launch merge kernel
        uint32_t n = static_cast<uint32_t>(plan.candidates.size());
        uint32_t blocks = (n + 255u) / 256u;
        execute_merges_kernel<<<blocks, 256>>>(
            d_candidates, n,
            csr.d_row_offsets, csr.d_entries,
            csr.d_sqrt_Q, csr.d_signs, csr.d_val_2_exps,
            ws.d_ws_entries, ws.d_ws_row_starts, ws.d_ws_row_lengths,
            ws.d_ws_sqrt_Q, ws.d_ws_signs, ws.d_ws_val_2_exps,
            ws.d_dual_counter, ws.max_merged_rows, ws.max_merged_entries,
            ws.d_row_ptr, ws.d_row_locks,
            ws.d_col_weight, ws.d_gf2_col_weight, mont, ws.d_abort_count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Clear locks
        clear_locks_kernel<<<(n_rows + 255u) / 256u, 256>>>(ws.d_row_locks, n_rows);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download results
        uint32_t h_abort_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_abort_count, ws.d_abort_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_col_weight.data(), ws.d_col_weight,
                               n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_gf2_col_weight.data(), ws.d_gf2_col_weight,
                               n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_row_ptr.data(), ws.d_row_ptr,
                               n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint32_t succeeded = n - h_abort_count;
        hw_total += succeeded;
        cumulative_merges += succeeded;
        total_aborts += h_abort_count;

        LOG(LOG_INFO) << "HW round " << hw_rounds << ": " << n << " candidates, "
                      << succeeded << " succeeded, " << h_abort_count << " aborted ("
                      << (n > 0 ? (100.0 * h_abort_count / n) : 0.0) << "%).";

        if (h_abort_count > n / 10) {
            LOG(LOG_WARNING) << "HW abort rate > 10% in round " << hw_rounds << ".";
        }

        // Re-download workspace data for next round's inverted index
        CUDA_CHECK(cudaMemcpy(&h_dual_packed, ws.d_dual_counter,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost));
        ws_row_count = static_cast<uint32_t>(h_dual_packed >> 32);
        ws_entry_count = static_cast<uint32_t>(h_dual_packed);

        h_ws_row_starts.resize(ws_row_count);
        h_ws_row_lengths.resize(ws_row_count);
        h_ws_entries.resize(ws_entry_count);

        if (ws_row_count > 0) {
            CUDA_CHECK(cudaMemcpy(h_ws_row_starts.data(), ws.d_ws_row_starts,
                                   ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ws_row_lengths.data(), ws.d_ws_row_lengths,
                                   ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        if (ws_entry_count > 0) {
            CUDA_CHECK(cudaMemcpy(h_ws_entries.data(), ws.d_ws_entries,
                                   ws_entry_count * sizeof(PackedEntry), cudaMemcpyDeviceToHost));
        }

        buildInvertedIndexFull(h_row_offsets, h_entries, h_row_ptr,
                               h_ws_row_starts, h_ws_row_lengths, h_ws_entries,
                               n_rows, n_cols, col_rows, row_weights);

        CUDA_CHECK(cudaFree(d_candidates));
        hw_rounds++;
    }

    LOG(LOG_INFO) << "Higher-weight phase complete: " << hw_total << " merges in "
                  << hw_rounds << " rounds.";
    {
        uint32_t gf2_n_cols = 0;
        for (uint32_t c = 0; c < n_cols; ++c)
            if (h_gf2_col_weight[c] > 0) ++gf2_n_cols;
        LOG(LOG_INFO) << "  HW done: GF(2) cols=" << gf2_n_cols
                      << " / " << n_cols << " packed cols.";
    }

    // --- Workspace utilization check ---
    CUDA_CHECK(cudaMemcpy(&h_dual_packed, ws.d_dual_counter,
                           sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint32_t final_ws_rows = static_cast<uint32_t>(h_dual_packed >> 32);
    uint32_t final_ws_entries = static_cast<uint32_t>(h_dual_packed);

    double row_util = (ws.max_merged_rows > 0)
        ? 100.0 * final_ws_rows / ws.max_merged_rows : 0.0;
    double entry_util = (ws.max_merged_entries > 0)
        ? 100.0 * final_ws_entries / ws.max_merged_entries : 0.0;

    LOG(LOG_INFO) << "Workspace utilization: " << final_ws_rows << "/"
                  << ws.max_merged_rows << " rows (" << row_util << "%), "
                  << final_ws_entries << "/" << ws.max_merged_entries
                  << " entries (" << entry_util << "%).";

    if (row_util > 80.0 || entry_util > 80.0) {
        LOG(LOG_WARNING) << "Workspace utilization > 80% — consider increasing safety margin.";
    }

    // --- Count surviving rows ---
    uint32_t alive_count = 0;
    for (uint32_t r = 0; r < n_rows; ++r) {
        if (h_row_ptr[r] != ROW_DEAD) alive_count++;
    }

    uint32_t total_rounds = w2_rounds + hw_rounds;
    LOG(LOG_INFO) << "GPU batch merge complete: "
                  << w2_total << " W2 + " << hw_total << " HW = "
                  << (w2_total + hw_total) << " total merges, "
                  << total_rounds << " rounds, "
                  << total_aborts << " total aborts, "
                  << alive_count << " surviving rows.";

    // --- Build result ---
    BatchMergeResult result;
    result.workspace    = std::move(ws);
    result.h_row_ptr    = std::move(h_row_ptr);
    result.h_col_weight     = std::move(h_col_weight);
    result.h_gf2_col_weight = std::move(h_gf2_col_weight);
    result.w2_merges    = w2_total;
    result.hw_merges    = hw_total;
    result.total_rounds = total_rounds;
    result.total_aborts = total_aborts;

    return result;
}

} // namespace matrix
} // namespace mpqs
