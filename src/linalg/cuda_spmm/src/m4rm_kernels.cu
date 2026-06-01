// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

// =================================================================================
// Kernel: Vectorized Inverse M4RM (Grid Stride)
// =================================================================================
#include "m4rm_data.h"
#include "vec_type.h" 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

// =================================================================================
// Tuning Constants
// =================================================================================
#define M4RM_BLOCK_SIZE 512
#define M4RM_WARPS_PER_BLOCK (M4RM_BLOCK_SIZE / 32)
#define M4RM_BUCKETS 256
#define M4RM_ROWS 8

// =================================================================================
// Kernel: Vectorized Inverse M4RM (Grid Stride)
// =================================================================================
template<int VEC_BITS>
__global__ void m4rm_inverse_kernel_tmpl(
    const uint8_t* __restrict__ patterns,
    const void* __restrict__ vectors, 
    void* __restrict__ results,       
    size_t num_relations,
    int stride_in_type_units 
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    const VecT* V = (const VecT*)vectors;
    VecT* C = (VecT*)results;

    extern __shared__ char smem[];
    VecT* bucket_base = (VecT*)smem;
    auto get_bucket = [&](int warp, int bucket) -> VecT* { return &bucket_base[warp * M4RM_BUCKETS + bucket]; };

    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid >> 5;
    uint32_t lane_id = tid & 31;

    // 1. Init Shared Memory
    #pragma unroll
    for (int i = 0; i < M4RM_BUCKETS; i += 32) *get_bucket(warp_id, i + lane_id) = VecType<VEC_BITS>::zero();
    __syncthreads();

    // 2. Grid-Stride Loop
    size_t total_threads = (size_t)gridDim.x * blockDim.x;
    size_t global_tid = (size_t)blockIdx.x * blockDim.x + tid;

    for (size_t idx = global_tid; idx < num_relations; idx += total_threads) {
        uint8_t p = patterns[idx];
        
        // Use stride to access the correct vector element
        VecT v = VecType<VEC_BITS>::load(&V[idx * stride_in_type_units]);
        
        // CRITICAL FIX: Use active mask for divergence safe sync
        uint32_t active = __activemask();
        uint32_t peer_mask = __match_any_sync(active, p);
        int leader = __ffs(peer_mask) - 1;
        
        VecT agg = VecType<VEC_BITS>::zero();
        uint32_t work_mask = peer_mask;
        
        while (work_mask) {
            int src = __ffs(work_mask) - 1;
            // Pass peer_mask explicitly to shfl to allow partial warp shuffle
            VecT val_src = VecType<VEC_BITS>::shfl(v, src, 32, peer_mask);
            if (lane_id == leader) agg = VecType<VEC_BITS>::xor_val(agg, val_src);
            work_mask &= ~(1u << src);
        }

        if (lane_id == leader) VecType<VEC_BITS>::atomic_xor_shared(get_bucket(warp_id, p), agg);
    }
    __syncthreads();

    // 3. Final Transform
    if (tid < M4RM_BUCKETS) {
        VecT acc = VecType<VEC_BITS>::zero();
        #pragma unroll
        for (int w = 0; w < M4RM_WARPS_PER_BLOCK; ++w) acc = VecType<VEC_BITS>::xor_val(acc, *get_bucket(w, tid));
        
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            int mask = 1 << k;
            VecT neighbor = VecType<VEC_BITS>::shfl_xor(acc, mask);
            if ((tid & mask) == 0) acc = VecType<VEC_BITS>::xor_val(acc, neighbor);
        }
        
        #pragma unroll
        for (int k = 5; k < 8; ++k) {
            int mask = 1 << k;
            *get_bucket(0, tid) = acc;
            __syncthreads();
            if ((tid & mask) == 0) acc = VecType<VEC_BITS>::xor_val(acc, *get_bucket(0, tid ^ mask));
            __syncthreads();
        }
        
        if (tid > 0 && (tid & (tid - 1)) == 0) {
            int row_idx = __ffs(tid) - 1;
            // Output uses stride to write to correct row in C
            if (row_idx < M4RM_ROWS) VecType<VEC_BITS>::atomic_xor_global(&C[row_idx * stride_in_type_units], acc);
        }
    }
}

// =================================================================================
// Host Launcher with Strip-Mining
// =================================================================================

void launch_m4rm_full(
    const M4RMContext& ctx, 
    const void* d_V_void, 
    void* d_C_void, 
    int total_width_bits, 
    cudaStream_t stream
) {
    if (ctx.num_relations == 0) return;

    const uint64_t* d_V = (const uint64_t*)d_V_void;
    uint64_t* d_C = (uint64_t*)d_C_void;

    int block_size = M4RM_BLOCK_SIZE;
    int num_blocks = 256; 
    size_t needed_blocks = (ctx.num_relations + block_size - 1) / block_size;
    if (needed_blocks < (size_t)num_blocks) num_blocks = (int)needed_blocks;
    if (num_blocks == 0) num_blocks = 1;

    auto launch_tmpl = [&](int bits, int offset_bytes) {
        void* d_V_offset = (void*)((uint8_t*)d_V + offset_bytes);
        void* d_C_offset = (void*)((uint8_t*)d_C + offset_bytes);
        
        // Calculate stride. 
        // Example: if total=256, bits=128. Stride = 2.
        // Input V is packed as [256][256]. We access [0..127] then [128..255].
        // Stride tells kernel to jump 2*128bits to get to the next element.
        int stride = total_width_bits / bits;

        switch (bits) {
            case 32: {
                size_t smem = M4RM_WARPS_PER_BLOCK * M4RM_BUCKETS * 4;
                m4rm_inverse_kernel_tmpl<32><<<num_blocks, block_size, smem, stream>>>(
                    ctx.d_pattern_stream, d_V_offset, d_C_offset, ctx.num_relations, stride);
                break;
            }
            case 64: {
                size_t smem = M4RM_WARPS_PER_BLOCK * M4RM_BUCKETS * 8;
                m4rm_inverse_kernel_tmpl<64><<<num_blocks, block_size, smem, stream>>>(
                    ctx.d_pattern_stream, d_V_offset, d_C_offset, ctx.num_relations, stride);
                break;
            }
            case 128: {
                size_t smem = M4RM_WARPS_PER_BLOCK * M4RM_BUCKETS * 16;
                cudaFuncSetAttribute(m4rm_inverse_kernel_tmpl<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                m4rm_inverse_kernel_tmpl<128><<<num_blocks, block_size, smem, stream>>>(
                    ctx.d_pattern_stream, d_V_offset, d_C_offset, ctx.num_relations, stride);
                break;
            }
            default: throw std::runtime_error("M4RM Internal: Invalid strip width");
        }
    };

    int remaining_bits = total_width_bits;
    int current_offset_bytes = 0;

    while (remaining_bits > 0) {
        if (remaining_bits >= 128) {
            launch_tmpl(128, current_offset_bytes);
            remaining_bits -= 128;
            current_offset_bytes += 16;
        } else if (remaining_bits >= 64) {
            launch_tmpl(64, current_offset_bytes);
            remaining_bits -= 64;
            current_offset_bytes += 8;
        } else {
            launch_tmpl(32, current_offset_bytes);
            remaining_bits -= 32;
            current_offset_bytes += 4;
        }
    }
}

