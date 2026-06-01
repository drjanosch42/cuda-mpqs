// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// 128-bit is native uint4
struct __align__(16) uint256_t {
    uint4 a, b;
};

struct __align__(16) uint512_t {
    uint4 a, b, c, d;
};

#ifdef __CUDACC__

template<int BITS> struct VecType;

// --- 32-bit Specialization ---
template<> struct VecType<32> {
    using Type = uint32_t;
    __device__ static __forceinline__ Type zero() { return 0U; }
    __device__ static __forceinline__ Type load(const Type* ptr) { return *ptr; } // Simplification
    __device__ static __forceinline__ void store(Type* ptr, Type val) { *ptr = val; }
    __device__ static __forceinline__ Type xor_val(Type a, Type b) { return a ^ b; }
    
    __device__ static __forceinline__ void atomic_xor_shared(Type* addr, Type val) { atomicXor(addr, val); }
    __device__ static __forceinline__ void atomic_xor_global(Type* addr, Type val) { atomicXor(addr, val); }
    
    // Updated Shuffles to take mask
    __device__ static __forceinline__ Type shfl_xor(Type val, int lane_mask, unsigned int width=32, unsigned int mask=0xFFFFFFFF) { 
        return __shfl_xor_sync(mask, val, lane_mask, width); 
    }
    __device__ static __forceinline__ Type shfl_down(Type val, int delta, unsigned int width=32, unsigned int mask=0xFFFFFFFF) { 
        return __shfl_down_sync(mask, val, delta, width); 
    }
    __device__ static __forceinline__ Type shfl(Type val, int src_lane, unsigned int width=32, unsigned int mask=0xFFFFFFFF) { 
        return __shfl_sync(mask, val, src_lane, width); 
    }
};

// --- 64-bit Specialization ---
template<> struct VecType<64> {
    using Type = uint64_t;
    __device__ static __forceinline__ Type zero() { return 0ULL; }
    __device__ static __forceinline__ Type load(const Type* ptr) { return *ptr; }
    __device__ static __forceinline__ void store(Type* ptr, Type val) { *ptr = val; }
    __device__ static __forceinline__ Type xor_val(Type a, Type b) { return a ^ b; }
    
    __device__ static __forceinline__ void atomic_xor_shared(Type* addr, Type val) { atomicXor((unsigned long long*)addr, (unsigned long long)val); }
    __device__ static __forceinline__ void atomic_xor_global(Type* addr, Type val) { atomicXor((unsigned long long*)addr, (unsigned long long)val); }

    __device__ static __forceinline__ Type shfl_xor(Type val, int lane_mask, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        uint32_t lo = __shfl_xor_sync(mask, (uint32_t)val, lane_mask, width);
        uint32_t hi = __shfl_xor_sync(mask, (uint32_t)(val >> 32), lane_mask, width);
        return ((uint64_t)hi << 32) | lo;
    }
    __device__ static __forceinline__ Type shfl_down(Type val, int delta, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        uint32_t lo = __shfl_down_sync(mask, (uint32_t)val, delta, width);
        uint32_t hi = __shfl_down_sync(mask, (uint32_t)(val >> 32), delta, width);
        return ((uint64_t)hi << 32) | lo;
    }
    __device__ static __forceinline__ Type shfl(Type val, int src_lane, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        uint32_t lo = __shfl_sync(mask, (uint32_t)val, src_lane, width);
        uint32_t hi = __shfl_sync(mask, (uint32_t)(val >> 32), src_lane, width);
        return ((uint64_t)hi << 32) | lo;
    }
};

// --- 128-bit Specialization ---
template<> struct VecType<128> {
    using Type = uint4;
    __device__ static __forceinline__ Type zero() { return make_uint4(0,0,0,0); }
    __device__ static __forceinline__ Type load(const Type* ptr) { return *ptr; }
    __device__ static __forceinline__ void store(Type* ptr, Type val) { *ptr = val; }
    __device__ static __forceinline__ Type xor_val(Type a, Type b) { 
        return make_uint4(a.x^b.x, a.y^b.y, a.z^b.z, a.w^b.w); 
    }
    
    __device__ static __forceinline__ void atomic_xor_shared(Type* addr, Type val) {
        unsigned long long* addr64 = (unsigned long long*)addr;
        unsigned long long v0, v1;
        asm("mov.b64 %0, {%1, %2};" : "=l"(v0) : "r"(val.x), "r"(val.y));
        asm("mov.b64 %0, {%1, %2};" : "=l"(v1) : "r"(val.z), "r"(val.w));
        atomicXor(&addr64[0], v0);
        atomicXor(&addr64[1], v1);
    }
    __device__ static __forceinline__ void atomic_xor_global(Type* addr, Type val) { atomic_xor_shared(addr, val); }

    __device__ static __forceinline__ Type shfl_xor(Type val, int lane_mask, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return make_uint4(
            __shfl_xor_sync(mask, val.x, lane_mask, width), __shfl_xor_sync(mask, val.y, lane_mask, width),
            __shfl_xor_sync(mask, val.z, lane_mask, width), __shfl_xor_sync(mask, val.w, lane_mask, width)
        );
    }
    __device__ static __forceinline__ Type shfl_down(Type val, int delta, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return make_uint4(
            __shfl_down_sync(mask, val.x, delta, width), __shfl_down_sync(mask, val.y, delta, width),
            __shfl_down_sync(mask, val.z, delta, width), __shfl_down_sync(mask, val.w, delta, width)
        );
    }
    __device__ static __forceinline__ Type shfl(Type val, int src_lane, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return make_uint4(
            __shfl_sync(mask, val.x, src_lane, width), __shfl_sync(mask, val.y, src_lane, width),
            __shfl_sync(mask, val.z, src_lane, width), __shfl_sync(mask, val.w, src_lane, width)
        );
    }
};

// --- 256-bit Specialization ---
template<> struct VecType<256> {
    using Type = uint256_t;
    __device__ static __forceinline__ Type zero() { return { VecType<128>::zero(), VecType<128>::zero() }; }
    
    __device__ static __forceinline__ Type load(const Type* ptr) { 
        const uint4* p4 = (const uint4*)ptr;
        return { p4[0], p4[1] };
    }
    __device__ static __forceinline__ void store(Type* ptr, Type val) {
        uint4* p4 = (uint4*)ptr; p4[0] = val.a; p4[1] = val.b;
    }
    __device__ static __forceinline__ Type xor_val(Type a, Type b) { 
        return { VecType<128>::xor_val(a.a, b.a), VecType<128>::xor_val(a.b, b.b) }; 
    }
    __device__ static __forceinline__ void atomic_xor_shared(Type* addr, Type val) {
        uint4* addr4 = (uint4*)addr;
        VecType<128>::atomic_xor_shared(&addr4[0], val.a);
        VecType<128>::atomic_xor_shared(&addr4[1], val.b);
    }
    __device__ static __forceinline__ void atomic_xor_global(Type* addr, Type val) { atomic_xor_shared(addr, val); }

    __device__ static __forceinline__ Type shfl_xor(Type val, int lane_mask, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return { VecType<128>::shfl_xor(val.a, lane_mask, width, mask), VecType<128>::shfl_xor(val.b, lane_mask, width, mask) };
    }
    __device__ static __forceinline__ Type shfl_down(Type val, int delta, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return { VecType<128>::shfl_down(val.a, delta, width, mask), VecType<128>::shfl_down(val.b, delta, width, mask) };
    }
    __device__ static __forceinline__ Type shfl(Type val, int src_lane, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return { VecType<128>::shfl(val.a, src_lane, width, mask), VecType<128>::shfl(val.b, src_lane, width, mask) };
    }
};

// --- 512-bit Specialization ---
template<> struct VecType<512> {
    using Type = uint512_t;
    __device__ static __forceinline__ Type zero() { 
        return { VecType<128>::zero(), VecType<128>::zero(), VecType<128>::zero(), VecType<128>::zero() }; 
    }
    __device__ static __forceinline__ Type load(const Type* ptr) {
        const uint4* p4 = (const uint4*)ptr;
        return { p4[0], p4[1], p4[2], p4[3] };
    }
    __device__ static __forceinline__ void store(Type* ptr, Type val) {
        uint4* p4 = (uint4*)ptr; p4[0] = val.a; p4[1] = val.b; p4[2] = val.c; p4[3] = val.d;
    }
    __device__ static __forceinline__ Type xor_val(Type a, Type b) {
        return { VecType<128>::xor_val(a.a, b.a), VecType<128>::xor_val(a.b, b.b), VecType<128>::xor_val(a.c, b.c), VecType<128>::xor_val(a.d, b.d) };
    }
    __device__ static __forceinline__ void atomic_xor_shared(Type* addr, Type val) {
        uint4* addr4 = (uint4*)addr;
        VecType<128>::atomic_xor_shared(&addr4[0], val.a); VecType<128>::atomic_xor_shared(&addr4[1], val.b);
        VecType<128>::atomic_xor_shared(&addr4[2], val.c); VecType<128>::atomic_xor_shared(&addr4[3], val.d);
    }
    __device__ static __forceinline__ void atomic_xor_global(Type* addr, Type val) { atomic_xor_shared(addr, val); }

    __device__ static __forceinline__ Type shfl_xor(Type val, int lane_mask, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return { 
            VecType<128>::shfl_xor(val.a, lane_mask, width, mask), VecType<128>::shfl_xor(val.b, lane_mask, width, mask),
            VecType<128>::shfl_xor(val.c, lane_mask, width, mask), VecType<128>::shfl_xor(val.d, lane_mask, width, mask) 
        };
    }
    __device__ static __forceinline__ Type shfl_down(Type val, int delta, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return { 
            VecType<128>::shfl_down(val.a, delta, width, mask), VecType<128>::shfl_down(val.b, delta, width, mask),
            VecType<128>::shfl_down(val.c, delta, width, mask), VecType<128>::shfl_down(val.d, delta, width, mask) 
        };
    }
    __device__ static __forceinline__ Type shfl(Type val, int src_lane, unsigned int width=32, unsigned int mask=0xFFFFFFFF) {
        return { 
            VecType<128>::shfl(val.a, src_lane, width, mask), VecType<128>::shfl(val.b, src_lane, width, mask),
            VecType<128>::shfl(val.c, src_lane, width, mask), VecType<128>::shfl(val.d, src_lane, width, mask) 
        };
    }
};

#endif
