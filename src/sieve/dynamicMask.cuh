#pragma once
#include <bit>
#include "prime_algorithms.h"
#include "sieving_data_structs.h"

namespace mpqs {
namespace sieve {

struct maskData {
    uint32_t period;     
    uint32_t smallPrimesUsed;
};

__host__
inline maskData generateMask(uint32_t* primes, devicePointers& dev_pointers){
    const auto log2_u32 = [](uint32_t n) -> uint8_t {
    return n == 0
        ? 0
        : static_cast<uint8_t>(31u - __builtin_clz(n));
    };
    constexpr uint32_t MEM_BOUND = 64u * 1024u;
    uint32_t memReq = 4;
    uint32_t smallPrimesUsed = 0;
    while(memReq * primes[smallPrimesUsed] < MEM_BOUND){
        memReq *= primes[smallPrimesUsed];
        smallPrimesUsed++;
    }
    uint32_t period = memReq/4;
    uint32_t* smallPrimeMask = new uint32_t[period]{};
    uint8_t* smallPrimeMask8 = reinterpret_cast<uint8_t*>(smallPrimeMask);

    for(int i = 0; i < smallPrimesUsed; i++){
        uint32_t p = primes[i];
        uint32_t log_p = log2_u32(p);
        for(uint32_t offset = 0; offset < 4*period; offset+=p){
            smallPrimeMask8[offset] += log_p;
        }
    }

    maskData MD;

    MD.period = period;
    MD.smallPrimesUsed = smallPrimesUsed;

    uint32_t* CRT_baseElements = new uint32_t[MD.smallPrimesUsed]{};

    for(int i = 0; i < smallPrimesUsed; i++){
        uint32_t p = primes[i];
        CRT_baseElements[i] = (period/p * modInv(period/p,p)) % period;
    }
    cudaMalloc((void**)&dev_pointers.dev_smallPrimeMask, memReq);
    cudaMalloc((void**)&dev_pointers.dev_CRT_baseElements, smallPrimesUsed*sizeof(uint32_t));

    cudaMemcpy(dev_pointers.dev_smallPrimeMask, smallPrimeMask, memReq, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pointers.dev_CRT_baseElements, CRT_baseElements, smallPrimesUsed*sizeof(uint32_t), cudaMemcpyHostToDevice);
    return MD;
}

__device__ 
inline void findMaskOffsets(int32_t sieveStart, __restrict__ int32_t* offsets1, __restrict__ int32_t* offsets2, __restrict__ int32_t* primes, uint32_t& maskOffset1, uint32_t& maskOffset2, uint32_t smallPrimesUsed, uint32_t period, __restrict__ uint32_t* CRT_baseElements){
    uint32_t rootShift;
    uint32_t offset;

    rootShift = 0;
    for(int i = 0; i < smallPrimesUsed; i++){
        uint32_t p = primes[i];
        uint32_t root = ((offsets1[i] % (int)p) + p) % p;
        rootShift += (CRT_baseElements[i]*root);
    }
    rootShift %= period;
    offset = ((sieveStart % (int)period) + period) % period;
    if (offset < rootShift) {
        offset += period;
    }
    offset -= rootShift;
    if (offset & 3u) offset += period;
    if (offset & 3u) offset += period;
    if (offset & 3u) offset += period;
    maskOffset1 = offset >> 2;

    rootShift = 0;
    for(int i = 0; i < smallPrimesUsed; i++){
        uint32_t p = primes[i];
        uint32_t root = ((offsets2[i] % (int)p) + p) % p;
        rootShift += (CRT_baseElements[i]*root);
    }
    rootShift %= period;
    offset = ((sieveStart % (int)period) + period) % period;
    if (offset < rootShift) {
        offset += period;
    }
    offset -= rootShift;
    if (offset & 3u) offset += period;
    if (offset & 3u) offset += period;
    if (offset & 3u) offset += period;
    maskOffset2 = offset >> 2;
}

}
}