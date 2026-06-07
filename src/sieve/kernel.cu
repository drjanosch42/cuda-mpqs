// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cassert>
#include <tuple>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>
#include "kernel.cuh"
#include "common.h"

// CUDA error-check helper (sieve module). Wraps cudaMalloc/cudaMemcpy/etc.
// On failure, logs file:line and throws so callers see a clean diagnostic.
#ifndef SIEVE_CUDA_CHECK
#define SIEVE_CUDA_CHECK(err) do { \
    cudaError_t _sieve_cuda_check_err = (err); \
    if (_sieve_cuda_check_err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "[Sieve] CUDA error: " << cudaGetErrorString(_sieve_cuda_check_err) \
                                 << " at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_sieve_cuda_check_err)); \
    } \
} while(0)
#endif
#include "math_utils.cuh"
#include "graycode.cuh"
#include "uint512.cuh"

#define STARTTIMER(var) \
    auto var = std::chrono::high_resolution_clock::now()

#define ENDTIMER(var, label) \
        std::cout << label << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - var).count() << " ms" << std::endl

#define ATOMIC_BYTE_ADD(array, index, x) \
    atomicAdd((uint32_t*)((array) + ((index) & (~3))), ((uint32_t)(x)) << (8 * ((index) & 3)))

#define ATOMIC_BYTE_ADD_RETURN(array, index, x) \
    (uint8_t)((atomicAdd((uint32_t*)((array) + ((index) & (~3))), ((uint32_t)(x)) << (8 * ((index) & 3))) >> (8 * ((index) & 3))) & 0xFF)

namespace mpqs {
namespace sieve {

/*
For sieving we use the following conventions of splitting up data to guarantee that everything fits into memory/cache:
- hypercube of polynomials -> subCubes -> polyBlocks (i.e. even smaller subCubes) -> polys
- sievingInterval -> SievingCycles -> sievingBlocks

One kernel launch sieves through one subCube and a given number of SievingCycles,
these are chosen such that:
- metaSieve has enough memory to store all the offsets of the SievingCycles and the subCube in global memory
- num_polyBlocks * num_sievingBlocks (the amount of lists simultaniously filled) is small enough to benifit from caching

The hypercube of polynomials that can be made from the leading coefficent a can be represented as

fullPolyId = [subCubeId] -> [polyBlockId] -> [polyId]

where
- polyId takes log2_polyBlockSize bits
- polyBlockId takes log2_numPolyBlocks bits
- subCubeId takes the remaining bits

This splits the hypercube into the product of 3 smaller cubes. Gray code can then be utilized to efficiently switch polynomials within a polyBlock
Since all of these are powers of 2, we can calculate the fullPolyId from each Id and bitshifts

In one sieving call, we only sieve one subCube.
Thus we can ignore subCubeId for memory accesses and the id of the bucket containing the results of metaSieve for each polynomial/sieveBlock are given by

globalBucketId = [polyBlockId] -> [polyId] -> [cycle] -> [sieveBlock]
*/
__global__ void __launch_bounds__(1024) sieveAndScanKernel(devicePointers dev_pointers, fixedSievingParams fs_params, dynamicSievingParams ds_params, generalSievingConfig gs_conf, sieveAndScanConfig ss_conf)//We want gridDim.x blocks with size of blockDim.x
{
    primeDataSIQS* primeData = dev_pointers.dev_primeData;
    uint512* B_values = dev_pointers.dev_B_values;
    uint64_t* globalBucketEntries = dev_pointers.dev_globalBucketEntries;
    int32_t* indexToCandidate = dev_pointers.dev_indexToCandidate + blockIdx.x * gs_conf.sievingBlockSize;
    candidateRelation* candidates = dev_pointers.dev_candidateRelations + blockIdx.x * gs_conf.maxRelationsPerBlock;

    extern __shared__ uint8_t sharedByteData[];
    // Cast to int* preserves signed logic for offsets (coordinates) and primes (sign flag)
    int* offsets1 = (int*)sharedByteData;
    int* offsets2 = offsets1 + gs_conf.bigPrimeStartIndex;
    int* primes = offsets2 + gs_conf.bigPrimeStartIndex;
    uint32_t candidatesFound;
    if(ds_params.newCube){
        candidatesFound = 0;
    }
    else{
        candidatesFound = dev_pointers.dev_blockRelationCounts[blockIdx.x];
    }
    int subCube = ds_params.subCube;
    polyData p_data;
    p_data.approxPolyRoot = fs_params.approxPolyRoot;
    p_data.threshold = fs_params.threshold;
    p_data.log2_a = ds_params.log2_a;

    uint8_t* blockEntries = (uint8_t*)(primes + gs_conf.bigPrimeStartIndex);
    uint512 b;
    int sieveIntervalStart = ds_params.startIndex;
    int reducedSieveStart = 0;

    int polyIdPrefix = (subCube*gs_conf.num_polysPerSieveCall) + (blockIdx.x * (gs_conf.num_polysPerSieveCall) / gridDim.x);
    bFromPolyId(polyIdPrefix, fs_params.shc_dim, B_values, b);
    __syncthreads();
    for (int i = threadIdx.x; i < gs_conf.bigPrimeStartIndex; i += blockDim.x) {
        primeDataSIQS curPrimeData = primeData[i];

        // p fits in int32 (< 2^31). Cast explicit.
        int p = (int)curPrimeData.p;

        // Roots are now unsigned uint32_t residues [0, p-1]
        uint32_t root1 = 0;
        uint32_t root2 = 0;
        rootsFromPolyId(polyIdPrefix, fs_params.shc_dim, curPrimeData, root1, root2);

        reducedSieveStart = ((sieveIntervalStart % p) + p) % p;

        // modSub_shifted returns uint32_t.
        // We cast back to int for coordinate calculation.
        // Logic: Start - (Distance to next root) + p. Result is positive coordinate relative to start.
        offsets1[i] = sieveIntervalStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root1, (uint32_t)p) + p;
        offsets2[i] = sieveIntervalStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root2, (uint32_t)p) + p;

        // Inactive logic: p * (1 - 2*0) = p, p * (1 - 2*1) = -p.
        // Cast inactive to int to ensure correct arithmetic.
        primes[i] = p * (1 - 2 * curPrimeData.inactive);
    }
    __syncthreads();
    int prevPolyId = 0;
    int polyId = 0;
    int truncatedPolyId = 0;
    for (int poly = 0; poly < (gs_conf.num_polysPerSieveCall) / gridDim.x; poly++) {//we dont care about the polyblocks here, incrementing is done below
        polyId = polyIdPrefix | gray(poly);
        truncatedPolyId = polyId & (gs_conf.num_polysPerSieveCall-1);
        if (poly > 0) {
            __syncthreads();
            for (int i = threadIdx.x; i < gs_conf.bigPrimeStartIndex; i += blockDim.x) {
                primeDataSIQS curPrimeData = primeData[i];
		        int p = (int)curPrimeData.p; // Explicit cast

                // Recalculate current roots based on offsets
                // ((offset % p) + p) % p ensures positive residue
                uint32_t root1 = (uint32_t)(((offsets1[i] % p) + p) % p);
                uint32_t root2 = (uint32_t)(((offsets2[i] % p) + p) % p);

                // Update roots for Gray code step (uses uint32_t internally)
                advanceRoots(prevPolyId, polyId, curPrimeData, root1, root2);

                reducedSieveStart = ((sieveIntervalStart % p) + p) % p;

                // Update offsets using new roots
                offsets1[i] = sieveIntervalStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root1, (uint32_t)p) + p;
                offsets2[i] = sieveIntervalStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root2, (uint32_t)p) + p;
            }
	    // Update b coefficient (mpqs::uint512)
            advance_b(prevPolyId, polyId, B_values, b);
        }
        __syncthreads();

        for (int sieveBlock = 0; sieveBlock < gs_conf.num_sievingBlocksPerSieveCall; sieveBlock++) {
            for (int i = threadIdx.x; i < gs_conf.sievingBlockSize; i += blockDim.x) {
                blockEntries[i] = 0;
            }
            int sieveBlockStart = sieveIntervalStart + sieveBlock * gs_conf.sievingBlockSize;
            int sieveBlockEnd = sieveBlockStart + gs_conf.sievingBlockSize;
            __syncthreads();

            uint64_t globalBucketId = (((long long)truncatedPolyId) * gs_conf.num_sievingBlocksPerSieveCall + sieveBlock);
            uint64_t listStart = globalBucketId * gs_conf.globalBucketSize; //CHANGE TO THE CURRENT GLOBAL BUCKET
            uint64_t* currentBucketEntries = globalBucketEntries + listStart;
            //dump globalBucketEntries into the current sieving block
            uint32_t currentFillLevel = (dev_pointers.dev_globalBucketCounts[globalBucketId]) & (16777216 - 1);
            for (int i = threadIdx.x; i < currentFillLevel; i += blockDim.x) {
                uint32_t val = (uint32_t)currentBucketEntries[i];
                ATOMIC_BYTE_ADD(blockEntries, val & ((1 << 24) - 1), val >> 24);
            }
            int midPrimeStart = gs_conf.midPrimeStartIndex;
            __syncthreads();
            for (int i = 0; i < midPrimeStart; i++) {
                int p = primes[i];
		        // If p is inactive, we do not touch blockEntries at all
		        // This prevents useless indexing work.
		        if (p < 0) {
		            __syncthreads();
		            continue;
		        }
		        // Here p is guaranteed > 0.
                uint8_t log_p = log2(p); // (p <= 0) ? 0 : log2(p);
                // p = abs(p);
                int offset1 = offsets1[i];
                int offset2 = offsets2[i];
		        // Forward Sieve: Offset is signed int, loop terminates when offset >= sieveBlockEnd
                int offset = offset1 + threadIdx.x * p;
                for (; offset < sieveBlockEnd; offset += blockDim.x * p) {
                    blockEntries[offset - sieveBlockStart] += log_p;
                }
                if (offset - p < sieveBlockEnd) {
                    offsets1[i] = offset; //exactly one thread has the correct "last" offset, keep it for the next iteration
                }
                offset = offset2 + threadIdx.x * p;
                for (; offset < sieveBlockEnd; offset += blockDim.x * p) {
                    blockEntries[offset - sieveBlockStart] += log_p;
                }
                if (offset - p < sieveBlockEnd) {
                    offsets2[i] = offset; //exactly one thread has the correct "last" offset, keep it for the next iteration
                }
                __syncthreads();
            }
	    // Small primes handling
            for (int i = midPrimeStart + threadIdx.x; i < gs_conf.bigPrimeStartIndex; i += blockDim.x) {
                int p = primes[i];
                uint8_t log_p = (p <= 0) ? 0 : log2(p);
                p = abs(p);
                int offset1 = offsets1[i];
                int offset2 = offsets2[i];
                //__syncthreads(); //NO SYNC NEEDED HERE, threads access disjoint data
                int offset = offset1;
                for (; offset < sieveBlockEnd; offset += p) {
                    ATOMIC_BYTE_ADD(blockEntries, offset - sieveBlockStart, log_p);
                }
                offsets1[i] = offset; //keep the offset for the next iteration
                offset = offset2;
                for (; offset < sieveBlockEnd; offset += p) {
                    ATOMIC_BYTE_ADD(blockEntries, offset - sieveBlockStart, log_p);
                }
                offsets2[i] = offset; //keep the offset for the next iteration
            }
            __syncthreads();

	    // Check for candidates
	    int newCandidateCount = excludeNonRelations(
		blockEntries,
		indexToCandidate,
		candidates, // Pass array
		b,          // Pass uint512 (by const ref logic)
		polyId,     // Pass ID
		candidatesFound,
		sieveBlockStart,
		gs_conf.sievingBlockSize,
		gs_conf.maxRelationsPerBlock,
		p_data
	    );
            candidatesFound += newCandidateCount;
            if (newCandidateCount == 0) {
                continue; //no candidates have been found, so skip this sieveBlock
            }

	    // Backward Sieve (Scanning candidates for factors)
            for (int i = 0; i < midPrimeStart; i++) {
                int p = primes[i];
                bool active = p > 0;
                p = abs(p);
                int offset1 = offsets1[i];
                int offset2 = offsets2[i];
                __syncthreads();
                if (active) {
                    // Backward loop using signed arithmetic.
                    // Loop terminates when offset <= sieveBlockStart.
                    // Safe because offset is int and subtracts p.
                    int offset = offset1 - threadIdx.x * p - p;
                    for (; offset >= sieveBlockStart; offset -= blockDim.x * p) {
                        int localOffset = offset - sieveBlockStart;
                        if (blockEntries[localOffset]) {
                            int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
			    int globalIdx = indexToCandidate[localOffset];
			    // Store factor
			    candidates[globalIdx].factors[31 & newPrimeIndex] = i;
			    // Update count
			    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                        }
                    }
                    offset = offset2 - threadIdx.x * p - p;
                    for (; offset >= sieveBlockStart; offset -= blockDim.x * p) {
                        int localOffset = offset - sieveBlockStart;
                        if (blockEntries[localOffset]) {
                            int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
			    int globalIdx = indexToCandidate[localOffset];
			    // Store factor
			    candidates[globalIdx].factors[31 & newPrimeIndex] = i;
			    // Update count
			    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                        }
                    }
                }
                __syncthreads();
            }
	    // Small primes backward scan
            for (int i = midPrimeStart + threadIdx.x; i < gs_conf.bigPrimeStartIndex; i += blockDim.x) {
                int p = primes[i];
                bool active = p > 0;
                p = abs(p);
                int offset1 = offsets1[i];
                int offset2 = offsets2[i];
                //__syncthreads(); AS ABOVE, NO SYNC NEEDED
                if (active) {
                    int offset = offset1 - p;
                    for (; offset >= sieveBlockStart; offset -= p) {
                        int localOffset = offset - sieveBlockStart;
                        if (blockEntries[localOffset]) {
                            int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
			    int globalIdx = indexToCandidate[localOffset];
			    // Store factor
			    candidates[globalIdx].factors[31 & newPrimeIndex] = i;
			    // Update count
			    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                        }
                    }
                    offset = offset2 - p;
                    for (; offset >= sieveBlockStart; offset -= p) {
                        int localOffset = offset - sieveBlockStart;
                        if (blockEntries[localOffset]) {
                            int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
			    int globalIdx = indexToCandidate[localOffset];
			    // Store factor
			    candidates[globalIdx].factors[31 & newPrimeIndex] = i;
			    // Update count
			    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                        }
                    }
                }
            }

            for (int i = threadIdx.x; i < currentFillLevel; i += blockDim.x) {
                uint64_t val = currentBucketEntries[i];
		if (val == 0) continue; // <- ignore illegal entries
                int localOffset = val & ((1 << 24) - 1);
                int prime_index = val >> 32;
                if (blockEntries[localOffset]) {
                    int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
		    int globalIdx = indexToCandidate[localOffset];
		    // Store factor
		    candidates[globalIdx].factors[31 & newPrimeIndex] = prime_index;
		    // Update count
		    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                }
            }
            __syncthreads();
        }
        prevPolyId = polyId;
        __syncthreads();
    }
    if(threadIdx.x == 0){
        dev_pointers.dev_blockRelationCounts[blockIdx.x] = candidatesFound;
    }
}


__global__ void globalMetaSieveKernel(devicePointers dev_pointers, fixedSievingParams fs_params, dynamicSievingParams ds_params, generalSievingConfig gs_conf, globalMetaSieveConfig gms_conf) {
    uint64_t* globalBucketEntries = dev_pointers.dev_globalBucketEntries;
    uint32_t* globalBucketCounts  = dev_pointers.dev_globalBucketCounts;

    primeDataSIQS* primeData = dev_pointers.dev_primeData;
    int shc_dim = fs_params.shc_dim;

    int subCube = ds_params.subCube;
    int sieveIntervalStart = ds_params.startIndex;

    int offsetMask = gs_conf.sievingBlockSize - 1;

    extern __shared__ int sharedData[];
    int* global_bucket_write_head = sharedData;

    /*
    ---------------------------------------------------------------------------
    CYCLES LOOP
    ---------------------------------------------------------------------------
    */
    for(int cycle = 0; cycle < gms_conf.num_metaSieveCycles; cycle++){
        int currentStart = sieveIntervalStart + cycle * gms_conf.num_activeBlocksPerCycle * gs_conf.sievingBlockSize;
        
        /*
        -----------------------------------------------------------------------
        POLY BLOCKS LOOP
        -----------------------------------------------------------------------
        */
        for(int curPolyBlock = 0; curPolyBlock < gms_conf.num_polyBlocksPerThreadBlock; curPolyBlock ++){
            __syncthreads();
            for (int i = threadIdx.x; i < gms_conf.num_activeBucketsPerThreadBlock; i += blockDim.x) {
                global_bucket_write_head[i] = 0;
            }
            __syncthreads();
            int polyBlockId = blockIdx.x + gms_conf.num_threadBlocks * curPolyBlock;
            int fullPolyIdPrefix = (subCube * gs_conf.num_polysPerSieveCall) + (polyBlockId << gms_conf.log2_polyBlockSize);

            /*
            -------------------------------------------------------------------
            PRIMES LOOP
            -------------------------------------------------------------------
            */
            for(int currentPrimeIndex = threadIdx.x + gs_conf.bigPrimeStartIndex; currentPrimeIndex < fs_params.fb_size; currentPrimeIndex += blockDim.x){
                int polyIndex = ((threadIdx.x/32) % gms_conf.polyBlockSize);
                int polyId = gray(polyIndex);
                int fullPolyId = fullPolyIdPrefix | polyId;
                primeDataSIQS curPrimeData = primeData[currentPrimeIndex];
                uint32_t p = curPrimeData.p;
                int log2p = (31 - __clz(p)) * (1 - curPrimeData.inactive);
                uint32_t root1;
                uint32_t root2;
                rootsFromPolyId(fullPolyId, shc_dim, curPrimeData, root1, root2);
                int maxOffsetCount = ((gms_conf.num_activeBlocksPerCycle * gs_conf.sievingBlockSize) / (primeData[currentPrimeIndex - threadIdx.x].p)) + 1; //using the first prime as an upper bound to have a constant inner loop length
                int reducedSieveStart = ((currentStart % (int)p) + (int)p ) % (int)p;// Formula: ((S % p) + p) % p ensures result is in [0, p-1]

                /*
                ---------------------------------------------------------------
                POLYS LOOP
                ---------------------------------------------------------------
                */
                for(int polyCounter = 0; polyCounter < gms_conf.polyBlockSize; polyCounter++){
                    int globalBucketIdPrefix = ((((long long)polyBlockId) * gms_conf.polyBlockSize + polyId) * gms_conf.num_metaSieveCycles + cycle) * gms_conf.num_activeBlocksPerCycle;
                    int offsetCounter = 0;
		            // Advance roots from previous to next poly
                    int offset1 = currentStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root1, p) + (int)p;
                    int offset2 = currentStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root2, p) + (int)p;

                    /*
                    -----------------------------------------------------------
                    OFFSETS LOOP
                    -----------------------------------------------------------
                    */
                    for(offsetCounter = 0; offsetCounter < maxOffsetCount; offsetCounter++){
                        int curOffset = offset1;
                        for (int i = 0; i < 2; i++) {
                            // Logic: curOffset is a coordinate >= currentStart.
                            // Subtraction yields a non-negative relative index.
                            int sievingBlockHit = (curOffset - currentStart) / gs_conf.sievingBlockSize;
                            if (sievingBlockHit < gms_conf.num_activeBlocksPerCycle) {
                                int activeBucketId =  polyId * gms_conf.num_activeBlocksPerCycle + sievingBlockHit;
                                int index = atomicAdd(&global_bucket_write_head[activeBucketId], 1); // reserve an index
                                if (index < gs_conf.globalBucketSize) {
                                    // the bucket is not full yet, so write in it
                                    int sievingBlock_offset = (curOffset - currentStart) % gs_conf.sievingBlockSize; // no subtract needed if currentStart is a multiple of the blockSize
                                    int entry = sievingBlock_offset | (log2p << 24); // leading 8 bits store the log2 value and the other 24 the offset
                                    /*
                                    The activeBucketId is given by;
                                    activeBucketId = [polyId] [sievingBlock]
                                    The global bucket id is given by:
                                    globalBucketId = [polyBlockId] [polyId] [cycle] [sievingBlock]
                                    */
                                    long long int globalIndex = (globalBucketIdPrefix + sievingBlockHit) * gs_conf.globalBucketSize + index;
                                    // If we want to store the prime value, we do:
                                    // globalBucketEntries[globalIndex] = (((uint64_t)p) << 32) | entry;
                                    // But we store the index instead:
                                    globalBucketEntries[globalIndex] = (((uint64_t)currentPrimeIndex) << 32) | entry;
                                }
                            }
                            curOffset = offset2;
                        }
                        offset1 += (int)p; // Signed increment
                        offset2 += (int)p;
                    }
                    //Offsets loop finished, prepare offsets of the next polynomial for the same prime
                    int prevFullPolyId = fullPolyId;
                    polyIndex = modAdd(polyIndex, 1, gms_conf.polyBlockSize);
                    polyId = gray(polyIndex);
                    fullPolyId = fullPolyIdPrefix | polyId;
                    advanceRoots(prevFullPolyId, fullPolyId, curPrimeData, root1, root2);
                }
                //poly loop finished, nothing to do here
            }
            //prime loop finished, record the bucket fill levels as new buckets are required next time
            __syncthreads();
            for (int i = threadIdx.x; i < gms_conf.num_activeBucketsPerThreadBlock; i += blockDim.x) {
                /*
                i iterates over the active Buckets, we need to convert the id to the globalBucketId
                i = activeBucketId = [polyId] [sievingBlock]
                Want: globalBucketId = [polyBlockId] [polyId] [cycle] [sievingBlock]
                */
                uint32_t i_polyId = i / gms_conf.num_activeBlocksPerCycle;
                uint32_t i_sievingBlock = i % gms_conf.num_activeBlocksPerCycle;
                uint64_t i_globalBucketId = (((((uint64_t)polyBlockId) * gms_conf.polyBlockSize + i_polyId) * gms_conf.num_metaSieveCycles + cycle) * gms_conf.num_activeBlocksPerCycle)+i_sievingBlock;
                uint32_t amountWritten = global_bucket_write_head[i];
                uint32_t encoded = (min(amountWritten, gs_conf.globalBucketSize)) | ((amountWritten > gs_conf.globalBucketSize) ? 0x80000000u : 0u);
                globalBucketCounts[i_globalBucketId] = encoded;
            }
        }
        //poly blocks loop finished, nothing to do here
    }
    //cycles loop finished, done.
}

/*
Initializes some values needed to calculate new polynomials quickly. In particular we need to
- for every prime p: reduce a modulo p and calculate an inverse of a modulo p
- for every prime p: reduce each summand B_1,...,B_d of b modulo p and store 2*B_i*invertedA (mod p) for all i
*/
__global__ void initPrimeDataKernel(devicePointers dev_pointers, generalSievingConfig gs_conf, fixedSievingParams fs_params, dynamicSievingParams ds_params)
{
    int shc_dim = fs_params.shc_dim;
    int fb_size = fs_params.fb_size;

    uint512* B_values = dev_pointers.dev_B_values;
    uint512 a = ds_params.a;

    uint32_t* factorBase = dev_pointers.dev_factorBase;
    uint32_t* rootN = dev_pointers.dev_rootN;
    primeDataSIQS* primeData = dev_pointers.dev_primeData;

    int stride = gridDim.x * blockDim.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    uint512 local_B_values[16];
    for (int i = 0; i < shc_dim; i++) {
        local_B_values[i] = B_values[i];
    }

    //for every prime p
    for (int i = id; i < fb_size; i += stride) {
        uint32_t p = factorBase[i];

        //reduce a (given by the a_limbs-array) modulo p
        //Use optimized mod_uint32 from uint512 class
        uint32_t reduced_a = a.mod_uint32(p);

        //now invert it modulo p:
        int exponent = p - 2;
        int base = reduced_a; //we want to invert the result from the modular reduction
        int result = 1;

        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
        for (int j = 0; j < 32; j++) {
            // Use uint64_t for intermediate product to avoid 32-bit overflow
            // ((exponent & 1) == 1) branchless logic preserved
            result = ((exponent & 1) == 1) * (uint32_t)(((uint64_t)result * base) % p) + ((exponent & 1) == 0) * (result);
            exponent = exponent >> 1;
            base = (uint32_t)(((uint64_t)base * base) % p);
        }
        primeDataSIQS tmpPrimeData{};
	// Note: Field name updated to match sieving_data_structs.h
        tmpPrimeData.mod_inverse_a = result;

        //now calculate the B_values used to update roots modulo p
        for (int j = 0; j < shc_dim; j++) {
            // Reduce B_value mod p first, then multiply by inverse_a
            uint32_t b_val_mod = local_B_values[j].mod_uint32(p);
            tmpPrimeData.B_values[j] = (uint32_t)(((uint64_t)b_val_mod * result) % p);
        }

        tmpPrimeData.p = p;
        tmpPrimeData.inv_aN = (uint32_t)(((uint64_t)result * rootN[i]) % p);

        //save the initialized primData to global memory
        primeData[i] = tmpPrimeData;
    }
}


/*
Marks cetrain primes as inactive so they are not sieved with.
*/
__global__ void markInactivePrimesKernel(int shc_dim, uint32_t* a_factors, primeDataSIQS* primeData){
    if(blockIdx.x == 0){
        for(int i = threadIdx.x; i < shc_dim; i += blockDim.x){
            primeData[a_factors[i]].inactive = 1;
        }
    }
}

/*
Loads sieving data to device
*/
void updateSievingData(devicePointers& dev_pointers, fixedSievingParams fs_params, std::vector<uint32_t>& a_factors, std::vector<uint512> B_values) {
    cudaMemcpy(dev_pointers.dev_a_factors, a_factors.data(),
               fs_params.shc_dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pointers.dev_B_values, B_values.data(),
               fs_params.shc_dim * sizeof(uint512), cudaMemcpyHostToDevice);
}

/*
Allocates and loads sieving data on device
*/
void loadSievingData(std::vector<uint32_t>& factorBase,
    std::vector<uint32_t>& rootN,
    std::vector<uint32_t>& a_factors,
    uint32_t shc_dim,
    generalSievingConfig gs_conf,
    sieveAndScanConfig ss_conf,
    devicePointers& dev_pointers){
    //allocate memory
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_factorBase, factorBase.size() * sizeof(uint32_t)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_rootN, rootN.size() * sizeof(uint32_t)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_a_factors, shc_dim * sizeof(uint32_t)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_B_values, shc_dim * sizeof(uint512)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_primeData, factorBase.size() * sizeof(primeDataSIQS)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_globalBucketEntries, ((long long)gs_conf.num_polysPerSieveCall * gs_conf.num_sievingBlocksPerSieveCall * gs_conf.globalBucketSize * sizeof(uint64_t))));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_globalBucketCounts, gs_conf.num_polysPerSieveCall * gs_conf.num_sievingBlocksPerSieveCall*sizeof(uint32_t)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_blockRelationCounts, ss_conf.num_threadBlocks * sizeof(uint32_t)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_candidateRelations, gs_conf.maxRelationsPerBlock * ss_conf.num_threadBlocks * sizeof(candidateRelation)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_indexToCandidate, gs_conf.sievingBlockSize * ss_conf.num_threadBlocks * sizeof(uint32_t)));
    //copy once
    SIEVE_CUDA_CHECK(cudaMemcpy(dev_pointers.dev_factorBase, factorBase.data(), factorBase.size() * sizeof(int), cudaMemcpyHostToDevice));
    SIEVE_CUDA_CHECK(cudaMemcpy(dev_pointers.dev_rootN, rootN.data(), rootN.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    SIEVE_CUDA_CHECK(cudaMemcpy(dev_pointers.dev_a_factors, a_factors.data(), a_factors.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
}
//allocating for the param test, copying isnt needed each time
void loadSievingDataParamTest(std::vector<uint32_t>& factorBase,
    std::vector<uint32_t>& rootN,
    std::vector<uint32_t>& a_factors,
    uint32_t shc_dim,
    generalSievingConfig gs_conf,
    sieveAndScanConfig ss_conf,
    devicePointers& dev_pointers){
    //allocate memory
    cudaFree(dev_pointers.dev_globalBucketEntries);
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_globalBucketEntries, ((long long)gs_conf.num_polysPerSieveCall * gs_conf.num_sievingBlocksPerSieveCall * gs_conf.globalBucketSize * sizeof(uint64_t))));
    cudaFree(dev_pointers.dev_globalBucketCounts);
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_globalBucketCounts, gs_conf.num_polysPerSieveCall * gs_conf.num_sievingBlocksPerSieveCall*sizeof(uint32_t)));
    cudaFree(dev_pointers.dev_blockRelationCounts);
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_blockRelationCounts, ss_conf.num_threadBlocks * sizeof(uint32_t)));
    cudaFree(dev_pointers.dev_candidateRelations);
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_candidateRelations, gs_conf.maxRelationsPerBlock * ss_conf.num_threadBlocks * sizeof(candidateRelation)));
    cudaFree(dev_pointers.dev_indexToCandidate);
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_indexToCandidate, gs_conf.sievingBlockSize * ss_conf.num_threadBlocks * sizeof(uint32_t)));
}

void warmup(){
    cudaFree(0);
}

void getDeviceInfo(gpuInfo& gInfo, int device){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    gInfo.totalGlobalMem = prop.totalGlobalMem;
    gInfo.sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
    gInfo.multiProcessorCount = prop.multiProcessorCount;
    gInfo.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    gInfo.maxSharedMemPerBlock = prop.sharedMemPerBlockOptin - 1024; //We use the optin here so we can use more than 48kb
}

/*
Used to determine the roots of a polynomial for a given prime anywhere in the hypercube. Since it doesnt re-use roots of adjacent polynomials it needs several operations
For a given coefficients a and b we want to find the root of (ax+b)^2-N.
These are given by inverted_a*(rootN-b) and inverted_a*(-rootN-b).
Since we already have inverted_a*B_k for the summands B_k of b, we dont need to use the multiple precision value "b" and instead just need to do a few single precision operations
*/
/*
Reconstructs roots for a specific Poly ID from the hypercube.
Calculates root of (ax+b)^2 - N (mod p).
Roots are strictly in [0, p-1].
*/
__device__ __forceinline__
void rootsFromPolyId(uint32_t id, int shc_dim, primeDataSIQS& primeData, uint32_t& result1, uint32_t& result2){
    uint32_t minus_b_aInv = 0;
    uint32_t p = primeData.p;

    // Calculate -b * a^-1 (mod p) by summing B_values
    for (int i = 0; i < shc_dim; i++) {
            int bit = id & 1;
            uint32_t B_val = primeData.B_values[i];
            // If bit is 0, add B_val. If bit is 1, add -B_val (which is p - B_val).
            // (p - B_val) is safe because B_val < p.
            uint32_t term = (bit == 0) ? B_val : (p - B_val);
            minus_b_aInv = modAdd(minus_b_aInv, term, p);
            id >>= 1;
    }
    // r = (-b*a^-1 +/- a^-1*sqrt(N)) mod p
    result1 = modAdd(minus_b_aInv, primeData.inv_aN, p);
    // subtraction of inv_aN is equivalent to adding (p - inv_aN)
    result2 = modAdd(minus_b_aInv, p - primeData.inv_aN, p);
}

/*
Reconstructs coefficient 'b' (512-bit) from Poly ID.
*/
__device__ __forceinline__
void bFromPolyId(uint32_t id, int shc_dim, mpqs::uint512* B_values, mpqs::uint512& result){
    // result assumed initialized to 0 by caller or logic?
    // Original code assumed accumulated result. We zero it here or assume clean state.
    // Based on usage in kernel, it's passed as reference `b`.
    // We assume `b` is initialized? Original code didn't clear `b`.
    // Assuming `b` is fresh or we reset it:
    // (In strict refactoring we just mimic the operations)

    for (int i = 0; i < shc_dim; i++) {
            int bit = id & 1;
            if(bit == 1){
                result.add(B_values[i]);
            }
            else{
                result.sub(B_values[i]);
            }
            id >>= 1;
    }
}

/*
Updates roots when moving between adjacent Gray codes.
Uses the precalculated B_value for the flipped bit.
*/
__device__ __forceinline__
void advanceRoots(uint32_t id1, uint32_t id2, primeDataSIQS& primeData, uint32_t& root1, uint32_t& root2) {
    uint32_t bitToFlip = __ffs((id1^id2))-1;

    // sign is +1 if bit flipped 1->0, -1 if bit flipped 0->1 (or vice versa depending on definition)
    // Original: 1 - 2 * ((id2 >> bitToFlip) & 1)
    // If new bit is 1, sign is -1. If new bit is 0, sign is 1.
    int sign = 1 - (int)2 * ((id2 >> bitToFlip) & 1);

    // Safe cast: B_values elements are < 2^31, fitting in signed int32 logic
    int32_t delta = sign * (int32_t)primeData.B_values[bitToFlip];

    root1 = modSum(root1, delta, primeData.p);
    root1 = modSum(root1, delta, primeData.p); // Applied twice (original logic?) - Yes, usually for N/A vs -N/A shift?
    // Actually in MPQS hypercube, root update is typically +/- 2*B_val.
    // The original code did it twice. We preserve this exact control flow.

    root2 = modSum(root2, delta, primeData.p);
    root2 = modSum(root2, delta, primeData.p);
}

/*
Updates coefficient 'b' (512-bit) for Gray code step.
*/
__device__ __forceinline__
void advance_b(uint32_t id1, uint32_t id2, mpqs::uint512* B_values, mpqs::uint512& b){
    uint32_t bitToFlip = __ffs((id1^id2))-1;

    // Original: -1 + 2 * ((id2 >> bitToFlip) & 1)
    // If new bit is 1, sign is +1. If new bit is 0, sign is -1.
    int sign = - 1 + (int)2 * (int)((id2 >> bitToFlip) & 1);

    if(sign == -1){
        b.sub(B_values[bitToFlip]);
	b.sub(B_values[bitToFlip]);
    }
    else{
        b.add(B_values[bitToFlip]);
        b.add(B_values[bitToFlip]);
    }
}

/*
This function can atomically add on a byte-level. Overflow must be avoided manually, otherwise it may corrupt an adjacent byte
*/
__device__ __forceinline__
int atomicByteAdd(uint8_t* array, int index, uint8_t x){
    uint32_t* ambientIntPointer = (uint32_t*)(array + (index & (~3))); //captures the "ambient" 4byte block in which the target byte resides
    int offset = index & 3;
    return atomicAdd(ambientIntPointer,((uint32_t)x) << (8*offset));
}

/*
Scans the sievingBlock and sets all offsets that dont quality as a relation to zero.
Refactored for Unsigned Arithmetic:
- Uses conditional subtraction for distance to avoid unsigned underflow.
- Handles globalIndex as uint32_t for magnitude checks.
*/
__device__
int excludeNonRelations(
    uint8_t* __restrict__ blockEntries,
    int32_t* __restrict__ indexToCandidate,
    candidateRelation* __restrict__ candidates,
    const mpqs::uint512& b,
    uint32_t poly_id,
    uint32_t candidatesFound,
    int startOffset,
    int sievingBlockSize,
    int maxPerBlock,
    polyData& p_data
) {
    uint32_t log2_a = p_data.log2_a;
    uint32_t approxPolyRoot = p_data.approxPolyRoot;
    uint32_t threshold = p_data.threshold;
    __shared__ int candidateWriteHead;
    if (threadIdx.x == 0) candidateWriteHead = candidatesFound;
    __syncthreads();

    for(int i = 0; i < sievingBlockSize; i += blockDim.x){
        int index = i + threadIdx.x;

        bool isCandidate = false;
        if(index < sievingBlockSize){
            int globalIndex = startOffset + index;

            // Manual distance calculation to replace abs()
            // dist = |globalIndex - approxPolyRoot| and dist = |globalIndex - (-approxPolyRoot)|
            uint32_t dist_minus = (globalIndex >= (int)approxPolyRoot) ? (globalIndex - approxPolyRoot) : (approxPolyRoot - globalIndex); // absolute distane to "left" root
            uint32_t dist_plus  = (globalIndex >= -(int)approxPolyRoot) ? (globalIndex + approxPolyRoot) : (-approxPolyRoot - globalIndex); // absolute distance to "right" root

            int approxPolyVal = log2_a + log2((int)dist_minus) + log2((int)dist_plus);// our log2 returns 0 if distance is 0
            isCandidate = blockEntries[index] > (approxPolyVal - threshold);
            blockEntries[index] = isCandidate;
        }

        if (isCandidate) {
            int candidateIndex = atomicAdd(&candidateWriteHead, 1);
            
            if(candidateIndex < maxPerBlock) { // Prevent buffer overflow.
                indexToCandidate[index] = candidateIndex;

                candidates[candidateIndex].b = b;
                candidates[candidateIndex].poly_id = poly_id;
                candidates[candidateIndex].sieve_offset = startOffset + index;
                candidates[candidateIndex].num_factors = 0; // Initialize counter
            }
            else
            {
                // CRITICAL:
                // If buffer is full, we MUST mark this entry as false.
                // Otherwise, the backward scan will try to process it using
                // uninitialized data from indexToCandidate, causing Illegal Access.
                blockEntries[index] = 0;
            }
        }
    }
    __syncthreads();
    // Return only candidates actually stored, not those dropped due to buffer overflow.
    // Returning 'count' would inflate candidatesFound past maxPerBlock, causing all
    // subsequent candidates in the cube to be dropped (globalIdx >= maxPerBlock for every thread).
    int actually_stored = min(candidateWriteHead - candidatesFound, maxPerBlock - (int)candidatesFound);
    return max(0, actually_stored);
}

// ---- BATCH SIEVING KERNELS ----

__global__ void globalMetaSieveBatchKernel(
    devicePointers dev_pointers,
    fixedSievingParams fs_params,
    // REMOVED: dynamicSievingParams ds_params
    // ADDED: Raw integers
    int current_step,       // Unused inside logic, but good for debug if needed
    int subCube,            // Formerly ds_params.subCube
    int sieveIntervalStart, // Formerly ds_params.startIndex

    generalSievingConfig gs_conf,
    globalMetaSieveConfig gms_conf
) {
    uint64_t* globalBucketEntries = dev_pointers.dev_globalBucketEntries;
    uint32_t* globalBucketCounts  = dev_pointers.dev_globalBucketCounts;

    primeDataSIQS* primeData = dev_pointers.dev_primeData;
    int shc_dim = fs_params.shc_dim;

    int offsetMask = gs_conf.sievingBlockSize - 1;

    extern __shared__ int sharedData[];
    int* global_bucket_write_head = sharedData;

    /*
    ---------------------------------------------------------------------------
    CYCLES LOOP
    ---------------------------------------------------------------------------
    */
    for(int cycle = 0; cycle < gms_conf.num_metaSieveCycles; cycle++){
        int currentStart = sieveIntervalStart + cycle * gms_conf.num_activeBlocksPerCycle * gs_conf.sievingBlockSize;
        
        /*
        -----------------------------------------------------------------------
        POLY BLOCKS LOOP
        -----------------------------------------------------------------------
        */
        for(int curPolyBlock = 0; curPolyBlock < gms_conf.num_polyBlocksPerThreadBlock; curPolyBlock ++){
            __syncthreads();
            for (int i = threadIdx.x; i < gms_conf.num_activeBucketsPerThreadBlock; i += blockDim.x) {
                global_bucket_write_head[i] = 0;
            }
            __syncthreads();
            int polyBlockId = blockIdx.x + gms_conf.num_threadBlocks * curPolyBlock;
            int fullPolyIdPrefix = (subCube * gs_conf.num_polysPerSieveCall) + (polyBlockId << gms_conf.log2_polyBlockSize);

            /*
            -------------------------------------------------------------------
            PRIMES LOOP
            -------------------------------------------------------------------
            */
            for(int currentPrimeIndex = threadIdx.x + gs_conf.bigPrimeStartIndex; currentPrimeIndex < fs_params.fb_size; currentPrimeIndex += blockDim.x){
                int polyIndex = ((threadIdx.x/32) % gms_conf.polyBlockSize);
                int polyId = gray(polyIndex);
                int fullPolyId = fullPolyIdPrefix | polyId;
                primeDataSIQS curPrimeData = primeData[currentPrimeIndex];
                uint32_t p = curPrimeData.p;
                int log2p = (31 - __clz(p)) * (1 - curPrimeData.inactive);
                uint32_t root1;
                uint32_t root2;
                rootsFromPolyId(fullPolyId, shc_dim, curPrimeData, root1, root2);
                int maxOffsetCount = ((gms_conf.num_activeBlocksPerCycle * gs_conf.sievingBlockSize) / (primeData[currentPrimeIndex - threadIdx.x].p)) + 1; //using the first prime as an upper bound to have a constant inner loop length
                int reducedSieveStart = ((currentStart % (int)p) + (int)p ) % (int)p;// Formula: ((S % p) + p) % p ensures result is in [0, p-1]

                /*
                ---------------------------------------------------------------
                POLYS LOOP
                ---------------------------------------------------------------
                */
                for(int polyCounter = 0; polyCounter < gms_conf.polyBlockSize; polyCounter++){
                    int globalBucketIdPrefix = ((((long long)polyBlockId) * gms_conf.polyBlockSize + polyId) * gms_conf.num_metaSieveCycles + cycle) * gms_conf.num_activeBlocksPerCycle;
                    int offsetCounter = 0;
		            // Advance roots from previous to next poly
                    int offset1 = currentStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root1, p) + (int)p;
                    int offset2 = currentStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root2, p) + (int)p;

                    /*
                    -----------------------------------------------------------
                    OFFSETS LOOP
                    -----------------------------------------------------------
                    */
                    for(offsetCounter = 0; offsetCounter < maxOffsetCount; offsetCounter++){
                        int curOffset = offset1;
                        for (int i = 0; i < 2; i++) {
                            // Logic: curOffset is a coordinate >= currentStart.
                            // Subtraction yields a non-negative relative index.
                            int sievingBlockHit = (curOffset - currentStart) / gs_conf.sievingBlockSize;
                            if (sievingBlockHit < gms_conf.num_activeBlocksPerCycle) {
                                int activeBucketId =  polyId * gms_conf.num_activeBlocksPerCycle + sievingBlockHit;
                                int index = atomicAdd(&global_bucket_write_head[activeBucketId], 1); // reserve an index
                                if (index < gs_conf.globalBucketSize) {
                                    // the bucket is not full yet, so write in it
                                    int sievingBlock_offset = (curOffset - currentStart) % gs_conf.sievingBlockSize; // no subtract needed if currentStart is a multiple of the blockSize
                                    int entry = sievingBlock_offset | (log2p << 24); // leading 8 bits store the log2 value and the other 24 the offset
                                    /*
                                    The activeBucketId is given by;
                                    activeBucketId = [polyId] [sievingBlock]
                                    The global bucket id is given by:
                                    globalBucketId = [polyBlockId] [polyId] [cycle] [sievingBlock]
                                    */
                                    long long int globalIndex = (globalBucketIdPrefix + sievingBlockHit) * gs_conf.globalBucketSize + index;
                                    // If we want to store the prime value, we do:
                                    // globalBucketEntries[globalIndex] = (((uint64_t)p) << 32) | entry;
                                    // But we store the index instead:
                                    globalBucketEntries[globalIndex] = (((uint64_t)currentPrimeIndex) << 32) | entry;
                                }
                            }
                            curOffset = offset2;
                        }
                        offset1 += (int)p; // Signed increment
                        offset2 += (int)p;
                    }
                    //Offsets loop finished, prepare offsets of the next polynomial for the same prime
                    int prevFullPolyId = fullPolyId;
                    polyIndex = modAdd(polyIndex, 1, gms_conf.polyBlockSize);
                    polyId = gray(polyIndex);
                    fullPolyId = fullPolyIdPrefix | polyId;
                    advanceRoots(prevFullPolyId, fullPolyId, curPrimeData, root1, root2);
                }
                //poly loop finished, nothing to do here
            }
            //prime loop finished, record the bucket fill levels as new buckets are required next time
            __syncthreads();
            for (int i = threadIdx.x; i < gms_conf.num_activeBucketsPerThreadBlock; i += blockDim.x) {
                /*
                i iterates over the active Buckets, we need to convert the id to the globalBucketId
                i = activeBucketId = [polyId] [sievingBlock]
                Want: globalBucketId = [polyBlockId] [polyId] [cycle] [sievingBlock]
                */
                uint32_t i_polyId = i / gms_conf.num_activeBlocksPerCycle;
                uint32_t i_sievingBlock = i % gms_conf.num_activeBlocksPerCycle;
                uint64_t i_globalBucketId = (((((uint64_t)polyBlockId) * gms_conf.polyBlockSize + i_polyId) * gms_conf.num_metaSieveCycles + cycle) * gms_conf.num_activeBlocksPerCycle)+i_sievingBlock;
                uint32_t amountWritten = global_bucket_write_head[i];
                uint32_t encoded = (min(amountWritten, gs_conf.globalBucketSize)) | ((amountWritten > gs_conf.globalBucketSize) ? 0x80000000u : 0u);
                globalBucketCounts[i_globalBucketId] = encoded;
            }
        }
        //poly blocks loop finished, nothing to do here
    }
    //cycles loop finished, done.
}

__global__ void __launch_bounds__(1024) sieveAndScanBatchKernel(
    devicePointers dev_pointers,
    fixedSievingParams fs_params,
    const mpqs::uint512* __restrict__ batch_a_array,
    const mpqs::uint512* __restrict__ batch_B_flat,
    uint32_t step_index,
    int32_t sieveIntervalStart, // Previously ds_params.startIndex (usually -M)
    uint32_t* __restrict__ dev_blockRelationCounts, // Output for compaction
    generalSievingConfig gs_conf,
    sieveAndScanConfig ss_conf) // We want gridDim.x blocks with size of blockDim.x
{
    primeDataSIQS* primeData = dev_pointers.dev_primeData;
    uint64_t* globalBucketEntries = dev_pointers.dev_globalBucketEntries;
    int32_t* indexToCandidate = dev_pointers.dev_indexToCandidate + blockIdx.x * gs_conf.sievingBlockSize;
    candidateRelation* candidates = dev_pointers.dev_candidateRelations + blockIdx.x * gs_conf.maxRelationsPerBlock;

    // Shared memory layout
    extern __shared__ uint8_t sharedByteData[];
    // Reserve space for B_values at the START (Alignment safe: uint512 is 16/8 byte aligned)
    mpqs::uint512* s_B_values = (mpqs::uint512*)sharedByteData;

    // Shift integer arrays to start AFTER the B_values
    // offset bytes = shc_dim * sizeof(uint512)
    // Cast to int* preserves signed logic for offsets (coordinates) and primes (sign flag)
    int* offsets1 = (int*)(s_B_values + fs_params.shc_dim);
    int* offsets2 = offsets1 + gs_conf.bigPrimeStartIndex;
    int* primes = offsets2 + gs_conf.bigPrimeStartIndex;

    // BlockEntries starts after primes
    uint8_t* blockEntries = (uint8_t*)(primes + gs_conf.bigPrimeStartIndex);

    // Load 'a' for this step into local register/stack
    mpqs::uint512 current_a = batch_a_array[step_index];

    polyData p_data;
    p_data.approxPolyRoot = fs_params.approxPolyRoot;
    p_data.threshold = fs_params.threshold;
    // We calculate log2_a on device)
    p_data.log2_a = current_a.msb();

    // Cooperative Load: B_values Global -> Shared
    // We access the slice corresponding to 'step_index'
    const mpqs::uint512* my_B_global = batch_B_flat + (step_index * fs_params.shc_dim);

    if (threadIdx.x < fs_params.shc_dim) {
        s_B_values[threadIdx.x] = my_B_global[threadIdx.x];
    }
    // Barrier to ensure B_values are ready and offsets pointers are safe
    __syncthreads();

    int polyIdPrefix = blockIdx.x * (gs_conf.num_polysPerSieveCall) / gridDim.x;

    mpqs::uint512 b((uint32_t)0);
    // Use s_B_values (Shared) instead of global pointer
    bFromPolyId(polyIdPrefix, fs_params.shc_dim, s_B_values, b);

    __syncthreads();

    int reducedSieveStart = 0;
    uint32_t candidatesFound = 0;

    for (int i = threadIdx.x; i < gs_conf.bigPrimeStartIndex; i += blockDim.x) {
        primeDataSIQS curPrimeData = primeData[i];

        // p fits in int32 (< 2^31). Cast explicit.
        int p = (int)curPrimeData.p;

        // Roots are now unsigned uint32_t residues [0, p-1]
        uint32_t root1 = 0;
        uint32_t root2 = 0;
        rootsFromPolyId(polyIdPrefix, fs_params.shc_dim, curPrimeData, root1, root2);

        reducedSieveStart = ((sieveIntervalStart % p) + p) % p;

        // modSub_shifted returns uint32_t.
        // We cast back to int for coordinate calculation.
        // Logic: Start - (Distance to next root) + p. Result is positive coordinate relative to start.
        offsets1[i] = sieveIntervalStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root1, (uint32_t)p) + p;
        offsets2[i] = sieveIntervalStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root2, (uint32_t)p) + p;

        // Inactive logic: p * (1 - 2*0) = p, p * (1 - 2*1) = -p.
        // Cast inactive to int to ensure correct arithmetic.
        primes[i] = p * (1 - 2 * curPrimeData.inactive);
    }
    __syncthreads();
    int prevPolyId = 0;
    int polyId = 0;
    for (int poly = 0; poly < (gs_conf.num_polysPerSieveCall) / gridDim.x; poly++) {//we dont care about the polyblocks here, incrementing is done below
        polyId = polyIdPrefix | gray(poly);
        if (poly > 0) {
            __syncthreads();
            for (int i = threadIdx.x; i < gs_conf.bigPrimeStartIndex; i += blockDim.x) {
                primeDataSIQS curPrimeData = primeData[i];
		int p = (int)curPrimeData.p; // Explicit cast

                // Recalculate current roots based on offsets
                // ((offset % p) + p) % p ensures positive residue
                uint32_t root1 = (uint32_t)(((offsets1[i] % p) + p) % p);
                uint32_t root2 = (uint32_t)(((offsets2[i] % p) + p) % p);

                // Update roots for Gray code step (uses uint32_t internally)
                advanceRoots(prevPolyId, polyId, curPrimeData, root1, root2);

                reducedSieveStart = ((sieveIntervalStart % p) + p) % p;

                // Update offsets using new roots
                offsets1[i] = sieveIntervalStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root1, (uint32_t)p) + p;
                offsets2[i] = sieveIntervalStart - (int)modSub_shifted((uint32_t)reducedSieveStart, root2, (uint32_t)p) + p;
            }
	    // Update b coefficient (mpqs::uint512)
            advance_b(prevPolyId, polyId, s_B_values, b);
        }
        __syncthreads();

        for (int sieveBlock = 0; sieveBlock < gs_conf.num_sievingBlocksPerSieveCall; sieveBlock++) {
            for (int i = threadIdx.x; i < gs_conf.sievingBlockSize; i += blockDim.x) {
                blockEntries[i] = 0;
            }
            int sieveBlockStart = sieveIntervalStart + sieveBlock * gs_conf.sievingBlockSize;
            int sieveBlockEnd = sieveBlockStart + gs_conf.sievingBlockSize;
            __syncthreads();

            uint64_t globalBucketId = (((long long)polyId) * gs_conf.num_sievingBlocksPerSieveCall + sieveBlock);
            uint64_t listStart = globalBucketId * gs_conf.globalBucketSize; //CHANGE TO THE CURRENT GLOBAL BUCKET
            uint64_t* currentBucketEntries = globalBucketEntries + listStart;
            //dump globalBucketEntries into the current sieving block
            uint32_t currentFillLevel = (dev_pointers.dev_globalBucketCounts[globalBucketId]) & (16777216 - 1);
            for (int i = threadIdx.x; i < currentFillLevel; i += blockDim.x) {
                uint32_t val = (uint32_t)currentBucketEntries[i];
                ATOMIC_BYTE_ADD(blockEntries, val & ((1 << 24) - 1), val >> 24);
            }
            int midPrimeStart = gs_conf.midPrimeStartIndex;
            __syncthreads();
            for (int i = 0; i < midPrimeStart; i++) {
                int p = primes[i];
		        // If p is inactive, we do not touch blockEntries at all
		        // This prevents useless indexing work.
		        if (p < 0) {
		            __syncthreads();
		            continue;
		        }
		        // Here p is guaranteed > 0.
                uint8_t log_p = log2(p); // (p <= 0) ? 0 : log2(p);
                // p = abs(p);
                int offset1 = offsets1[i];
                int offset2 = offsets2[i];
		        // Forward Sieve: Offset is signed int, loop terminates when offset >= sieveBlockEnd
                int offset = offset1 + threadIdx.x * p;
                for (; offset < sieveBlockEnd; offset += blockDim.x * p) {
                    blockEntries[offset - sieveBlockStart] += log_p;
                }
                if (offset - p < sieveBlockEnd) {
                    offsets1[i] = offset; //exactly one thread has the correct "last" offset, keep it for the next iteration
                }
                offset = offset2 + threadIdx.x * p;
                for (; offset < sieveBlockEnd; offset += blockDim.x * p) {
                    blockEntries[offset - sieveBlockStart] += log_p;
                }
                if (offset - p < sieveBlockEnd) {
                    offsets2[i] = offset; //exactly one thread has the correct "last" offset, keep it for the next iteration
                }
                __syncthreads();
            }
	    // Small primes handling
            for (int i = midPrimeStart + threadIdx.x; i < gs_conf.bigPrimeStartIndex; i += blockDim.x) {
                int p = primes[i];
                uint8_t log_p = (p <= 0) ? 0 : log2(p);
                p = abs(p);
                int offset1 = offsets1[i];
                int offset2 = offsets2[i];
                //__syncthreads(); //NO SYNC NEEDED HERE, threads access disjoint data
                int offset = offset1;
                for (; offset < sieveBlockEnd; offset += p) {
                    ATOMIC_BYTE_ADD(blockEntries, offset - sieveBlockStart, log_p);
                }
                offsets1[i] = offset; //keep the offset for the next iteration
                offset = offset2;
                for (; offset < sieveBlockEnd; offset += p) {
                    ATOMIC_BYTE_ADD(blockEntries, offset - sieveBlockStart, log_p);
                }
                offsets2[i] = offset; //keep the offset for the next iteration
            }
            __syncthreads();

	    // Check for candidates
	    int newCandidateCount = excludeNonRelations(
		blockEntries,
		indexToCandidate,
		candidates, // Pass array
		b,          // Pass uint512 (by const ref logic)
		polyId,     // Pass ID
		candidatesFound,
		sieveBlockStart,
		gs_conf.sievingBlockSize,
		gs_conf.maxRelationsPerBlock,
		p_data
	    );
            candidatesFound += newCandidateCount;
            if (newCandidateCount == 0) {
                continue; //no candidates have been found, so skip this sieveBlock
            }

	    // Backward Sieve (Scanning candidates for factors)
            for (int i = 0; i < midPrimeStart; i++) {
                int p = primes[i];
                bool active = p > 0;
                p = abs(p);
                int offset1 = offsets1[i];
                int offset2 = offsets2[i];
                __syncthreads();
                if (active) {
                    // Backward loop using signed arithmetic.
                    // Loop terminates when offset <= sieveBlockStart.
                    // Safe because offset is int and subtracts p.
                    int offset = offset1 - threadIdx.x * p - p;
                    for (; offset >= sieveBlockStart; offset -= blockDim.x * p) {
                        int localOffset = offset - sieveBlockStart;
                        if (blockEntries[localOffset]) {
                            int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
			    int globalIdx = indexToCandidate[localOffset];
			    // Store factor
			    candidates[globalIdx].factors[31 & newPrimeIndex] = i;
			    // Update count
			    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                        }
                    }
                    offset = offset2 - threadIdx.x * p - p;
                    for (; offset >= sieveBlockStart; offset -= blockDim.x * p) {
                        int localOffset = offset - sieveBlockStart;
                        if (blockEntries[localOffset]) {
                            int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
			    int globalIdx = indexToCandidate[localOffset];
			    // Store factor
			    candidates[globalIdx].factors[31 & newPrimeIndex] = i;
			    // Update count
			    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                        }
                    }
                }
                __syncthreads();
            }
	    // Small primes backward scan
            for (int i = midPrimeStart + threadIdx.x; i < gs_conf.bigPrimeStartIndex; i += blockDim.x) {
                int p = primes[i];
                bool active = p > 0;
                p = abs(p);
                int offset1 = offsets1[i];
                int offset2 = offsets2[i];
                //__syncthreads(); AS ABOVE, NO SYNC NEEDED
                if (active) {
                    int offset = offset1 - p;
                    for (; offset >= sieveBlockStart; offset -= p) {
                        int localOffset = offset - sieveBlockStart;
                        if (blockEntries[localOffset]) {
                            int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
			    int globalIdx = indexToCandidate[localOffset];
			    // Store factor
			    candidates[globalIdx].factors[31 & newPrimeIndex] = i;
			    // Update count
			    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                        }
                    }
                    offset = offset2 - p;
                    for (; offset >= sieveBlockStart; offset -= p) {
                        int localOffset = offset - sieveBlockStart;
                        if (blockEntries[localOffset]) {
                            int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
			    int globalIdx = indexToCandidate[localOffset];
			    // Store factor
			    candidates[globalIdx].factors[31 & newPrimeIndex] = i;
			    // Update count
			    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                        }
                    }
                }
            }

            for (int i = threadIdx.x; i < currentFillLevel; i += blockDim.x) {
                uint64_t val = currentBucketEntries[i];
		if (val == 0) continue; // <- ignore illegal entries
                int localOffset = val & ((1 << 24) - 1);
                int prime_index = val >> 32;
                if (blockEntries[localOffset]) {
                    int newPrimeIndex = ATOMIC_BYTE_ADD_RETURN(blockEntries, localOffset, 1) - 1;
		    int globalIdx = indexToCandidate[localOffset];
		    // Store factor
		    candidates[globalIdx].factors[31 & newPrimeIndex] = prime_index;
		    // Update count
		    atomicAdd((uint32_t*)&candidates[globalIdx].num_factors, 1);
                }
            }
            __syncthreads();
        }
        prevPolyId = polyId;
        __syncthreads();
    }

    // --- FINAL OUTPUT FOR COMPACTION ---
    // Save the number of relations found by this specific block
    if (threadIdx.x == 0) {
        dev_blockRelationCounts[blockIdx.x] = candidatesFound;
    }
}

// -----------------------------------------------------------------------------
// Compaction Batch Kernel
// -----------------------------------------------------------------------------
// (from postprocessing, moved here)
// CONTRACT: GridDim.x == Number of Sieving Blocks
__global__ void compactCandidatesBatchKernel(
    const mpqs::sieve::candidateRelation* __restrict__ input,
    uint32_t maxRelationsPerBlock,
    const mpqs::uint512* __restrict__ batch_a_array,
    const uint32_t* __restrict__ dev_factor_indices_flat, // [steps * shc_dim]
    uint32_t step_index,
    uint32_t shc_dim, // num_a_factors
    const uint32_t* __restrict__ dev_blockRelationCounts, // The map produced by the sieve kernel
    int32_t start_index,
    DenseCandidate* __restrict__ output,
    uint32_t* __restrict__ counter,
    uint32_t max_capacity
) {
    // 1. Identify Work Unit
    // We assume GridDim.x == Number of Sieving Blocks
    uint32_t sieveBlockIdx = blockIdx.x;

    // 2. Check Validity (Fast Exit)
    // We read the count produced by the sieve kernel.
    // If 0, the whole block exits. No scanning zeros!
    uint32_t count = dev_blockRelationCounts[sieveBlockIdx];
    if (count == 0) return;

    // 3. Load Context ('a' and its factors)
    // We use Shared Memory to cache 'a_factors' for the block
    extern __shared__ uint32_t s_a_factors[]; // Effective Size: shc_dim * sizeof(uint32_t)

    // Load 'a' (Broadcast or single thread read)
    // Note: uint512 is large, reading it into a register per thread is fine
    // since only active threads (i < count) need it.
    mpqs::uint512 current_a = batch_a_array[step_index];

    // Cooperative Load of 'a' factors from Global -> Shared
    const uint32_t* my_a_factors = dev_factor_indices_flat + (step_index * shc_dim);
    for (uint32_t i = threadIdx.x; i < shc_dim; i += blockDim.x) {
        s_a_factors[i] = my_a_factors[i];
    }
    __syncthreads(); // Ensure factors are loaded

    // 4. Processing Loop
    // Calculate the base offset in the sparse input array
    uint64_t blockBaseOffset = (uint64_t)sieveBlockIdx * maxRelationsPerBlock;

    // Iterate ONLY over the valid candidates
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {

        // Load Candidate
        mpqs::sieve::candidateRelation in_rel = input[blockBaseOffset + i];

        // --- ATOMIC RESERVE ---
        uint32_t pos = atomicAdd(counter, 1);
        if (pos >= max_capacity) {
            // Optional: Set a global overflow flag here
            return;
        }

	// We first sort in_rel.factors in increasing order
	// Since this array is statistically almost sorted,
	// We chose a sorting strategy which takes this into
	// account.
	// --- Sort Step (In-Place on Registers) ---
	// Adaptive: Runs in O(N) if already sorted.
	// No extra memory buffer needed.
        uint32_t n_sieve = in_rel.num_factors;
	if(n_sieve > 32)
	    // we have an overflow
	    continue;
        for (int k = 1; k < n_sieve; ++k) {
            uint32_t key = in_rel.factors[k];
            int j = k - 1;
            while (j >= 0 && in_rel.factors[j] > key) {
                in_rel.factors[j + 1] = in_rel.factors[j];
                j--;
            }
            in_rel.factors[j + 1] = key;
        }

        // --- OUTPUT GENERATION ---
        // Write directly to global memory
        output[pos].a = current_a;
        output[pos].b = in_rel.b;
        output[pos].true_x = (int32_t)in_rel.sieve_offset;

	// --- Merge Step ---
        // Two-pointer merge of two sorted arrays into sorted output:
        //   a-factors from SHARED memory  (s_a_factors[0..shc_dim-1])
        //   sieve factors from REGISTERS  (in_rel.factors[0..n_sieve-1])
        uint8_t merged_count = 0;
        uint32_t ai = 0, si = 0;

        while (ai < shc_dim && si < n_sieve && merged_count < 48) {
            if (s_a_factors[ai] <= in_rel.factors[si]) {
                output[pos].factor_indices[merged_count++] = s_a_factors[ai++];
            } else {
                output[pos].factor_indices[merged_count++] = in_rel.factors[si++];
            }
        }
        while (ai < shc_dim && merged_count < 48) {
            output[pos].factor_indices[merged_count++] = s_a_factors[ai++];
        }
        while (si < n_sieve && merged_count < 48) {
            output[pos].factor_indices[merged_count++] = in_rel.factors[si++];
        }

        output[pos].num_factors = merged_count;
    }
}

/*
 * Marks specific primes as inactive for the current batch step.
 * Primes dividing 'a' must be skipped during sieving.
 */
__global__ void markInactivePrimesBatchKernel(
    int shc_dim,
    const uint32_t* __restrict__ batch_factor_indices_flat,
    uint32_t step_index,
    primeDataSIQS* primeData
) {
    // Only 1 block is launched for this kernel
    if (blockIdx.x > 0) return;

    // Calculate pointer to the factors for the current step
    // Input is flattened: [step 0 factors] [step 1 factors] ...
    const uint32_t* my_factors = batch_factor_indices_flat + (step_index * shc_dim);

    // Parallelize marking (though shc_dim is small, usually < 20)
    for(int i = threadIdx.x; i < shc_dim; i += blockDim.x){
        uint32_t p_index = my_factors[i];

        // Mark as inactive in the global primeData array
        // Note: We assume primeData was just refreshed/reset by initPrimeDataBatchKernel
        primeData[p_index].inactive = 1;
    }
}
 
__global__ void resetBatchCountersKernel(
    uint32_t* __restrict__ bucketCounts,
    uint32_t numBuckets,
    uint32_t* __restrict__ blockCounts,
    uint32_t numBlocks
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Reset Bucket Counts (Larger array)
    /*
    if (tid < numBuckets) {
        bucketCounts[tid] = 0;
    }
    */

    // Reset Block Counts (Smaller array)
    // Only first few threads do this
    if (tid < numBlocks) {
        blockCounts[tid] = 0;
    }
}

/*
 * Batch Initialization Kernel.
 *
 * Prepares the 'primeData' array for the specific polynomial step 'step_index'.
 * - Caches 'a' and 'B_values' for this step in Shared Memory.
 * - Computes modular inverses and reductions for every prime in the Factor Base.
 */
__global__ void initPrimeDataBatchKernel(
    devicePointers dev_pointers,
    generalSievingConfig gs_conf,
    fixedSievingParams fs_params,
    const mpqs::uint512* __restrict__ batch_a_array,
    uint32_t step_index
)
{
    // --- 1. Shared Memory Setup ---
    // We store 'a' (1 element) and 'B_values' (shc_dim elements)
    extern __shared__ __align__(16) uint8_t sharedByteData[];
    mpqs::uint512* s_storage = (mpqs::uint512*)sharedByteData;

    mpqs::uint512* s_a = &s_storage[0];
    mpqs::uint512* s_B_values = &s_storage[1];

    // ...
    int shc_dim = fs_params.shc_dim;
    int fb_size = fs_params.fb_size;

    // --- 2. Cooperative Loading ---
    // Load 'a'
    if (threadIdx.x == 0) {
        *s_a = batch_a_array[step_index];
    }

    // Load 'B_values' corresponding to this step
    // Global ptr: base + (step * dim)
    const mpqs::uint512* my_B_global = dev_pointers.dev_job_B_flat + (step_index * shc_dim);

    if (threadIdx.x < shc_dim) {
        s_B_values[threadIdx.x] = my_B_global[threadIdx.x];
    }

    // Barrier: Ensure Shared Memory is populated before any thread proceeds
    __syncthreads();

    // --- 3. Per-Prime Calculation ---
    uint32_t* factorBase = dev_pointers.dev_factorBase;
    uint32_t* rootN = dev_pointers.dev_rootN;
    primeDataSIQS* primeData = dev_pointers.dev_primeData;

    int stride = gridDim.x * blockDim.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = id; i < fb_size; i += stride) {
        uint32_t p = factorBase[i];

        // A. Reduce 'a' modulo p (reading from Shared Memory)
        uint32_t reduced_a = s_a->mod_uint32(p);

        // B. Invert 'a' modulo p (Binary Exponentiation Method: a^(p-2))
        int exponent = p - 2;
        int base = reduced_a;
        int result = 1;

        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
        for (int j = 0; j < 32; j++) {
            // Branchless conditional multiply
            result = ((exponent & 1) == 1) * (uint32_t)(((uint64_t)result * base) % p) + ((exponent & 1) == 0) * (result);
            exponent = exponent >> 1;
            base = (uint32_t)(((uint64_t)base * base) % p);
        }

        primeDataSIQS tmpPrimeData;
        tmpPrimeData.mod_inverse_a = result;
	// Set inactive
	tmpPrimeData.inactive = 0;

        // C. Calculate B_values used to update roots (reading from Shared Memory)
        // B_val_stored = (B_val_original * inv_a) % p
        for (int j = 0; j < shc_dim; j++) {
            // Reduce 512-bit B value mod p
            uint32_t b_val_mod = s_B_values[j].mod_uint32(p);

            // Multiply by inverse a
            tmpPrimeData.B_values[j] = (uint32_t)(((uint64_t)b_val_mod * result) % p);
        }

        tmpPrimeData.p = p;
        // D. Calculate inv_aN = (inv_a * rootN) % p
        tmpPrimeData.inv_aN = (uint32_t)(((uint64_t)result * rootN[i]) % p);

        // E. Save to Global Memory
        primeData[i] = tmpPrimeData;
    }
}

__global__ void generatePolynomialsKernel(
    uint32_t num_steps,
    uint32_t shc_dim,
    const uint32_t* __restrict__ factor_indices_flat, // [steps * dim]
    const uint32_t* __restrict__ dev_factorBase,
    const uint32_t* __restrict__ dev_rootN,
    mpqs::uint512* __restrict__ out_a,                // [steps]
    mpqs::uint512* __restrict__ out_B_values_flat     // [steps * dim]
) {
    // 1 Block per Step
    uint32_t step_idx = blockIdx.x;
    if (step_idx >= num_steps) return;

    extern __shared__ mpqs::uint512 s_storage[]; // Configurable size
    mpqs::uint512* s_a = &s_storage[0];

    // --- 1. Compute 'a' via Reduction ---
    uint32_t tid = threadIdx.x;

    // Each thread loads a prime (if tid < dim)
    uint32_t p = 1;
    if (tid < shc_dim) {
        uint32_t fb_idx = factor_indices_flat[step_idx * shc_dim + tid];
        p = dev_factorBase[fb_idx];
    } else if (tid < 16) {
        // Zero out padding to ensure reduction works
        s_storage[1 + tid] = mpqs::uint512((uint32_t)1);
    }

    // Initialize shared memory with primes as uint512
    if (tid < 16) {
      s_storage[1 + tid] = (tid < shc_dim ? mpqs::uint512(p) : mpqs::uint512((uint32_t)1)); // Padding for reduction
    }
    __syncthreads();

    // Simple Tree Reduction (for dim=16)
    // Needs proper sync and care for uint512 copy cost
    if (tid < 8) s_storage[1+tid].mult(s_storage[1+tid+8]); __syncthreads();
    if (tid < 4) s_storage[1+tid].mult(s_storage[1+tid+4]); __syncthreads();
    if (tid < 2) s_storage[1+tid].mult(s_storage[1+tid+2]); __syncthreads();
    if (tid == 0) {
        s_storage[1].mult(s_storage[2]);
        *s_a = s_storage[1]; // Store 'a' in shared[0]
        out_a[step_idx] = *s_a; // Write to global
    }
    __syncthreads();

    // --- 2. Compute B_values (Parallel) ---
    // We need 'a' visible to all threads. s_a is set.

    if (tid < shc_dim) {
        uint32_t fb_idx = factor_indices_flat[step_idx * shc_dim + tid];
        uint32_t p = dev_factorBase[fb_idx];
        uint32_t r = dev_rootN[fb_idx];

        // a / p_k
        // We use a specialized "div_uint32_copy" that returns quotient,
        // does not modify s_a (which is shared).
        mpqs::uint512 Tk = s_a->div_uint32_const(p);

        // (a/p) mod p -> Tk mod p
        uint32_t rem = Tk.mod_uint32(p);

        // Modular Inverse
        uint32_t inv = mpqs::math::mod_inverse(rem, p);

        // Gamma
        uint64_t gamma_long = ((uint64_t)inv * r) % p;
        uint32_t gamma = (uint32_t)gamma_long;

        // Centering (Knuth-Schroeppel)
        if (gamma > (p >> 1)) gamma = p - gamma;

        // B_k = Tk * gamma
        Tk.mult_uint32(gamma);

        // Write to global
        out_B_values_flat[step_idx * shc_dim + tid] = Tk;
    }
}

// Kernel launchers

void initPrimeData(
    devicePointers dev_pointers,
    initConfig init_conf,
    generalSievingConfig gs_conf,
    fixedSievingParams fs_params,
    dynamicSievingParams ds_params,
    cudaStream_t stream
) {
    #ifdef SIEVING_DEBUG_FLAG
        cudaError_t cudaStatus;
        cudaGetLastError();
    #endif
    initPrimeDataKernel<<< init_conf.num_threadBlocks, init_conf.num_threadsPerBlock, 0, stream>>>(dev_pointers, gs_conf, fs_params, ds_params);
    #ifdef SIEVING_DEBUG_FLAG
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "Error in initPrimeDataKernel: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
	    }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "Error in initPrimeDataKernel: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
	    }
    #endif
    markInactivePrimesKernel << <1, 32, 0, stream >> > (fs_params.shc_dim, dev_pointers.dev_a_factors, dev_pointers.dev_primeData);
    #ifdef SIEVING_DEBUG_FLAG
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "Error in markInactivePrimesKernel: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
	    }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "Error in markInactivePrimesKernel: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
	    }
    #endif
}

void globalMetaSieve(
    devicePointers dev_pointers,
    fixedSievingParams fs_params,
    dynamicSievingParams ds_params,
    generalSievingConfig gs_conf,
    globalMetaSieveConfig gms_conf,
    cudaStream_t stream
) {
    #ifdef SIEVING_DEBUG_FLAG
        cudaError_t cudaStatus;
        cudaGetLastError();
    #endif
    globalMetaSieveKernel <<<gms_conf.num_threadBlocks, gms_conf.num_threadsPerBlock, gms_conf.sharedMemReq, stream >>> (dev_pointers, fs_params, ds_params, gs_conf, gms_conf);
    #ifdef SIEVING_DEBUG_FLAG
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "Error in globalMetaSieveKernel: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
	    }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "Error in globalMetaSieveKernel: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
	    }
    #endif
}

void sieveAndScan(
    devicePointers dev_pointers,
    fixedSievingParams fs_params,
    dynamicSievingParams ds_params,
    generalSievingConfig gs_conf,
    sieveAndScanConfig ss_conf,
    cudaStream_t stream
) {
    #ifdef SIEVING_DEBUG_FLAG
        cudaError_t cudaStatus;
        cudaGetLastError();
    #endif
    sieveAndScanKernel <<< ss_conf.num_threadBlocks, ss_conf.num_threadsPerBlock, ss_conf.sharedMemReq, stream >>> (dev_pointers, fs_params, ds_params, gs_conf, ss_conf);
    #ifdef SIEVING_DEBUG_FLAG
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "Error in sieveAndScanKernel: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
	    }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "Error in sieveAndScanKernel: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
	    }
    #endif
}

// ---- BATCH SIEVING LAUNCHERS -----

void prepareSievingBatch(
    const devicePointers* dev_pointers_ptr,
    const std::vector<uint32_t>* next_factor_indices_ptr,
    const uint32_t shc_dim,
    const uint32_t batch_size,
    cudaStream_t stream,
    uint32_t* h_pinned_factor_indices
) {
    // 1. Upload Indices (Async)
    // The vector size is (batch_size * shc_dim).
    size_t copy_bytes = next_factor_indices_ptr->size() * sizeof(uint32_t);

    if (h_pinned_factor_indices) {
        // Copy to pinned staging buffer, then truly async H2D transfer.
        // Without this, cudaMemcpyAsync from pageable memory forces an implicit
        // stream synchronization (runtime drains all queued kernels first).
        memcpy(h_pinned_factor_indices, next_factor_indices_ptr->data(), copy_bytes);
        cudaMemcpyAsync(
            dev_pointers_ptr->dev_job_factor_indices,
            h_pinned_factor_indices,
            copy_bytes,
            cudaMemcpyHostToDevice,
            stream
        );
    } else {
        // Fallback: pageable copy (implicit sync — legacy path)
        cudaMemcpyAsync(
            dev_pointers_ptr->dev_job_factor_indices,
            next_factor_indices_ptr->data(),
            copy_bytes,
            cudaMemcpyHostToDevice,
            stream
        );
    }

    // 2. Launch Polynomial Generation Kernel
    // Shared Memory Calculation:
    // We need a power-of-2 array of uint512s for the tree reduction.
    // Minimum 32 to handle warp-level operations safely.
    uint32_t reduction_dim = 32;
    while (reduction_dim < shc_dim) {
        reduction_dim <<= 1; // * 2
    }
    size_t smem_size = (reduction_dim + 1) * sizeof(mpqs::uint512);

    // Threads per block: 64 is sufficient for the reduction and B-value computation
    // Grid size: One block per step in the batch
    mpqs::sieve::generatePolynomialsKernel<<<batch_size, 64, smem_size, stream>>>(
        batch_size,
        shc_dim,
        dev_pointers_ptr->dev_job_factor_indices,
        dev_pointers_ptr->dev_factorBase,
        dev_pointers_ptr->dev_rootN,
        dev_pointers_ptr->dev_job_a_array,
        dev_pointers_ptr->dev_job_B_flat
    );
}

void prepareSievingBatchFromStaged(
    const devicePointers* dev_pointers_ptr,
    const uint32_t* d_indices,
    mpqs::uint512* a_array_out,
    mpqs::uint512* B_flat_out,
    const uint32_t shc_dim,
    const uint32_t batch_size,
    cudaStream_t stream
) {
    // No H2D copy — indices are already on device in d_indices.
    // Launch polynomial generation directly.
    uint32_t reduction_dim = 32;
    while (reduction_dim < shc_dim) reduction_dim <<= 1;
    size_t smem_size = (reduction_dim + 1) * sizeof(mpqs::uint512);

    mpqs::sieve::generatePolynomialsKernel<<<batch_size, 64, smem_size, stream>>>(
        batch_size, shc_dim,
        d_indices,
        dev_pointers_ptr->dev_factorBase,
        dev_pointers_ptr->dev_rootN,
        a_array_out,
        B_flat_out
    );
}

void runSievingBatch(
    devicePointers* dev_pointers_ptr,
    initConfig* init_conf_ptr,
    generalSievingConfig* gs_conf_ptr,
    fixedSievingParams* fs_params_ptr,
    dynamicSievingParams* ds_params_ptr,
    globalMetaSieveConfig* gms_conf_ptr,
    sieveAndScanConfig* ss_conf_ptr,
    int num_steps,
    int start_batch_index,
    cudaStream_t stream
) {
    // Pre-calculate grid dimensions to avoid overhead inside the loop
    // 1. Reset Kernel Config
    // We need to clear:
    //   - globalBucketCounts (Size: num_polys * num_sievingBlocks)
    //   - blockRelationCounts (Size: num_sievingBlocks)
    // A simple 1D grid covers the larger buffer, usually globalBucketCounts.
    uint32_t total_buckets = gs_conf_ptr->num_polysPerSieveCall * gs_conf_ptr->num_sievingBlocksPerSieveCall;
    dim3 reset_grid((total_buckets + 255) / 256);
    dim3 reset_block(256);

    // 2. InitPrimeData Config
    dim3 init_grid = init_conf_ptr->num_threadBlocks;
    dim3 init_block = init_conf_ptr->num_threadsPerBlock;

    // 3. MetaSieve Config
    dim3 meta_grid = gms_conf_ptr->num_threadBlocks;
    dim3 meta_block = gms_conf_ptr->num_threadsPerBlock;
    size_t meta_smem = gms_conf_ptr->sharedMemReq;

    // 4. SieveAndScan Config (Grid = Number of Sieving Blocks)
    dim3 sieve_grid = ss_conf_ptr->num_threadBlocks;
    dim3 sieve_block = ss_conf_ptr->num_threadsPerBlock;
    size_t sieve_smem = ss_conf_ptr->sharedMemReq + (fs_params_ptr->shc_dim * sizeof(mpqs::uint512)); // Add space for B_values

    // 5. Compact Config (One block per Sieving Block)
    //dim3 compact_grid = gs_conf_ptr->num_sievingBlocksPerSieveCall;
    dim3 compact_grid = ss_conf_ptr->num_threadBlocks;
    dim3 compact_block = 256;
    size_t compact_smem = fs_params_ptr->shc_dim * sizeof(uint32_t); // For caching 'a' factors

    // Calculate shared memory required for initPrimeDataBatchKernel
    // We need space for 'a' (1 uint512) + 'B_values' (shc_dim uint512s)
    size_t init_smem = sizeof(mpqs::uint512) * (1 + fs_params_ptr->shc_dim);

    // --- The "Small" Inner Loop ---
    // No CPU<->GPU syncs here. Just pushing commands to the queue.
    for (int i = 0; i < num_steps; i++) {
        uint32_t current_step = start_batch_index + i;

        // Step 1: Reset Counters
        // We use a custom kernel instead of cudaMemsetAsync to minimize driver overhead in the loop
        resetBatchCountersKernel<<<reset_grid, reset_block, 0, stream>>>(
            dev_pointers_ptr->dev_globalBucketCounts,
            total_buckets,
            dev_pointers_ptr->dev_blockRelationCounts,
            ss_conf_ptr->num_threadBlocks
        );

        #ifdef SIEVING_DEBUG_FLAG
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	  fprintf(stdout, "cudaDeviceSynchronize failed after resetBatchCounters kernel launch.\n");
	  fprintf(stdout, "Error: %s\n", cudaGetErrorString(cudaStatus));
	  exit(-1);
	}
        #endif

        // Step 2: Update Prime Data (Roots & Inverses for the new 'a')
        initPrimeDataBatchKernel<<<init_grid, init_block, init_smem, stream>>>(
            *dev_pointers_ptr,
            *gs_conf_ptr,
            *fs_params_ptr,
            dev_pointers_ptr->dev_job_a_array, // Pointer to pre-calc 'a' array
            current_step
        );

        #ifdef SIEVING_DEBUG_FLAG
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	  fprintf(stdout, "cudaDeviceSynchronize failed after initPrimeDataBatch kernel launch.\n");
	  fprintf(stdout, "Error: %s\n", cudaGetErrorString(cudaStatus));
	  exit(-1);
	}
        #endif

        // Step 2b: Mark inactive primes (divisors of 'a')
        // Only 1 block needed
        markInactivePrimesBatchKernel<<<1, 64, 0, stream>>>(
            fs_params_ptr->shc_dim,
            dev_pointers_ptr->dev_job_factor_indices, // Flat array of factors
            current_step,
            dev_pointers_ptr->dev_primeData
        );

        #ifdef SIEVING_DEBUG_FLAG
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	  fprintf(stdout, "cudaDeviceSynchronize failed after markInactivePrimesBatch kernel launch.\n");
	  fprintf(stdout, "Error: %s\n", cudaGetErrorString(cudaStatus));
	  exit(-1);
	}
        #endif

        // Step 3: Meta Sieve (Fill Buckets)
        globalMetaSieveBatchKernel<<<meta_grid, meta_block, meta_smem, stream>>>(
            *dev_pointers_ptr,
            *fs_params_ptr,
            // ds_params removed, passing step index
            current_step,
            // subCube index logic (assuming 1 subCube per step for now, or derived from step)
            0,
            ds_params_ptr->startIndex, // -M
            *gs_conf_ptr,
            *gms_conf_ptr
        );

        #ifdef SIEVING_DEBUG_FLAG
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	  fprintf(stdout, "cudaDeviceSynchronize failed after globalMetaSieveBatch kernel launch.\n");
	  fprintf(stdout, "Error: %s\n", cudaGetErrorString(cudaStatus));
	  exit(-1);
	}
        #endif

        // Step 4: Sieve & Scan (Produce Relation Candidates + Block Counts)
        sieveAndScanBatchKernel<<<sieve_grid, sieve_block, sieve_smem, stream>>>(
            *dev_pointers_ptr,
            *fs_params_ptr,
            dev_pointers_ptr->dev_job_a_array,
            dev_pointers_ptr->dev_job_B_flat,
            current_step,
            ds_params_ptr->startIndex, // -M
            dev_pointers_ptr->dev_blockRelationCounts, // Output count array
            *gs_conf_ptr,
            *ss_conf_ptr
        );

        #ifdef SIEVING_DEBUG_FLAG
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	  fprintf(stdout, "cudaDeviceSynchronize failed after sieveAndScanBatch kernel launch.\n");
	  fprintf(stdout, "Error: %s\n", cudaGetErrorString(cudaStatus));
	  exit(-1);
	}
        #endif

        // Step 5: Compact & Accumulate (Write to PostProc Buffer)
        // Accesses PostProcessing Controller's buffers.
        // Note: You might need to pass the pointers in via dev_pointers or a separate struct.
        compactCandidatesBatchKernel<<<compact_grid, compact_block, compact_smem, stream>>>(
            dev_pointers_ptr->dev_candidateRelations,
            gs_conf_ptr->maxRelationsPerBlock, // Stride
            dev_pointers_ptr->dev_job_a_array,
            dev_pointers_ptr->dev_job_factor_indices,
            current_step,
            fs_params_ptr->shc_dim,
            dev_pointers_ptr->dev_blockRelationCounts, // Map
            ds_params_ptr->startIndex,
            // These pointers should be set in dev_pointers or passed as arguments
            (mpqs::sieve::DenseCandidate*)dev_pointers_ptr->dev_pp_accumulation_buffer,
            dev_pointers_ptr->dev_pp_counter,
            dev_pointers_ptr->pp_max_capacity
        );

        #ifdef SIEVING_DEBUG_FLAG
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	  fprintf(stdout, "cudaDeviceSynchronize failed after comCandidatesBatch kernel launch.\n");
	  fprintf(stdout, "Error: %s\n", cudaGetErrorString(cudaStatus));
	  exit(-1);
	}
        #endif

    }
}

} // namespace sieve
} // namespace mpqs
