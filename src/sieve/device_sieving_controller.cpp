// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "device_sieving_controller.h"
#include "common.h"
#include <bit>
#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <stdexcept>
#include <string>
#include "json_helper.h"

// CUDA error-check helper specific to the sieve module.
// Wraps any CUDA runtime call that returns cudaError_t. On failure,
// logs the call site and throws std::runtime_error so the orchestrator
// can surface a clean diagnostic instead of silently propagating bad pointers.
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

namespace mpqs {
namespace sieve {

DeviceSievingController::DeviceSievingController(int device, cudaStream_t stream)
    : device(device), stream(stream)
{
    getDeviceInfo(g_info, device);
    size_t max_shared_bytes = g_info.maxSharedMemPerBlock;
    cudaFuncSetAttribute((const void*)sieveAndScanKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
    cudaFuncSetAttribute((const void*)sieveAndScanBatchKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
}

DeviceSievingController::DeviceSievingController(int device)
    : device(device), stream(0)
{
    getDeviceInfo(g_info, device);
    size_t max_shared_bytes = g_info.maxSharedMemPerBlock;
    cudaFuncSetAttribute((const void*)sieveAndScanKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
    cudaFuncSetAttribute((const void*)sieveAndScanBatchKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
}

DeviceSievingController::~DeviceSievingController()
{
    // Proper cleanup of device pointers
    if (dev_pointers.dev_factorBase) cudaFree(dev_pointers.dev_factorBase);
    if (dev_pointers.dev_rootN) cudaFree(dev_pointers.dev_rootN);
    if (dev_pointers.dev_a_factors) cudaFree(dev_pointers.dev_a_factors);
    if (dev_pointers.dev_B_values) cudaFree(dev_pointers.dev_B_values);
    if (dev_pointers.dev_primeData) cudaFree(dev_pointers.dev_primeData);
    if (dev_pointers.dev_globalBucketEntries) cudaFree(dev_pointers.dev_globalBucketEntries);
    if (dev_pointers.dev_globalBucketCounts) cudaFree(dev_pointers.dev_globalBucketCounts);
    if (dev_pointers.dev_candidateRelations) cudaFree(dev_pointers.dev_candidateRelations);
    if (dev_pointers.dev_indexToCandidate) cudaFree(dev_pointers.dev_indexToCandidate);
    if (dev_pointers.dev_blockRelationCounts) cudaFree(dev_pointers.dev_blockRelationCounts);
    if (dev_pointers.dev_job_a_array) cudaFree(dev_pointers.dev_job_a_array);
    if (dev_pointers.dev_job_B_flat) cudaFree(dev_pointers.dev_job_B_flat);
    if (dev_pointers.dev_job_factor_indices) cudaFree(dev_pointers.dev_job_factor_indices);
    if (h_pinned_factor_indices_) { cudaFreeHost(h_pinned_factor_indices_); h_pinned_factor_indices_ = nullptr; }
}

void DeviceSievingController::initiate(factoringData& f_data)
{
    LOG_SET_MODULE("Sieve");
    this->f_data = f_data;
    fs_params.fb_size = (uint32_t)f_data.factorBase.size();
    fs_params.shc_dim = (uint32_t)f_data.a_factors.size();
    fs_params.M = f_data.M;
    // Safe intermediate calculation using long long
    fs_params.approxPolyRoot = (uint32_t)((((int64_t)f_data.M) * 10000) / 14142);
    fs_params.threshold = 31 - std::countl_zero((uint32_t)f_data.F);
    // Note: updateState() is NOT called here — it must run after loadData()
    // has allocated device memory (dev_a_factors, dev_B_values). The
    // orchestrator calls updateState() explicitly after loadData().
}

void DeviceSievingController::setThresholdOverride(uint64_t threshold_bound)
{
    if (threshold_bound > 0)
        fs_params.threshold = 63 - std::countl_zero(threshold_bound);
}

void DeviceSievingController::allocateBatchBuffers() {
    // 1. Batch Context Arrays
    size_t size_a = gs_conf.batch_size * sizeof(mpqs::uint512);
    size_t size_B = gs_conf.batch_size * fs_params.shc_dim * sizeof(mpqs::uint512);
    size_t size_idx = gs_conf.batch_size * fs_params.shc_dim * sizeof(uint32_t);

    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_job_a_array, size_a));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_job_B_flat, size_B));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_job_factor_indices, size_idx));

    // 2. Inter-Kernel Counters
    // This array maps 1-to-1 with the Grid Dimensions of the Sieve Kernel
    size_t size_counts = ss_conf.num_threadBlocks * sizeof(uint32_t);
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_blockRelationCounts, size_counts));
    // Initialize counters to 0 just in case
    SIEVE_CUDA_CHECK(cudaMemset(dev_pointers.dev_blockRelationCounts, 0, size_counts));

    // Pinned host buffer for truly async H2D factor index copies
    pinned_factor_indices_capacity_ = gs_conf.batch_size * fs_params.shc_dim;
    SIEVE_CUDA_CHECK(cudaMallocHost(&h_pinned_factor_indices_, pinned_factor_indices_capacity_ * sizeof(uint32_t)));

    // Logging for debug
    LOG(LOG_DEBUG_1) << "Batch Buffers Allocated. BatchSize=" << gs_conf.batch_size
                     << ", Mem=" << (size_a + size_B + size_idx + size_counts) / 1024 << " KB";
}

/**
 * @brief Pre-batch sieving handshake with postprocessor
 */
void DeviceSievingController::setPostProcessingLinks(mpqs::postprocessing::DoubleBuffer* active_buffer) {
    current_pp_buffer = active_buffer;
    dev_pointers.dev_pp_accumulation_buffer = active_buffer->d_data;
    dev_pointers.dev_pp_counter = active_buffer->d_counter;
    dev_pointers.pp_max_capacity = active_buffer->capacity;
}

/**
 * @brief Enqueues a DMA transfer of the Device batch counter to Pinned Host Memory.
 *
 * @details
 * Evaluates \mathcal{S}_{host} \leftarrow \mathcal{S}_{dev}. By recording this payload
 * onto the exact same `cudaStream_t` `stream` as the computational sieving kernels, we naturally
 * satisfy the Read-After-Write (RAW) dependency asynchronously.
 * The CPU orchestrator may thus poll `h_pinned_counter` infinitely in an active spin-loop
 * without incurring the latency cost of implicit `cudaDeviceSynchronize()` or blocking API calls.
 */
void DeviceSievingController::pushCounterToHostAsync(volatile uint32_t* h_pinned_counter) {
    if (current_pp_buffer && current_pp_buffer->d_counter && h_pinned_counter) {
        cudaMemcpyAsync((void*)h_pinned_counter,
                        current_pp_buffer->d_counter,
                        sizeof(uint32_t),
                        cudaMemcpyDeviceToHost,
                        stream);
    }
}

void DeviceSievingController::updateState()
{
    ds_params.a = f_data.a;
    ds_params.log2_a = ds_params.a.msb();
    ds_params.startIndex = -((int32_t)f_data.M);
    ds_params.subCube = 0; // Default to 0 for single step

    updateSievingData(dev_pointers, fs_params, f_data.a_factors, f_data.B_values);
}

void DeviceSievingController::advance_a(int step) {
    // we update factor data, a, B-values
    advance_a_factors(&f_data, step);
}

void DeviceSievingController::loadData(){
    loadSievingData(f_data.factorBase,
        f_data.rootN,
        f_data.a_factors,
        fs_params.shc_dim,
        gs_conf,
        ss_conf,
        dev_pointers);
}

void DeviceSievingController::clearCandidates() {
    size_t size = gs_conf.maxRelationsPerBlock * ss_conf.num_threadBlocks * sizeof(candidateRelation);
    cudaMemset(dev_pointers.dev_candidateRelations, 0, size);
    cudaDeviceSynchronize();
}

void DeviceSievingController::clearBuckets() {
    // Clear global buckets (per sieveStep for safety, not just once)
    const size_t totalBuckets =
        (size_t)gs_conf.num_polysPerSieveCall *
        (size_t)gs_conf.num_sievingBlocksPerSieveCall;
    const size_t totalBytes =
        totalBuckets * (size_t)gs_conf.globalBucketSize * sizeof(uint64_t);

    cudaMemset(dev_pointers.dev_globalBucketEntries, 0, totalBytes);
    cudaDeviceSynchronize();
}

#ifdef SIEVING_DEBUG_FLAG
void DeviceSievingController::sieveFullCubeSnapshot
(bool meta,
bool meta_P_enabled,
uint32_t meta_P,
bool meta_O_enabled,
uint32_t meta_O,
bool sas,
uint32_t step,
JSON_IO j_io){
    std::vector<int> prevRelCounts((size_t)ss_conf.num_threadBlocks * sizeof(uint32_t));
    ds_params.newCube = true;
    for(uint32_t sievingBlockBatch = 0; sievingBlockBatch < gs_conf.num_sievingBlockBatches; sievingBlockBatch++){
        for(uint32_t subCube = 0; subCube < gs_conf.num_subCubes; subCube++){
            ds_params.subCube = subCube;
            ds_params.startIndex = -((int32_t)f_data.M)+(int32_t)(sievingBlockBatch*gs_conf.num_sievingBlocksPerSieveCall*gs_conf.sievingBlockSize);
            initPrimeData(dev_pointers, init_conf, gs_conf, fs_params, ds_params, stream);
            globalMetaSieve(dev_pointers, fs_params, ds_params, gs_conf, gms_conf, stream);

            if(meta){
                LOG(LOG_DEBUG_1) << "=== Taking a snapshot of buckets ===";
                size_t entriesSize = (size_t)gs_conf.num_polysPerSieveCall * gs_conf.num_sievingBlocksPerSieveCall * gs_conf.globalBucketSize * sizeof(uint64_t);
                size_t countsSize = (size_t)gs_conf.num_polysPerSieveCall * gs_conf.num_sievingBlocksPerSieveCall * sizeof(uint32_t);

                uint64_t* h_bucketEntries = (uint64_t*)malloc(entriesSize);
                uint32_t* h_bucketCounts  = (uint32_t*)malloc(countsSize);

                cudaMemcpy(h_bucketEntries, dev_pointers.dev_globalBucketEntries, entriesSize, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_bucketCounts, dev_pointers.dev_globalBucketCounts, countsSize, cudaMemcpyDeviceToHost);

                JSONString dataArray(JSONString::Type::Array);
                for(uint32_t polyIdInSubcube = 0; polyIdInSubcube < gs_conf.num_polysPerSieveCall; polyIdInSubcube++){
                    for(uint32_t subInterval = 0; subInterval < gs_conf.num_sievingBlocksPerSieveCall; subInterval++){
                        uint64_t currentBucketId = polyIdInSubcube*gs_conf.num_sievingBlocksPerSieveCall+subInterval;
                        JSONString bucketData(JSONString::Type::Object);
                        bucketData.addNamedData("subInterval", std::to_string(subInterval));
                        bucketData.addNamedData("polyId", std::to_string(polyIdInSubcube));
                        bucketData.addNamedData("numEntries", std::to_string(h_bucketCounts[currentBucketId]));
                        JSONString entries(JSONString::Type::Array);
                        uint32_t jsonWrites = 0;
                        for(uint32_t index = 0; index < h_bucketCounts[currentBucketId]; index++){
                            uint64_t entry = h_bucketEntries[currentBucketId*gs_conf.globalBucketSize + index];
                            uint32_t p = entry >> 32;
                            uint32_t p_log = (entry >> 24) & 0xFF;
                            uint32_t localOffset = entry & ((1<<24)-1);
                            JSONString bucketEntry(JSONString::Type::Object);
                            bucketEntry.addNamedData("p_index", std::to_string(p));
                            bucketEntry.addNamedData("p_log",std::to_string(p_log));
                            bucketEntry.addNamedData("localOffset",std::to_string(localOffset));

                            bool valid = true;
                            if(meta_P_enabled && (p != meta_P)){
                                valid = false;
                            }
                            if(meta_O_enabled && (p != meta_O)){
                                valid = false;
                            }
                            if(valid && jsonWrites < 16){ //safety measure so we dont get huge json files
                                entries.appendData(bucketEntry.str());
                                jsonWrites++;
                            }
                        }
                        bucketData.addNamedData("entries", entries.str());
                        dataArray.appendData(bucketData.str());
                    }
                }
                JSONString fullSieveCall(JSONString::Type::Object);
                fullSieveCall.addNamedData("sieveCallID", std::to_string(step));
                fullSieveCall.addNamedData("data", dataArray.str());
                j_io.appendToFile("metaSieve.json", fullSieveCall);

                free(h_bucketEntries);
                free(h_bucketCounts);
            }

            sieveAndScan(dev_pointers, fs_params, ds_params, gs_conf, ss_conf, stream);
            if(sas){
                LOG(LOG_DEBUG_1) << "=== Taking a snapshot of candidate relations ===";
                size_t candidateRelationsSize = (size_t)gs_conf.maxRelationsPerBlock * ss_conf.num_threadBlocks * sizeof(candidateRelation);
                size_t relCountsSize = (size_t)ss_conf.num_threadBlocks * sizeof(uint32_t);

                candidateRelation* h_candidateRelations = (candidateRelation*)malloc(candidateRelationsSize);
                uint32_t* h_relCounts  = (uint32_t*)malloc(relCountsSize);

                cudaMemcpy(h_candidateRelations, dev_pointers.dev_candidateRelations, candidateRelationsSize, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_relCounts, dev_pointers.dev_blockRelationCounts, relCountsSize, cudaMemcpyDeviceToHost);
                for(uint32_t block = 0; block < ss_conf.num_threadBlocks; block ++){
                    for(uint32_t i = prevRelCounts[block]; i < h_relCounts[block]; i++){
                        candidateRelation rel = h_candidateRelations[block*gs_conf.maxRelationsPerBlock+i];
                        JSONString jsonRel(JSONString::Type::Object);
                        jsonRel.addNamedData("subCube", std::to_string(subCube));
                        jsonRel.addNamedData("sievingBlockBatch", std::to_string(sievingBlockBatch));

                        jsonRel.addNamedData("block", std::to_string(block));
                        jsonRel.addNamedData("id", std::to_string(i));
                        jsonRel.addNamedData("poly_id", std::to_string(rel.poly_id));
                        jsonRel.addNamedData("sieve_offset", std::to_string(rel.sieve_offset));
                        jsonRel.addNamedData("num_factors", std::to_string(rel.num_factors));
                        JSONString jsonFactors(JSONString::Type::Array);
                        for(uint32_t j = 0; j < rel.num_factors; j++){
                            jsonFactors.appendData(std::to_string(rel.factors[j]));
                        }
                        jsonRel.addNamedData("factors", jsonFactors.str());
                        j_io.appendToFile("sieveAndScan.json", jsonRel);

                        prevRelCounts[block] = h_relCounts[block];
                    }
                }

                free(h_candidateRelations);
                free(h_relCounts);
            }
            ds_params.newCube = false;
        }
    }
    if(meta){
        LOG(LOG_DEBUG_1) << "=== Bucket snapshot complete ===";
    }
    if(sas){
        LOG(LOG_DEBUG_1) << "=== Candidate relation snapshot complete ===";
    }
}
#endif

void DeviceSievingController::sieveFullCube()
{
    ds_params.newCube = true;
    for(uint32_t sievingBlockBatch = 0; sievingBlockBatch < gs_conf.num_sievingBlockBatches; sievingBlockBatch++){
        for(uint32_t subCube = 0; subCube < gs_conf.num_subCubes; subCube++){
            ds_params.subCube = subCube;
            ds_params.startIndex = -((int32_t)f_data.M)+(int32_t)(sievingBlockBatch*gs_conf.num_sievingBlocksPerSieveCall*gs_conf.sievingBlockSize);
            sieveStep();
        }
    }
}

void DeviceSievingController::sieveStep()
{
    initPrimeData(dev_pointers, init_conf, gs_conf, fs_params, ds_params, stream);
    globalMetaSieve(dev_pointers, fs_params, ds_params, gs_conf, gms_conf, stream);
    sieveAndScan(dev_pointers, fs_params, ds_params, gs_conf, ss_conf, stream);
    ds_params.newCube = false; //A sieving call was made locally, so the cube is not "new" anymore
}

float DeviceSievingController::sieveMini(uint32_t num_subcubes) {
    // 0 means full cube
    if (num_subcubes == 0) num_subcubes = gs_conf.num_subCubes;

    // Save state
    int32_t  saved_startIndex = ds_params.startIndex;
    uint32_t saved_subCube    = ds_params.subCube;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ds_params.newCube = true;
    cudaEventRecord(start, stream);

    for (uint32_t sc = 0; sc < num_subcubes; ++sc) {
        ds_params.subCube = sc;
        for (uint32_t batch = 0; batch < gs_conf.num_sievingBlockBatches; ++batch) {
            ds_params.startIndex = -((int32_t)f_data.M)
                + (int32_t)(batch * gs_conf.num_sievingBlocksPerSieveCall * gs_conf.sievingBlockSize);
            sieveStep();
        }
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Restore state
    ds_params.startIndex = saved_startIndex;
    ds_params.subCube    = saved_subCube;

    return ms * 1000.0f; // microseconds
}

float DeviceSievingController::evaluateConfig(const Params8& params,
                                               uint32_t num_subcubes,
                                               bool& reload_needed) {
    loadPartialCustomConfig(params[0], params[1], params[2], params[3],
                            params[4], params[5], params[6], params[7]);
    if (!validateConfigs()) return -1.0f;

    if (reload_needed) {
        loadSievingDataParamTest(f_data.factorBase,
                                 f_data.rootN,
                                 f_data.a_factors,
                                 fs_params.shc_dim,
                                 gs_conf, ss_conf, dev_pointers);
        reload_needed = false;
    }

    return sieveMini(num_subcubes);
}

DeviceSievingController::ParamTestResult
DeviceSievingController::runParamTest(factoringData& f_data){
    LOG_SET_SUBMODULE("PARAM_TEST");
    int32_t hyperCubeSize = 1 << (f_data.a_factors.size() - 1);
    int32_t M = f_data.M;
    int32_t sievingBlockSize = 1 << 15;

    uint32_t metaBlocks[3] = {32, 64, 128};//i
    uint32_t metaThreads[3] = {256, 512, 1024};//j
    uint32_t sasBlocks[3] = {32, 256, 512};//k
    uint32_t sasThreads[3] = {256, 512, 1024};//l
    int64_t best = 1ull << 32;
    Params8 bestConfig = {0,0,0,0,0,0,0,0};
    uint32_t configs_tested = 0;
    JSON_IO j_io;
    bool reloadneeded = true;
    for(uint32_t i = 0; i < 3; i++){
        for(uint32_t j = 0; j < 3; j++){
            for(uint32_t k = 0; k < 3; k++){
                for(uint32_t l = 0; l < 3; l++){
                    for(uint32_t subCubeSize = sasBlocks[k]; subCubeSize <= (uint32_t)hyperCubeSize; subCubeSize <<= 1){
                        for(uint32_t numIntervals = 4; numIntervals <= (uint32_t)2*M/sievingBlockSize; numIntervals <<= 1){
                            reloadneeded = true;
                            for(uint32_t polyBlockSize = 4; polyBlockSize <= (uint32_t)subCubeSize/metaBlocks[i]; polyBlockSize <<= 1){
                                for(uint32_t blocksPerCycle = 4; blocksPerCycle <= numIntervals; blocksPerCycle <<= 1){

                                    Params8 params = {subCubeSize, numIntervals, polyBlockSize,
                                                      blocksPerCycle, metaBlocks[i], metaThreads[j],
                                                      sasBlocks[k], sasThreads[l]};

                                    float timing_us = evaluateConfig(params, 0, reloadneeded);
                                    if (timing_us < 0.0f) continue; // invalid config

                                    int64_t elapsed_us = static_cast<int64_t>(timing_us);
                                    ++configs_tested;

                                    if(elapsed_us < best){
                                        best = elapsed_us;
                                        bestConfig = params;
                                    }
                                    if(elapsed_us == 0){
                                        LOG(LOG_ERROR_CRITICAL) << "ILLEGAL PARAMETERS ENCOUNTERED";
                                    }
                                    JSONString jsonSample(JSONString::Type::Object);
                                    jsonSample.addNamedData("subCubeSize", std::to_string(params[0]));
                                    jsonSample.addNamedData("numIntervals", std::to_string(params[1]));
                                    jsonSample.addNamedData("polyBlockSize", std::to_string(params[2]));
                                    jsonSample.addNamedData("blocksPerCycle", std::to_string(params[3]));

                                    jsonSample.addNamedData("metaGridDim", std::to_string(params[4]));
                                    jsonSample.addNamedData("metaBlockDim", std::to_string(params[5]));
                                    jsonSample.addNamedData("sasGridDim", std::to_string(params[6]));
                                    jsonSample.addNamedData("sasBlockDim", std::to_string(params[7]));

                                    jsonSample.addNamedData("microseconds", std::to_string(elapsed_us));
                                    j_io.appendToFile("paramTest.json", jsonSample);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::string config = "(";
    for(int i = 0; i < 8; i++){
        if(i > 0){
            config += ",";
        }
        config += std::to_string(bestConfig[i]);
    }
    config += ")";
    LOG(LOG_INFO) << "Test complete, best combination found: " << config;

    return ParamTestResult{bestConfig, static_cast<float>(best), configs_tested, "paramTest.json"};
}

// -----------------------
//      BATCH SIEVING
// -----------------------

void DeviceSievingController::prepareSievingBatch()
{
    // 1. Host Preparation
    // This function (from prime_algorithms.h) advances the internal f_data state
    // and returns the flattened vector of indices for the next batch_size steps.
    std::vector<uint32_t> next_factor_indices = mpqs::sieve::prepareNextBatchIndices(
								 &f_data, gs_conf.batch_size);

    // 2. Kernel launcher, handles data transfer of next_factor_indices to device
    mpqs::sieve::prepareSievingBatch(
	&dev_pointers,
	&next_factor_indices,
	fs_params.shc_dim,
	gs_conf.batch_size,
	stream,
	h_pinned_factor_indices_
    );
}

void DeviceSievingController::prepareSievingBatchFromStaged(
    const uint32_t* d_indices,
    mpqs::uint512* a_array_out,
    mpqs::uint512* B_flat_out
) {
    ::mpqs::sieve::prepareSievingBatchFromStaged(
        &dev_pointers, d_indices, a_array_out, B_flat_out,
        fs_params.shc_dim, init_conf.batch_size, stream);
}

void DeviceSievingController::setJobArrays(
    mpqs::uint512* a_array,
    mpqs::uint512* B_flat,
    const uint32_t* factor_indices
) {
    dev_pointers.dev_job_a_array = a_array;
    dev_pointers.dev_job_B_flat = B_flat;
    dev_pointers.dev_job_factor_indices = const_cast<uint32_t*>(factor_indices);
}

void DeviceSievingController::runSievingBatch(int num_steps, int start_batch_index)
{
    // External stop check (cluster mode) — returns without launching kernels
    if (external_stop_ && external_stop_->load(std::memory_order_relaxed)) return;

    mpqs::sieve::runSievingBatch(
        &dev_pointers,
	&init_conf,
	&gs_conf,
	&fs_params,
	&ds_params,
	&gms_conf,
	&ss_conf,
	num_steps,
	start_batch_index,
	stream
    );
}


// ============================================================================
// Cluster: Snapshot + Reset-and-Advance (Spec Section 11.1)
// ============================================================================

void DeviceSievingController::saveSnapshot() {
    snapshot_.a_factors = f_data.a_factors;
    snapshot_.lowerHalfStart = f_data.lowerHalfStart;
    snapshot_.upperHalfStart = f_data.upperHalfStart;
    snapshot_.shc_dim = static_cast<uint32_t>(f_data.a_factors.size());
}

void DeviceSievingController::resetAndAdvanceTo(uint64_t global_a_index) {
    const uint32_t d = snapshot_.shc_dim;
    if (d == 0) return;  // No snapshot saved
    const uint32_t H = 1u << d;
    const uint64_t target_hc = global_a_index / H;
    const uint32_t local_offset = static_cast<uint32_t>(global_a_index % H);

    // 1. Restore initial canonical state from snapshot
    f_data.a_factors = snapshot_.a_factors;
    f_data.lowerHalfStart = snapshot_.lowerHalfStart;
    f_data.upperHalfStart = snapshot_.upperHalfStart;
    f_data.current_a_index = 0;
    f_data.max_a_index = H;

    // 2. Simulate window slides to reach target hypercube.
    //    advance_a_factors(&f_data, H) triggers Case 2 (0 + H >= H):
    //    resets odd parities, calls advance() on a_factors to slide the
    //    window, resets current_a_index = 0, calls recalc_a().
    //    Note: upperHalfStart/lowerHalfStart are NOT modified by
    //    advance_a_factors — they serve as fixed pivot bases.
    for (uint64_t h = 0; h < target_hc; h++) {
        advance_a_factors(&f_data, static_cast<int>(H));
    }

    // 3. Simulate Gray code steps to reach local offset within hypercube.
    //    advance_a_factors(&f_data, local_offset) triggers Case 1
    //    (0 + offset < H): applies Gray code steps, calls recalc_a().
    if (local_offset > 0) {
        advance_a_factors(&f_data, static_cast<int>(local_offset));
    }

    // 4. Upload new polynomial data to GPU
    //    (recalc_a was already called by advance_a_factors)
    updateState();
}


// ============================================================================
// Validation Logic
// ============================================================================

int DeviceSievingController::validateResults(factoringData& f_data) {
    LOG_SET_SUBMODULE("Validation");
    // 1. Copy candidates back to host
    size_t num_candidates = gs_conf.maxRelationsPerBlock * ss_conf.num_threadBlocks;
    std::vector<candidateRelation> host_candidates(num_candidates);

    cudaMemcpy(host_candidates.data(), dev_pointers.dev_candidateRelations,
               num_candidates * sizeof(candidateRelation), cudaMemcpyDeviceToHost);

    // 2. Statistics Containers
    int total_candidates = 0;
    int b_mismatch_count = 0;

    int valid_gpu_claims = 0;   // GPU factors actually divide Q
    int valid_full_relations = 0; // After dividing factors and 'a', rem == 1
    int valid_partial_relations = 0; // Remainder != 1 but valid GPU claims

    // Failure Categories
    int err_ghost_factors = 0; // GPU claimed factor, but Q % p != 0
    // int err_wrong_Q = 0;       // No factors worked (likely x or b is wrong)

    std::map<uint32_t, int> poly_stats;
    int max_print_errors = 10; // Prevent console flood

    LOG(LOG_DEBUG_1) << "=== Starting Deep Validation ===";

    // 3. Iterate and Verify
    for (const auto& cand : host_candidates) {
        // Skip empty slots.
        // Note: With atomic increment fix, num_factors should be accurate.
        // If num_factors is 0 but poly_id != 0, we should look at it (potential sieving bug).
        if (cand.num_factors == 0) {
            // Optional: Check if it looks like a "lost" candidate
            // if (cand.poly_id != 0) std::cout << "Warning: PolyID " << cand.poly_id << " has 0 factors.\n";
            continue;
        }

        total_candidates++;
        poly_stats[cand.poly_id]++;
        // bool print_this_error = ((err_ghost_factors + err_wrong_Q) < max_print_errors); // helper concept

        // --- Step 1: Reconstruct 'b' on Host for Consistency Check ---
        mpqs::uint512 b_recalc((uint32_t)0);
        uint32_t id = cand.poly_id;
        for (uint32_t i = 0; i < fs_params.shc_dim; i++) {
            if (id & 1) b_recalc.add(f_data.B_values[i]);
            else        b_recalc.sub(f_data.B_values[i]);
            id >>= 1;
        }

        bool b_match = (cand.b == b_recalc);
        if (!b_match) b_mismatch_count++;

        // --- Step 2: Calculate Q(x) using CANDIDATE's b (Requirement 1) ---
        // x = startIndex + sieve_offset
        // Note: Ensure sieve_offset was written correctly by the kernel!
        int32_t x = cand.sieve_offset;

        mpqs::uint512 ax = f_data.a;
        uint32_t abs_x = (x < 0) ? -x : x;
        ax.mult_uint32(abs_x);

        mpqs::uint512 ax_plus_b;
        if (x >= 0) {
            ax_plus_b = ax;
            ax_plus_b.add(cand.b);
        } else {
            if (cand.b >= ax) {
                ax_plus_b = cand.b;
                ax_plus_b.sub(ax);
            } else {
                ax_plus_b = ax;
                ax_plus_b.sub(cand.b);
            }
        }

        mpqs::uint512 Q = ax_plus_b;
        Q.mult(ax_plus_b); // Square

        if (Q < f_data.N) {
            mpqs::uint512 tmp = f_data.N;
            tmp.sub(Q);
            Q = tmp;
        } else {
            Q.sub(f_data.N);
        }

        // Save original Q for printing
        // mpqs::uint512 original_Q = Q;

        // --- Step 3: Verify GPU Factors ---
        bool gpu_lied = false;
        int factors_confirmed = 0;

        for (uint32_t i = 0; i < cand.num_factors; i++) {
            uint32_t p_idx = cand.factors[i];
            if (p_idx >= f_data.factorBase.size()) {
                 gpu_lied = true; break;
            }
            uint32_t p = f_data.factorBase[p_idx];

            if (Q.mod_uint32(p) != 0) {
                gpu_lied = true;
                if (err_ghost_factors < max_print_errors) {
		    LOG(LOG_ERROR_CRITICAL) << "[ERR] Ghost Factor: p=" << p << " does not divide Q.";
                    LOG(LOG_ERROR_CRITICAL) << "    PolyID=" << cand.poly_id << " Offset=" << cand.sieve_offset;
                }
                // Break or continue? Let's break, Q is wrong or factor is wrong.
                break;
            } else {
                // Divide out fully
                while (Q.mod_uint32(p) == 0) {
                    Q.div_uint32_inplace(p);
                }
                factors_confirmed++;
            }
        }

        if (gpu_lied) {
            err_ghost_factors++;
            continue; // Stop processing this candidate
        } else {
            valid_gpu_claims++;
        }

        // --- Step 4: Handle 'a' factors (Requirement 3) ---
        // The GPU skips primes in 'a'. So Q usually still contains 'a'.
        // Divide Q by 'a' (or factors of a)

        mpqs::uint512 Q_after_gpu = Q;

        // Try dividing by 'a' directly
        mpqs::uint512 rem_a;
        Q.div_mod_core(f_data.a, &rem_a);

        // bool divisible_by_a = false;

        if (rem_a.is_zero()) {
             Q = Q; // Q is now Q/a (from div_mod_core: *this becomes quotient)
             // divisible_by_a = true;
        } else {
            // Restore Q and try individual factors of a (if a is large)
            Q = Q_after_gpu;
            // (Optional: loop over f_data.a_factors and divide)
            // For now, let's assume if it's not divisible by 'a', it's a partial
        }

        // --- Step 5: Final Classification ---
        if (Q.is_one()) {
            valid_full_relations++;
        } else {
            // It is a valid sieve output, but leaves a remainder (Large Prime?)
            valid_partial_relations++;

            if (valid_partial_relations <= 3) {
	         LOG(LOG_DEBUG_1) << "Partial Relation found. Remainder: " << Q.to_string();
            }
        }
    }

    // --- Reporting ---
    LOG(LOG_DEBUG_1) << "=== Validation Summary ===";
    LOG(LOG_DEBUG_1) << "Total Candidates: " << total_candidates;
    LOG(LOG_DEBUG_1) << "B Mismatches:     " << b_mismatch_count << (b_mismatch_count > 0 ? " [CRITICAL]" : " [OK]");
    LOG(LOG_DEBUG_1) << "--------------------------";
    LOG(LOG_DEBUG_1) << "Valid GPU Claims: " << valid_gpu_claims << " (Math check passed)";
    LOG(LOG_DEBUG_1) << "  - Full Relations: " << valid_full_relations;
    LOG(LOG_DEBUG_1) << "  - Part Relations: " << valid_partial_relations;
    LOG(LOG_DEBUG_1) << "--------------------------";
    LOG(LOG_DEBUG_1) << "Failures:";
    LOG(LOG_DEBUG_1) << "  - Ghost Factors:  " << err_ghost_factors << " (Calculated Q is wrong or Factor is wrong)";

    // Helper lambda for error counting
    // auto get_err_count = [&](){ return err_ghost_factors; };

    return valid_full_relations;
}

void DeviceSievingController::loadStandardConfig()
{
    auto pow2leq = [](uint32_t x) -> uint32_t {
        if (x < 1) return 0;
        return 1 << (31 - std::countl_zero(x));
    };
    auto pow2geq = [pow2leq](uint32_t x) -> uint32_t {
        uint32_t pow2 = pow2leq(x);
        return (pow2 == x) ? pow2 : pow2 << 1; // * 2
    };
    auto log2 = [](uint32_t x) -> uint32_t {
        if (x <= 0) return 0; // Fix: undefined for 0 usually
        return 31 - std::countl_zero(x);
    };

    /* initConfig */
    init_conf.num_threadsPerBlock = 256;
    init_conf.num_threadBlocks = 2*pow2geq(g_info.multiProcessorCount);
    init_conf.batch_size = 0; // this may be changed a posteriori via setSievingBatchSize(uint32_t batch_size)

    /* generalSievingConfig */
    gs_conf.sievingBlockSize = pow2leq((3*g_info.maxSharedMemPerBlock)/4);
    gs_conf.log2_sievingBlockSize = log2(gs_conf.sievingBlockSize);

    gs_conf.num_polysPerSieveCall = std::min(32768u,(1u << fs_params.shc_dim)/2); // Explicit unsigned
    gs_conf.globalBucketSize = gs_conf.sievingBlockSize/2;

    gs_conf.bigPrimeStartIndex = 1024;
    gs_conf.midPrimeStartIndex = 32;
    gs_conf.maxRelationsPerBlock = 64;

    /* globalMetaSieveConfig */
    gms_conf.num_threadsPerBlock = 256;
    gms_conf.num_threadBlocks = 64;

    gms_conf.maxActiveBucketsTotal = 2 << 15; //first, we choose how many active buckets per thread block we want

    gs_conf.num_sievingBlocksPerSieveCall = (2*fs_params.M)/gs_conf.sievingBlockSize; //next, we choose the sievingBlocksPerCall to cover the whole sieving interval
    //reduce polys sieved until everything fits into global memory
    while(gs_conf.num_polysPerSieveCall*gs_conf.num_sievingBlocksPerSieveCall*gs_conf.globalBucketSize*sizeof(uint64_t) > g_info.totalGlobalMem){
        gs_conf.num_polysPerSieveCall >>= 1;
    }
    gs_conf.num_subCubes = std::min(32768u,(1u << fs_params.shc_dim)/2)/gs_conf.num_polysPerSieveCall;
    //set activeBlocks to the max (that is how many subintervals are considered at once)
    gms_conf.num_activeBlocksPerCycle = 1 << std::countr_zero(gs_conf.num_sievingBlocksPerSieveCall);
    gs_conf.num_sievingBlockBatches = (2*fs_params.M)/(gs_conf.sievingBlockSize*gs_conf.num_sievingBlocksPerSieveCall);
    //incrase polyBlockSize until we run out of poly or we have too many active buckets
    gms_conf.polyBlockSize = 1;
    while(gms_conf.polyBlockSize*gms_conf.num_activeBlocksPerCycle*gms_conf.num_threadBlocks*2 <= gms_conf.maxActiveBucketsTotal
            && gms_conf.polyBlockSize*gms_conf.num_threadBlocks*2 <= gs_conf.num_polysPerSieveCall){
        gms_conf.polyBlockSize <<= 1;
    }
    //calculations that automatically follow the above definitions
    gms_conf.log2_polyBlockSize = log2(gms_conf.polyBlockSize);
    gms_conf.num_polyBlocksPerThreadBlock = (gs_conf.num_polysPerSieveCall/gms_conf.num_threadBlocks)/gms_conf.polyBlockSize;
    gms_conf.log2_num_polyBlocksPerThreadBlock = log2(gms_conf.num_polyBlocksPerThreadBlock);
    gms_conf.num_metaSieveCycles = gs_conf.num_sievingBlocksPerSieveCall/gms_conf.num_activeBlocksPerCycle;
    gms_conf.num_activeBucketsPerThreadBlock = gms_conf.num_activeBlocksPerCycle*gms_conf.polyBlockSize;
    gms_conf.sharedMemReq = gms_conf.num_activeBucketsPerThreadBlock * sizeof(int);

    /* sieveAndScanConfig */
    ss_conf.num_threadsPerBlock = 256;
    ss_conf.num_threadBlocks = std::min(256u, gs_conf.num_polysPerSieveCall);
    ss_conf.sharedMemReq = gs_conf.sievingBlockSize * sizeof(uint8_t) + 3 * gs_conf.bigPrimeStartIndex * sizeof(int);

    /* processRelationsConfig */
    pr_conf.num_threadsPerBlock = 256;
    pr_conf.num_threadBlocks = pow2geq(g_info.multiProcessorCount);
    //loadPartialCustomConfig(512,32,8,32,32,256,512,1024);//RSA100
    //loadPartialCustomConfig(512,16,8,16,32,256,512,1024);//BASE
}

void DeviceSievingController::loadPartialCustomConfig(uint32_t totalPolys, uint32_t totalIntervals, uint32_t polyBlockSize, uint32_t blocksPerCycle, uint32_t metaB, uint32_t metaT, uint32_t sasB, uint32_t sasT)
{
    auto pow2leq = [](uint32_t x) -> uint32_t {
        if (x < 1) return 0;
        return 1 << (31 - std::countl_zero(x));
    };
    auto pow2geq = [pow2leq](uint32_t x) -> uint32_t {
        uint32_t pow2 = pow2leq(x);
        return (pow2 == x) ? pow2 : pow2 << 1; // * 2
    };
    auto log2 = [](uint32_t x) -> uint32_t {
        if (x <= 0) return 0; // Fix: undefined for 0 usually
        return 31 - std::countl_zero(x);
    };

    /* initConfig */
    init_conf.num_threadsPerBlock = 512;
    init_conf.num_threadBlocks = 2*pow2geq(g_info.multiProcessorCount);
    init_conf.batch_size = 0; // this may be changed a posteriori via setSievingBatchSize(uint32_t batch_size)

    /* generalSievingConfig */
    gs_conf.sievingBlockSize = std::min(fs_params.M, pow2leq((3*g_info.maxSharedMemPerBlock)/4));
    gs_conf.log2_sievingBlockSize = log2(gs_conf.sievingBlockSize);

    gs_conf.num_polysPerSieveCall = totalPolys; // Explicit unsigned
    gs_conf.globalBucketSize = gs_conf.sievingBlockSize/2;

    gs_conf.bigPrimeStartIndex = gs_conf.sievingBlockSize/32;
    gs_conf.midPrimeStartIndex = 32;
    gs_conf.maxRelationsPerBlock = 64;

    /* globalMetaSieveConfig */
    gms_conf.num_threadsPerBlock = metaT;
    gms_conf.num_threadBlocks = metaB;

    gms_conf.maxActiveBucketsTotal = 2 << 15; //first, we choose how many active buckets per thread block we want

    gs_conf.num_sievingBlocksPerSieveCall = totalIntervals; //next, we choose the sievingBlocksPerCall to cover the whole sieving interval
    gs_conf.num_subCubes = std::min(32768u,(1u << fs_params.shc_dim)/2)/gs_conf.num_polysPerSieveCall;
    //set activeBlocks to the max (that is how many subintervals are considered at once)
    gms_conf.num_activeBlocksPerCycle = blocksPerCycle;
    gs_conf.num_sievingBlockBatches = (2*fs_params.M)/(gs_conf.sievingBlockSize*gs_conf.num_sievingBlocksPerSieveCall);
    // Invariant: the sieve covers [-M,M) in num_sievingBlockBatches batches and must run >= 1,
    // otherwise the batch loops (sieveDataBatch / runSievingBatch / runSievingLegacy) execute
    // zero iterations, the sieve does no work, and the pipeline yields no relations. The integer
    // division above floors to 0 when sievingBlockSize*num_sievingBlocksPerSieveCall > 2*M.
    // Unlike loadStandardConfig() — which itself derives num_sievingBlocksPerSieveCall =
    // (2*M)/sievingBlockSize, keeping the quotient >= 1 — this function takes totalIntervals
    // (= num_sievingBlocksPerSieveCall) as an external parameter. Full autotune sweeps M down
    // while holding the kernel params (hence totalIntervals) fixed, so a small enough M makes
    // sievingBlockSize = min(M, ...) == M and the quotient floor to 0. One batch with this
    // geometry over-covers [-M,M) (totalIntervals*sievingBlockSize >= 2*M in that regime), which
    // is the correct conservative behavior. This clamp is a no-op for every M used by
    // --autotune_stage1 / no-autotune / loadStandardConfig (their M keeps sievingBlockSize < M
    // with 2*M >= sievingBlockSize*totalIntervals), so those modes are behaviorally unchanged.
    if (gs_conf.num_sievingBlockBatches == 0) gs_conf.num_sievingBlockBatches = 1;
    //incrase polyBlockSize until we run out of poly or we have too many active buckets
    gms_conf.polyBlockSize = polyBlockSize;
    //calculations that automatically follow the above definitions
    gms_conf.log2_polyBlockSize = log2(gms_conf.polyBlockSize);
    gms_conf.num_polyBlocksPerThreadBlock = (gs_conf.num_polysPerSieveCall/gms_conf.num_threadBlocks)/gms_conf.polyBlockSize;
    gms_conf.log2_num_polyBlocksPerThreadBlock = log2(gms_conf.num_polyBlocksPerThreadBlock);
    gms_conf.num_metaSieveCycles = gs_conf.num_sievingBlocksPerSieveCall/gms_conf.num_activeBlocksPerCycle;
    gms_conf.num_activeBucketsPerThreadBlock = gms_conf.num_activeBlocksPerCycle*gms_conf.polyBlockSize;
    gms_conf.sharedMemReq = gms_conf.num_activeBucketsPerThreadBlock * sizeof(int);

    /* sieveAndScanConfig */
    ss_conf.num_threadsPerBlock = sasT;
    ss_conf.num_threadBlocks = sasB;
    ss_conf.sharedMemReq = gs_conf.sievingBlockSize * sizeof(uint8_t) + 3 * gs_conf.bigPrimeStartIndex * sizeof(int);

    /* processRelationsConfig */
    pr_conf.num_threadsPerBlock = 512;
    pr_conf.num_threadBlocks = pow2geq(g_info.multiProcessorCount);
}

void DeviceSievingController::printConfigs() {
    LOG_SET_SUBMODULE("Config");
    LOG(LOG_DEBUG_1) << "============================================================";
    LOG(LOG_DEBUG_1) << "initConfig:";
    LOG(LOG_DEBUG_1) << "threads: " << init_conf.num_threadsPerBlock;
    LOG(LOG_DEBUG_1) << "blocks: " << init_conf.num_threadBlocks;
    LOG(LOG_DEBUG_1) << "batch size: " << init_conf.batch_size;
    LOG(LOG_DEBUG_1) << "============================================================";
    LOG(LOG_DEBUG_1) << "generalSievingConfig:";
    LOG(LOG_DEBUG_1) << "sievingBlockSize: " << gs_conf.sievingBlockSize;
    LOG(LOG_DEBUG_1) << "globalBucketSize: " << gs_conf.globalBucketSize;
    LOG(LOG_DEBUG_1) << "polys sieved per call: " << gs_conf.num_polysPerSieveCall;
    LOG(LOG_DEBUG_1) << "sievingBlocks sieved per call: " << gs_conf.num_sievingBlocksPerSieveCall;
    LOG(LOG_DEBUG_1) << "bigPrimeStart: " << gs_conf.bigPrimeStartIndex;
    LOG(LOG_DEBUG_1) << "midPrimeStart: " << gs_conf.midPrimeStartIndex;
    LOG(LOG_DEBUG_1) << "maxRelationsPerBlock: " << gs_conf.maxRelationsPerBlock;
    LOG(LOG_DEBUG_1) << "============================================================";
    LOG(LOG_DEBUG_1) << "globalMetaSieveConfig:";
    LOG(LOG_DEBUG_1) << "threads: " << gms_conf.num_threadsPerBlock;
    LOG(LOG_DEBUG_1) << "blocks: " << gms_conf.num_threadBlocks;
    LOG(LOG_DEBUG_1) << "num activeblocks: " << gms_conf.num_activeBlocksPerCycle;
    LOG(LOG_DEBUG_1) << "shared memory required: " << gms_conf.sharedMemReq << " bytes";
    LOG(LOG_DEBUG_1) << "============================================================";
    LOG(LOG_DEBUG_1) << "sieveAndScanConfig:";
    LOG(LOG_DEBUG_1) << "threads: " << ss_conf.num_threadsPerBlock;
    LOG(LOG_DEBUG_1) << "blocks: " << ss_conf.num_threadBlocks;
    LOG(LOG_DEBUG_1) << "shared memory required: " << ss_conf.sharedMemReq << " bytes";
    LOG(LOG_DEBUG_1) << "============================================================";
}
void DeviceSievingController::printConfigsDEBUG() {
    std::cout << "============================================================" << std::endl;
    std::cout << "initConfig:" << std::endl;
    std::cout << "threads: " << init_conf.num_threadsPerBlock << std::endl;
    std::cout << "blocks: " << init_conf.num_threadBlocks << std::endl;
    std::cout << "batch size: " << init_conf.batch_size << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "generalSievingConfig:" << std::endl;
    std::cout << "sievingBlockSize: " << gs_conf.sievingBlockSize << std::endl;
    std::cout << "globalBucketSize: " << gs_conf.globalBucketSize << std::endl;
    std::cout << "polys sieved per call: " << gs_conf.num_polysPerSieveCall << std::endl;
    std::cout << "sievingBlocks sieved per call: " << gs_conf.num_sievingBlocksPerSieveCall << std::endl;
    std::cout << "bigPrimeStart: " << gs_conf.bigPrimeStartIndex << std::endl;
    std::cout << "midPrimeStart: " << gs_conf.midPrimeStartIndex << std::endl;
    std::cout << "maxRelationsPerBlock: " << gs_conf.maxRelationsPerBlock << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "globalMetaSieveConfig:" << std::endl;
    std::cout << "threads: " << gms_conf.num_threadsPerBlock << std::endl;
    std::cout << "blocks: " << gms_conf.num_threadBlocks << std::endl;
    std::cout << "num activeblocks: " << gms_conf.num_activeBlocksPerCycle << std::endl;
    std::cout << "shared memory required: "<< gms_conf.sharedMemReq << " bytes" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "sieveAndScanConfig:" << std::endl;
    std::cout << "threads: " << ss_conf.num_threadsPerBlock << std::endl;
    std::cout << "blocks: " << ss_conf.num_threadBlocks << std::endl;
    std::cout << "shared memory required: " << ss_conf.sharedMemReq << " bytes" << std::endl;
    std::cout << "============================================================" << std::endl;
}

void DeviceSievingController::printCustomConfigs(){
    LOG(LOG_DEBUG_1) << "============================================================";
    LOG(LOG_DEBUG_1) << "custom Configs:";
    LOG(LOG_DEBUG_1) << "subcube size: " << gs_conf.num_polysPerSieveCall;
    LOG(LOG_DEBUG_1) << "meta-sieve: blocks: " << gs_conf.num_sievingBlocksPerSieveCall;
    LOG(LOG_DEBUG_1) << "meta-sieve: blocks: " << gms_conf.polyBlockSize;
    LOG(LOG_DEBUG_1) << "meta-sieve: blocks: " << gms_conf.num_activeBlocksPerCycle;
    LOG(LOG_DEBUG_1) << "meta-sieve: blocks: " << gms_conf.num_threadBlocks;
    LOG(LOG_DEBUG_1) << "meta-sieve: threads: " << gms_conf.num_threadsPerBlock;
    LOG(LOG_DEBUG_1) << "sieve and scan: blocks: " << ss_conf.num_threadBlocks;
    LOG(LOG_DEBUG_1) << "sieve and scan: threads: " << ss_conf.num_threadsPerBlock;
    LOG(LOG_DEBUG_1) << "============================================================";
}

bool DeviceSievingController::validateConfigs() {
    auto isPowerOfTwo = [](uint32_t n) {
        return n > 0 && (n & (n - 1)) == 0;
    };

    bool validFlag = true;

    #define POW2_CHECK(var, validFlag) \
    do { \
        if (!isPowerOfTwo(var)) { \
            validFlag = false; \
            LOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #var << " must be a power of 2. " << "Current value: " << (var) << std::endl; \
        } \
    } while (0)

    #define EQUAL_CHECK(var1, var2, validFlag) \
    do { \
        if ( (var1) != (var2) ) { \
            validFlag = false; \
            LOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #var1 << " must be equal to " << #var2 << ". Current values: " << (var1) << ", " << (var2) << std::endl; \
        } \
    } while (0)

    #define LEQ_CHECK(var1, var2, validFlag) \
    do { \
        if (!( (var1) <= (var2) )) { \
            validFlag = false; \
            LOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #var1 << " must be less than or equal to " << #var2 << ". Current values: " << (var1) << ", " << (var2) << std::endl; \
        } \
    } while (0)

    #define NON0_CHECK(var, validFlag) \
    do { \
    if ((var) == 0) { \
        validFlag = false; \
        LOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #var << " must be > 0." << std::endl; \
    } \
    } while (0)


    POW2_CHECK(gs_conf.sievingBlockSize, validFlag);
    POW2_CHECK(gs_conf.num_polysPerSieveCall, validFlag);
    POW2_CHECK(gms_conf.polyBlockSize, validFlag);
    POW2_CHECK(gms_conf.num_polyBlocksPerThreadBlock, validFlag);
    POW2_CHECK(gms_conf.num_activeBlocksPerCycle, validFlag);
    //grid/block dims
    POW2_CHECK(gms_conf.num_threadsPerBlock, validFlag);
    POW2_CHECK(ss_conf.num_threadsPerBlock, validFlag);
    POW2_CHECK(gms_conf.num_threadBlocks, validFlag);
    POW2_CHECK(ss_conf.num_threadBlocks, validFlag);

    //equations:
    EQUAL_CHECK(gms_conf.num_polyBlocksPerThreadBlock*gms_conf.polyBlockSize*gms_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall, validFlag);
    EQUAL_CHECK(gms_conf.sharedMemReq, (gms_conf.num_activeBucketsPerThreadBlock * sizeof(int)), validFlag);
    EQUAL_CHECK(ss_conf.sharedMemReq, (gs_conf.sievingBlockSize * sizeof(uint8_t) + 3 * gs_conf.bigPrimeStartIndex * sizeof(int)), validFlag);
    EQUAL_CHECK(gms_conf.num_activeBucketsPerThreadBlock, gms_conf.num_activeBlocksPerCycle * gms_conf.polyBlockSize,validFlag);
    EQUAL_CHECK(gms_conf.num_metaSieveCycles*gms_conf.num_activeBlocksPerCycle, gs_conf.num_sievingBlocksPerSieveCall, validFlag);

    //space/subdivision requirements
    LEQ_CHECK(gms_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall, validFlag);
    LEQ_CHECK(gms_conf.num_threadBlocks*gms_conf.polyBlockSize, gs_conf.num_polysPerSieveCall, validFlag);
    LEQ_CHECK(gs_conf.num_polysPerSieveCall, ((1u << (fs_params.shc_dim - 1))), validFlag);
    LEQ_CHECK(gms_conf.num_activeBlocksPerCycle, gs_conf.num_sievingBlocksPerSieveCall, validFlag);

    //memcheck
    LEQ_CHECK(gs_conf.num_polysPerSieveCall*gs_conf.num_sievingBlocksPerSieveCall*gs_conf.globalBucketSize*sizeof(uint64_t), 3*g_info.totalGlobalMem/4, validFlag);//keep a buffer
    LEQ_CHECK(gms_conf.sharedMemReq, g_info.maxSharedMemPerBlock, validFlag);
    LEQ_CHECK(ss_conf.sharedMemReq, g_info.maxSharedMemPerBlock, validFlag);

    NON0_CHECK(gs_conf.num_sievingBlockBatches, validFlag);
    NON0_CHECK(gs_conf.num_subCubes, validFlag);
    NON0_CHECK(gms_conf.num_polyBlocksPerThreadBlock, validFlag);
    NON0_CHECK(gms_conf.num_metaSieveCycles, validFlag);

    #undef POW2_CHECK
    #undef EQUAL_CHECK
    #undef LEQ_CHECK
    #undef NON0_CHECK

    if(!validFlag){
        LOG(LOG_ERROR_CRITICAL) << "============================================================";
        LOG(LOG_ERROR_CRITICAL) << "initConfig:";
        LOG(LOG_ERROR_CRITICAL) << "threads: " << init_conf.num_threadsPerBlock;
        LOG(LOG_ERROR_CRITICAL) << "blocks: " << init_conf.num_threadBlocks;
        LOG(LOG_ERROR_CRITICAL) << "batch size: " << init_conf.batch_size;
        LOG(LOG_ERROR_CRITICAL) << "============================================================";
        LOG(LOG_ERROR_CRITICAL) << "generalSievingConfig:";
        LOG(LOG_ERROR_CRITICAL) << "sievingBlockSize: " << gs_conf.sievingBlockSize;
        LOG(LOG_ERROR_CRITICAL) << "globalBucketSize: " << gs_conf.globalBucketSize;
        LOG(LOG_ERROR_CRITICAL) << "polys sieved per call: " << gs_conf.num_polysPerSieveCall;
        LOG(LOG_ERROR_CRITICAL) << "sievingBlocks sieved per call: " << gs_conf.num_sievingBlocksPerSieveCall;
        LOG(LOG_ERROR_CRITICAL) << "bigPrimeStart: " << gs_conf.bigPrimeStartIndex;
        LOG(LOG_ERROR_CRITICAL) << "midPrimeStart: " << gs_conf.midPrimeStartIndex;
        LOG(LOG_ERROR_CRITICAL) << "maxRelationsPerBlock: " << gs_conf.maxRelationsPerBlock;
        LOG(LOG_ERROR_CRITICAL) << "============================================================";
        LOG(LOG_ERROR_CRITICAL) << "globalMetaSieveConfig:";
        LOG(LOG_ERROR_CRITICAL) << "threads: " << gms_conf.num_threadsPerBlock;
        LOG(LOG_ERROR_CRITICAL) << "blocks: " << gms_conf.num_threadBlocks;
        LOG(LOG_ERROR_CRITICAL) << "num activeblocks: " << gms_conf.num_activeBlocksPerCycle;
        LOG(LOG_ERROR_CRITICAL) << "shared memory required: " << gms_conf.sharedMemReq << " bytes";
        LOG(LOG_ERROR_CRITICAL) << "============================================================";
        LOG(LOG_ERROR_CRITICAL) << "sieveAndScanConfig:";
        LOG(LOG_ERROR_CRITICAL) << "threads: " << ss_conf.num_threadsPerBlock;
        LOG(LOG_ERROR_CRITICAL) << "blocks: " << ss_conf.num_threadBlocks;
        LOG(LOG_ERROR_CRITICAL) << "shared memory required: " << ss_conf.sharedMemReq << " bytes";
        LOG(LOG_ERROR_CRITICAL) << "============================================================";
    }
    return validFlag;
}

} // namespace sieve
} // namespace mpqs
