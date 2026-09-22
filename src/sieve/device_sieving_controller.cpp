// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "device_sieving_controller.h"
#include "common.h"
#include "sieve_memory_model.h"   // single source-of-truth sieve memory model
#include <bit>
#include <algorithm>
#include <chrono>   // wide-probe wall-clock measurement window
#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <stdexcept>
#include <string>
#include "json_helper.h"
#include "dynamicMask.cuh"

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
    // RSA-155 dual-path: opt-in max dynamic shared for the wide (uint16) batch kernel.
    // Must run in the ctor (before any cuda-graph stream capture) — the wide launch relies on it.
    cudaFuncSetAttribute((const void*)sieveAndScanBatchKernelWide, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
    // Same opt-in for the saturating-uint8 wide kernel (larger SB => larger smem).
    cudaFuncSetAttribute((const void*)sieveAndScanBatchKernelWideU8Sat, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
}

DeviceSievingController::DeviceSievingController(int device)
    : device(device), stream(0)
{
    getDeviceInfo(g_info, device);
    size_t max_shared_bytes = g_info.maxSharedMemPerBlock;
    cudaFuncSetAttribute((const void*)sieveAndScanKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
    cudaFuncSetAttribute((const void*)sieveAndScanBatchKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
    // RSA-155 dual-path: opt-in max dynamic shared for the wide (uint16) batch kernel.
    // Must run in the ctor (before any cuda-graph stream capture) — the wide launch relies on it.
    cudaFuncSetAttribute((const void*)sieveAndScanBatchKernelWide, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
    // Same opt-in for the saturating-uint8 wide kernel (larger SB => larger smem).
    cudaFuncSetAttribute((const void*)sieveAndScanBatchKernelWideU8Sat, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes);
}

DeviceSievingController::~DeviceSievingController()
{
    // Proper cleanup of device pointers
    if (dev_pointers.dev_sieveOffsets) cudaFree(dev_pointers.dev_sieveOffsets);
    if (dev_pointers.dev_factorBase) cudaFree(dev_pointers.dev_factorBase);
    if (dev_pointers.dev_rootN) cudaFree(dev_pointers.dev_rootN);
    if (dev_pointers.dev_a_factors) cudaFree(dev_pointers.dev_a_factors);
    if (dev_pointers.dev_B_values) cudaFree(dev_pointers.dev_B_values);
    if (dev_pointers.dev_primeData) cudaFree(dev_pointers.dev_primeData);
    if (dev_pointers.dev_primeBValues) cudaFree(dev_pointers.dev_primeBValues);
    if (dev_pointers.dev_globalBucketEntries) cudaFree(dev_pointers.dev_globalBucketEntries);
    if (dev_pointers.dev_globalBucketCounts) cudaFree(dev_pointers.dev_globalBucketCounts);
    if (dev_pointers.dev_candidateRelations) cudaFree(dev_pointers.dev_candidateRelations);
    if (dev_pointers.dev_indexToCandidate) cudaFree(dev_pointers.dev_indexToCandidate);
    if (dev_pointers.dev_blockRelationCounts) cudaFree(dev_pointers.dev_blockRelationCounts);
    if (dev_pointers.dev_job_a_array) cudaFree(dev_pointers.dev_job_a_array);
    if (dev_pointers.dev_job_B_flat) cudaFree(dev_pointers.dev_job_B_flat);
    if (dev_pointers.dev_job_factor_indices) cudaFree(dev_pointers.dev_job_factor_indices);
    releasePinnedIndexStaging();
    // Wide-probe scratch (owned by this controller; dev_pp_* in dev_pointers only alias it).
    if (dev_probe_pp_accum_)   { cudaFree(dev_probe_pp_accum_);   dev_probe_pp_accum_ = nullptr; }
    if (dev_probe_pp_counter_) { cudaFree(dev_probe_pp_counter_); dev_probe_pp_counter_ = nullptr; }
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

    // --- Dual-path sieve accumulator dispatch predicate (COMPUTE + LOG) ---
    // Decide whether the uint16 wide accumulator path is needed for this (a, M).
    // The result drives the batch-sieve accumulator-width dispatch downstream.
    //
    // Guard: f_data.a MUST be finalized before the predicate (caller order recalc_a ->
    // initiate). Defensive tripwire; if it ever fires, a future caller reordered and the
    // predicate + config sizing must move to the first point where 'a' is final.
    if (f_data.a.is_zero()) {
        LOG(LOG_ERROR_CRITICAL) << "Sieve predicate: f_data.a not populated before initiate()";
        // (production escalates; do not silently pick u8)
    }
    uint32_t log2_a  = (uint32_t)f_data.a.msb();                // == device current_a.msb()/p_data.log2_a
    uint32_t ilog2_M = 31u - __builtin_clz((uint32_t)f_data.M); // M is pow2 in every validated config (logging/cross-check only)
    // EXACT-EVAL form (mirrors kernel.cu:834-837). Evaluate approxPolyVal at the three
    // worst-case indices gi in {-M, 0, M-1} using the device's own arithmetic and
    // log2(0)=0 convention, then take the max.
    auto host_ilog2 = [](int v) -> int { return (v <= 0) ? 0 : (31 - __builtin_clz((uint32_t)v)); };
    int32_t  M    = (int32_t)f_data.M;
    uint32_t apr  = fs_params.approxPolyRoot;                   // = M*10000/14142 (set above)
    int APV_max = 0;
    for (int32_t gi : { -M, 0, M - 1 }) {
        // exact analog of kernel.cu:834-835 (unsigned-safe absolute distances)
        uint32_t dist_minus = (gi >= (int)apr)  ? (uint32_t)(gi - (int)apr)  : (uint32_t)((int)apr - gi);
        uint32_t dist_plus  = (gi >= -(int)apr) ? (uint32_t)(gi + (int)apr)  : (uint32_t)(-(int)apr - gi);
        int apv = (int)log2_a + host_ilog2((int)dist_minus) + host_ilog2((int)dist_plus);  // == kernel.cu:837
        APV_max = std::max(APV_max, apv);
    }
    // (Cross-check only:) closed form log2_a + 2*ilog2(M) - 1 agrees to +/-1.
    const int WIDE_MARGIN = 4;                                  // safe window [1,4]; 5 is UNSAFE (RSA-140@1M 251+5=256)
    bool predicate_wide = (APV_max + WIDE_MARGIN) >= 256;
    use_wide_accumulator_ = (accumulator_mode_ == 2) ? true
                          : (accumulator_mode_ == 1) ? false
                          : predicate_wide;
    LOG(LOG_INFO) << "Sieve accumulator: log2_a=" << log2_a << " ilog2_M=" << ilog2_M
                  << " APV_max=" << APV_max << " margin=" << WIDE_MARGIN
                  << " mode=" << accumulator_mode_
                  << " => use_wide=" << use_wide_accumulator_;
    // --- end accumulator predicate ---

    fs_params.threshold = 31 - std::countl_zero((uint32_t)f_data.F);

    // --- Wide saturating-uint8 accumulator width dispatch --------------------
    // Only meaningful in the wide regime. The saturating-uint8 accumulator is
    // bit-for-bit equivalent to the uint16 wide path in candidate selection iff
    // the per-block MAX threshold target APV_max - threshold <= 254 (equivalence
    // lemma, 2026-07-10 wide-smem-reduction assessment, §3). When it
    // holds, u8sat restores SB to the 1-byte width; when it fails (larger M /
    // smaller F pushing the target toward 255) we fall back to uint16 — never to
    // the wrap bug. Gate is checked even under a force (mode 1) so a forced-u8sat
    // never silently reintroduces the RSA-155 smooth-loss pathology.
    //
    // HARDENED GATE (post-review F1, 2026-07-10 wide-u8sat prototype review):
    // APV_max is a 3-point sample {-M,0,M-1} of approxPolyVal and a single
    // f_data.a.msb(), NOT a proven per-location / per-poly upper bound. The reviewer's
    // brute force over all gi found two independent latent slop sources, each <=1 count:
    //   (S1) non-power-of-2 M can undercount the true interior max by up to 1
    //        (power-of-2 M — every validated/probe config — undercounts 0);
    //   (S2) the batch holds many polynomials whose a.msb() may differ from the sampled
    //        f_data.a.msb() by +/-1 (the use_wide predicate absorbs this with WIDE_MARGIN=4;
    //        this gate had no analogous margin).
    // A single undercount at the exact boundary would let u8sat clamp a candidate uint16
    // keeps (a few false-negative relations — not the wrap catastrophe, but it breaks the
    // "bit-for-bit identical" guarantee). We therefore require a 2-count safety margin:
    // max_target <= 252 (== 254 - 2), covering S1+S2 for ANY M (pow2 or not). This is a
    // no-op at every production/probe point (RSA-155 max_target=249, RSA-150=239 — both
    // still u8sat with room to spare) and only matters if a future F<=25M descent + non-pow2
    // M pushes the margin toward 0, where it degrades gracefully to uint16. The same
    // hardened threshold applies to a forced --wide_accum u8sat (mode 1).
    static constexpr int kU8SatGateMaxTarget = 252;   // 254 - 2 (S1 non-pow2 + S2 a.msb slop)
    wide_u8sat_selected_ = false;
    if (use_wide_accumulator_) {
        const int max_target = APV_max - (int)fs_params.threshold;
        const bool gate_ok   = (max_target <= kU8SatGateMaxTarget);
        if (wide_accum_mode_ == 2) {
            wide_u8sat_selected_ = false;                 // force uint16
        } else if (wide_accum_mode_ == 1) {               // force u8sat (gate-honoured)
            wide_u8sat_selected_ = gate_ok;
            if (!gate_ok)
                LOG(LOG_WARNING) << "Wide accumulator: --wide_accum u8sat requested but "
                                    "hardened exactness gate fails (max_target=" << max_target
                                 << " > " << kU8SatGateMaxTarget
                                 << "); falling back to uint16 to avoid overflow.";
        } else {                                          // auto
            wide_u8sat_selected_ = gate_ok;
        }
        LOG(LOG_INFO) << "Wide accumulator width: APV_max=" << APV_max
                      << " threshold=" << fs_params.threshold
                      << " max_target=" << max_target
                      << " gate(<=" << kU8SatGateMaxTarget << ")=" << (gate_ok ? 1 : 0)
                      << " wide_accum_mode=" << wide_accum_mode_
                      << " => " << (wide_u8sat_selected_ ? "u8sat" : "uint16");
    }
    // Note: updateState() is NOT called here — it must run after loadData()
    // has allocated device memory (dev_a_factors, dev_B_values). The
    // orchestrator calls updateState() explicitly after loadData().
}

void DeviceSievingController::setThresholdOverride(uint64_t threshold_bound)
{
    if (threshold_bound > 0)
        fs_params.threshold = 63 - std::countl_zero(threshold_bound);
}

bool DeviceSievingController::getBucketOverflowStats(BucketOverflowStats& out) const
{
    // WIDE by default; NARROW only under --sieve_bucket_overflow_stats (v1.0.6).
    //
    // The device data exists on BOTH widths: globalMetaSieveBatchKernel writes the bit-31
    // overflow flag and that kernel is width-agnostic (it has a legacy twin). Only this host
    // reader was gated, and for a real reason: the read below costs a cudaMemcpyAsync plus a
    // cudaStreamSynchronize ON THE SIEVER STREAM at the ~5 s stats cadence, and the narrow
    // (uint8, <= RSA-140) production path runs a zero-sync double-buffered batch pipeline that
    // carries every validated record. Making the relaxation unconditional would put that sync
    // into every narrow production run as an unmeasured perturbation.
    //
    // v1.0.6 therefore relaxes it behind a DEFAULT-OFF knob. Wide behaviour is byte-identical;
    // narrow behaviour is byte-identical unless --sieve_bucket_overflow_stats is passed. The
    // knob exists because halving bigPrimeStartIndex moves ~1024 factor-base primes onto the
    // bucketed SCATTER path, which raises bucket fill by an INFERRED amount with an unbounded
    // error bar -- and narrow bucket overflow % has never been observed anywhere in this
    // project, so the telemetry has to ship WITH the knob that makes it move, not after.
    //
    // Rejected alternative: gating on the debug log level. It is a genuine one-liner with no
    // plumbing, but narrow production runs at RSA-140 and below DO use --verbose --debug for
    // other telemetry, so it would silently turn the sync on in production.
    if (!use_wide_accumulator_ && !narrow_overflow_stats_) return false;
    if (dev_pointers.dev_globalBucketCounts == nullptr) return false;

    const uint64_t total = (uint64_t)gs_conf.num_polysPerSieveCall
                         * (uint64_t)gs_conf.num_sievingBlocksPerSieveCall;
    if (total == 0) return false;

    // Pure read of the EXISTING per-bucket counts buffer (bit 31 = overflowed, set at
    // kernel.cu globalMetaSieveBatchKernel; masked off by the GATHER dump). Confined to
    // the siever stream so it cannot perturb work on other streams; read-only, so it
    // cannot alter any device sieving result. Snapshots the most-recently-populated
    // buckets (a representative batch — the overflow tax is ~F/M-invariant per the
    // root-cause analysis).
    std::vector<uint32_t> h(total);
    cudaError_t e = cudaMemcpyAsync(h.data(), dev_pointers.dev_globalBucketCounts,
                                    total * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return false;
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return false;

    uint64_t overflowed = 0;
    uint32_t max_fill   = 0;
    for (uint64_t i = 0; i < total; ++i) {
        const uint32_t c = h[i];
        if (c & 0x80000000u) ++overflowed;
        const uint32_t fill = c & 0x00FFFFFFu;   // clamped fill level (min(head, globalBucketSize))
        if (fill > max_fill) max_fill = fill;
    }

    out.total_buckets      = total;
    out.overflowed_buckets = overflowed;
    out.global_bucket_size = gs_conf.globalBucketSize;
    out.max_fill           = max_fill;
    out.fraction           = (double)overflowed / (double)total;
    return true;
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

    // Pinned host buffer for truly async H2D factor index copies (double-buffered + event-gated)
    allocatePinnedIndexStaging((size_t)gs_conf.batch_size * fs_params.shc_dim);

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
    // A new 'a' invalidates everything initPrimeData() derives, so this IS a new cube. Set it
    // here rather than relying on each sweep entry to do so: sieveStep() gates the (expensive)
    // re-derivation on this flag, and updateState() is the only place 'a' ever changes.
    ds_params.newCube = true;
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

    maskData MD = generateMask(f_data.factorBase.data(), dev_pointers);
    fs_params.period = MD.period;
    fs_params.smallPrimesUsed = MD.smallPrimesUsed;
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
    // [Non-batch guard — PRODUCTION: escalate] sieveFullCube drives the legacy (non-batch)
    // sieveAndScanKernel via sieveStep()->sieveAndScan(), which has no uint16 fork.
    // The wide accumulator is only available on the batch sieve path, so escalate rather than
    // silently run the overflow-prone uint8 non-batch kernel. Guard the ENTRY point (not the
    // shared sieveStep(), which sieveMini() also uses).
    if (use_wide_accumulator_) {
        LOG(LOG_ERROR_CRITICAL) << "wide accumulator requires the batch sieve; rerun with "
                                   "--sieve_batch_size >= 1 (non-batch sieveFullCube has no uint16 path)";
        throw std::runtime_error(
            "wide accumulator requires the batch sieve; rerun with --sieve_batch_size >= 1");
    }
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
    // initPrimeData() depends ONLY on 'a' (via ds_params.a / dev_B_values / factorBase / rootN):
    // mod_inverse_a, inv_aN and the whole dev_primeBValues table are fixed for the lifetime of
    // one hypercube, and nothing in the sieve writes primeData back. Re-deriving it per call
    // costs fb_size * (a 32-round modular exponentiation + shc_dim uint512 reductions), repeated
    // once per subCube and per sievingBlockBatch of the SAME 'a'. Gate it on newCube — set by
    // updateState() whenever 'a' changes, cleared below after the first call of each cube.
    // The whole wrapper is gated deliberately: initPrimeDataKernel writes a value-initialised
    // primeDataSIQS (clearing `inactive`), so its paired markInactivePrimesKernel must run with
    // it or the a-factors would come back active.
    if (ds_params.newCube)
        initPrimeData(dev_pointers, init_conf, gs_conf, fs_params, ds_params, stream);
    globalMetaSieve(dev_pointers, fs_params, ds_params, gs_conf, gms_conf, stream);
    sieveAndScan(dev_pointers, fs_params, ds_params, gs_conf, ss_conf, stream);
    ds_params.newCube = false; //A sieving call was made locally, so the cube is not "new" anymore
}

float DeviceSievingController::sieveMini(uint32_t num_subcubes) {
    // [Non-batch guard — PROBE: skip] sieveMini() (the probe/autotune entry, also reached via
    // evaluateConfig) drives the legacy non-batch sieveAndScanKernel via sieveStep(), which has no
    // uint16 fork. Skip this infeasible probe (return the -1.0f sentinel) rather than
    // aborting the whole candidate sweep (MINOR-5). Guard the ENTRY, not the shared sieveStep().
    if (use_wide_accumulator_) {
        // Wide-autotune foundation: the wide accumulator has NO non-batch kernel, so
        // instead of skipping the probe (the old -1.0f sentinel that made the optimizer bail
        // with "0 configs tested") drive the REAL batch-wide pipeline. num_subcubes carries
        // through as the repeat count.
        return sieveMiniBatch(num_subcubes);
    }
    // 0 means full cube
    if (num_subcubes == 0) num_subcubes = gs_conf.num_subCubes;

    // Save state
    int32_t  saved_startIndex = ds_params.startIndex;
    uint32_t saved_subCube    = ds_params.subCube;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // UNTIMED SETUP PASS. Two things used to sit inside the measured window and should not:
    //
    //   initPrimeData() -- newCube is set here, so the first sieveStep() re-derives
    //     mod_inverse_a, inv_aN and the whole dev_primeBValues table: fb_size primes x (a
    //     32-round modular exponentiation + shc_dim uint512 reductions). That cost depends on
    //     'a' alone, NOT on the geometry under test, so it is the same constant for every
    //     candidate -- it compresses the spread between them and, because the score is divided
    //     by work, penalises small np. Production amortises it over a whole hypercube; charging
    //     each probe one full copy is the unrepresentative choice.
    //
    //   the cold start -- first-touch caches, clock ramp and JIT all land on the first pass.
    //     sieveMiniBatch has discarded a warm-up prefix for exactly this reason since the wide
    //     autotune work; narrow sieveMini never did, which is why a seed's first draw reads
    //     well above its own later draws.
    //
    // One full sieveStep() outside the window covers both: it performs the derivation, clears
    // newCube, and warms the machine. The timed passes then measure steady-state sieving.
    ds_params.newCube = true;
    ds_params.subCube = 0;
    ds_params.startIndex = -((int32_t)f_data.M);
    sieveStep();                       // setup + warm-up, deliberately not measured

    // The GATHER kernel resumes candidatesFound from dev_blockRelationCounts whenever newCube
    // is false, and a block that has spent its maxRelationsPerBlock budget stops emitting (and
    // stops running the backward sieve). Zero the counters so the timed region starts from the
    // same state the untimed pass did -- otherwise the warm-up would leave every block full and
    // the measurement would skip the backward sieve entirely.
    cudaMemsetAsync(dev_pointers.dev_blockRelationCounts, 0,
                    (size_t)ss_conf.num_threadBlocks * sizeof(uint32_t), stream);

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

// ---------------------------------------------------------------------------
// Wide-autotune foundation: batch-wide probe.
// ---------------------------------------------------------------------------

void DeviceSievingController::ensureProbeBatchSetup() {
    if (probe_batch_ready_) return;

    // Probe SAMPLE size = number of DISTINCT polynomials staged once here. Every candidate
    // (Phase-1 seed, all Phase-2 configs, the verify pass, and the floor probe) re-sieves this
    // SAME staged batch (A/B fidelity), so it is the actual sample the survivors/sec objective
    // sees — a longer measurement window only re-times it more precisely, it never enlarges it.
    //
    // Per-poly sieve-survivor density falls steeply with N (the sieve-log threshold tightens),
    // so the old fixed 8-poly sample — signal-rich at RSA-100 — becomes ~65-90% likely to hold
    // ZERO survivors at RSA-150/155, starving the harness (measured 0 survivors/s for every
    // candidate incl. the floor → NO_SIGNAL; report rsa150_a100_at_eff_v2_analysis_2026_07_10.md
    // §4). Fix: scale the DISTINCT sample with N so E[survivors] stays comfortably nonzero —
    // one doubling per ~16 bits of N above 400, off the 8-poly RSA-100 baseline, capped at 1024.
    // At the measured RSA-150 density (~0.015-0.10 survivors/poly), 512 polys give
    // E[survivors] ≈ 8-50 (P(empty) < 0.1%); RSA-155 (lower density) gets 1024. Smaller N keeps
    // the old fast, survivor-rich 8-poly probe unchanged (preserving the validated RSA-100 wide
    // autotune behavior). Staging cost is trivial (≤1024 × shc × 64 B ≈ 2 MB) and the per-chunk
    // survivor count stays far below probe_pp_capacity_ (density × sample ≪ 2^18) at every scale.
    // CLI --autotune_probe_polys overrides the auto-scale (0 = auto) for A/B testing on any N.
    // nbits is the true BIT-LENGTH: uint512::msb() returns the top-set-bit INDEX (bit_length-1),
    // so +1 — else RSA-150 (496b) would stage 256 not 512 and RSA-155 (512b) 512 not 1024.
    const uint32_t nbits = static_cast<uint32_t>(f_data.N.msb()) + 1u;  // bit-length of N
    if (autotune_probe_polys_override_ > 0u) {
        probe_batch_size_ = autotune_probe_polys_override_;
    } else {
        probe_batch_size_ = 8u;
        if (nbits > 400u) {
            const uint32_t steps = std::min((nbits - 400u) / 16u, 7u); // 8<<7 = 1024 cap
            probe_batch_size_ = 8u << steps;
        }
    }
    LOG(LOG_INFO) << "[Autotune][wide-probe] staged sample = " << probe_batch_size_
                  << " distinct polynomials (N=" << nbits << "b, override="
                  << autotune_probe_polys_override_ << ")";
    setSievingBatchSize(probe_batch_size_);   // init/gs/gms/ss batch_size

    const uint32_t shc = fs_params.shc_dim;

    // Batch-context job arrays (a, flattened B, factor indices). loadData() /
    // loadSievingDataParamTest() (the autotune path) never allocate these, so stage
    // them here. Free-before-realloc keeps ensureProbeBatchSetup() idempotent and avoids
    // the dev_job_* triple-ownership leak shared with allocateBatchBuffers()/clearSievingBuffers().
    if (dev_pointers.dev_job_a_array)        { cudaFree(dev_pointers.dev_job_a_array);        dev_pointers.dev_job_a_array = nullptr; }
    if (dev_pointers.dev_job_B_flat)         { cudaFree(dev_pointers.dev_job_B_flat);         dev_pointers.dev_job_B_flat = nullptr; }
    if (dev_pointers.dev_job_factor_indices) { cudaFree(dev_pointers.dev_job_factor_indices); dev_pointers.dev_job_factor_indices = nullptr; }
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_job_a_array,        (size_t)probe_batch_size_ * sizeof(mpqs::uint512)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_job_B_flat,         (size_t)probe_batch_size_ * shc * sizeof(mpqs::uint512)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_pointers.dev_job_factor_indices, (size_t)probe_batch_size_ * shc * sizeof(uint32_t)));

    // Pinned host factor-index staging buffer used by prepareSievingBatch()'s launcher.
    allocatePinnedIndexStaging((size_t)probe_batch_size_ * shc);

    // Postprocessing scratch. compactCandidatesBatchKernel atomicAdds dev_pp_counter
    // UNCONDITIONALLY (→ NULL-deref on the first survivor if it is null) and writes a
    // DenseCandidate only while pos < pp_max_capacity, so a non-null counter AND a bounded
    // output buffer are both mandatory. A fixed, scale-independent capacity keeps the probe
    // memory-safe at any N/M — overflow past capacity is dropped by the kernel's guard.
    probe_pp_capacity_ = 1u << 18;   // 262144 DenseCandidates (~88 MB @ 336 B/entry)
    if (dev_probe_pp_accum_)   { cudaFree(dev_probe_pp_accum_);   dev_probe_pp_accum_ = nullptr; }
    if (dev_probe_pp_counter_) { cudaFree(dev_probe_pp_counter_); dev_probe_pp_counter_ = nullptr; }
    SIEVE_CUDA_CHECK(cudaMalloc(&dev_probe_pp_accum_, (size_t)probe_pp_capacity_ * sizeof(mpqs::sieve::DenseCandidate)));
    SIEVE_CUDA_CHECK(cudaMalloc((void**)&dev_probe_pp_counter_, sizeof(uint32_t)));
    dev_pointers.dev_pp_accumulation_buffer = dev_probe_pp_accum_;
    dev_pointers.dev_pp_counter             = dev_probe_pp_counter_;
    dev_pointers.pp_max_capacity            = probe_pp_capacity_;

    // Stage ONE batch of polynomials; every candidate re-sieves these SAME polynomials so
    // the search is apples-to-apples (A/B fidelity). Advances only this ephemeral siever's
    // f_data copy — never production's.
    prepareSievingBatch();

    probe_batch_ready_ = true;
}

float DeviceSievingController::sieveMiniBatch(uint32_t /*repeats — ignored, see below*/) {
    // Scale-representative harness. Replace the old fixed-repeat cudaEvent µs timing with
    // a wall-clock survivors/sec RATE measured through the REAL batch pipeline. Rationale:
    // isolated-kernel µs mis-ranks occupancy on wide (the halved
    // wide SB shifts the sieve↔postproc↔bucket balance); the production-bound signal is
    // candidate survivors/sec through runSievingBatch (sieve + compact). Auto-scaling the
    // measurement by WALL TIME (not by a fixed batch/subcube count) keeps every candidate
    // representative at any N/M — larger M ⇒ fewer polys/call ⇒ a fixed batch count would be
    // unrepresentative. CUDA-graph is OFF (per-candidate recapture JIT is prohibitive and the
    // graph is only a param-invariant launch-latency amortizer, so its omission cannot change
    // the ranking); the batch stays double-buffered in the sense that runSievingBatch keeps the
    // sieve→compact pipeline in one stream and the host enqueues a small chunk ahead of each
    // sync to keep the GPU saturated.

    // Idempotent setup (job arrays + pinned + pp scratch + one staged batch).
    ensureProbeBatchSetup();
    // loadPartialCustomConfig()/loadStandardConfig() zeroed init_conf.batch_size for THIS
    // candidate; re-assert the probe batch size on init/gs/gms/ss before any batch launch.
    setSievingBatchSize(probe_batch_size_);

    // The batch kernels sieve from ds_params.startIndex = -M (production sets this in
    // updateState()); assert it here and restore afterwards.
    int32_t saved_startIndex = ds_params.startIndex;
    ds_params.startIndex = -((int32_t)f_data.M);

    using clk = std::chrono::steady_clock;
    auto secs = [](clk::duration d) {
        return std::chrono::duration<double>(d).count();
    };

    // Chunk of runSievingBatch calls enqueued between host syncs. Enough polys in flight
    // (~32) to hide launch latency / keep the GPU saturated (like the double-buffered
    // production overlap), but no more: a large probe sample already saturates one batch, so
    // re-sieving it 4× per chunk (the old fixed value) would just multiply wall time without
    // adding distinct samples. Scale inversely with the sample size — 4 at the 8-poly baseline
    // (byte-identical to the prior harness, preserving validated RSA-100 wide behavior), down
    // to 1 for samples ≥ 32. Kept small enough that one chunk's survivors stay below pp
    // capacity, so the compact kernel does full per-survivor work and the reset-per-chunk keeps
    // the counter well below dev_probe_pp_counter_'s uint32 range.
    const int SYNC_CHUNK = std::max(1, static_cast<int>(32u / probe_batch_size_));

    // ---- Warm-up prefix (JIT, clock ramp, cache/TLB fill) — discarded from the score. ----
    {
        auto w0 = clk::now();
        do {
            for (int c = 0; c < SYNC_CHUNK; ++c) runSievingBatch((int)probe_batch_size_, 0);
            cudaStreamSynchronize(stream);
        } while (secs(clk::now() - w0) < probe_warmup_sec_);
    }

    // ---- Measurement window: sum survivors over probe_window_sec_ of wall clock. ----
    // Reset the compact fill counter per chunk and host-sum the chunk deltas, so the counter
    // stays bounded (< capacity ⇒ no dropped-write bias) while the TOTAL survivor count is
    // exact even when it far exceeds pp capacity across the whole window.
    uint64_t total_survivors = 0;
    auto m0 = clk::now();
    double elapsed = 0.0;
    do {
        cudaMemsetAsync(dev_probe_pp_counter_, 0, sizeof(uint32_t), stream);
        for (int c = 0; c < SYNC_CHUNK; ++c) runSievingBatch((int)probe_batch_size_, 0);
        uint32_t chunk_survivors = 0;
        cudaMemcpyAsync(&chunk_survivors, dev_probe_pp_counter_, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        total_survivors += chunk_survivors;
        elapsed = secs(clk::now() - m0);
    } while (elapsed < probe_window_sec_);

    ds_params.startIndex = saved_startIndex;

    if (elapsed <= 0.0) return -1.0f;
    // survivors per second — HIGHER is better (the wide objective is a rate).
    return static_cast<float>(static_cast<double>(total_survivors) / elapsed);
}

// ---------------------------------------------------------------------------
// Floor gate: time loadStandardConfig-wide under the SAME window harness.
// ---------------------------------------------------------------------------

float DeviceSievingController::sieveMiniStandardWide() {
    // Wide-only. The narrow path has its own µs objective and never needs a rate floor.
    if (!use_wide_accumulator_) return -1.0f;

    // Load the EXACT geometry that ships when autotune is OFF (useParams==false): the
    // standard wide config (SB cap, bigPrimeStart=SB/32, num_polys≤512, A1 SCATTER grid).
    // This is the acceptance floor the search winner must beat — measured the same way.
    loadStandardConfig();
    if (!validateConfigs()) {
        LOG(LOG_WARNING) << "[Autotune][floor] loadStandardConfig-wide failed validateConfigs(); "
                            "floor unavailable (winner applied on positive rate)";
        return -1.0f;
    }

    // Resize the sieve scratch (bucket/counts/candidates/index) to the standard config.
    // loadSievingDataParamTest touches ONLY those buffers — it does NOT reallocate the staged
    // job arrays (dev_job_*) or the probe pp scratch (dev_pp_*), so the batch staged in
    // ensureProbeBatchSetup() and the survivor counter both survive for the floor window.
    loadSievingDataParamTest(f_data.factorBase, f_data.rootN, f_data.a_factors,
                             fs_params.shc_dim, gs_conf, ss_conf, dev_pointers);

    return sieveMiniBatch(0);  // repeats ignored; uses the current probe window (set by caller)
}

// ---------------------------------------------------------------------------
// Wide autotune search-space seeding: occupancy-optimal GATHER blockDim.
// ---------------------------------------------------------------------------

uint32_t DeviceSievingController::wideGatherOccupancyBlockDim(
    const uint32_t* candidates, uint32_t count) const {
    // Wide-only. The narrow autotune seeds/searches with its own (unchanged) blockDim.
    if (!use_wide_accumulator_ || candidates == nullptr || count == 0) return 0;

    // Wide GATHER shared memory — the SAME footprint validateConfigs()'s wide-feasibility guard
    // uses: the realized sieveAndScanBatchKernelWide launch shares ss_conf.sharedMemReq
    // (blockEntries at uint16 width + 3*bigPrime buckets) PLUS the per-block B_values
    // (shc_dim*sizeof(uint512)). It is INDEPENDENT of blockDim (blockEntries is sized per
    // sieving-block, every work loop strides by blockDim.x), so compute it once. ss_conf reflects
    // loadStandardConfig-wide at autotune-Stage-1 entry (the caller's state), which is the exact
    // geometry the seed will run at (loadStandardConfig-wide and loadPartialCustomConfig-wide
    // derive the identical wide sievingBlockSize from the opt-in smem budget).
    const size_t wideSieveSmem =
        ss_conf.sharedMemReq + (size_t)fs_params.shc_dim * sizeof(mpqs::uint512);
    if (wideSieveSmem > g_info.maxSharedMemPerBlock) return 0;  // infeasible → caller keeps its seed

    uint32_t best_bd = 0;
    uint64_t best_threads = 0;  // resident threads/SM = blocks_per_SM * blockDim (the occupancy knee)
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t bd = candidates[i];
        if (bd == 0) continue;
        int blocks_per_sm = 0;
        // Query the kernel actually dispatched at this width.
        const void* wideKern = wide_u8sat_selected_
            ? (const void*)sieveAndScanBatchKernelWideU8Sat
            : (const void*)sieveAndScanBatchKernelWide;
        const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, wideKern,
            (int)bd, wideSieveSmem);
        if (err != cudaSuccess || blocks_per_sm < 1) continue;
        const uint64_t threads = (uint64_t)blocks_per_sm * bd;
        // Strictly-greater keeps the SMALLEST blockDim among ties (fewer threads to schedule
        // for identical residency); the wide kernel is smem-bound, so the max typically lands
        // at the largest feasible blockDim (H100 1/SM → 1024; A100 2/SM → 1024 = 100%).
        if (threads > best_threads) { best_threads = threads; best_bd = bd; }
    }
    return best_bd;  // 0 if the API failed for every candidate → caller keeps its seed
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

// Smallest power of two >= x (the loaders keep their own local copies of this).
static inline uint32_t pow2geqU32(uint32_t x) {
    if (x <= 1) return 1u;
    return 1u << (32 - std::countl_zero(x - 1u));
}

void DeviceSievingController::runParamTest(factoringData& f_data, const ParamSet* seed,
                                                      uint32_t radius)
{
    LOG_SET_SUBMODULE("PARAM_TEST");

    const int          kRadius        = (int)(radius ? radius : 1u);  // --param_test_radius
    constexpr int      kMaxRounds     = 64;  // cycles over the face complex
    constexpr double   kJumpMargin    = 0.005; // a face must beat the record by this to be adopted
    constexpr uint64_t kProgressProbes = 100; // heartbeat cadence within a face, in probes
    constexpr uint32_t kMaxSubCubes = 2;     // sieve calls per hypercube; see the prune below
    constexpr uint64_t kRefEvery   = 25;     // re-probe the incumbent every N candidates
    constexpr int      kRefWindow  = 3;      // reference = min of the last N such draws
    constexpr int      kReps          = 2;   // draws a contender gets; judged on their minimum
    constexpr uint32_t kProbeSubCubes = 2;   // subcubes per timing probe

    using clk = std::chrono::steady_clock;
    auto secs = [](clk::duration d) {
        return std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
    };

    if (use_wide_accumulator_) {
        LOG(LOG_ERROR_CRITICAL) << "Param test: narrow (uint8) path only";
        return;
    }

    const uint32_t hyperCubeSize = 1u << (f_data.a_factors.size() - 1);
    const uint32_t M  = f_data.M;
    const uint32_t mp = (uint32_t)g_info.multiProcessorCount;

    auto pow2Range = [](std::vector<uint32_t>& out, uint32_t lo, uint32_t hi) {
        out.clear();
        for (uint32_t v = 1u; v <= hi; v <<= 1) {
            if (v >= lo) out.push_back(v);
            if (v == (1u << 31)) break;
        }
    };
    auto addRung = [](std::vector<uint32_t>& v, uint32_t x) {
        if (x && std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
    };

    // ---- rung ladders ---------------------------------------------------------------
    std::vector<uint32_t> rSB, rNP, rSASG, rPBS, rBPSI, rMETAG, rMETAB, rSASB;
    pow2Range(rSB,    8192, std::min(M, 131072u));
    pow2Range(rNP,      32, hyperCubeSize);
    pow2Range(rSASG,    32, 1024);
    pow2Range(rPBS,      1, 32);
    pow2Range(rMETAG,   32, 1024);
    pow2Range(rMETAB,  128, 1024);
    pow2Range(rSASB,   128, 1024);

    // ---- prime-band boundaries: an arithmetic ladder, NOT a power-of-two one ---------
    // midPrimeStartIndex and bigPrimeStartIndex are prime-INDEX band boundaries, and the
    // small->mid band is walked in groups of 32 consecutive primes (kernel.cu:1276 strides
    // blockDim/32 such groups over [smallPrimesUsed, midPrimeStart)), so both boundaries
    // belong on the lattice  smallPrimesUsed + k*32  (the mask covers the first
    // smallPrimesUsed primes, so the groups start there); a power-of-two ladder both misses
    // every lattice point and steps far too coarsely.
    const uint32_t spu = fs_params.smallPrimesUsed;
    auto snapBand = [spu](long v) -> uint32_t {   // nearest lattice point, >= spu + 32
        return spu + 32u * (uint32_t)std::max<long>(((v - (long)spu) + 16) / 32, 1);
    };
    auto bandRange = [&](std::vector<uint32_t>& out, long lo, long hi, long step) {
        out.clear();
        for (long v = lo; v <= hi; v += step) addRung(out, snapBand(v));
        std::sort(out.begin(), out.end());
    };
    // midPrimeStart: ranged well past the 128 heuristic -- the Turing and H100 searches BOTH
    // pinned this axis at the old top rung (196), which says the range was the binding
    // constraint rather than the optimum.
    std::vector<uint32_t> rMIDPS;
    bandRange(rMIDPS, 128 - 2*32, 128 + 8*32, 16);

    // bigPrimeStart: bounded by what the GATHER shared-memory budget actually permits, not by a
    // fixed offset from the SB heuristic. The old top (rSB.back()/32 + 256) stopped at 4324 on
    // H100 while the hardware allows ~8389, and the search pinned there. The bound is taken at
    // the SMALLEST SB rung, i.e. the largest headroom any SB could use; rungs that do not fit at
    // the SB actually chosen are rejected by validateConfigs for free. With offsetsInGlobal the
    // shared-memory term disappears entirely, so only the factor base bounds it.
    {
        const long smemForBands = (long)g_info.maxSharedMemPerBlock
                                - (long)fs_params.shc_dim * (long)sizeof(mpqs::uint512);
        const long fbCap = (long)f_data.factorBase.size() / 2;
        const long hiFit = offsets_in_global_ ? fbCap
                         : std::min((smemForBands - (long)rSB.front()) / 12, fbCap);
        const long lo    = std::max((long)rSB.front()/32 - 2*128, (long)rMIDPS.back() + 32);
        const long hi    = std::max(hiFit, (long)rSB.back()/32 + 2*128);
        // Bound the rung COUNT (~48), not the step: a fixed 128 would give ~140 rungs on H100,
        // and the occupancy face is 5 x |bPSI| x 4 -- it would become the entire search.
        const long step  = std::max(128L, (((hi - lo) / 48 + 31) / 32) * 32);
        bandRange(rBPSI, lo, hi, step);
    }

    // The exact-wave family is not on a power-of-two ladder: add SM-derived rungs so the
    // neighbourhood can reach it. SM counts are read from the device, never hardcoded.
    for (uint32_t k : {1u, 2u}) { addRung(rMETAG, mp*k); addRung(rSASG, mp*k); }
    for (uint32_t k : {1u, 2u, 4u, 8u, 16u}) if (mp*k <= hyperCubeSize) addRung(rNP, mp*k);
    std::sort(rMETAG.begin(), rMETAG.end());
    std::sort(rSASG.begin(),  rSASG.end());
    std::sort(rNP.begin(),    rNP.end());

    // Seed. With --params11: that tuple, so the search can be started at a known-good
    // geometry and its score read directly against the neighbourhood. Without it: whatever a
    // bare run would use — loadStandardConfig resolves all eleven fields self-consistently,
    // which the autotune HEURISTIC_DEFAULTS tuple does NOT (np=512 with metaGridDim=256,
    // polyBlockSize=4 gives np % (metaGridDim*polyBlockSize) != 0). Either way the eleven
    // seed values are read back from the RESOLVED config, so a partial --params11 (0 =
    // unset) is seeded at the loader's own derivation for the fields it leaves out.
    if (seed) {
        loadPartialCustomConfigDynamic(*seed);
    } else {
        loadStandardConfig();
        // loadStandardConfig pins the narrow SCATTER grid to a literal 64 ("keeps the exact
        // prior value") so bare runs stay byte-identical to the validated records. That is below
        // the SM count on every card with more than 64 SMs, where the grid prune then rejects
        // the seed and the search cannot start at all. Raise it for the SEED only.
        //
        // Both grids and polyBlockSize move together: metaGridDim * polyBlockSize must divide
        // num_polysPerSieveCall EXACTLY, and loadStandardConfig grew polyBlockSize against the
        // old grid of 64. Raising the grid alone therefore breaks the partition (128 * 32 =
        // 4096 > np = 2048 leaves num_polyBlocksPerThreadBlock = 0). Adjust all of them and
        // re-load through loadPartialCustomConfig, so every derived quantity -- npbptb,
        // activeBuckets, both sharedMemReq -- follows from one consistent tuple instead of
        // being patched field by field.
        const uint32_t mpq  = pow2geqU32((uint32_t)g_info.multiProcessorCount);
        const uint32_t np   = gs_conf.num_polysPerSieveCall;
        const uint32_t metaGrid = std::min(std::max(gms_conf.num_threadBlocks, mpq), np);
        const uint32_t sasGrid  = std::min(std::max(ss_conf.num_threadBlocks,  mpq), np);
        const uint32_t pbs  = std::max(1u, std::min(gms_conf.polyBlockSize, np / metaGrid));
        loadPartialCustomConfig(np,
                                gs_conf.num_sievingBlocksPerSieveCall,
                                pbs,
                                gms_conf.num_activeBlocksPerCycle,
                                metaGrid, gms_conf.num_threadsPerBlock,
                                sasGrid,  ss_conf.num_threadsPerBlock);
    }
    LOG(LOG_INFO) << "Param test: seed from "
                  << (seed ? "--params11" : "loadStandardConfig()");
    const uint32_t seedVal[] = {
        gs_conf.sievingBlockSize, gs_conf.num_polysPerSieveCall,
        ss_conf.num_threadBlocks, gms_conf.polyBlockSize,
        gs_conf.bigPrimeStartIndex, gms_conf.num_threadBlocks,
        gms_conf.num_threadsPerBlock, ss_conf.num_threadsPerBlock,
        gs_conf.midPrimeStartIndex
    };
    // Make the seed exactly representable, then freeze the ladders (the axis table below
    // holds pointers into them).
    {
        std::vector<uint32_t>* lad[] = {&rSB,&rNP,&rSASG,&rPBS,
                                        &rBPSI,&rMETAG,&rMETAB,&rSASB,&rMIDPS};
        for (size_t i = 0; i < 9; ++i) { addRung(*lad[i], seedVal[i]); std::sort(lad[i]->begin(), lad[i]->end()); }
    }

    struct Axis { SieveParam id; const char* name; const std::vector<uint32_t>* vals; };
    const std::vector<Axis> axis = {
        // numIntervals and blocksPerCycle are absent BY DESIGN: both are functions of
        // sievingBlockSize (coverage == 2M, one meta-sieve cycle), derived in the loader. As
        // free axes they made every SB move cost three simultaneous changes, which no bounded
        // neighbourhood could afford -- so SB, the axis with the largest effect, never moved.
        {P_SIEVING_BLOCK_SIZE,      "SB",        &rSB   },   // reload-forcing
        {P_SUB_CUBE_SIZE,           "np",        &rNP   },
        {P_SAS_GRID_DIM,            "sasGrid",   &rSASG },
        {P_POLY_BLOCK_SIZE,         "pbs",       &rPBS  },
        {P_BIG_PRIME_START,         "bPSI",      &rBPSI },
        {P_META_GRID_DIM,           "metaGrid",  &rMETAG},
        {P_META_BLOCK_DIM,          "metaBlock", &rMETAB},
        {P_SAS_BLOCK_DIM,           "sasBlock",  &rSASB },
        {P_MID_PRIME_START,         "midPS",     &rMIDPS}
    };
    const size_t n = axis.size();

    std::vector<size_t> best(n);
    for (size_t i = 0; i < n; ++i)
        best[i] = (size_t)(std::find(axis[i].vals->begin(), axis[i].vals->end(), seedVal[i])
                           - axis[i].vals->begin());

    // ---- probe: work-normalised, so configurations sweeping different amounts of the
    //      cube stay comparable. Lower is better. Negative = rejected/unmeasurable.
    //      A candidate is no longer rejected for its bucket FILL: loadPartialCustomConfigDynamic
    //      sizes globalBucketSize to the predicted peak occupancy, so a large bPSI simply buys a
    //      bigger bucket. The only bucket-related rejection left is the VRAM budget, enforced
    //      (like every other admissibility rule) by validateConfigsSilent().
    double last_probe_us = 0.0;   // absolute time of the most recent probe (for logging)
    auto toParamSet = [&](const std::vector<size_t>& p) {
        ParamSet ps;
        for (size_t i = 0; i < n; ++i) ps[axis[i].id] = (*axis[i].vals)[p[i]];
        return ps;
    };
    // Heuristic prunes. These reject configs that are ADMISSIBLE but structurally wasteful,
    // before any launch, so the ball spends its probes on candidates that could plausibly win.
    // Kept here rather than in validateConfigs: they are judgements about what is sensible,
    // not statements about what is correct.
    // Counted, not logged: at a compressed radius the ball holds thousands of candidates and a
    // line per rejection buries the round. The per-round tally says the same thing in one line.
    uint64_t pruned_grid = 0, pruned_cycles = 0, pruned_subcubes = 0, pruned_smem = 0;
    auto heuristicallyHopeless = [&]() -> bool {
        // (1) A grid below the SM count leaves whole SMs idle for the kernel's entire
        //     duration -- no occupancy, cache or latency effect can pay that back.
        if (gms_conf.num_threadBlocks < mp || ss_conf.num_threadBlocks < mp) {
            ++pruned_grid; return true;
        }
        // (2) num_metaSieveCycles > 1 re-traverses the entire factor base above
        //     bigPrimeStartIndex once per cycle, paying rootsFromPolyId and the modular setup
        //     again while the useful hit work per cycle merely halves.
        if (gms_conf.num_metaSieveCycles > 1) { ++pruned_cycles; return true; }
        // (3) num_subCubes = min(32768, 2^(shc_dim-1))/np is the number of sieve CALLS one
        //     hypercube costs, and each call carries a fixed pipeline tax the probe cannot see:
        //     its own processAndCommit, counter resets and stream sync, amortised over np
        //     polynomials. Small np multiplies that tax without changing the relation set --
        //     a cost that is real in production and invisible to a kernel-only timing probe,
        //     so it is pruned rather than measured.
        if (gs_conf.num_subCubes > kMaxSubCubes) { ++pruned_subcubes; return true; }
        // (4) The search PROBES through the legacy kernel, which reads B_values from global
        //     memory, but its OUTPUT is a tuple for a production BATCH run, where the GATHER
        //     kernel stages shc_dim uint512 B_values in shared memory on top of
        //     ss_conf.sharedMemReq. validateConfigs gates that term on batch_size > 0 -- which
        //     is correct for whichever kernel is about to run, and therefore zero here. Check
        //     the batch footprint explicitly so the search cannot recommend a geometry that
        //     passes every probe and then fails to launch in production.
        if (ss_conf.sharedMemReq + (size_t)fs_params.shc_dim * sizeof(mpqs::uint512)
                > g_info.maxSharedMemPerBlock) { ++pruned_smem; return true; }
        return false;
    };

    auto evaluate = [&](const std::vector<size_t>& p) -> double {
        loadPartialCustomConfigDynamic(toParamSet(p));
        if (!validateConfigsSilent()) return -1.0;
        if (heuristicallyHopeless()) return -1.0;
        loadSievingDataParamTest(f_data.factorBase, f_data.rootN, f_data.a_factors,
                                 fs_params.shc_dim, gs_conf, ss_conf, dev_pointers);
        // Never probe past the cube: num_subCubes is cap/np, so it is 1 at the largest np.
        const uint32_t sub = std::min(kProbeSubCubes, gs_conf.num_subCubes);
        if (sub == 0) return -1.0;
        cudaGetLastError();                      // clear anything sticky from the last candidate
        const float t_us = sieveMini(sub);
        // sieveMini only times; it never checks. A config that passes validateConfigs but fails
        // AT LAUNCH (smem over the opt-in limit, bad grid) returns a near-zero time and would
        // otherwise win the round. Reject it, and clear the error so the next probe starts clean.
        const cudaError_t perr = cudaGetLastError();
        if (perr != cudaSuccess) {
            LOG(LOG_DEBUG_1) << "      probe failed: " << cudaGetErrorString(perr);
            return -1.0;
        }
        if (t_us <= 0.0f) return -1.0;
        last_probe_us = t_us;
        const double work = (double)sub * gs_conf.num_polysPerSieveCall
                          * gs_conf.num_sievingBlockBatches
                          * gs_conf.num_sievingBlocksPerSieveCall
                          * gs_conf.sievingBlockSize;
        return work > 0.0 ? (double)t_us / work : -1.0;
    };

    auto axisOf = [&](SieveParam id) -> size_t {
        for (size_t i = 0; i < n; ++i) if (axis[i].id == id) return i;
        return n;
    };
    auto describe = [&](const std::vector<size_t>& p) {
        std::string s;
        for (size_t i = 0; i < n; ++i)
            s += (i ? " " : "") + std::string(axis[i].name) + "=" +
                 std::to_string((*axis[i].vals)[p[i]]);
        return s;
    };

    // Same tuple in --params11 order (SieveParam), ready to paste onto a command line.
    // numIntervals and blocksPerCycle are no longer axes but they ARE still fields, so they are
    // filled with the values the loader derives -- otherwise the line would come back 9 values
    // long and --params11, which wants all 11, would reject it.
    auto describeDynamic = [&](const std::vector<size_t>& p) {
        const size_t sbAxis = axisOf(P_SIEVING_BLOCK_SIZE);
        const uint32_t sb   = (sbAxis < n) ? (*axis[sbAxis].vals)[p[sbAxis]] : 0u;
        const uint32_t nI   = sb ? (uint32_t)((2ull * fs_params.M) / sb) : 0u;
        std::string s;
        for (uint32_t k = 0; k < (uint32_t)P_COUNT; ++k) {
            if (k) s += ",";
            if (k == (uint32_t)P_NUM_INTERVALS || k == (uint32_t)P_BLOCKS_PER_CYCLE) {
                s += std::to_string(nI);      // derived: coverage == 2M, one meta-sieve cycle
                continue;
            }
            for (size_t a = 0; a < n; ++a)
                if ((uint32_t)axis[a].id == k)
                    s += std::to_string((*axis[a].vals)[p[a]]);
        }
        return s;
    };

    LOG(LOG_INFO) << "Param test: radius " << kRadius
                  << " (max face dimension)"
                  << ", <= " << kMaxRounds
                  << " rounds, jump margin " << (100.0 * kJumpMargin)
                  << " %, probing seed: " << describe(best);
    LOG(LOG_INFO) << "  --params11 " << describeDynamic(best);
    double bestScore = evaluate(best);
    if (bestScore < 0.0) {
        LOG(LOG_ERROR_CRITICAL) << "Param test: seed config rejected -- "
                                << describe(best) << "; validator says:";
        loadPartialCustomConfigDynamic(toParamSet(best));
        validateConfigs();          // loud: print which check failed
        exit(1);                    // do NOT fall back into runParamTestLegacy
    }
    LOG(LOG_INFO) << "  seed score " << bestScore << " us/poly-position ("
                  << last_probe_us << " us absolute), bucket " << gs_conf.globalBucketSize
                  << " slots for a predicted "
                  << expectedBucketEntries(gs_conf.sievingBlockSize, gs_conf.bigPrimeStartIndex)
                  << " mean entries";

    // ---- hill climb, FIRST-IMPROVEMENT. The walk re-centres on the FIRST candidate that
    //      beats the incumbent by more than kJumpMargin and restarts the neighbourhood from
    //      there, so a single round can move several axes; the argmin-over-the-whole-ball
    //      rule it replaces could move the incumbent by at most one L1 ball per round no
    //      matter how many probes that ball cost. The margin is what keeps a jump from
    //      chasing probe noise. A ball that completes with no such jump falls back to the
    //      argmin over that ball (steepest descent), which is also what makes "the incumbent
    //      is the neighbourhood optimum" a meaningful convergence test.
    // ---- THE TUNING COMPLEX ---------------------------------------------------------
    //
    // Which parameters are worth moving TOGETHER is not a question of distance in ladder space;
    // it is fixed by the constraints in validateConfigs and by which quantity each parameter
    // feeds. A uniform L1 ball gets this backwards: it spends most of its probes on pairs that
    // share no constraint at all (polyBlockSize with bigPrimeStart, say) while being far too
    // small to reach along the axes that genuinely must travel together.
    //
    // So the neighbourhood is a complex of FACES -- small sets of axes with a real coupling --
    // and each face is swept over the FULL PRODUCT of its ladders with every other axis held at
    // the incumbent. A face therefore reaches further on each of its axes than a radius-3 ball
    // does, for a small fraction of the probes: ~1.1k for a whole cycle against ~84k for
    // radius 3, which still only reaches +-3 rungs.
    //
    // Vertices are shared between faces (np sits in two, sasBlock and metaGrid in two each), so
    // this is a complex rather than a partition. Every axis appears in at least one face.
    struct Face { const char* name; std::vector<SieveParam> ids; };
    const std::vector<Face> faces = {
        // GATHER smem is SB + 3*bPSI*4 and blocks/SM = min(smem limit, threads limit from
        // sasBlockDim): all three set the same occupancy rung and none of them means anything
        // alone. This is the face a ball can never assemble.
        {"occupancy",         {P_SIEVING_BLOCK_SIZE, P_BIG_PRIME_START, P_SAS_BLOCK_DIM}},
        // metaGridDim * num_polyBlocksPerThreadBlock * polyBlockSize == np, EXACTLY.
        {"scatter-partition", {P_SUB_CUBE_SIZE, P_META_GRID_DIM, P_POLY_BLOCK_SIZE}},
        // sasGridDim divides np, np/sasGridDim is a power of two, and polys-per-block decides
        // how early the maxRelationsPerBlock budget is exhausted.
        {"gather-partition",  {P_SUB_CUBE_SIZE, P_SAS_GRID_DIM}},
        {"scatter-waves",     {P_META_GRID_DIM, P_META_BLOCK_DIM}},
        {"gather-waves",      {P_SAS_GRID_DIM, P_SAS_BLOCK_DIM}},
        // Shallow gradients that need full-ladder resolution rather than neighbours; midPS is
        // genuinely independent, constrained only by smallPrimesUsed <= midPS <= bPSI.
        {"big-prime-split",   {P_BIG_PRIME_START}},
        {"mid-prime-split",   {P_MID_PRIME_START}},
    };

    const std::vector<size_t> seedIdx = best;     // kept for the final head-to-head
    const double seedScore = bestScore;
    std::vector<size_t> cand(n);
    uint64_t probes = 0;

    // SCORING RULE: RELATIVE, not absolute.
    //
    // An absolute record cannot be compared across a long run. Clocks ramp up from idle over the
    // first seconds and drift down as the part heats -- on a mobile card that is several percent
    // over a multi-minute search, monotone and correlated with WHEN a probe was taken rather
    // than with what it measured. A record kept as a min-over-all-draws captures the machine's
    // single best thermal moment, and every later candidate is then handicapped against it; and
    // because a face sweep walks its ladders in ascending order, whatever is enumerated first
    // (small SB, small bPSI) wins on timing alone.
    //
    // So every candidate is scored RELATIVE to the incumbent measured near the same moment:
    //   * during a sweep, the incumbent is re-probed every kRefEvery candidates and the running
    //     reference is the min of the last kRefWindow such draws (min, because the noise is
    //     one-sided: interference only ever adds time);
    //   * a face's winner then faces the incumbent in an ALTERNATING duel, so both see the same
    //     thermal state, and adoption needs the ratio to clear kJumpMargin.
    //
    // INVARIANT: each adopted step is >= kJumpMargin better than the incumbent measured beside
    // it, so the chain product is a real cumulative improvement rather than a sum of drifts.
    double refNow = bestScore;                  // incumbent floor, near the current probe
    std::deque<double> refWindow{bestScore};
    double chainGain = 1.0;                     // product of the adopted relative improvements

    auto probeReference = [&]() {
        const double d = evaluate(best);
        ++probes;
        if (d <= 0.0) return;
        refWindow.push_back(d);
        while (refWindow.size() > (size_t)kRefWindow) refWindow.pop_front();
        refNow = *std::min_element(refWindow.begin(), refWindow.end());
    };

    // Alternating paired duel: challenger and incumbent interleaved so any drift over the duel
    // hits both equally. Returns the ratio (challenger / incumbent); < 1 means faster.
    auto duel = [&](const std::vector<size_t>& challenger, double& outAbs) -> double {
        double cMin = -1.0, iMin = -1.0;
        for (int r = 0; r < kReps; ++r) {
            const double c = evaluate(challenger); ++probes;
            const double i = evaluate(best);       ++probes;
            if (c > 0.0 && (cMin < 0.0 || c < cMin)) cMin = c;
            if (i > 0.0 && (iMin < 0.0 || i < iMin)) iMin = i;
        }
        if (cMin <= 0.0 || iMin <= 0.0) return 1.0;
        outAbs  = cMin;
        refNow  = iMin;                       // the duel just measured the incumbent: reuse it
        refWindow.assign(1, iMin);
        return cMin / iMin;
    };

    // A face has no early exit, so its size is exactly the product of its ladders -- known
    // before the first probe, which makes both the plan and the ETA exact.
    auto faceSize = [&](const Face& f) {
        uint64_t sz = 1;
        for (SieveParam id : f.ids) { const size_t a = axisOf(id); if (a < n) sz *= axis[a].vals->size(); }
        return sz;
    };
    {
        uint64_t cycleProbes = 0;
        std::string plan;
        for (const Face& f : faces) {
            if ((int)f.ids.size() > kRadius) continue;
            cycleProbes += faceSize(f);
            plan += (plan.empty() ? "" : ", ") + std::string(f.name) + "("
                  + std::to_string(faceSize(f)) + ")";
        }
        LOG(LOG_INFO) << "Tuning complex: " << plan;
        LOG(LOG_INFO) << "  " << cycleProbes << " probes per full cycle";
    }

    for (int cycle = 0; cycle < kMaxRounds; ++cycle) {
        bool improved = false;
        for (const Face& f : faces) {
            // --param_test_radius now bounds the FACE DIMENSION: 1 = single-axis sweeps only,
            // 3 = every face below. Faces above the bound are skipped, not truncated.
            if ((int)f.ids.size() > kRadius) continue;

            std::vector<size_t> fax;
            for (SieveParam id : f.ids) { const size_t a = axisOf(id); if (a < n) fax.push_back(a); }
            if (fax.empty()) continue;

            std::vector<size_t> faceBest;
            double faceScore = -1.0;
            uint64_t tried = 0, accepted = 0;
            const uint64_t total = faceSize(f);
            pruned_grid = pruned_cycles = pruned_subcubes = pruned_smem = 0;
            const clk::time_point t0 = clk::now();
            cand = best;

            auto sweep = [&](auto&& self, size_t k) -> void {
                if (k == fax.size()) {
                    ++tried;
                    if (tried % kRefEvery == 0) probeReference();   // track the machine, not the clock
                    const double sc = evaluate(cand);
                    if (sc > 0.0) {
                        ++accepted;
                        const double rel = sc / refNow;             // 1.0 == the incumbent, here, now
                        if (cand != best && (faceScore < 0.0 || rel < faceScore)) {
                            faceScore = rel;
                            faceBest  = cand;
                        }
                    }
                    if (tried % kProgressProbes == 0) {
                        const double el = secs(clk::now() - t0);
                        LOG(LOG_INFO) << "    " << f.name << ": " << tried << "/" << total
                                      << " (" << (100 * tried / total) << " %), " << accepted
                                      << " valid, best x" << faceScore << " (ref "
                                      << refNow << "), " << el << " s elapsed, ~"
                                      << (el * (double)(total - tried) / (double)tried)
                                      << " s left";
                    }
                    return;
                }
                const size_t a = fax[k];
                for (size_t v = 0; v < axis[a].vals->size(); ++v) {
                    cand[a] = v;
                    self(self, k + 1);
                }
                cand[a] = best[a];
            };
            sweep(sweep, 0);
            probes += tried;

            const double el = secs(clk::now() - t0);
            if (faceScore < 0.0) {
                LOG(LOG_INFO) << "  cycle " << cycle << " face " << f.name << ": no valid "
                              << "alternative (" << tried << " probes, " << el << " s)";
                continue;
            }
            double challengerAbs = 0.0;
            const double rel = duel(faceBest, challengerAbs);      // paired, same thermal state
            if (rel >= 1.0 - kJumpMargin) {
                LOG(LOG_INFO) << "  cycle " << cycle << " face " << f.name << ": unchanged ("
                              << accepted << "/" << tried << " valid, " << el
                              << " s); best alternative x" << rel << " in the duel (sweep said x"
                              << faceScore << "), needs x" << (1.0 - kJumpMargin) << "; pruned "
                              << (pruned_grid + pruned_cycles + pruned_subcubes + pruned_smem) << " (grid "
                              << pruned_grid << ", cycles " << pruned_cycles << ", subcubes "
                              << pruned_subcubes << ")";
                continue;
            }
            LOG(LOG_INFO) << "  cycle " << cycle << " face " << f.name << ": x" << rel
                          << " (" << (100.0 * (1.0 - rel)) << " % better than the incumbent "
                          << "measured beside it), " << accepted << "/" << tried << " valid, "
                          << el << " s";
            best       = faceBest;
            bestScore  = challengerAbs;
            chainGain *= rel;
            improved   = true;
            LOG(LOG_INFO) << "    " << describe(best);
            LOG(LOG_INFO) << "    --params11 " << describeDynamic(best);
        }
        if (!improved) {
            LOG(LOG_INFO) << "  converged (no face improves on the record)";
            break;
        }
    }

    // ---- Single-variable descent: one full pass, sweeping each axis over its ENTIRE ladder
    //      (not radius-limited), so a long shallow axis can cross to its far end in one move.
    //      Adoption obeys the same record rule as the rounds above: the axis argmin is re-drawn
    //      and must beat the record by kJumpMargin, so the chain invariant survives this phase.
    LOG(LOG_INFO) << "Single-variable descent over " << n << " axes";
    for (size_t i = 0; i < n; ++i) {
        const size_t before = best[i];
        std::vector<size_t> axBest;
        double axScore = -1.0;
        for (size_t v = 0; v < axis[i].vals->size(); ++v) {
            if (v == before) continue;                 // the incumbent holds the record already
            cand = best;
            cand[i] = v;
            const double sc = evaluate(cand);
            ++probes;
            const double rel = sc / refNow;
            if (sc > 0.0 && (axScore < 0.0 || rel < axScore)) { axScore = rel; axBest = cand; }
        }
        if (axScore < 0.0) {
            LOG(LOG_WARNING) << "  " << axis[i].name << ": no valid value; left at "
                             << (*axis[i].vals)[before];
            continue;
        }
        double challengerAbs = 0.0;
        const double rel = duel(axBest, challengerAbs);
        if (rel >= 1.0 - kJumpMargin) {
            LOG(LOG_INFO) << "  " << axis[i].name << ": " << (*axis[i].vals)[before]
                          << " (unchanged); best alternative x" << rel << " in the duel";
            continue;
        }
        LOG(LOG_INFO) << "  " << axis[i].name << ": " << (*axis[i].vals)[before] << " -> "
                      << (*axis[i].vals)[axBest[i]] << ", x" << rel << " ("
                      << (100.0 * (1.0 - rel)) << " % better)";
        best       = axBest;
        bestScore  = challengerAbs;
        chainGain *= rel;
    }

    // The chain endpoints. Every link was >= kJumpMargin, all measured under the same record
    // rule, so this total is a statement about the geometry rather than about the probe spread.
    // The chain is the product of paired improvements -- the only total that survives a drifting
    // machine. The seed is ALSO re-probed here: comparing that fresh draw against the one taken
    // before the search measures how far the machine itself moved, which is the quantity that
    // makes an absolute before/after comparison meaningless.
    {
        const double seedNow = evaluate(seedIdx);
        ++probes;
        LOG(LOG_INFO) << "Chain: " << (100.0 * (1.0 / chainGain - 1.0))
                      << " % faster than the seed (product of paired steps)";
        if (seedScore > 0.0 && seedNow > 0.0) {
            LOG(LOG_INFO) << "  machine drift: seed measured " << seedScore << " before the "
                          << "search, " << seedNow << " now => "
                          << (100.0 * (seedNow - seedScore) / seedScore)
                          << " % (clock ramp and thermals, NOT geometry)";
        }
        LOG(LOG_INFO) << "  seed   --params11 " << describeDynamic(seedIdx);
    }
    LOG(LOG_INFO) << "Param test complete: " << probes << " probes, OVERALL BEST "
                  << bestScore << " us/poly-position";
    LOG(LOG_INFO) << "  " << describe(best);
    LOG(LOG_INFO) << "  --params11 " << describeDynamic(best);
    exit(0);
}

DeviceSievingController::ParamTestResult
DeviceSievingController::runParamTestLegacy(factoringData& f_data){
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

/**
 * @brief Allocate the double-buffered pinned factor-index staging + its slot events.
 *
 * @p elems_per_slot = batch_size * shc_dim (uint32 indices for one batch). The
 * allocation is kPinnedIndexSlots * elems_per_slot contiguous elements; slot i
 * starts at h_pinned_factor_indices_ + i * pinned_factor_indices_capacity_.
 * Events are timing-disabled — they are pure ordering tokens, never queried for
 * elapsed time. Free-before-realloc keeps the call idempotent (the autotune probe
 * path re-stages with a different batch_size).
 */
void DeviceSievingController::allocatePinnedIndexStaging(size_t elems_per_slot)
{
    releasePinnedIndexStaging();
    pinned_factor_indices_capacity_ = elems_per_slot;
    SIEVE_CUDA_CHECK(cudaMallocHost(&h_pinned_factor_indices_,
                                    elems_per_slot * kPinnedIndexSlots * sizeof(uint32_t)));
    for (uint32_t i = 0; i < kPinnedIndexSlots; ++i) {
        SIEVE_CUDA_CHECK(cudaEventCreateWithFlags(&pinned_h2d_done_[i], cudaEventDisableTiming));
    }
    pinned_slot_ = 0;
}

/// @brief Free the pinned staging and destroy the slot events. Idempotent.
void DeviceSievingController::releasePinnedIndexStaging()
{
    for (uint32_t i = 0; i < kPinnedIndexSlots; ++i) {
        if (pinned_h2d_done_[i]) { cudaEventDestroy(pinned_h2d_done_[i]); pinned_h2d_done_[i] = nullptr; }
    }
    if (h_pinned_factor_indices_) { cudaFreeHost(h_pinned_factor_indices_); h_pinned_factor_indices_ = nullptr; }
    pinned_factor_indices_capacity_ = 0;
    pinned_slot_ = 0;
}

void DeviceSievingController::prepareSievingBatch()
{
    // 1. Host Preparation
    // This function (from prime_algorithms.h) advances the internal f_data state
    // and returns the flattened vector of indices for the next batch_size steps.
    std::vector<uint32_t> next_factor_indices = mpqs::sieve::prepareNextBatchIndices(
								 &f_data, gs_conf.batch_size);

    // 2. Claim a pinned staging slot and WAIT for the H2D that last consumed it.
    //    The launcher's memcpy into the slot happens now (host time) while its
    //    cudaMemcpyAsync executes at GPU time, stream-ordered after the previous
    //    batch's sieve — so without this gate the slot is a single shared cell that
    //    every queued copy reads at its own (much later) execution time. See the
    //    kPinnedIndexSlots comment in the header for the failure mode.
    //    cudaEventSynchronize is a HOST wait on one already-recorded event: it does
    //    not serialize the device and does not drain the stream (unlike the pageable
    //    fallback's implicit sync). With two slots the event waited on was recorded
    //    two batches ago, so while the GPU is the bottleneck the wait is ~0; when it
    //    does bite it is exactly the launch-queue back-pressure that previously
    //    accumulated inside cudaLaunchKernel, relocated one call earlier.
    uint32_t  slot    = pinned_slot_;
    uint32_t* staging = nullptr;
    if (h_pinned_factor_indices_) {
        staging = h_pinned_factor_indices_ + (size_t)slot * pinned_factor_indices_capacity_;
        if (pinned_h2d_done_[slot]) {
            SIEVE_CUDA_CHECK(cudaEventSynchronize(pinned_h2d_done_[slot]));
        }
    }

    // 3. Kernel launcher, handles data transfer of next_factor_indices to device
    //    (staging == nullptr falls back to the pageable copy, unchanged).
    mpqs::sieve::prepareSievingBatch(
	&dev_pointers,
	&next_factor_indices,
	fs_params.shc_dim,
	gs_conf.batch_size,
	stream,
	staging
    );

    // 4. Publish slot reuse. Recorded on `stream` AFTER the H2D (and the polyGen
    //    launch) is enqueued, so completion of this event implies the copy has
    //    already read the slot. Mirrors the graph path's stage_done[] record.
    if (staging) {
        SIEVE_CUDA_CHECK(cudaEventRecord(pinned_h2d_done_[slot], stream));
        pinned_slot_ = (slot + 1u) % kPinnedIndexSlots;
    }
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
	stream,
	use_wide_accumulator_,  // dispatch the wide (uint16) batch kernel when predicate/override selected it
	wide_u8sat_selected_    // within the wide path, dispatch the saturating-uint8 kernel
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
    custom_config_invalid_ = false;   // v1.0.6: a fresh load starts from a clean verdict
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
    // Zero ALL FOUR batch_size fields (mirrors setSievingBatchSize(0)); may be changed a
    // posteriori via setSievingBatchSize(uint32_t). Only init_conf was zeroed before v1.0.6,
    // leaving gs_conf/gms_conf/ss_conf.batch_size INDETERMINATE in a legacy (non-batch) run —
    // harmless while nothing read them there, but validateConfigs' batch predicate does.
    init_conf.batch_size = 0;
    gs_conf.batch_size   = 0;
    gms_conf.batch_size  = 0;
    ss_conf.batch_size   = 0;

    /* generalSievingConfig */
    if (use_wide_accumulator_) {
        // RSA-155 dual-path, test-coverage path. The wide-SB budget reserves a CONSERVATIVE
        // 1024-entry bigPrime floor: solve  sizeof(uint16)*SB + 3*1024*sizeof(int) <= (3/4)*maxShared.
        // bigPrimeStartIndex itself is now set to SB/32 below (mirroring the tuned custom-path
        // geometry in loadPartialCustomConfig) rather than the fixed 1024. Since SB was sized under
        // the 1024-floor budget (2*SB <= 3/4*maxShared - 12288 => SB <= 3/8*maxShared - 6144), the
        // realized sieveAndScan shared-mem 2*SB + 3*(SB/32)*4 = 2.375*SB stays <= ~0.89*maxShared
        // < maxShared, so the config remains feasible (validateConfigs LEQ + wide-feasibility guard
        // both pass). Keeping the budget calc at the 1024 floor is conservative (SB is never larger
        // than the true SB/32 budget would allow), so no shared-mem overflow is possible.
        // accumElemBytes() is 1 for the saturating-uint8 wide accumulator
        // (SB doubles vs uint16, restoring toward narrow's SB) and 2 for uint16.
        gs_conf.sievingBlockSize = pow2leq(
            (uint32_t)(((3u * g_info.maxSharedMemPerBlock) / 4u - 3u * 1024u * sizeof(int)) / accumElemBytes()));
    } else {
        gs_conf.sievingBlockSize = pow2leq((3*g_info.maxSharedMemPerBlock)/4);
    }
    gs_conf.log2_sievingBlockSize = log2(gs_conf.sievingBlockSize);

    gs_conf.num_polysPerSieveCall = std::min(32768u,(1u << fs_params.shc_dim)/2); // Explicit unsigned
    // globalBucketSize: legacy SB/2 unless the --bucket_size_factor ablation knob is set
    // (default 0.0 => SB/2 EXACTLY, byte-identical). Set BEFORE reduceNumPolysToBudget below,
    // so a larger bucket is charged against the VRAM budget (num_polys degrades, no OOM). See
    // the 2026-07-10 A100 uint16 degenerate-baseline root-cause analysis.
    gs_conf.globalBucketSize = computeGlobalBucketSize(gs_conf.sievingBlockSize);
    if (bucket_size_factor_override_ > 0.0) {
        LOG(LOG_WARNING) << "[bucket_size_factor] override active: globalBucketSize="
            << gs_conf.globalBucketSize << " (factor=" << bucket_size_factor_override_
            << " x SB=" << gs_conf.sievingBlockSize << "; legacy SB/2="
            << (gs_conf.sievingBlockSize/2) << ")";
    }

    // WIDE geometry fix: mirror the tuned custom-path bigPrime split (SB/32) instead of the fixed
    // 1024 floor. The un-tuned standard wide default otherwise loads a slow bigPrime geometry that
    // diverges from loadPartialCustomConfig (~:1065, SB/32) — one component of the ~70x wide
    // throughput deficit vs the autotuned narrow path. GATED on use_wide_accumulator_: the narrow
    // (uint8) path — which carries ALL validated records — keeps exactly 1024, byte-for-byte.
    gs_conf.bigPrimeStartIndex = use_wide_accumulator_ ? (gs_conf.sievingBlockSize / 32) : 1024;
    gs_conf.midPrimeStartIndex = 32;
    gs_conf.maxRelationsPerBlock = 64;

    /* globalMetaSieveConfig */
    gms_conf.num_threadsPerBlock = 256;
    // A1 GPU-occupancy fix (WIDE/uint16 path only). The SCATTER kernel
    // globalMetaSieveBatchKernel is memory-latency-bound (long-scoreboard stalls
    // dominate and rise with M); the legacy hardcoded 64-block grid leaves SMs idle
    // on large-SM GPUs (e.g. only 64/108 SMs busy on an A100 -> 12.5% occupancy),
    // starving the machine of the resident warps needed to hide that latency. Mirror
    // init_conf's SM-aware grid (2*pow2geq(SM)) so every SM gets a block. GATED on
    // use_wide_accumulator_: the narrow (uint8, <=RSA-140) path -- which carries ALL
    // validated records -- keeps the exact prior value (64), so it is byte-for-byte
    // unchanged; only the wide (uint16, RSA-150/155) path gets the SM-aware grid.
    gms_conf.num_threadBlocks = use_wide_accumulator_
        ? 2 * pow2geq(g_info.multiProcessorCount)
        : 64;

    gms_conf.maxActiveBucketsTotal = 2 << 15; //first, we choose how many active buckets per thread block we want

    gs_conf.num_sievingBlocksPerSieveCall = (2*fs_params.M)/gs_conf.sievingBlockSize; //next, we choose the sievingBlocksPerCall to cover the whole sieving interval
    if (use_wide_accumulator_) {
        // RSA-155 dual-path: wide SB halved, so num_sievingBlocksPerSieveCall (= 2M/SB, just
        // above) auto-doubles and full [-M,M) coverage is preserved (C1 is not an issue on this
        // derived path). Assert the invariant (holds because M and SB are both powers of two, so
        // SB divides 2M).
        const uint64_t coverage = (uint64_t)gs_conf.num_sievingBlocksPerSieveCall * gs_conf.sievingBlockSize;
        if (coverage != 2ull * fs_params.M) {
            LOG(LOG_ERROR_CRITICAL) << "[C1] WIDE standard-config interval coverage != 2M: intervals("
                << gs_conf.num_sievingBlocksPerSieveCall << ")*SB(" << gs_conf.sievingBlockSize
                << ")=" << coverage << " != 2M(" << (2ull * fs_params.M) << ")";
        }
    }
    // Reduce polys sieved until the global bucket buffer fits with headroom.
    //
    // The bucket buffer (dev_globalBucketEntries, kernel.cu:593) is
    //   num_polysPerSieveCall * num_sievingBlocksPerSieveCall * globalBucketSize * 8 bytes.
    // Substituting num_sievingBlocksPerSieveCall = 2M/sievingBlockSize and
    // globalBucketSize = sievingBlockSize/2, it collapses to num_polysPerSieveCall * M * 8
    // bytes, i.e. it scales linearly with the sieve interval M. At large FB the loop is
    // entered with num_polysPerSieveCall = min(32768, 2^(shc_dim-1)).
    //
    // The bound is 3/4 * totalGlobalMem, NOT totalGlobalMem: the config validator
    // (validateConfig, the LEQ_CHECK below; KernelLaunchValidator::checkGlobalMem)
    // already requires this exact 3/4 headroom, so capping the loop only at totalGlobalMem
    // could leave a config the validator then rejects, AND a buffer consuming nearly all of
    // VRAM with nothing left for the factor base, primeData, and the postprocessing/LP
    // buffers (~13 GB at RSA-140).
    //
    // 32-BIT OVERFLOW FIX (verified root cause of the M=262K/RSA-140 OOM). The ORIGINAL
    // hand-rolled loop condition computed
    //   num_polysPerSieveCall * num_sievingBlocksPerSieveCall * globalBucketSize
    // as a product of three uint32_t fields, evaluated left-to-right in 32-bit BEFORE the
    // trailing * sizeof(uint64_t) promoted to size_t. For any bucket >= 4 GB the uint32_t
    // product wraps. At the M=262144 seed (32768 * 8 * 32768 = 2^33) it wraps to 0, so the
    // condition `0 > 3*total/4` is false, the loop never reduces num_polys, and the
    // (long long)-cast allocation at kernel.cu:593 requests the true ~68.7 GB -> OOM on a
    // 40 GB A100. This is why commit 383438f (which only tightened the RHS budget from
    // totalGlobalMem to 3/4) did nothing for M=262K: the LHS had already wrapped to 0.
    //
    // The single source-of-truth helper (sieve_memory_model.h) computes the bucket term in
    // 64-bit (reduceNumPolysToBudget casts num_polys to uint64_t first), so it correctly
    // reduces num_polys at large buckets. The budget is now sieveBucketBudget(totalGlobalMem,
    // 0, 4, 5) == (4*totalGlobalMem)/5 == 0.80*VRAM (kSieveBudget was flipped from 3/4 to
    // 4/5). Net effect of routing through the helper: the 64-bit LHS corrects the
    // 32-bit overflow that OOMed M=262K/RSA-140; the RHS at 0.80 is a deliberate +5%-of-VRAM
    // loosening shared with the production validator and the autotune OOM guard, so the
    // validated operating points (e.g. M=131072 — well under budget) are unaffected.
    //
    // This loop governs ONLY loadStandardConfig() — the autotune Stage-1 benchmarking seed
    // (whose chosen production params come from independent loadPartialCustomConfig() tuples,
    // not from this npc) and the no-autotune/no-history/no-pinned-params fallback. The
    // validated production path (--autotune_stage1 / pinned params / AutoApply history) calls
    // loadPartialCustomConfig(), which sets num_polysPerSieveCall directly from the tuned
    // tuple (e.g. 512) and never runs this loop.
    //
    // WIDE geometry fix: cap the wide default poly-count seed at 512 (the tuned custom-path value)
    // BEFORE the budget reduction. The un-capped seed min(32768, 2^(shc_dim-1)) drives ~4096-8192
    // polys/call at RSA-155 M, inflating bucket-buffer write traffic ~8-16x — the dominant share of
    // the ~70x wide throughput deficit. reduceNumPolysToBudget (and the seed-clamp below) only ever
    // LOWER num_polys, so a 512 seed is kept or further reduced; num_subCubes (=total/num_polys)
    // absorbs the change so total polys sieved is unaffected. GATED on use_wide_accumulator_: the
    // narrow (uint8) path keeps its full min(32768, 2^(shc_dim-1)) seed, byte-for-byte unchanged.
    if (use_wide_accumulator_)
        gs_conf.num_polysPerSieveCall = std::min(gs_conf.num_polysPerSieveCall, 512u);
    gs_conf.num_polysPerSieveCall = reduceNumPolysToBudget(
        gs_conf.num_polysPerSieveCall,
        gs_conf.num_sievingBlocksPerSieveCall,
        gs_conf.globalBucketSize,
        sieveBucketBudget(g_info.totalGlobalMem, 0, kSieveBudgetNum, kSieveBudgetDen),
        /*min_num_polys=*/0);

    // ----- Autotune OOM-guard knob. DEFAULTED OFF (0). ---------------------------
    // When the autotune Stage-1 seed guard set max_total_sieve_bytes_ (= 0.80*free
    // VRAM minus the postprocessing/LP footprint and the CUDA-context reserve), the
    // bucket-only reduction above is necessary but NOT sufficient: it bounds only
    // dev_globalBucketEntries against 0.80*totalGlobalMem, ignoring the ~GB of
    // persistent FB/primeData + scratch + the pp/LP buffers that share the same
    // device. Halve num_polys further until the ENTIRE sieve footprint (bucket +
    // persistent + scratch, via estimateSieveFootprint) fits the cap. The scratch
    // grid (ss_conf.num_threadBlocks) is set below at min(256, num_polys); mirror
    // that here so the estimate matches the realized geometry. Floor at the
    // validator minimum so >= 1 feasible geometry always survives (the autotune's
    // fallback, autotune.cpp, handles the genuinely-infeasible case).
    last_seed_clamp_ = SeedClampInfo{};
    if (max_total_sieve_bytes_ > 0) {
        auto footprint_total = [&](uint32_t npc) -> uint64_t {
            uint32_t ntb = std::min(256u, npc);
            return estimateSieveFootprint(
                       npc, gs_conf.num_sievingBlocksPerSieveCall,
                       gs_conf.globalBucketSize, gs_conf.sievingBlockSize,
                       ntb, gs_conf.maxRelationsPerBlock,
                       fs_params.fb_size, fs_params.shc_dim).total();
        };
        // Validator minimum num_polys: must satisfy num_threadBlocks*polyBlockSize
        // <= num_polysPerSieveCall; the smallest power-of-two seed the rest of this
        // routine produces a valid config for is 1 (num_subCubes caps it above).
        const uint32_t min_seed = 1u;
        const uint32_t before = gs_conf.num_polysPerSieveCall;
        const uint64_t total_before = footprint_total(before);
        uint32_t npc = before;
        while (npc > min_seed && footprint_total(npc) > max_total_sieve_bytes_) {
            npc >>= 1;
        }
        if (npc != before) {
            gs_conf.num_polysPerSieveCall = npc;
            last_seed_clamp_.clamped          = true;
            last_seed_clamp_.num_polys_before = before;
            last_seed_clamp_.num_polys_after  = npc;
            last_seed_clamp_.total_before     = total_before;
            last_seed_clamp_.total_after      = footprint_total(npc);
            last_seed_clamp_.budget           = max_total_sieve_bytes_;
        }
    }
    // WIDE/uint16 path only: clamp the SCATTER grid to the settled poly count.
    // gms_conf.num_threadBlocks was seeded SM-aware (2*pow2geq(SMs) = 512 on H100 /
    // 256 on A100) at ~:1242, but reduceNumPolysToBudget (and the autotune seed-clamp
    // just above) can VRAM-force num_polysPerSieveCall BELOW that grid at large bf*M
    // (e.g. F=200M/M=16M/bf1.0 on H100: bf*M=16.8e6 > the 9.76e6 np=512 retention bound
    // -> np degrades 512->256). Without this clamp num_polys < num_threadBlocks makes
    // num_polyBlocksPerThreadBlock = (np/num_threadBlocks)/polyBlockSize underflow to 0
    // (POW2_CHECK fail) and LEQ_CHECK(num_threadBlocks <= np) abort config validation --
    // a spurious floor, not an OOM. Clamping the grid to np keeps every SM-derived block
    // that still has a poly to process and lets the pow2/product/LEQ checks pass (both
    // operands are powers of two, so the min is a power of two). Must sit AFTER the two
    // np reductions and BEFORE the polyBlockSize loop / num_polyBlocksPerThreadBlock
    // derivation below. GATED on use_wide_accumulator_: the narrow (uint8, <=RSA-140)
    // path keeps its exact grid (64) and is byte-for-byte unchanged.
    if (use_wide_accumulator_)
        gms_conf.num_threadBlocks = std::min(gms_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall);
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
    applyMetaCycleCap();  // A2: optional SCATTER cycle cap (no-op when meta_cycle_cap_ == 0)
    gms_conf.num_metaSieveCycles = gs_conf.num_sievingBlocksPerSieveCall/gms_conf.num_activeBlocksPerCycle;
    gms_conf.num_activeBucketsPerThreadBlock = gms_conf.num_activeBlocksPerCycle*gms_conf.polyBlockSize;
    gms_conf.sharedMemReq = gms_conf.num_activeBucketsPerThreadBlock * sizeof(int);

    /* sieveAndScanConfig */
    ss_conf.num_threadsPerBlock = 256;
    ss_conf.num_threadBlocks = std::min(256u, gs_conf.num_polysPerSieveCall);
    // Width-aware: the wide (uint16) accumulator path sizes blockEntries at sizeof(uint16_t).
    ss_conf.offsetsInGlobal = (offsets_in_global_ && !use_wide_accumulator_) ? 1u : 0u;
    ss_conf.sharedMemReq = gs_conf.sievingBlockSize * accumElemBytes()
                         + (ss_conf.offsetsInGlobal ? 0u : 3 * gs_conf.bigPrimeStartIndex * sizeof(int));
    applyGatherBlockDimOverride();  // A/B knob: override ss_conf.num_threadsPerBlock (no-op when 0)

    /* processRelationsConfig */
    pr_conf.num_threadsPerBlock = 256;
    pr_conf.num_threadBlocks = pow2geq(g_info.multiProcessorCount);
    //loadPartialCustomConfig(512,32,8,32,32,256,512,1024);//RSA100
    //loadPartialCustomConfig(512,16,8,16,32,256,512,1024);//BASE
}

void DeviceSievingController::loadPartialCustomConfig(uint32_t totalPolys, uint32_t totalIntervals, uint32_t polyBlockSize, uint32_t blocksPerCycle, uint32_t metaB, uint32_t metaT, uint32_t sasB, uint32_t sasT)
{
    custom_config_invalid_ = false;   // v1.0.6: a fresh load starts from a clean verdict
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
    // Zero ALL FOUR batch_size fields (mirrors setSievingBatchSize(0)); may be changed a
    // posteriori via setSievingBatchSize(uint32_t). Only init_conf was zeroed before v1.0.6,
    // leaving gs_conf/gms_conf/ss_conf.batch_size INDETERMINATE in a legacy (non-batch) run —
    // harmless while nothing read them there, but validateConfigs' batch predicate does.
    init_conf.batch_size = 0;
    gs_conf.batch_size   = 0;
    gms_conf.batch_size  = 0;
    ss_conf.batch_size   = 0;

    /* generalSievingConfig */
    if (use_wide_accumulator_) {
        // RSA-155 dual-path: the uint16 accumulator doubles the per-position byte cost, so
        // the shared-memory budget admits a HALVED sieving block (H100 131072->65536, RTX ~16384).
        // Solve  accumBytes*SB + 3*(SB/32)*sizeof(int) <= (3/4)*maxShared
        //   => SB*(accumBytes + 12/32) = SB*((32*accumBytes+12)/32) <= (3/4)*maxShared
        //   => SB <= 24*maxShared/(32*accumBytes+12).
        // Denominator: 76 for uint16 (32*2+12), 44 for the saturating-uint8 path (32*1+12)
        // — u8sat DOUBLES the wide SB toward narrow's. Integer-safe (multiply-before-divide).
        // bigPrimeStartIndex = SB/32 (below) auto-tracks the wide SB.
        const uint32_t wide_sb_den = (uint32_t)(32u * accumElemBytes() + 12u);
        gs_conf.sievingBlockSize = std::min(fs_params.M,
            pow2leq((3u * g_info.maxSharedMemPerBlock * 32u) / (4u * wide_sb_den)));
    } else {
        gs_conf.sievingBlockSize = std::min(fs_params.M, pow2leq((3*g_info.maxSharedMemPerBlock)/4));
    }
    // v1.0.6: --sieve_block_size override, NARROW BATCH ONLY, default 0 = OFF.
    //
    // The derivation above ties SB to the shared-memory budget (SB = min(M, pow2leq(3/4 maxSh))),
    // which on an A100 pins SB = 65536 => ss_conf.sharedMemReq = 90112 B and hence EXACTLY ONE
    // resident GATHER block per SM (measured: launch__occupancy_limit_shared_mem = 1 while
    // launch__occupancy_limit_registers = 2, job 34135902 §0). Shared memory is the SOLE
    // obstruction to co-residency, so SB must be settable BELOW its budget-derived maximum to
    // make the 2-blocks/SM measurement at production M at all. Applied HERE, at the derivation,
    // so :log2_sievingBlockSize (next line) and computeGlobalBucketSize(SB) (below) both follow
    // it automatically -- no second site to keep in sync.
    //
    // Admissible set (CLI-enforced, NEVER silently floored/clamped): power of
    // two (kernel.cu:1006 offsetMask = SB-1; downstream POW2_CHECK), 256 <= N <= M, and the smem
    // sum within maxSharedMemPerBlock (downstream LEQ_CHECK + the narrow-batch occupancy
    // preflight). Scope (--params present, batch mode, narrow accumulator, no --autotune) is
    // enforced at the CLI and re-asserted in validateConfigs.
    //
    // Coverage: SB is a divisor of the sieved interval, so LOWERING it under a fixed
    // num_sievingBlocksPerSieveCall shrinks a single launch's coverage below 2M. That is the
    // silent half-sieve and is caught by the narrowBatchCoverageOk() invariant
    // in validateConfigs; the escape is --params field 2 (numIntervals).
    if (!use_wide_accumulator_ && sb_override_ != 0)
        gs_conf.sievingBlockSize = sb_override_;
    gs_conf.log2_sievingBlockSize = log2(gs_conf.sievingBlockSize);

    gs_conf.num_polysPerSieveCall = totalPolys; // Explicit unsigned
    // WIDE geometry fix: mirror the loadStandardConfig num_polys<=512 clamp (~:964-965) on
    // the CUSTOM apply path. Without it an autotune/pinned tuple could set num_polysPerSieveCall
    // directly from totalPolys (e.g. 1024) and re-inflate the wide bucket-write traffic the 27f810e
    // geometry fix removed (~8-16x at RSA-155 M). num_subCubes (=.../num_polysPerSieveCall, below)
    // absorbs the change so total polys sieved is unaffected. GATED on use_wide_accumulator_ via
    // the shared clampWideNumPolys() helper: the narrow (uint8) path keeps totalPolys verbatim,
    // byte-for-byte unchanged (helper is a NO-OP when wide==false). Host-unit-tested (test
    // wide_num_polys_clamp) against the same helper.
    gs_conf.num_polysPerSieveCall =
        clampWideNumPolys(gs_conf.num_polysPerSieveCall, use_wide_accumulator_);
    // globalBucketSize: legacy SB/2 unless the --bucket_size_factor ablation knob is set
    // (default 0.0 => SB/2 EXACTLY, byte-identical). On this custom/tuned path there is no
    // reduceNumPolysToBudget loop, so an over-large factor is caught by validateConfigs
    // (LEQ_CHECK on bucketEntriesBytes) with a LOG_ERROR_CRITICAL rather than silently OOMing.
    gs_conf.globalBucketSize = computeGlobalBucketSize(gs_conf.sievingBlockSize);
    if (bucket_size_factor_override_ > 0.0) {
        LOG(LOG_WARNING) << "[bucket_size_factor] override active (custom cfg): globalBucketSize="
            << gs_conf.globalBucketSize << " (factor=" << bucket_size_factor_override_
            << " x SB=" << gs_conf.sievingBlockSize << "; legacy SB/2="
            << (gs_conf.sievingBlockSize/2) << ")";
    }

    gs_conf.bigPrimeStartIndex = gs_conf.sievingBlockSize/32;
    // v1.0.6: --sieve_big_prime_start override, NARROW BATCH ONLY, default 0 = OFF.
    //
    // bigPrimeStartIndex splits the factor base: [0, bPSI) is sieved in-block by GATHER (and
    // costs 3*bPSI*4 B of the shared-memory budget, device_sieving_controller.cpp ss_conf.
    // sharedMemReq below), [bPSI, fb_size) is bucketed by SCATTER. The derivation above couples
    // it UNCONDITIONALLY to SB/32, so an SB override silently moves 1024 factor-base primes from
    // GATHER's path onto SCATTER's -- a second, confounded change. This knob CANCELS that
    // coupling, which is the only way to attribute the barrier-density cost (Route A) separately
    // from the prime-transition cost (Route B). bPSI = 1024 on narrow is not a novel value: it is
    // exactly what the standard loader already ships (:1295, `use_wide ? SB/32 : 1024`).
    //
    // Applied HERE, at the derivation, so ss_conf.sharedMemReq (below) follows it automatically,
    // and so does validateConfigs' EQUAL_CHECK, which re-derives from the same two fields.
    //
    // Admissible set (CLI-enforced, NEVER silently floored/clamped):
    // 32 < N <= fb_size. The STRICT lower bound is midPrimeStartIndex = 32 (next line): the
    // mid-prime loops (kernel.cu:1293, :1371) run [midPrimeStart, bPSI), so N <= 32 inverts the
    // range and silently drops the whole mid-prime band. Power-of-two is NOT required -- every
    // consumer is a grid-stride loop over [0,bPSI), [32,bPSI) or [bPSI,fb) and the GATHER smem
    // layout (kernel.cu:1152-1158) is plain pointer arithmetic. The smem sum is enforced
    // downstream (LEQ_CHECK on ss_conf.sharedMemReq + the narrow-batch occupancy preflight).
    if (!use_wide_accumulator_ && bpsi_override_ != 0)
        gs_conf.bigPrimeStartIndex = bpsi_override_;
    gs_conf.midPrimeStartIndex = 96;
    gs_conf.maxRelationsPerBlock = 64;

    /* globalMetaSieveConfig */
    gms_conf.num_threadsPerBlock = metaT;
    gms_conf.num_threadBlocks = metaB;

    gms_conf.maxActiveBucketsTotal = 2 << 15; //first, we choose how many active buckets per thread block we want

    if (use_wide_accumulator_) {
        // [C1 - CRITICAL] Wide SB = uint8 SB / 2, so reusing the fixed tuple's totalIntervals
        // would cover only [-M,0) — half the interval, ~50% SILENT yield loss (a naive
        // nonzero-smooths gate still passes on the sieved half). Override to preserve full
        // [-M,M): intervals = 2M/wideSB (doubles vs the uint8 tuple value). num_metaSieveCycles
        // (below) then doubles automatically; the existing validateConfigs EQUAL_CHECK
        // (num_metaSieveCycles*num_activeBlocksPerCycle == num_sievingBlocksPerSieveCall)
        // enforces exact divisibility.
        gs_conf.num_sievingBlocksPerSieveCall = (2u * fs_params.M) / gs_conf.sievingBlockSize;
    } else {
        gs_conf.num_sievingBlocksPerSieveCall = totalIntervals; //next, we choose the sievingBlocksPerCall to cover the whole sieving interval
    }
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
    // [C1] Interval-coverage invariant (scoped by path). The batch kernel sieves exactly
    // num_sievingBlocksPerSieveCall * sievingBlockSize of [-M,M) per launch.
    {
        const uint64_t coverage =
            (uint64_t)gs_conf.num_sievingBlocksPerSieveCall * gs_conf.sievingBlockSize;
        const uint64_t two_M = 2ull * fs_params.M;
        if (use_wide_accumulator_) {
            // WIDE: hard == 2M in a single batch. This is the C1 catch — it fires if the wide
            // coverage collapses to [-M,0) (half-interval yield loss).
            if (coverage != two_M || gs_conf.num_sievingBlockBatches != 1) {
                LOG(LOG_ERROR_CRITICAL) << "[C1] WIDE interval coverage invariant violated: intervals("
                    << gs_conf.num_sievingBlocksPerSieveCall << ")*SB(" << gs_conf.sievingBlockSize
                    << ")=" << coverage << " != 2M(" << two_M << "), num_sievingBlockBatches="
                    << gs_conf.num_sievingBlockBatches;
            }
        } else {
            // uint8 / shared: the sieve covers [-M,M) across num_sievingBlockBatches launches
            // (the batch loops in sieveDataBatch / runSievingBatch / benchmarkSievingConfig
            // offset each launch by intervals*SB), so one launch legitimately covers only a
            // FRACTION of 2M when an --autotune geometry picks intervals*SB < 2M (giving
            // num_sievingBlockBatches >= 2, e.g. Turing intervals=8/SB=32768 at M=262144 -> 2
            // batches). A single launch may also OVER-cover (sievingBlockSize == M =>
            // intervals*SB > 2M, with the num_sievingBlockBatches 0->1 clamp above); [C1] still
            // only asserts the TOTAL here, so valid multi-batch configs do not false-trip this
            // critical guard while a genuine coverage shortfall is still caught. Over-coverage
            // is rejected one level up, by the two-sided coverage invariant in validateConfigs()
            // — it sieves past [-M,M), which is cheap per position and barren in relations.
            const uint64_t total_coverage =
                (uint64_t)gs_conf.num_sievingBlockBatches * coverage;
            if (total_coverage < two_M) {
                LOG(LOG_ERROR_CRITICAL) << "[C1] total interval coverage below 2M: batches("
                    << gs_conf.num_sievingBlockBatches << ")*intervals("
                    << gs_conf.num_sievingBlocksPerSieveCall << ")*SB(" << gs_conf.sievingBlockSize
                    << ")=" << total_coverage << " < 2M(" << two_M << ")";
            }
        }
    }
    //incrase polyBlockSize until we run out of poly or we have too many active buckets
    gms_conf.polyBlockSize = polyBlockSize;
    //calculations that automatically follow the above definitions
    gms_conf.log2_polyBlockSize = log2(gms_conf.polyBlockSize);
    // v1.0.6: EXACTNESS-CHECKED num_polyBlocksPerThreadBlock.
    //
    // The batch SCATTER kernel partitions the polynomials with a FIXED trip count and no
    // `if (id < n)` guard, so the identity
    //     metaGridDim * num_polyBlocksPerThreadBlock * polyBlockSize == num_polysPerSieveCall
    // must hold EXACTLY: under-coverage silently drops polynomials from the large-prime
    // scatter (yield collapse with no error), over-coverage writes past the bucket array.
    // The previous derivation (np/metaB)/polyBlockSize floors TWICE, which in the
    // power-of-two world could never lose a remainder but in the SM-aligned world can
    // (e.g. np=864, metaB=100, pbs=8 -> 864/100 = 8, 8/8 = 1, and 100*1*8 = 800 != 864).
    // A floored derivation would therefore manufacture an invalid geometry and hand it to
    // validateConfigs' EQUAL_CHECK as a puzzling equality failure. Compute it exactly and
    // reject the tuple HERE, naming the divisibility that failed.
    const uint32_t meta_divisor =
        gms_conf.num_threadBlocks * gms_conf.polyBlockSize;
    if (meta_divisor == 0 ||
        (gs_conf.num_polysPerSieveCall % meta_divisor) != 0) {
        custom_config_invalid_ = true;
        LOG(LOG_ERROR_CRITICAL)
            << "[Sieve] Invalid pinned geometry: metaGridDim(" << gms_conf.num_threadBlocks
            << ") * polyBlockSize(" << gms_conf.polyBlockSize << ") = " << meta_divisor
            << " does not divide num_polysPerSieveCall(" << gs_conf.num_polysPerSieveCall
            << ") exactly. The SCATTER partition is unguarded, so a floored"
               " num_polyBlocksPerThreadBlock would silently drop or over-write polynomials."
               " Choose num_polysPerSieveCall as an exact multiple of metaGridDim*polyBlockSize"
               " (e.g. 864 = 108*8*1 on a 108-SM A100).";
        gms_conf.num_polyBlocksPerThreadBlock = 0;   // NON0_CHECK in validateConfigs also fires
    } else {
        gms_conf.num_polyBlocksPerThreadBlock = gs_conf.num_polysPerSieveCall / meta_divisor;
    }
    // log2_num_polyBlocksPerThreadBlock is DEAD STATE: it is written here and shipped inside
    // globalMetaSieveConfig, but NO kernel reads it (grep over src/sieve/*.cu* confirms). For a
    // non-power-of-two num_polyBlocksPerThreadBlock this floor-log2 is therefore merely stale,
    // not wrong. Kept (and kept in sync) so the config struct's layout and every debug dump of
    // it stay unchanged; removing the field is a separate cleanup.
    gms_conf.log2_num_polyBlocksPerThreadBlock = log2(gms_conf.num_polyBlocksPerThreadBlock);
    applyMetaCycleCap();  // A2: optional SCATTER cycle cap (no-op when meta_cycle_cap_ == 0)
    gms_conf.num_metaSieveCycles = gs_conf.num_sievingBlocksPerSieveCall/gms_conf.num_activeBlocksPerCycle;
    gms_conf.num_activeBucketsPerThreadBlock = gms_conf.num_activeBlocksPerCycle*gms_conf.polyBlockSize;
    gms_conf.sharedMemReq = gms_conf.num_activeBucketsPerThreadBlock * sizeof(int);

    /* sieveAndScanConfig */
    ss_conf.num_threadsPerBlock = sasT;
    ss_conf.num_threadBlocks = sasB;
    // Width-aware: the wide (uint16) accumulator path sizes blockEntries at sizeof(uint16_t).
    ss_conf.offsetsInGlobal = (offsets_in_global_ && !use_wide_accumulator_) ? 1u : 0u;
    ss_conf.sharedMemReq = gs_conf.sievingBlockSize * accumElemBytes()
                         + (ss_conf.offsetsInGlobal ? 0u : 3 * gs_conf.bigPrimeStartIndex * sizeof(int));
    applyGatherBlockDimOverride();  // A/B knob: override ss_conf.num_threadsPerBlock (no-op when 0)

    /* processRelationsConfig */
    pr_conf.num_threadsPerBlock = 512;
    pr_conf.num_threadBlocks = pow2geq(g_info.multiProcessorCount);
}

void DeviceSievingController::ensureTailInvP() const
{
    const size_t n = f_data.factorBase.size();
    if (tail_inv_p_.size() == n + 1) return;
    tail_inv_p_.assign(n + 1, 0.0);
    for (size_t i = n; i-- > 0; )
        tail_inv_p_[i] = tail_inv_p_[i + 1]
                       + (f_data.factorBase[i] ? 1.0 / (double)f_data.factorBase[i] : 0.0);
}

double DeviceSievingController::expectedBucketEntries(uint32_t sievingBlockSize,
                                                      uint32_t bigPrimeStartIndex) const
{
    ensureTailInvP();
    if (bigPrimeStartIndex >= tail_inv_p_.size()) return 0.0;
    return 2.0 * (double)sievingBlockSize * tail_inv_p_[bigPrimeStartIndex];
}

uint32_t DeviceSievingController::autoGlobalBucketSize(uint32_t sievingBlockSize,
                                                       uint32_t bigPrimeStartIndex,
                                                       uint32_t numPolys,
                                                       uint32_t numIntervals) const
{
    const double mu = expectedBucketEntries(sievingBlockSize, bigPrimeStartIndex);
    if (mu <= 0.0) return 32u;                      // nothing is bucketed at this bPSI
    const double N = (double)std::max(1u, numPolys) * (double)std::max(1u, numIntervals);
    const double z = std::sqrt(2.0 * std::log(std::max(N, 2.0)));
    const double peak = mu + z * std::sqrt(mu);
    const uint64_t slots = (uint64_t)std::ceil(peak);
    const uint64_t rounded = ((slots + 31u) / 32u) * 32u;   // keep the stride 32-aligned
    return (uint32_t)std::min<uint64_t>(rounded, 0xFFFFFFFFull);
}

void DeviceSievingController::loadPartialCustomConfigDynamic(const ParamSet& p)
{
    custom_config_invalid_ = false;
    auto pow2leq = [](uint32_t x) -> uint32_t {
        if (x < 1) return 0;
        return 1 << (31 - std::countl_zero(x));
    };
    auto pow2geq = [pow2leq](uint32_t x) -> uint32_t {
        uint32_t pow2 = pow2leq(x);
        return (pow2 == x) ? pow2 : pow2 << 1;
    };
    auto log2 = [](uint32_t x) -> uint32_t {
        if (x <= 0) return 0;
        return 31 - std::countl_zero(x);
    };
    // A parameter present in `p` is hard-set; an absent one keeps the derivation above it.
    auto set = [&p](SieveParam k, uint32_t& dst) { if (p[k]) dst = *p[k]; };

    /* initConfig */
    init_conf.num_threadsPerBlock = 512;
    init_conf.num_threadBlocks = 2*pow2geq(g_info.multiProcessorCount);
    init_conf.batch_size = 0;
    gs_conf.batch_size   = 0;
    gms_conf.batch_size  = 0;
    ss_conf.batch_size   = 0;

    /* generalSievingConfig */
    if (use_wide_accumulator_) {
        const uint32_t wide_sb_den = (uint32_t)(32u * accumElemBytes() + 12u);
        gs_conf.sievingBlockSize = std::min(fs_params.M,
            pow2leq((3u * g_info.maxSharedMemPerBlock * 32u) / (4u * wide_sb_den)));
    } else {
        gs_conf.sievingBlockSize = std::min(fs_params.M, pow2leq((3*g_info.maxSharedMemPerBlock)/4));
    }
    set(P_SIEVING_BLOCK_SIZE, gs_conf.sievingBlockSize);
    gs_conf.log2_sievingBlockSize = log2(gs_conf.sievingBlockSize);

    gs_conf.num_polysPerSieveCall =
        clampWideNumPolys(gs_conf.num_polysPerSieveCall, use_wide_accumulator_);
    set(P_SUB_CUBE_SIZE, gs_conf.num_polysPerSieveCall);

    gs_conf.bigPrimeStartIndex = gs_conf.sievingBlockSize/32;
    set(P_BIG_PRIME_START, gs_conf.bigPrimeStartIndex);
    gs_conf.midPrimeStartIndex = 96;
    set(P_MID_PRIME_START, gs_conf.midPrimeStartIndex);
    gs_conf.maxRelationsPerBlock = 64;

    /* globalMetaSieveConfig */
    set(P_META_BLOCK_DIM, gms_conf.num_threadsPerBlock);
    set(P_META_GRID_DIM,  gms_conf.num_threadBlocks);
    gms_conf.maxActiveBucketsTotal = 2 << 15;

    if (use_wide_accumulator_) {
        gs_conf.num_sievingBlocksPerSieveCall = (2u * fs_params.M) / gs_conf.sievingBlockSize;
    }
    // numIntervals is pinned by the coverage invariant (numIntervals * SB == 2M), and
    // blocksPerCycle by the one-cycle rule (num_metaSieveCycles = numIntervals/blocksPerCycle;
    // anything above 1 re-walks the whole factor base above bigPrimeStart). Both are therefore
    // FUNCTIONS of sievingBlockSize, derived here unless explicitly pinned, so that moving SB
    // costs one degree of freedom rather than three.
    if (!use_wide_accumulator_ && gs_conf.sievingBlockSize)
        gs_conf.num_sievingBlocksPerSieveCall =
            (uint32_t)((2ull * fs_params.M) / gs_conf.sievingBlockSize);
    set(P_NUM_INTERVALS, gs_conf.num_sievingBlocksPerSieveCall);
    gms_conf.num_activeBlocksPerCycle = gs_conf.num_sievingBlocksPerSieveCall;

    // Bucket sizing, once SB / bPSI / np / numIntervals are all resolved. An explicit
    // --bucket_size_factor is operator intent and wins; otherwise the bucket is sized to the
    // PREDICTED PEAK occupancy, so it is neither so small that large-prime hits are silently
    // discarded nor so large that VRAM is spent on unreachable headroom. Whether it fits is
    // decided by the VRAM budget check in validateConfigs(), not here.
    gs_conf.globalBucketSize = (bucket_size_factor_override_ > 0.0)
        ? computeGlobalBucketSize(gs_conf.sievingBlockSize)
        : autoGlobalBucketSize(gs_conf.sievingBlockSize, gs_conf.bigPrimeStartIndex,
                               gs_conf.num_polysPerSieveCall,
                               gs_conf.num_sievingBlocksPerSieveCall);

    gs_conf.num_subCubes = gs_conf.num_polysPerSieveCall
        ? std::min(32768u,(1u << fs_params.shc_dim)/2)/gs_conf.num_polysPerSieveCall : 0;

    set(P_BLOCKS_PER_CYCLE, gms_conf.num_activeBlocksPerCycle);

    {
        const uint64_t cover = (uint64_t)gs_conf.sievingBlockSize
                             * gs_conf.num_sievingBlocksPerSieveCall;
        gs_conf.num_sievingBlockBatches = cover ? (uint32_t)((2ull*fs_params.M)/cover) : 1;
        if (gs_conf.num_sievingBlockBatches == 0) gs_conf.num_sievingBlockBatches = 1;
    }

    set(P_POLY_BLOCK_SIZE, gms_conf.polyBlockSize);
    gms_conf.log2_polyBlockSize = log2(gms_conf.polyBlockSize);

    // EXACTNESS-CHECKED num_polyBlocksPerThreadBlock, as in loadPartialCustomConfig: the
    // SCATTER kernel partitions the polynomials with a fixed trip count and no bound guard,
    // so metaGridDim * num_polyBlocksPerThreadBlock * polyBlockSize == num_polysPerSieveCall
    // must hold exactly. Reject here, naming the divisibility, rather than leave a floored
    // value to surface downstream as a puzzling EQUAL_CHECK failure.
    const uint32_t meta_divisor = gms_conf.num_threadBlocks * gms_conf.polyBlockSize;
    if (meta_divisor == 0 || (gs_conf.num_polysPerSieveCall % meta_divisor) != 0) {
        custom_config_invalid_ = true;
        gms_conf.num_polyBlocksPerThreadBlock = 0;   // NON0_CHECK also fires
    } else {
        gms_conf.num_polyBlocksPerThreadBlock = gs_conf.num_polysPerSieveCall / meta_divisor;
    }
    gms_conf.log2_num_polyBlocksPerThreadBlock = log2(gms_conf.num_polyBlocksPerThreadBlock);

    applyMetaCycleCap();
    gms_conf.num_metaSieveCycles = gms_conf.num_activeBlocksPerCycle
        ? gs_conf.num_sievingBlocksPerSieveCall/gms_conf.num_activeBlocksPerCycle : 0;
    gms_conf.num_activeBucketsPerThreadBlock = gms_conf.num_activeBlocksPerCycle*gms_conf.polyBlockSize;
    gms_conf.sharedMemReq = gms_conf.num_activeBucketsPerThreadBlock * sizeof(int);

    /* sieveAndScanConfig */
    set(P_SAS_BLOCK_DIM, ss_conf.num_threadsPerBlock);
    set(P_SAS_GRID_DIM,  ss_conf.num_threadBlocks);
    ss_conf.offsetsInGlobal = (offsets_in_global_ && !use_wide_accumulator_) ? 1u : 0u;
    ss_conf.sharedMemReq = gs_conf.sievingBlockSize * accumElemBytes()
                         + (ss_conf.offsetsInGlobal ? 0u : 3 * gs_conf.bigPrimeStartIndex * sizeof(int));
    applyGatherBlockDimOverride();

    /* processRelationsConfig */
    pr_conf.num_threadsPerBlock = 512;
    pr_conf.num_threadBlocks = pow2geq(g_info.multiProcessorCount);
}


/*
 * A2 meta-sieve SCATTER cycle cap (--sieve_meta_cycle_cap, default 0 = OFF).
 *
 * The SCATTER kernels (globalMetaSieveKernel / globalMetaSieveBatchKernel) write each
 * large-prime hit into a per-(poly, sievingBlock) bucket. Per cycle, a thread's write
 * destinations span num_activeBlocksPerCycle * globalBucketSize contiguous bucket slots;
 * with the default num_activeBlocksPerCycle == num_sievingBlocksPerSieveCall == 2M/SB this
 * spread grows linearly with M, degrading L1 locality (A100 ncu: L1 73%->56%, long-scoreboard
 * stalls x4.1 from M=256K->1M on flat DRAM bytes). Capping C := num_activeBlocksPerCycle at a
 * fixed constant and running num_metaSieveCycles = blocks/C cycles bounds the spread
 * independent of M.
 *
 * Correctness (coverage preservation): the kernels already implement the cycle partition —
 * cycle c sieves the contiguous block slice [c*C, (c+1)*C) via
 *   currentStart = sieveIntervalStart + c*C*SB          (kernel.cu:981 batch, :384 legacy)
 * keeping exactly the hits with sievingBlockHit = (off-currentStart)/SB < C (kernel.cu:1038),
 * and addresses the bucket as
 *   globalBucketId = (polyIdx*num_metaSieveCycles + c)*C + hit = polyIdx*blocks + (c*C + hit)
 * (kernel.cu:1021/:1051) — i.e. the flat (poly, absolute-block) layout is INDEPENDENT of the
 * cycle split, so the GATHER consumer needs no change. The union over c of the slices is
 * [0, cycles*C) = [0, blocks) exactly, because the division blocks/C is exact (below).
 *
 * Exactness of the split: the loader-derived C is a power of two (validateConfigs POW2_CHECK)
 * that divides num_sievingBlocksPerSieveCall (EQUAL_CHECK cycles*C == blocks). The cap is
 * rounded DOWN to a power of two and clamped to <= the derived C; a smaller power of two
 * divides the derived C and hence blocks, so blocks/C' is exact and no partial ("remainder")
 * last cycle can exist. If the requested cap >= derived C, this is a no-op. If
 * meta_cycle_cap_ == 0 (default), the function returns immediately: byte-identical geometry.
 *
 * Must run before num_metaSieveCycles / num_activeBucketsPerThreadBlock / sharedMemReq are
 * derived (they follow from C), and after the polyBlockSize sizing so poly batching, bucket
 * buffer sizes (num_polys*blocks*bucketSize — split-invariant) and everything else match the
 * uncapped baseline.
 */
void DeviceSievingController::applyMetaCycleCap()
{
    if (meta_cycle_cap_ == 0) return;  // OFF: exact legacy behavior
    const uint32_t cap_pow2 = 1u << (31 - std::countl_zero(meta_cycle_cap_));  // pow2_floor(cap), cap >= 1
    if (cap_pow2 < gms_conf.num_activeBlocksPerCycle) {
        const uint32_t cycles = gs_conf.num_sievingBlocksPerSieveCall / cap_pow2;  // exact (pow2 chain)
        LOG(LOG_INFO) << "[A2] meta-cycle cap active: num_activeBlocksPerCycle "
                      << gms_conf.num_activeBlocksPerCycle << " -> " << cap_pow2
                      << " (requested " << meta_cycle_cap_ << "), num_metaSieveCycles -> " << cycles;
        gms_conf.num_activeBlocksPerCycle = cap_pow2;
    } else {
        LOG(LOG_INFO) << "[A2] meta-cycle cap " << meta_cycle_cap_ << " >= derived num_activeBlocksPerCycle "
                      << gms_conf.num_activeBlocksPerCycle << " -- no-op";
    }
}

void DeviceSievingController::applyGatherBlockDimOverride()
{
    if (gather_block_dim_override_ == 0) return;  // OFF: keep loader-derived blockDim (legacy)
    // blockDim is result-invariant for the sieve-and-scan kernels (accumulator sized per
    // sieving-block, not per-thread; all loops stride by blockDim.x), so only occupancy
    // changes. The value is a power of two in [32,1024], validated at the CLI; here we only
    // reassign ss_conf.num_threadsPerBlock — sharedMemReq and every buffer size are
    // blockDim-independent, so nothing else needs re-deriving.
    LOG(LOG_INFO) << "[GATHER] sieve-and-scan blockDim override: "
                  << ss_conf.num_threadsPerBlock << " -> " << gather_block_dim_override_
                  << " (result-invariant occupancy knob)";
    ss_conf.num_threadsPerBlock = gather_block_dim_override_;
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
    LOG(LOG_DEBUG_1) << "num metaSieveCycles: " << gms_conf.num_metaSieveCycles;
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
    std::cout << "num metaSieveCycles: " << gms_conf.num_metaSieveCycles << std::endl;
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
    // Silent mode (validateConfigsSilent): verdict unchanged, diagnostics suppressed.
    #define VLOG(lvl) if (validate_quiet_) {} else LOG(lvl)
    auto isPowerOfTwo = [](uint32_t n) {
        return n > 0 && (n & (n - 1)) == 0;
    };

    bool validFlag = true;

    // v1.0.6: a pinned tuple the CONFIG LOADER already proved invalid (an
    // inexact SCATTER partition — see loadPartialCustomConfig) fails here too, so the single
    // escalation path (LOG_ERROR_CRITICAL + throw at the orchestrator) is preserved. Consumed
    // and cleared: both loaders reset it, so this only guards the config just loaded.
    if (custom_config_invalid_) {
        validFlag = false;
        VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: the pinned kernel geometry was rejected by "
                                   "the config loader (see the [Sieve] Invalid pinned geometry "
                                   "message above).";
        custom_config_invalid_ = false;
    }

    #define POW2_CHECK(var, validFlag) \
    do { \
        if (!isPowerOfTwo(var)) { \
            validFlag = false; \
            VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #var << " must be a power of 2. " << "Current value: " << (var) << std::endl; \
        } \
    } while (0)

    #define EQUAL_CHECK(var1, var2, validFlag) \
    do { \
        if ( (var1) != (var2) ) { \
            validFlag = false; \
            VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #var1 << " must be equal to " << #var2 << ". Current values: " << (var1) << ", " << (var2) << std::endl; \
        } \
    } while (0)

    #define LEQ_CHECK(var1, var2, validFlag) \
    do { \
        if (!( (var1) <= (var2) )) { \
            validFlag = false; \
            VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #var1 << " must be less than or equal to " << #var2 << ". Current values: " << (var1) << ", " << (var2) << std::endl; \
        } \
    } while (0)

    #define NON0_CHECK(var, validFlag) \
    do { \
    if ((var) == 0) { \
        validFlag = false; \
        VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #var << " must be > 0." << std::endl; \
    } \
    } while (0)


    // ---- v1.0.6: SM-aligned launch geometry, NARROW BATCH PATH ONLY ----
    // The power-of-two requirement on {num_polysPerSieveCall, metaGridDim, sasGridDim,
    // num_polyBlocksPerThreadBlock} is NOT mathematical -- it is a host convention. The only
    // kernel-side pow2 dependencies on these four quantities live in the LEGACY GATHER mask
    // (kernel.cu: polyId & (np-1)) and the legacy subCube partition; the BATCH kernels read
    // neither (batch SCATTER never reads gridDim.x and has no grid mask; batch GATHER derives
    // its trip count as np/gridDim.x and composes polyIdPrefix | gray(poly), which needs only
    // the CHUNK np/sasGridDim to be a power of two -- see G2 below). Relaxing them lets the
    // grids be aligned to the device's SM count so both kernels run exact waves; on a 108-SM
    // A100 the pow2 grid metaGridDim=64 leaves 44/108 SMs idle for a kernel that is 34.1 % of
    // RSA-100 GPU time (measured under ncu, 2026-08-22).
    //
    // The relaxation is deliberately narrow:
    //   * batch only  -- legacy (batch_size == 0) keeps mandatory pow2, so the legacy sieve
    //                    kernels in kernel.cu are never touched;
    //   * narrow only -- the wide (uint16 / u8sat) path keeps mandatory pow2: its bucket VRAM
    //                    is proportional to np and its production regime (RSA-150/155) already
    //                    rides the retention boundary at np = 512.
    // For any pow2 tuple the relaxed predicates are strict supersets of the old ones and
    // G1-G3 are tautologies, so NO existing configuration changes behaviour.
    // The normative admissible set is the predicate implemented below (v1.0.6, 2026-08-23).
    // Batch-mode predicate. setSievingBatchSize() sets all four *_conf.batch_size fields and
    // runs BEFORE this call (orchestrator: setSievingBatchSize -> printConfigs -> validateConfigs);
    // in LEGACY runs it is never called, so the loaders' explicit zeroing of all four (mirroring
    // setSievingBatchSize(0)) is what makes this predicate well-defined. Reading an unzeroed
    // field here would let a legacy run pick up a stale non-zero batch size and silently accept
    // an SM-aligned tuple that the legacy GATHER mask cannot execute.
    const bool relaxed_geometry = (init_conf.batch_size > 0 && gs_conf.batch_size > 0)
                               && !use_wide_accumulator_;

    // ---- v1.0.6: narrow-batch geometry-override scope + coverage invariant ----
    // Evaluated HERE and not in the loader because `relaxed_geometry` depends on batch_size,
    // which setSievingBatchSize() only establishes AFTER the loader has run
    // (orchestrator: loader -> setSievingBatchSize -> printConfigs -> validateConfigs).
    //
    // SCOPE REJECTION. The overrides are narrow-BATCH-only, exactly as the v1.0.6 non-pow2
    // geometry is, and are rejected LOUDLY (never silently ignored) anywhere else:
    //   * legacy (batch_size == 0): the legacy GATHER masks with (np-1) and partitions subCubes
    //     on the pow2 world; more importantly the legacy sieve kernels in kernel.cu must stay
    //     byte-identical there, and the legacy path was never measured.
    //   * wide (uint16 / u8sat): its SB is already halved off a DIFFERENT budget solve and its
    //     bucket VRAM rides the RSA-150/155 retention boundary. Nothing is claimed or changed
    //     for the wide path in v1.0.6.
    // The loaders additionally gate both overrides on !use_wide_accumulator_ at the point of
    // use, so a wide run cannot even reach an overridden field; this check is what makes the
    // attempt AUDIBLE rather than quietly inert.
    if (sb_override_ != 0 || bpsi_override_ != 0) {
        if (use_wide_accumulator_) {
            validFlag = false;
            VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: --sieve_block_size / "
                "--sieve_big_prime_start are NARROW-BATCH ONLY and were rejected on the WIDE "
                "(uint16/u8sat) accumulator path. Requested sieve_block_size="
                << sb_override_ << ", sieve_big_prime_start=" << bpsi_override_
                << ". Drop the flags, or force the narrow path with --sieve_accumulator u8."
                << std::endl;
        } else if (!(init_conf.batch_size > 0 && gs_conf.batch_size > 0)) {
            validFlag = false;
            VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: --sieve_block_size / "
                "--sieve_big_prime_start are NARROW-BATCH ONLY and were rejected in LEGACY mode "
                "(--sieve_batch_size 0). Requested sieve_block_size=" << sb_override_
                << ", sieve_big_prime_start=" << bpsi_override_
                << ". Set --sieve_batch_size > 0." << std::endl;
        }
    }

    // BAND-ORDERING INVARIANT. The GATHER kernel sizes its shared offsets1/offsets2/primes
    // arrays at bigPrimeStartIndex entries each, and walks [smallPrimesUsed, midPrimeStart)
    // then [midPrimeStart, bigPrimeStart). midPrimeStart ABOVE bigPrimeStart makes the first
    // loop index past the end of its array into the next one — garbage log values, silently.
    // Below smallPrimesUsed it double-counts the primes the mask already applied.
    LEQ_CHECK(gs_conf.midPrimeStartIndex, gs_conf.bigPrimeStartIndex, validFlag);
    LEQ_CHECK(fs_params.smallPrimesUsed, gs_conf.midPrimeStartIndex, validFlag);

    // COVERAGE INVARIANT (narrow, BOTH paths): numIntervals * sievingBlockSize == 2M, exactly.
    // Per sieve call the kernel sweeps coverage = numIntervals * sievingBlockSize positions from
    // ds_params.startIndex, and [-M,M) is 2M wide. Both deviations are defects:
    //
    //   coverage > 2M — num_sievingBlockBatches floors to 0 and is clamped to 1, so the call
    //     runs off the end of [-M,M) into positions where Q is far above the threshold: cheap
    //     per position in kernel time, nearly barren in relations. Nothing re-normalises for
    //     that, so the config looks fast to any timing probe while yielding almost nothing —
    //     which is exactly how a parameter search gets walked into it.
    //
    //   coverage < 2M — on the BATCH path runSievingBatch never advances ds_params.startIndex
    //     (the per-batch re-offsetting that sieveFullCubeSnapshot / sieveFullCube /
    //     benchmarkSievingConfig each perform is absent there), so num_sievingBlockBatches is
    //     fictitious and the sieve silently covers only part of [-M,M). The LEGACY loops do
    //     advance it and would cover 2M across several launches, but then the per-call work is
    //     a config-dependent fraction of the interval, which is not a quantity a probe can
    //     compare across candidates — so equality is required there too.
    if (!use_wide_accumulator_) {
        const uint64_t coverage = (uint64_t)gs_conf.num_sievingBlocksPerSieveCall
                                * (uint64_t)gs_conf.sievingBlockSize;
        const uint64_t two_M    = 2ull * (uint64_t)fs_params.M;
        if (coverage != two_M) {
            validFlag = false;
            const uint64_t required = gs_conf.sievingBlockSize == 0 ? 0
                : two_M / gs_conf.sievingBlockSize;
            VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: NARROW interval coverage must equal 2M. "
                "numIntervals(" << gs_conf.num_sievingBlocksPerSieveCall << ") * sievingBlockSize("
                << gs_conf.sievingBlockSize << ") = " << coverage
                << (coverage > two_M ? " > 2M(" : " < 2M(") << two_M << "). "
                << (coverage > two_M
                        ? "Sieving past [-M,M) is fast per position and barren in relations. "
                        : "Part of [-M,M) would go unsieved. ")
                << "Required: numIntervals = " << required << " (--params field 2)." << std::endl;
        }
    }

    // Keeps the OLD predicate textually present on the non-relaxed branch.
    #define POW2_UNLESS_CHECK(relaxed, var, validFlag) \
    do { \
        if (relaxed) { NON0_CHECK(var, validFlag); } else { POW2_CHECK(var, validFlag); } \
    } while (0)

    // DIVIDES_CHECK / POW2_VALUE_CHECK carry the two GATHER decomposition invariants that the
    // pow2 world used to guarantee for free (G1/G2 below).
    #define DIVIDES_CHECK(divisor, dividend, validFlag) \
    do { \
        if ((divisor) == 0 || ((dividend) % (divisor)) != 0) { \
            validFlag = false; \
            VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << #divisor << " must divide " << #dividend \
                                    << " exactly. Current values: " << (divisor) << ", " << (dividend) << std::endl; \
        } \
    } while (0)

    #define POW2_VALUE_CHECK(expr, label, validFlag) \
    do { \
        if (!isPowerOfTwo(expr)) { \
            validFlag = false; \
            VLOG(LOG_ERROR_CRITICAL) << "VALIDATION ERROR: " << label << " must be a power of 2. " \
                                    << "Current value: " << (expr) << std::endl; \
        } \
    } while (0)

    POW2_CHECK(gs_conf.sievingBlockSize, validFlag);
    // V1: legacy GATHER masks with (np-1) and the legacy subCube partition halves np.
    POW2_UNLESS_CHECK(relaxed_geometry, gs_conf.num_polysPerSieveCall, validFlag);
    // polyBlockSize stays pow2 FOREVER: advanceRoots' cyclic Gray wrap is single-bit only for
    // a power-of-two block, and the SCATTER seed composes (polyBlockId << log2_polyBlockSize).
    POW2_CHECK(gms_conf.polyBlockSize, validFlag);
    // V2: no kernel reads log2_num_polyBlocksPerThreadBlock; its exactness is carried by V5.
    NON0_CHECK(gms_conf.num_polyBlocksPerThreadBlock, validFlag);
    POW2_CHECK(gms_conf.num_activeBlocksPerCycle, validFlag);
    //grid/block dims
    POW2_CHECK(gms_conf.num_threadsPerBlock, validFlag);
    POW2_CHECK(ss_conf.num_threadsPerBlock, validFlag);
    // V3: SCATTER never reads gridDim.x; the exact partition (V5) is the only constraint.
    POW2_UNLESS_CHECK(relaxed_geometry, gms_conf.num_threadBlocks, validFlag);
    // V4: batch GATHER needs sasGridDim | np with a pow2 quotient (G1/G2), not sasGridDim pow2.
    POW2_UNLESS_CHECK(relaxed_geometry, ss_conf.num_threadBlocks, validFlag);

    // G1: a non-divisor GATHER grid gives overlapping prefixes AND an uncovered polynomial
    //     tail -- silently wrong relations, no error. (Tautology for pow2 sG <= pow2 np.)
    DIVIDES_CHECK(ss_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall, validFlag);
    // G2: the batch GATHER composes polyIdPrefix | gray(poly) with prefix = blockIdx.x*chunk;
    //     the OR is a valid addition only when chunk = np/sasGridDim is a power of two.
    //     (Tautology in the pow2 world.) Guarded by G1 so the division is exact.
    if (ss_conf.num_threadBlocks != 0 &&
        (gs_conf.num_polysPerSieveCall % ss_conf.num_threadBlocks) == 0) {
        POW2_VALUE_CHECK(gs_conf.num_polysPerSieveCall / ss_conf.num_threadBlocks,
                         "GATHER chunk (num_polysPerSieveCall / ss_conf.num_threadBlocks)",
                         validFlag);
    }
    // G3: sasGridDim > np gives a ZERO trip count => a silent zero-relation sieve. The
    //     autotune preflight already rejects this (kernel_launch_validator), but validateConfigs
    //     runs FIRST and is the authoritative gate for every load path, so state it here too
    //     (defense in depth).
    LEQ_CHECK(ss_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall, validFlag);

    //equations:
    // V5 (I1) -- UNCHANGED. This IS the SCATTER exact-partition identity: the batch SCATTER
    // kernel has a FIXED trip count and no `if (id < n)` guard, so under-coverage silently
    // drops polynomials (yield collapse) and over-coverage writes out of bounds.
    EQUAL_CHECK(gms_conf.num_polyBlocksPerThreadBlock*gms_conf.polyBlockSize*gms_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall, validFlag);
    EQUAL_CHECK(gms_conf.sharedMemReq, (gms_conf.num_activeBucketsPerThreadBlock * sizeof(int)), validFlag);
    // Width-aware: accumElemBytes() is 2 for the uint16 wide path, 1 for narrow AND the
    // saturating-uint8 wide path. The narrow path RHS is byte-identical to before.
    const size_t expected_ss_sharedMemReq =
        gs_conf.sievingBlockSize * accumElemBytes()
        + (ss_conf.offsetsInGlobal ? 0u : 3 * gs_conf.bigPrimeStartIndex * sizeof(int));
    EQUAL_CHECK(ss_conf.sharedMemReq, expected_ss_sharedMemReq, validFlag);
    EQUAL_CHECK(gms_conf.num_activeBucketsPerThreadBlock, gms_conf.num_activeBlocksPerCycle * gms_conf.polyBlockSize,validFlag);
    EQUAL_CHECK(gms_conf.num_metaSieveCycles*gms_conf.num_activeBlocksPerCycle, gs_conf.num_sievingBlocksPerSieveCall, validFlag);

    //space/subdivision requirements
    LEQ_CHECK(gms_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall, validFlag);
    LEQ_CHECK(gms_conf.num_threadBlocks*gms_conf.polyBlockSize, gs_conf.num_polysPerSieveCall, validFlag);
    LEQ_CHECK(gs_conf.num_polysPerSieveCall, ((1u << (fs_params.shc_dim - 1))), validFlag);
    LEQ_CHECK(gms_conf.num_activeBlocksPerCycle, gs_conf.num_sievingBlocksPerSieveCall, validFlag);

    //memcheck — routed through the single source-of-truth memory model (sieve_memory_model.h).
    // Like the loadStandardConfig loop above, the ORIGINAL LHS here multiplied three uint32_t
    // fields (num_polysPerSieveCall * num_sievingBlocksPerSieveCall * globalBucketSize) in
    // 32-bit before the trailing * sizeof(uint64_t) promoted, so it WRAPPED for buckets >= 4 GB
    // (e.g. the M=262K seed wraps to 0, masking the over-budget config). bucketEntriesBytes()
    // computes the same dev_globalBucketEntries term in 64-bit; the RHS
    // sieveBucketBudget(totalGlobalMem,0,4,5) == (4*totalGlobalMem)/5 == 0.80*VRAM (was
    // 3/4). The 64-bit LHS corrects the wrap (now rejects the over-budget config); the
    // 0.80 RHS matches loadStandardConfig and the autotune OOM guard.
    // BUCKET SIZING. The hard constraint is VRAM (below): a bucket that does not fit is
    // rejected. Being too SMALL is not rejected here -- it is a yield defect, not an
    // inadmissible geometry -- but it is silent in the kernel (overflow drops large-prime
    // hits with no error and no counter), so it is called out loudly. loadPartialCustomConfigDynamic
    // sizes the bucket at the predicted peak automatically unless --bucket_size_factor pins it.
    {
        const double mu = expectedBucketEntries(gs_conf.sievingBlockSize,
                                                gs_conf.bigPrimeStartIndex);
        const uint32_t want = autoGlobalBucketSize(gs_conf.sievingBlockSize,
                                                   gs_conf.bigPrimeStartIndex,
                                                   gs_conf.num_polysPerSieveCall,
                                                   gs_conf.num_sievingBlocksPerSieveCall);
        if (mu > 0.0 && gs_conf.globalBucketSize < want) {
            VLOG(LOG_WARNING) << "Bucket UNDERSIZED: globalBucketSize=" << gs_conf.globalBucketSize
                << " slots vs a predicted peak of " << want << " (mean " << mu
                << " entries at bigPrimeStart=" << gs_conf.bigPrimeStartIndex << ", SB="
                << gs_conf.sievingBlockSize << "). Overflow DISCARDS large-prime hits silently. "
                   "Raise --bucket_size_factor, or raise bigPrimeStart so fewer primes are bucketed.";
        }
    }
    // offsetsInGlobal moves the GATHER offsets/primes arrays into VRAM: charge them against the
    // same budget the bucket is measured against, so the two cannot overcommit together.
    const uint64_t gatherOffsetsBytes = ss_conf.offsetsInGlobal
        ? (uint64_t)ss_conf.num_threadBlocks
          * gatherOffsetsStride(gs_conf.bigPrimeStartIndex) * sizeof(int)
        : 0ull;
    LEQ_CHECK(bucketEntriesBytes(gs_conf.num_polysPerSieveCall, gs_conf.num_sievingBlocksPerSieveCall, gs_conf.globalBucketSize), sieveBucketBudget(g_info.totalGlobalMem, gatherOffsetsBytes, kSieveBudgetNum, kSieveBudgetDen), validFlag);//keep a buffer
    LEQ_CHECK(gms_conf.sharedMemReq, g_info.maxSharedMemPerBlock, validFlag);
    // GATHER shared memory must be checked against what is actually LAUNCHED. The batch path
    // adds shc_dim uint512 B_values on top of ss_conf.sharedMemReq (kernel.cu, "Add space for
    // B_values"), and that term was missing here: a config could pass validation and then fail
    // at launch with cudaErrorInvalidValue. Nothing checks the launch error outside
    // SIEVING_DEBUG_FLAG, so the run proceeded, every batch yielded 0 candidates, and it looked
    // like a healthy sieve producing nothing. The non-pow2 preflight below computes the right
    // figure but is gated on non-power-of-two geometry, so pow2 tuples were unguarded entirely.
    const size_t gatherLaunchSmem = ss_conf.sharedMemReq
        + ((gs_conf.batch_size > 0) ? (size_t)fs_params.shc_dim * sizeof(mpqs::uint512) : 0u);
    LEQ_CHECK(gatherLaunchSmem, g_info.maxSharedMemPerBlock, validFlag);

    // RSA-155 dual-path: wide-launch feasibility + occupancy (GRACEFUL, not a hard abort).
    // The realized batch-sieve launch shared memory is ss_conf.sharedMemReq PLUS the per-block
    // B_values (shc_dim * sizeof(uint512)) — mirrors kernel.cu runSievingBatch's sieve_smem.
    // Validate the wide kernel actually fits opt-in shared and yields >= 1 resident block/SM. On
    // infeasibility we WARN + fail the config (validFlag=false): the production caller escalates
    // cleanly, while the autotune candidate sweep never calls validateConfigs, so a probe sweep is
    // not torn down here. No LOG_ERROR_CRITICAL / throw on this path: the launch is
    // pre-validated via the CUDA occupancy API and an infeasible config is discarded.
    if (use_wide_accumulator_) {
        const size_t wideSieveSmem =
            ss_conf.sharedMemReq + (size_t)fs_params.shc_dim * sizeof(mpqs::uint512);
        if (wideSieveSmem > g_info.maxSharedMemPerBlock) {
            validFlag = false;
            VLOG(LOG_WARNING) << "Wide (uint16) sieve launch infeasible: smem " << wideSieveSmem
                             << " B > maxSharedMemPerBlock " << g_info.maxSharedMemPerBlock
                             << " B. Reduce M so the wide sieving block fits shared memory.";
        } else {
            int wide_blocks_per_sm = 0;
            // Query the kernel actually dispatched at this width.
            const void* wideKern = wide_u8sat_selected_
                ? (const void*)sieveAndScanBatchKernelWideU8Sat
                : (const void*)sieveAndScanBatchKernelWide;
            cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &wide_blocks_per_sm, wideKern,
                (int)ss_conf.num_threadsPerBlock, wideSieveSmem);
            if (occ_err != cudaSuccess || wide_blocks_per_sm < 1) {
                validFlag = false;
                VLOG(LOG_WARNING) << "Wide (uint16) sieve launch infeasible: occupancy "
                                 << wide_blocks_per_sm << " block/SM (cuda: "
                                 << cudaGetErrorString(occ_err)
                                 << "). Reduce M so the wide launch config is schedulable.";
            } else {
                VLOG(LOG_DEBUG_1) << "Wide (uint16) sieve launch feasible: smem " << wideSieveSmem
                                 << " B, occupancy " << wide_blocks_per_sm << " block/SM.";
            }
        }
    }

    // v1.0.6: NARROW-BATCH launch feasibility + occupancy, mirroring the wide
    // gate immediately above (same GRACEFUL contract: LOG_WARNING + validFlag=false, never a
    // hard abort mid-probe; pre-validated via the CUDA occupancy API). GATED on a genuinely
    // SM-aligned geometry: when every relaxed parameter is still a power of two this block
    // does not run, so all pre-v1.0.6 configurations execute byte-identical code. The
    // realized batch-sieve launch shared memory is ss_conf.sharedMemReq PLUS the per-block
    // B_values (shc_dim * sizeof(uint512)) — mirrors runSievingBatch's sieve_smem in
    // kernel.cu; the SCATTER launch uses gms_conf.sharedMemReq verbatim (meta_smem there).
    // The log line records the achievable waves/SM so a run's utilisation claim is
    // self-documenting rather than a paper calculation.
    if (relaxed_geometry &&
        !(isPowerOfTwo(gs_conf.num_polysPerSieveCall)
          && isPowerOfTwo(gms_conf.num_threadBlocks)
          && isPowerOfTwo(ss_conf.num_threadBlocks))) {
        const size_t narrowSieveSmem =
            ss_conf.sharedMemReq + (size_t)fs_params.shc_dim * sizeof(mpqs::uint512);
        // gpuInfo does not carry maxGridSize[0]; query it here rather than widening a struct
        // that every sieve translation unit includes.
        auto grid_over_limit = [this]() {
            int maxGridX = 0;
            cudaDeviceGetAttribute(&maxGridX, cudaDevAttrMaxGridDimX, device);
            return maxGridX > 0
                && ((uint64_t)gms_conf.num_threadBlocks > (uint64_t)maxGridX
                 || (uint64_t)ss_conf.num_threadBlocks  > (uint64_t)maxGridX);
        };
        if (narrowSieveSmem > g_info.maxSharedMemPerBlock) {
            validFlag = false;
            VLOG(LOG_WARNING) << "SM-aligned narrow sieve launch infeasible: GATHER smem "
                             << narrowSieveSmem << " B > maxSharedMemPerBlock "
                             << g_info.maxSharedMemPerBlock << " B.";
        } else if (grid_over_limit()) {
            validFlag = false;
            VLOG(LOG_WARNING) << "SM-aligned narrow sieve launch infeasible: grid ("
                             << gms_conf.num_threadBlocks << " SCATTER / "
                             << ss_conf.num_threadBlocks << " GATHER) exceeds maxGridSize[0].";
        } else {
            int meta_blocks_per_sm = 0, sas_blocks_per_sm = 0;
            cudaError_t occ_meta = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &meta_blocks_per_sm, (const void*)globalMetaSieveBatchKernel,
                (int)gms_conf.num_threadsPerBlock, gms_conf.sharedMemReq);
            cudaError_t occ_sas = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &sas_blocks_per_sm, (const void*)sieveAndScanBatchKernel,
                (int)ss_conf.num_threadsPerBlock, narrowSieveSmem);
            if (occ_meta != cudaSuccess || meta_blocks_per_sm < 1) {
                validFlag = false;
                VLOG(LOG_WARNING) << "SM-aligned SCATTER launch infeasible: occupancy "
                                 << meta_blocks_per_sm << " block/SM (cuda: "
                                 << cudaGetErrorString(occ_meta) << ") at blockDim "
                                 << gms_conf.num_threadsPerBlock << ", smem "
                                 << gms_conf.sharedMemReq << " B.";
            } else if (occ_sas != cudaSuccess || sas_blocks_per_sm < 1) {
                validFlag = false;
                VLOG(LOG_WARNING) << "SM-aligned GATHER launch infeasible: occupancy "
                                 << sas_blocks_per_sm << " block/SM (cuda: "
                                 << cudaGetErrorString(occ_sas) << ") at blockDim "
                                 << ss_conf.num_threadsPerBlock << ", smem "
                                 << narrowSieveSmem << " B.";
            } else {
                const uint32_t sms = (g_info.multiProcessorCount > 0)
                                   ? (uint32_t)g_info.multiProcessorCount : 1u;
                VLOG(LOG_INFO) << "[SM-aligned geometry] np=" << gs_conf.num_polysPerSieveCall
                              << " SCATTER grid=" << gms_conf.num_threadBlocks
                              << " (occ " << meta_blocks_per_sm << " blk/SM => "
                              << ((double)gms_conf.num_threadBlocks
                                  / ((double)meta_blocks_per_sm * sms)) << " waves/SM)"
                              << ", GATHER grid=" << ss_conf.num_threadBlocks
                              << " (occ " << sas_blocks_per_sm << " blk/SM => "
                              << ((double)ss_conf.num_threadBlocks
                                  / ((double)sas_blocks_per_sm * sms)) << " waves/SM)"
                              << ", SMs=" << sms
                              << ", GATHER chunk=" << (ss_conf.num_threadBlocks
                                    ? gs_conf.num_polysPerSieveCall / ss_conf.num_threadBlocks : 0)
                              << ", npbptb=" << gms_conf.num_polyBlocksPerThreadBlock << ".";
            }
        }
    }

    NON0_CHECK(gs_conf.num_sievingBlockBatches, validFlag);
    NON0_CHECK(gs_conf.num_subCubes, validFlag);
    NON0_CHECK(gms_conf.num_polyBlocksPerThreadBlock, validFlag);
    NON0_CHECK(gms_conf.num_metaSieveCycles, validFlag);

    #undef POW2_CHECK
    #undef EQUAL_CHECK
    #undef LEQ_CHECK
    #undef NON0_CHECK
    #undef POW2_UNLESS_CHECK
    #undef DIVIDES_CHECK
    #undef POW2_VALUE_CHECK

    if(!validFlag){
        VLOG(LOG_ERROR_CRITICAL) << "============================================================";
        VLOG(LOG_ERROR_CRITICAL) << "initConfig:";
        VLOG(LOG_ERROR_CRITICAL) << "threads: " << init_conf.num_threadsPerBlock;
        VLOG(LOG_ERROR_CRITICAL) << "blocks: " << init_conf.num_threadBlocks;
        VLOG(LOG_ERROR_CRITICAL) << "batch size: " << init_conf.batch_size;
        VLOG(LOG_ERROR_CRITICAL) << "============================================================";
        VLOG(LOG_ERROR_CRITICAL) << "generalSievingConfig:";
        VLOG(LOG_ERROR_CRITICAL) << "sievingBlockSize: " << gs_conf.sievingBlockSize;
        VLOG(LOG_ERROR_CRITICAL) << "globalBucketSize: " << gs_conf.globalBucketSize;
        VLOG(LOG_ERROR_CRITICAL) << "polys sieved per call: " << gs_conf.num_polysPerSieveCall;
        VLOG(LOG_ERROR_CRITICAL) << "sievingBlocks sieved per call: " << gs_conf.num_sievingBlocksPerSieveCall;
        VLOG(LOG_ERROR_CRITICAL) << "bigPrimeStart: " << gs_conf.bigPrimeStartIndex;
        VLOG(LOG_ERROR_CRITICAL) << "midPrimeStart: " << gs_conf.midPrimeStartIndex;
        VLOG(LOG_ERROR_CRITICAL) << "maxRelationsPerBlock: " << gs_conf.maxRelationsPerBlock;
        VLOG(LOG_ERROR_CRITICAL) << "============================================================";
        VLOG(LOG_ERROR_CRITICAL) << "globalMetaSieveConfig:";
        VLOG(LOG_ERROR_CRITICAL) << "threads: " << gms_conf.num_threadsPerBlock;
        VLOG(LOG_ERROR_CRITICAL) << "blocks: " << gms_conf.num_threadBlocks;
        VLOG(LOG_ERROR_CRITICAL) << "num activeblocks: " << gms_conf.num_activeBlocksPerCycle;
        VLOG(LOG_ERROR_CRITICAL) << "shared memory required: " << gms_conf.sharedMemReq << " bytes";
        VLOG(LOG_ERROR_CRITICAL) << "============================================================";
        VLOG(LOG_ERROR_CRITICAL) << "sieveAndScanConfig:";
        VLOG(LOG_ERROR_CRITICAL) << "threads: " << ss_conf.num_threadsPerBlock;
        VLOG(LOG_ERROR_CRITICAL) << "blocks: " << ss_conf.num_threadBlocks;
        VLOG(LOG_ERROR_CRITICAL) << "shared memory required: " << ss_conf.sharedMemReq << " bytes";
        VLOG(LOG_ERROR_CRITICAL) << "============================================================";
    }
    #undef VLOG
    return validFlag;
}

bool DeviceSievingController::validateConfigsSilent() {
    validate_quiet_ = true;
    const bool ok = validateConfigs();
    validate_quiet_ = false;
    return ok;
}

} // namespace sieve
} // namespace mpqs
