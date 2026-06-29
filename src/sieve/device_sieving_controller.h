// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include "sieving_data_structs.h"
#include "prime_algorithms.h"
#include "kernel.cuh"
#include "json_helper.h"
#include <array>
#include <atomic>

namespace mpqs {
namespace sieve {

class DeviceSievingController {

public:

    DeviceSievingController(int device, cudaStream_t stream);
    DeviceSievingController(int device); // legacy constructor on stream 0
    ~DeviceSievingController();

    void initiate(factoringData& f_data);
    void allocateBatchBuffers();
    void updateState();
    void advance_a(int step);

    // Core sieving methods
    void sieveFullCube();
    void sieveStep();

    /// 8-parameter tuple: {subCubeSize, numIntervals, polyBlockSize, blocksPerCycle,
    ///                      metaGridDim, metaBlockDim, sasGridDim, sasBlockDim}
    using Params8 = std::array<uint32_t, 8>;

    /// Result of exhaustive parameter test
    struct ParamTestResult {
        Params8 best_params;
        float   best_timing_us;
        uint32_t configs_tested;
        std::string json_path;  ///< path to paramTest.json output
    };

    /// Lightweight sieve benchmark: run num_subcubes subcubes (0 = full cube).
    /// Returns elapsed microseconds. Saves/restores ds_params state.
    float sieveMini(uint32_t num_subcubes);

    /// Evaluate a single 8-param configuration.
    /// Handles loadPartialCustomConfig, validateConfigs, optional GPU reload.
    /// Returns elapsed microseconds, or -1.0f if the config is infeasible.
    float evaluateConfig(const Params8& params, uint32_t num_subcubes, bool& reload_needed);

    #ifdef SIEVING_DEBUG_FLAG
        void sieveFullCubeSnapshot(bool meta,
        bool meta_P_enabled,
        uint32_t meta_P,
        bool meta_O_enabled,
        uint32_t meta_O,
        bool sas,
        uint32_t step,
        JSON_IO j_io);
    #endif

    /**
     * @brief Prepares the next batch of polynomials for sieving.
     *
     * 1. Updates Host state (f_data) to determine the next K sets of 'a' factors.
     * 2. Uploads these factor indices to the GPU (dev_job_factor_indices).
     * 3. Launches the Polynomial Generator Kernel to compute 'a' and 'B' values on-device.
     *
     */
    void prepareSievingBatch();

    /// @brief Graph-capturable polynomial preparation.
    /// Assumes factor indices are already in d_indices (device memory).
    /// Only launches generatePolynomialsKernel — no H2D copy, no host state advance.
    void prepareSievingBatchFromStaged(
        const uint32_t* d_indices,
        mpqs::uint512* a_array_out,
        mpqs::uint512* B_flat_out
    );

    /// @brief Redirect job array pointers used by runSievingBatch.
    /// Call before runSievingBatch to use per-batch staged arrays.
    void setJobArrays(
        mpqs::uint512* a_array,
        mpqs::uint512* B_flat,
        const uint32_t* factor_indices
    );

    /// @brief Mutable reference to the controller's factoringData.
    /// Required for graph path: the orchestrator must call prepareNextBatchIndices
    /// on the SAME f_data the siever owns (not the orchestrator's stale copy).
    factoringData& getFactoringDataRef() { return f_data; }
    /**
     * @brief Executes the "Inner Sieving Loop" entirely on the GPU.
     *
     * Runs the sequence: Reset -> InitState -> MetaSieve -> Sieve -> Compact
     * repeatedly for 'num_steps' without any CPU intervention.
     *
     * @param num_steps Number of polynomial steps to process in this batch.
     * @param start_batch_index Offset into the pre-calculated Batch Arrays (a, B, etc).
     */
    void runSievingBatch(int num_steps, int start_batch_index);

    // Configuration
    void loadStandardConfig();
    void loadPartialCustomConfig(uint32_t totalPolys, uint32_t totalIntervals, uint32_t polyBlockSize, uint32_t blocksPerCycle, uint32_t metaB, uint32_t metaT, uint32_t sasB, uint32_t sasT);
    ParamTestResult runParamTest(factoringData& f_data);
    void setConfig(const initConfig& c) { init_conf = c; }
    void setConfig(const generalSievingConfig& c) { gs_conf = c; }
    void setConfig(const globalMetaSieveConfig& c) { gms_conf = c; }
    void setConfig(const sieveAndScanConfig& c) { ss_conf = c; }
    void setConfig(const processRelationsConfig& c) { pr_conf = c; }

    // Sets batch size for batch sieving of batch_size many values of "a"
    void setSievingBatchSize(uint32_t batch_size) {
        init_conf.batch_size = batch_size;
        gs_conf.batch_size = batch_size;
        gms_conf.batch_size = batch_size;
        ss_conf.batch_size = batch_size;
    }

    void setThresholdOverride(uint64_t threshold_bound);

    /// Autotune OOM-guard knob: an optional cap (in bytes) on the TOTAL device
    /// footprint of one sieve instance (bucket + persistent + scratch), applied
    /// inside loadStandardConfig() AFTER the bucket-only budget reduction. The
    /// seed num_polysPerSieveCall is halved further (down to the validator's
    /// minimum) until estimateSieveFootprint(...).total() fits this cap.
    ///
    /// DEFAULTED OFF (0 => no extra cap). The orchestrator/production paths never
    /// set it, so loadStandardConfig is byte-identical there. Only the autotune
    /// Stage-1 seed guard (AutotuneController::runStage1_KernelParams) sets it to
    /// 0.80*free_VRAM - (postprocessing/LP bytes + CUDA-context reserve), so the
    /// seed's ENTIRE footprint fits free VRAM before loadData() allocates it.
    ///
    /// Returns the clamp diagnostic (seed num_polys before/after, est totals) via
    /// the out-params when a clamp fired; *clamped is false if the knob was off or
    /// did not bind. Read after loadStandardConfig() to log the clamp.
    void setMaxTotalSieveBytes(uint64_t max_total_bytes) {
        max_total_sieve_bytes_ = max_total_bytes;
    }
    /// Diagnostic for the last loadStandardConfig() seed clamp under the knob.
    struct SeedClampInfo {
        bool     clamped = false;       ///< true iff the knob bound and reduced num_polys
        uint32_t num_polys_before = 0;  ///< seed num_polys after the bucket-only budget loop
        uint32_t num_polys_after = 0;   ///< seed num_polys after the total-footprint knob loop
        uint64_t total_before = 0;      ///< estimateSieveFootprint().total() before the knob loop
        uint64_t total_after = 0;       ///< estimateSieveFootprint().total() after the knob loop
        uint64_t budget = 0;            ///< the max_total_sieve_bytes_ cap that was applied
    };
    SeedClampInfo getLastSeedClamp() const { return last_seed_clamp_; }

    /// Set external stop flag (cluster mode). If non-null, runSievingBatch()
    /// checks *external_stop_ before launching kernels and returns early if set.
    /// Host-side check only — no device-side modification.
    /// One of 2 permitted submodule changes (Spec Section 8.3, Section 11.1).
    void setExternalStop(std::atomic<bool>* flag) { external_stop_ = flag; }

    /// Save snapshot of current a-factor state. Call immediately after initiate()
    /// and init_a_factors() completes, before any sieving begins.
    /// One of 2 permitted submodule changes (Spec Section 11.1).
    void saveSnapshot();

    /// Reset polynomial state and advance to a global a-index.
    /// Reconstructs the exact polynomial state at position global_a_index
    /// in the global Hamiltonian path: (1) restore snapshot, (2) simulate
    /// window slides to target hypercube, (3) simulate Gray code steps to
    /// local offset, (4) recalc_a(), (5) updateState().
    /// Cost: ~5ms dominated by GPU upload, independent of jump distance.
    /// @param global_a_index  Target position. 0 = initial state after init_a_factors.
    void resetAndAdvanceTo(uint64_t global_a_index);

    /// Get the saved snapshot (for serialization to workers).
    const AFactorsSnapshot& getSnapshot() const { return snapshot_; }

    // For handshake with postprocessor
    // OLD signature (SoA): void setPostProcessingLinks(void* buffer, uint32_t* counter, uint32_t capacity);
    void setPostProcessingLinks(mpqs::postprocessing::DoubleBuffer* active_buffer);

    // Appends an async copy to the siever stream to update the Host
    void pushCounterToHostAsync(volatile uint32_t* h_pinned_counter);

    // Data management
    void loadData();

    // Validation & Debugging
    void printConfigs();
    void printCustomConfigs();
    void printConfigsDEBUG();
    bool validateConfigs();

    // Public getter for non-batch processing postprocessor handshake
    cudaStream_t getCudaStream() const { return stream; }

    // Public getters for batch postprocessing of relation candidates
    candidateRelation* getRawCandidates() const { return dev_pointers.dev_candidateRelations; }
    int getRawCandidateBufferSize() const {
        // Based on config: maxRelations * numBlocks
        return gs_conf.maxRelationsPerBlock * ss_conf.num_threadBlocks;
    }
    // Allow access to all device pointers for postprocessing
    devicePointers getDevicePointers() const { return dev_pointers; }
    // Allow access to factoring data.
    factoringData getFactoringData() const { return f_data; }

    // For debugging: Allow access to config structs
    generalSievingConfig getGeneralConfig() const { return gs_conf; };
    sieveAndScanConfig getSieveAndScanConfig() const { return ss_conf; };
    fixedSievingParams getFixedParams() const { return fs_params; };
    dynamicSievingParams getDynamicParams() const {return ds_params; };

    // Clear internal buffers to make space for matrix gen
    void clearSievingBuffers() {
        cudaFree(dev_pointers.dev_globalBucketEntries);
        cudaFree(dev_pointers.dev_globalBucketCounts);
        cudaFree(dev_pointers.dev_candidateRelations);
        cudaFree(dev_pointers.dev_indexToCandidate);
	if(init_conf.batch_size) {
	    if(dev_pointers.dev_blockRelationCounts)
	        cudaFree(dev_pointers.dev_blockRelationCounts);
	    if(dev_pointers.dev_job_a_array)
	        cudaFree(dev_pointers.dev_job_a_array);
	    if(dev_pointers.dev_job_B_flat)
	        cudaFree(dev_pointers.dev_job_B_flat);
	    if(dev_pointers.dev_job_factor_indices)
	        cudaFree(dev_pointers.dev_job_factor_indices);
	    if(h_pinned_factor_indices_)
	        { cudaFreeHost(h_pinned_factor_indices_); h_pinned_factor_indices_ = nullptr; }
	}
        // Do NOT free FactorBase or RootN here as they might be used by post processing
    }

    // Expose Device Pointer for PostProcessor
    const uint32_t* getDeviceA_Factors() const {
        return dev_pointers.dev_a_factors;
    }

    /**
     * @brief Pulls candidates from GPU and validates them via CPU trial division.
     * @return Number of valid relations found and verified.
     */
    int validateResults(factoringData& f_data);

private:

    factoringData f_data;

    int device;
    gpuInfo g_info;
    cudaStream_t stream;
    devicePointers dev_pointers;
    fixedSievingParams fs_params;
    dynamicSievingParams ds_params;

    initConfig init_conf;
    generalSievingConfig gs_conf;
    globalMetaSieveConfig gms_conf;
    sieveAndScanConfig ss_conf;
    processRelationsConfig pr_conf;

    mpqs::postprocessing::DoubleBuffer* current_pp_buffer;

    // Pinned host buffer for truly async H2D copies in prepareSievingBatch().
    // Without pinning, cudaMemcpyAsync from pageable memory forces an implicit
    // stream synchronization, creating a pipeline bubble every batch.
    uint32_t* h_pinned_factor_indices_ = nullptr;
    size_t    pinned_factor_indices_capacity_ = 0;
    std::atomic<bool>* external_stop_ = nullptr;  ///< External stop signal (cluster mode)
    AFactorsSnapshot snapshot_;  ///< Saved after initiate() + init_a_factors()

    /// Autotune OOM-guard knob (bytes). 0 => off (orchestrator/production default,
    /// byte-identical loadStandardConfig). See setMaxTotalSieveBytes().
    uint64_t max_total_sieve_bytes_ = 0;
    SeedClampInfo last_seed_clamp_;  ///< Filled by loadStandardConfig() when the knob binds.

    // NOT NECESSARY / OBSOLETE / ONLY FOR DEBUGGING PURPOSES
    // Helper to clear candidate buffer before launch for validation clarity / safety
    void clearCandidates();
    // Helper to clear bucket buffer before launch for validation clarity / safety
    void clearBuckets();
};

} // namespace sieve
} // namespace mpqs
