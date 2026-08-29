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
    /// Wide-autotune foundation: on the WIDE (uint16) accumulator path — where
    /// there is no non-batch kernel — this dispatches to sieveMiniBatch() (the real
    /// batch-wide probe) instead of returning the -1.0f "wide has no probe" sentinel.
    /// The NARROW (uint8) path is byte-for-byte the original non-batch probe.
    float sieveMini(uint32_t num_subcubes);

    /// Scale-representative harness (wide-only): drive the REAL batch pipeline
    /// (mpqs::sieve::runSievingBatch → sieveAndScanBatchKernelWide + compact) over a
    /// warm-up-then-measure WALL-CLOCK window and return a candidate-survivors/sec RATE
    /// (Δdev_probe_pp_counter / Δwall-seconds). HIGHER = faster/better (the wide objective
    /// is a rate, not microseconds). Auto-scales by wall time (larger M ⇒ fewer polys/call
    /// ⇒ a fixed batch count is unrepresentative), so `repeats` is IGNORED — the window
    /// (probe_window_sec_ + probe_warmup_sec_, set via setProbeWindow) governs the duration.
    /// Returns -1.0f if setup/timing failed. Only ever called from sieveMini() when
    /// isWideAccumulator() and from sieveMiniStandardWide() (the acceptance floor).
    float sieveMiniBatch(uint32_t repeats);

    /// Floor gate (wide-only): load the STANDARD wide config (loadStandardConfig —
    /// the exact geometry that ships when autotune is OFF / useParams==false) into this
    /// probe siever, resize the sieve scratch to match, and measure it under the SAME
    /// window harness as the search candidates. Returns the standard config's
    /// survivors/sec RATE (the acceptance floor), or -1.0f if not wide / setup failed.
    /// Preserves the staged probe batch + pp scratch (loadSievingDataParamTest does not
    /// touch them). NARROW callers never reach this (wide-gated).
    float sieveMiniStandardWide();

    /// Wide autotune search-space seeding (wide-only): the occupancy-optimal GATHER
    /// (sieve-and-scan) blockDim for the wide kernel sieveAndScanBatchKernelWide. Evaluates
    /// each candidate blockDim via cudaOccupancyMaxActiveBlocksPerMultiprocessor at the wide
    /// GATHER shared-memory footprint (ss_conf.sharedMemReq + shc_dim*sizeof(uint512) — the
    /// SAME smem the validateConfigs() wide-feasibility guard uses) and returns the blockDim
    /// maximizing resident threads/SM (blocks_per_SM * blockDim), i.e. the occupancy knee.
    /// The smem is blockDim-independent (blockEntries is sized per sieving-block, not per
    /// thread), so the API alone drives the choice — no hardcoded per-GPU value. Returns 0 if
    /// not wide, `candidates` is empty, the smem exceeds the block limit, or the occupancy API
    /// fails for every candidate — so the caller keeps its existing seed. Reads the CURRENT
    /// ss_conf/fs_params (loadStandardConfig-wide at autotune-Stage-1 entry); never mutates
    /// state; never reached on the narrow (uint8) path.
    uint32_t wideGatherOccupancyBlockDim(const uint32_t* candidates, uint32_t count) const;

    /// Wide-only: set the measurement window (survivors/sec is measured over
    /// window_sec of wall clock after a warmup_sec warm-up prefix). The optimizer sets
    /// this before each wide eval so the harness auto-scales by wall time and can shrink
    /// toward a floor when the autotune budget is tight. Never read on the narrow path.
    void setProbeWindow(double window_sec, double warmup_sec) {
        probe_window_sec_ = window_sec;
        probe_warmup_sec_ = warmup_sec;
    }

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

    /// Dual-path accumulator: CLI/config override for the accumulator-width
    /// dispatch predicate. 0 = auto (predicate decides), 1 = force u8, 2 = force u16.
    /// Consumed in initiate() when computing use_wide_accumulator_.
    void setAccumulatorMode(int mode) { accumulator_mode_ = mode; }

    /// Wide saturating-uint8 accumulator: CLI/config override for the
    /// wide-accumulator WIDTH dispatch. Only consulted when use_wide_accumulator_
    /// is true. 0 = auto (use saturating-uint8 iff the config-time exactness gate
    /// APV_max-threshold<=254 holds, else uint16), 1 = force u8sat (still honours
    /// the gate: falls back to uint16 if unsafe), 2 = force uint16.
    /// Consumed in initiate() when computing wide_u8sat_selected_.
    void setWideAccumMode(int mode) { wide_accum_mode_ = mode; }

    /// True iff this siever resolved to the SATURATING-uint8 wide
    /// accumulator (use_wide AND the exactness gate holds AND not forced to u16).
    /// Read by runSievingBatch() to pick the u8sat kernel and by the config
    /// loaders to size SB at the 1-byte width. Narrow and uint16-wide see false.
    bool isWideU8Sat() const { return wide_u8sat_selected_; }

    /// Bucket-overflow telemetry (host-side, read-only). Surfaces the silent
    /// large-prime bucket-overflow discard documented in the 2026-07-10 A100
    /// uint16 degenerate-baseline root-cause analysis: the
    /// SCATTER kernel drops every hit past globalBucketSize and the GATHER masks
    /// the overflow flag (bit 31 of dev_globalBucketCounts) off, so the resulting
    /// under-accumulation is otherwise invisible. This surfaces the flag.
    struct BucketOverflowStats {
        uint64_t total_buckets      = 0;   ///< num_polysPerSieveCall * num_sievingBlocksPerSieveCall
        uint64_t overflowed_buckets = 0;   ///< buckets with bit 31 (0x80000000) set
        uint32_t global_bucket_size = 0;   ///< per-bucket capacity (legacy SB/2; = factor*SB under --bucket_size_factor)
        uint32_t max_fill           = 0;   ///< max clamped fill level over all buckets (& 0x00FFFFFF)
        double   fraction           = 0.0; ///< overflowed_buckets / total_buckets
    };
    /// Snapshot the most-recently-populated dev_globalBucketCounts and count the
    /// overflow flags. WIDE-ONLY (u16 + u8sat): returns false on the narrow path
    /// so the validated-record narrow production hot path takes NO extra DtoH
    /// sync. Pure read (cudaMemcpyAsync DtoH + stream sync) — cannot alter any
    /// device sieving result. Returns false if wide-off / buffer unallocated /
    /// CUDA error. Intended for the periodic stats cadence, not every batch.
    bool getBucketOverflowStats(BucketOverflowStats& out) const;

    /// Wide-autotune probe sample size override (CLI --autotune_probe_polys). 0 = auto-scale
    /// by N in ensureProbeBatchSetup() (see there). >0 forces exactly N distinct staged
    /// polynomials. Wide path only (ensureProbeBatchSetup is never reached on narrow).
    void setAutotuneProbePolys(uint32_t n) { autotune_probe_polys_override_ = n; }

    /// True iff this siever resolved to the wide
    /// (uint16) accumulator path in initiate() (predicate OR --sieve_accumulator
    /// override). Read by the autotune probe/validator to select width-aware
    /// geometry and by loadPartialCustomConfig()'s num_polys clamp. Narrow
    /// (uint8) callers see false, so every narrow code path is unaffected.
    bool isWideAccumulator() const { return use_wide_accumulator_; }

    /// A2 (meta-sieve SCATTER locality): CLI/config cap on the number of active
    /// sieving blocks the meta-sieve kernels scatter into per cycle.
    /// 0 = OFF (legacy: config loaders keep their derived num_activeBlocksPerCycle,
    /// byte-identical geometry). N>0: num_activeBlocksPerCycle is capped at
    /// min(pow2_floor(N), derived value) and num_metaSieveCycles rises to
    /// num_sievingBlocksPerSieveCall / cap — exact, since the derived value is a
    /// power of two dividing num_sievingBlocksPerSieveCall and any smaller power
    /// of two divides it too (no partial last cycle can exist). This bounds each
    /// thread's bucket-write destination spread to cap*sievingBlockSize positions
    /// independent of M, recovering SCATTER L1 locality at large M.
    /// Consumed by loadStandardConfig() / loadPartialCustomConfig() via
    /// applyMetaCycleCap(); call before either loader.
    void setMetaCycleCap(uint32_t cap) { meta_cycle_cap_ = cap; }

    /// GATHER (sieve-and-scan) blockDim A/B override. 0 = OFF (loaders keep their
    /// derived ss_conf.num_threadsPerBlock = 256, byte-identical legacy launch).
    /// N>0: ss_conf.num_threadsPerBlock is set to N by applyGatherBlockDimOverride(),
    /// run inside both config loaders. blockDim is a RESULT-INVARIANT performance knob
    /// for the sieve-and-scan kernels (the shared accumulator blockEntries is sized per
    /// sieving-block, not per-thread; every work loop strides by blockDim.x; candidate
    /// append uses an atomic write-head, correct for any warp count), so relations and
    /// witnesses are identical across values — only occupancy changes. Caller must pass a
    /// power of two in [32,1024] (validated at the CLI; downstream validateConfigs()
    /// POW2_CHECKs ss_conf.num_threadsPerBlock). Call before either loader.
    void setGatherBlockDim(uint32_t n) { gather_block_dim_override_ = n; }

    /// Ablation knob: decouple the large-prime bucket capacity globalBucketSize from the
    /// legacy SB/2. f == 0.0 (default) keeps globalBucketSize = sievingBlockSize/2 EXACTLY,
    /// so every path (narrow uint8, wide uint16, wide u8sat) is byte-identical to prior
    /// behavior. f > 0.0 sizes globalBucketSize = f*SB via computeGlobalBucketSize() in both
    /// config loaders (f=0.5 reproduces legacy; f=1.0 doubles the bucket — the isolation
    /// experiment L2 in a100_u16_degenerate_baseline_rootcause_2026_07_10.md). The resized
    /// bucket feeds the VRAM budget reduction (reduceNumPolysToBudget) and validateConfigs,
    /// so an over-large factor degrades num_polys / is validator-rejected, never OOM-crashes.
    /// Consumed only at config-load time; call before either loader.
    void setBucketSizeFactor(double f) { bucket_size_factor_override_ = f; }

    /// v1.0.6: override gs_conf.sievingBlockSize on the NARROW BATCH --params path.
    /// n == 0 (default) = OFF: the loader keeps SB = min(M, pow2leq(3/4 * maxSharedMemPerBlock)),
    /// byte-identical to v1.0.5 on every path. n > 0 replaces that derivation inside
    /// loadPartialCustomConfig ONLY (loadStandardConfig is deliberately untouched, so no
    /// non---params run can change). SB is the GATHER shared-memory accumulator length, so
    /// lowering it is the ONLY lever that can bring ss_conf.sharedMemReq under the co-residency
    /// threshold at production M -- shared memory is the measured sole obstruction
    /// (launch__occupancy_limit_shared_mem = 1 vs _registers = 2, A100 job 34135902).
    /// Caller must pass a power of two in [256, M] (validated at the CLI; downstream
    /// POW2_CHECK + LEQ_CHECK + the narrow-batch occupancy preflight). LOWERING SB shrinks a
    /// single launch's interval coverage, so --params field 2 (numIntervals) must be raised in
    /// step to keep numIntervals * SB >= 2M -- enforced by narrowBatchCoverageOk() in
    /// validateConfigs, never silently accepted. Ignored (and rejected by validateConfigs) on
    /// the wide and legacy paths. Call before initiate().
    void setSievingBlockSizeOverride(uint32_t n) { sb_override_ = n; }

    /// v1.0.6: override gs_conf.bigPrimeStartIndex on the NARROW BATCH --params path.
    /// n == 0 (default) = OFF: the loader keeps bPSI = sievingBlockSize/32, byte-identical.
    /// n > 0 replaces that derivation inside loadPartialCustomConfig ONLY. bPSI splits the
    /// factor base between GATHER's in-block path [0,bPSI) and SCATTER's bucketed path
    /// [bPSI,fb_size), and costs 3*bPSI*4 bytes of the GATHER shared-memory budget. Because the
    /// loader derivation is unconditionally bPSI = SB/32, an SB override alone ALSO moves primes
    /// across that split; this knob cancels the coupling so the two effects can be attributed
    /// separately. Caller must pass n > midPrimeStartIndex (= 32) and n <= fb_size (validated at
    /// the CLI): at n <= 32 the mid-prime loops [midPrimeStart, bPSI) invert and silently drop
    /// the entire mid-prime band. Power-of-two is NOT required (every consumer is a grid-stride
    /// loop). Ignored (and rejected by validateConfigs) on the wide and legacy paths.
    /// Call before initiate().
    void setBigPrimeStartOverride(uint32_t n) { bpsi_override_ = n; }

    /// v1.0.6: enable getBucketOverflowStats() on the NARROW path. false (default) = OFF:
    /// the reader returns false on narrow exactly as in v1.0.5, so the narrow production hot
    /// path takes no extra DtoH copy and no cudaStreamSynchronize on the siever stream at the
    /// ~5 s stats cadence. true = the same read the wide path already performs. Wide behaviour
    /// is unaffected either way (wide always reports). Set from --sieve_bucket_overflow_stats.
    void setNarrowOverflowStats(bool on) { narrow_overflow_stats_ = on; }

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
    globalMetaSieveConfig getMetaSieveConfig() const { return gms_conf; };
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
	    releasePinnedIndexStaging();
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

    // Pinned host staging for truly async H2D copies in prepareSievingBatch().
    // Without pinning, cudaMemcpyAsync from pageable memory forces an implicit
    // stream synchronization, creating a pipeline bubble every batch.
    //
    // DOUBLE-BUFFERED + EVENT-GATED (v1.0.6). A single reused slot is a
    // correctness hazard: the host memcpy into the slot is immediate while the
    // cudaMemcpyAsync that consumes it is stream-deferred, so with the host Δ
    // batches ahead in the launch queue every in-flight H2D reads whichever index
    // set the host wrote LAST. In steady state that is a pure relabeling (batch b
    // sieves the set prepared for b+Δ, batch b+Δ's own set is skipped); at any host
    // stop — sieve-loop exit, checkpoint quiesce — the Δ still-queued copies all
    // read the SAME final set and the final polynomial batch is sieved Δ times,
    // re-emitting its candidates and (with LP on) its partials verbatim. Gating the
    // host write on completion of the H2D that last consumed the slot closes the
    // overwrite window. kPinnedIndexSlots = 2 keeps one batch in flight while the
    // host stages the next — the discipline the graph path already uses for its
    // staged slots (orchestrator.cpp, stage_done[]).
    // Root-caused and fixed in v1.0.6 (2026-08-23).
    static constexpr uint32_t kPinnedIndexSlots = 2;

    /// Allocate (or re-allocate) the pinned index staging and its per-slot
    /// completion events. @p elems_per_slot = batch_size * shc_dim.
    void allocatePinnedIndexStaging(size_t elems_per_slot);
    /// Free the pinned staging and destroy the per-slot events. Idempotent.
    void releasePinnedIndexStaging();

    uint32_t* h_pinned_factor_indices_ = nullptr;   ///< kPinnedIndexSlots contiguous slots
    size_t    pinned_factor_indices_capacity_ = 0;  ///< elements PER SLOT
    cudaEvent_t pinned_h2d_done_[kPinnedIndexSlots] = {};
    uint32_t    pinned_slot_ = 0;                   ///< slot claimed by the next call
    std::atomic<bool>* external_stop_ = nullptr;  ///< External stop signal (cluster mode)
    AFactorsSnapshot snapshot_;  ///< Saved after initiate() + init_a_factors()

    /// Autotune OOM-guard knob (bytes). 0 => off (orchestrator/production default,
    /// byte-identical loadStandardConfig). See setMaxTotalSieveBytes().
    uint64_t max_total_sieve_bytes_ = 0;
    SeedClampInfo last_seed_clamp_;  ///< Filled by loadStandardConfig() when the knob binds.

    // Dual-path sieve accumulator: dispatch-predicate state.
    // accumulator_mode_: 0 = auto (predicate decides), 1 = force u8, 2 = force u16
    //   (set via setAccumulatorMode() before initiate()).
    // use_wide_accumulator_: computed in initiate() from f_data.a.msb()/f_data.M.
    //   Computed once per initiate(); it selects the batch-sieve kernel width.
    int  accumulator_mode_     = 0;
    /// v1.0.6: set by loadPartialCustomConfig when a pinned tuple violates an
    /// invariant the loader itself can prove (currently: metaGridDim*polyBlockSize must divide
    /// num_polysPerSieveCall exactly). validateConfigs consumes and CLEARS it, so a later
    /// successful load starts clean. Loader-level rejection keeps the diagnostic specific —
    /// validateConfigs would otherwise only report the resulting EQUAL_CHECK mismatch.
    bool custom_config_invalid_ = false;

    bool use_wide_accumulator_ = false;

    // Wide saturating-uint8 accumulator: width dispatch inside the wide
    // regime. wide_accum_mode_: 0 = auto (gate decides), 1 = force u8sat (gate-honoured),
    // 2 = force uint16. wide_u8sat_selected_ computed in initiate() (only when
    // use_wide_accumulator_). Narrow path leaves both at their defaults (u8sat=false).
    int  wide_accum_mode_      = 0;
    bool wide_u8sat_selected_  = false;

    // Accumulator element width (bytes) for shared-memory sizing. Narrow (uint8) and
    // the saturating-uint8 wide path both use 1 byte; only the uint16 wide path uses 2.
    // Single source of truth for every SB / sharedMemReq derivation.
    size_t accumElemBytes() const {
        return (use_wide_accumulator_ && !wide_u8sat_selected_) ? sizeof(uint16_t)
                                                                : sizeof(uint8_t);
    }

    /// A2 meta-sieve SCATTER cycle cap. 0 = OFF (legacy geometry). See setMetaCycleCap().
    uint32_t meta_cycle_cap_ = 0;
    /// A2: apply meta_cycle_cap_ to gms_conf.num_activeBlocksPerCycle. Must run AFTER
    /// the loader has set num_activeBlocksPerCycle (and, in loadStandardConfig, after
    /// the polyBlockSize sizing loop, so polyBlockSize is unaffected) and BEFORE
    /// num_metaSieveCycles / num_activeBucketsPerThreadBlock / sharedMemReq are derived.
    void applyMetaCycleCap();

    /// GATHER blockDim override. 0 = OFF (no-op). See setGatherBlockDim(). Run inside
    /// both loaders AFTER ss_conf.num_threadsPerBlock is set to the loader default, so
    /// N>0 replaces it. Only ss_conf.num_threadsPerBlock changes; sharedMemReq and all
    /// buffer sizes are blockDim-independent, so nothing else is re-derived.
    uint32_t gather_block_dim_override_ = 0;
    void applyGatherBlockDimOverride();

    /// --bucket_size_factor override. 0.0 = OFF (legacy globalBucketSize = SB/2, byte-identical
    /// on all paths). See setBucketSizeFactor(). Consumed by computeGlobalBucketSize() inside
    /// both config loaders where globalBucketSize is assigned.
    double bucket_size_factor_override_ = 0.0;

    /// v1.0.6 --sieve_block_size / --sieve_big_prime_start overrides. 0 = OFF (the loader's own
    /// derivation runs, byte-identical to v1.0.5). Consumed ONLY inside loadPartialCustomConfig,
    /// at the derivation site of each field, so every quantity derived from them
    /// (log2_sievingBlockSize, globalBucketSize, ss_conf.sharedMemReq) follows automatically and
    /// there is no second site to keep in sync. Both are additionally gated on
    /// !use_wide_accumulator_ at the point of use and rejected by validateConfigs on the wide or
    /// legacy paths. See setSievingBlockSizeOverride() / setBigPrimeStartOverride().
    uint32_t sb_override_ = 0;
    uint32_t bpsi_override_ = 0;

    /// v1.0.6 --sieve_bucket_overflow_stats. false = OFF (narrow reports no overflow stats,
    /// byte-identical to v1.0.5). See setNarrowOverflowStats() and getBucketOverflowStats().
    bool narrow_overflow_stats_ = false;

    /// Resolve globalBucketSize for a given sieving-block size, honoring the ablation knob.
    /// override <= 0 (default) returns SB/2 EXACTLY — this is the byte-identical legacy path.
    /// override > 0 returns round(override*SB), floored at 1 slot. globalBucketSize need not
    /// be a power of two (it is a plain per-bucket stride in the entry index math); the legacy
    /// SB/2 and the common factor=1.0 doubling both stay powers of two anyway.
    uint32_t computeGlobalBucketSize(uint32_t sievingBlockSize) const {
        if (bucket_size_factor_override_ <= 0.0)
            return sievingBlockSize / 2;
        double sized = bucket_size_factor_override_ * static_cast<double>(sievingBlockSize);
        uint32_t v = static_cast<uint32_t>(sized + 0.5);
        return v < 1u ? 1u : v;
    }

    // ----- Wide-autotune foundation: batch-wide probe scratch state -----
    // The wide probe (sieveMiniBatch) drives the REAL batch path, which needs the
    // batch-context job arrays + a postprocessing scratch that the autotune ephemeral
    // siever never allocates (it only calls loadData(), not allocateBatchBuffers() /
    // setPostProcessingLinks()). ensureProbeBatchSetup() allocates these ONCE (idempotent)
    // and prepares one batch of polynomials reused by every candidate (A/B fidelity).
    // WIDE-only: none of this is touched on the narrow path (sieveMini never calls it).
    bool      probe_batch_ready_    = false;  ///< true once ensureProbeBatchSetup() has run
    uint32_t  probe_batch_size_     = 0;      ///< #polynomials staged for the probe (set at first setup)
    uint32_t  autotune_probe_polys_override_ = 0; ///< CLI --autotune_probe_polys; 0 = auto-scale by N
    uint32_t* dev_probe_pp_counter_ = nullptr;///< probe-owned pp fill counter (compact atomicAdd target)
    void*     dev_probe_pp_accum_   = nullptr;///< probe-owned DenseCandidate scratch (compact output)
    uint32_t  probe_pp_capacity_    = 0;      ///< capacity (entries) of dev_probe_pp_accum_
    /// Scale-representative harness: the wall-clock measurement window and warm-up
    /// prefix (seconds) used by sieveMiniBatch(). Defaults give a stable, repeatable rate;
    /// the optimizer overrides via setProbeWindow() to auto-scale and honour the budget.
    double    probe_window_sec_     = 10.0;   ///< survivors/sec measured over this wall window
    double    probe_warmup_sec_     = 2.5;    ///< warm-up prefix discarded before measuring
    /// Idempotent one-time setup for sieveMiniBatch(): stage job arrays + pinned buffer,
    /// point dev_pp_* at a bounded probe scratch, and prepare one batch of polynomials.
    void ensureProbeBatchSetup();

    // NOT NECESSARY / OBSOLETE / ONLY FOR DEBUGGING PURPOSES
    // Helper to clear candidate buffer before launch for validation clarity / safety
    void clearCandidates();
    // Helper to clear bucket buffer before launch for validation clarity / safety
    void clearBuckets();
};

} // namespace sieve
} // namespace mpqs
