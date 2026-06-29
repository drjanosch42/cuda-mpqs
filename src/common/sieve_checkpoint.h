// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "mpqs_soa.h"
#include "relation_io.h"
#include "uint512.cuh"
#include <string>
#include <vector>
#include <cstdint>

/// @file sieve_checkpoint.h
/// @brief Atomic, crash-safe mid-sieve checkpoint artifact (`sieve.ckpt`).
///
/// Layout (plan §2.3):
///   [ serialize_v2 payload (verbatim) ] [ progress trailer ] [ cluster block (S3) ]
///   [ fixed-size EOF footer ]
///
/// The trailer is read by SEEKING from the trailer_offset recorded in the fixed-size EOF
/// footer (the trailer is variable-size once the S3 cluster block lands, so "read a
/// fixed-size struct at EOF" does not work — the footer is the only fixed-position record).
/// The footer magic at the very end of the file is the COMPLETENESS SENTINEL: a torn write
/// never has it, so a partially-written checkpoint is never loadable.
///
/// `deserialize_v2` reads section-by-section and returns `f.good()` WITHOUT checking EOF
/// (relation_io.cpp), so the trailer + footer are invisible to it and to ordinary
/// `relations.v2` consumers — the checkpoint reuses the tested serializer verbatim.

namespace mpqs::ckpt {

/// 9-byte magics (not NUL-terminated on disk).
constexpr char     CKPT_TRAILER_MAGIC[9] = {'M','P','Q','S','_','C','K','P','T'};
constexpr char     CKPT_FOOTER_MAGIC[9]  = {'M','P','Q','S','_','C','K','F','T'};
constexpr uint32_t CKPT_SCHEMA_VERSION   = 1;

/// On-disk size of the fixed EOF footer: magic(9) + trailer_offset(8) + trailer_len(8) +
/// schema_version(4) = 29 bytes. Load seeks (file_size - CKPT_FOOTER_SIZE) to read it.
constexpr uint64_t CKPT_FOOTER_SIZE = 9 + 8 + 8 + 4;

/// Progress trailer (in-memory form). Serialized field-by-field by writeCheckpointAtomic
/// (NOT a raw struct dump — avoids padding/ABI ambiguity, matching relation_io.cpp style).
struct CheckpointTrailer {
    uint64_t global_a_index       = 0;   ///< Solo cursor: a-values consumed == first un-sieved a-index.
    uint64_t target_relations     = 0;   ///< config_.target_relations at write time.
    uint64_t loaded_smooths_raw   = 0;   ///< RAW (pre-dedup) device smooth count (M2 target accounting).
    uint64_t loaded_smooths_dedup = 0;   ///< Deduped smooth count == #smooths actually in the payload.
    uint64_t loaded_partials      = 0;   ///< Raw partial/witness count in the payload.
    uint64_t lp1_bound            = 0;   ///< LP bound (0 = no-LP run).
    uint32_t sieve_bound          = 0;   ///< M (sieve interval half-width).
    mpqs::uint512 N{};                   ///< Modulus (64 B, redundant sanity vs the v2 metadata).
    uint8_t  cluster_section_present = 0;///< 0 = solo file; 1 = an S3 cluster block follows (S3).
    uint64_t elapsed_sieve_sec    = 0;   ///< Wall-clock sieve time accumulated so far.
};

/// Variable-size cluster resume block (S3 — coordinator only). Appended between the
/// progress trailer and the fixed EOF footer, present iff trailer.cluster_section_present
/// == 1. Its size grows with node count, which is exactly why the EOF footer carries
/// {trailer_offset, trailer_len}: the trailer+cluster-block region is addressable by
/// seeking, not by a fixed-size struct read.
///
/// On-disk form (little-endian, field-by-field):
///   completed_prefix_cursor (u64)        — WorkPool::completedPrefixCursor() (B2)
///   node_count              (u32)        — size of the high-water array below
///   initial_high_water[node_count] (u64) — per-node initial-range contiguous high-water
///                                          (M1; index == node_id, node 0 == coordinator)
struct CheckpointClusterBlock {
    uint64_t completed_prefix_cursor = 0;
    std::vector<uint64_t> initial_high_water;  ///< index == node_id (node 0 = coordinator)
};

/// Result of loading a checkpoint.
struct CheckpointLoadResult {
    bool ok = false;
    CheckpointTrailer trailer;
    CheckpointClusterBlock cluster;   ///< Populated iff trailer.cluster_section_present == 1 (S3).
    mpqs::structures::HostRelationBatch smooths;
    mpqs::structures::HostRelationBatch partials;
    mpqs::io::V2Metadata meta;
};

/// Deduplicate `b` in place, keeping the first occurrence of each relation (by the shared
/// `mpqs::computeRelationHash`). Maintains CSR validity. Used to shrink the checkpoint file
/// — it operates on a SCRATCH host copy and never touches live device state (B1).
void dedupRelationsInPlace(mpqs::structures::HostRelationBatch& b);

/// Write `<ckpt_dir>/sieve.ckpt` atomically and crash-safely (plan §2.3):
///   0. unlink any stale `sieve.ckpt.tmp`
///   1. serialize_v2(smooths, partials, meta) → `sieve.ckpt.tmp`  (intra-FS tmp)
///   2. append trailer (+ S3 cluster block) + fixed EOF footer; fflush + fsync(fd)
///   3. rename existing `sieve.ckpt` → `sieve.ckpt.prev`, then rename tmp → `sieve.ckpt`
///   4. fsync the containing directory
/// A kill at any earlier point leaves only `sieve.ckpt.tmp` (ignored + unlinked next write)
/// and the previous committed `sieve.ckpt` intact. Returns false on any I/O failure.
///
/// `cluster` (S3): when non-null AND `trailer.cluster_section_present == 1`, the variable-
/// size cluster block is serialized between the trailer and the fixed EOF footer (the
/// footer's trailer_len then spans trailer + cluster block). Pass nullptr (the default,
/// used by the solo S1 path) to write a solo file with no cluster block.
bool writeCheckpointAtomic(const std::string& ckpt_dir,
                           const mpqs::structures::HostRelationBatch& smooths,
                           const mpqs::structures::HostRelationBatch& partials,
                           const mpqs::io::V2Metadata& meta,
                           const CheckpointTrailer& trailer,
                           const CheckpointClusterBlock* cluster = nullptr);

/// Read+validate a single checkpoint file: footer (fixed offset from EOF) → trailer (at
/// footer.trailer_offset) → `deserialize_v2` payload. A missing/garbled footer magic or a
/// failed schema/offset sanity check returns false (treat as torn/incomplete). Fully
/// consumed by S2 resume; S1 ships enough for the validator.
bool readCheckpoint(const std::string& path, CheckpointLoadResult& out);

/// Pick the freshest valid checkpoint in `dir`: try `sieve.ckpt`, fall back to
/// `sieve.ckpt.prev` if the live file is missing/torn. Returns false if neither loads.
bool loadLatestCheckpoint(const std::string& dir, CheckpointLoadResult& out);

// ============================================================================
// S4 cluster-resume pure helpers (testable in isolation — no orchestrator deps)
// ============================================================================

/// Result of trimming one node's initial contiguous range by its loaded high-water.
struct ResumeTrim {
    uint64_t start;    ///< First a-index to (re-)sieve == orig_start + eff_hw.
    uint64_t count;    ///< Number of a-values to sieve  == orig_count - eff_hw.
    uint64_t eff_hw;   ///< Effective high-water actually applied. == the offset to add back
                       ///< to this leg's consumed count when re-recording the absolute
                       ///< high-water for the NEXT checkpoint (so multi-resume never mis-trims).
};

/// Compute the resumed initial-range assignment for one node (M1, plan §2.6.4, corrected
/// option (a)): re-issue `[orig_start + hw, orig_count − hw)`. `hw` is the per-node
/// initial-range contiguous high-water loaded from the cluster block. It is **conservative**
/// (telemetry can lag a CHUNK_COMPLETE by a beat — N1), so re-sieving from it never skips
/// un-sieved a-values; dedup absorbs the small overlap.
///
/// **Boundary hardening (S4):** a trimmed `count == 0` would set the DataTap range limit to 0,
/// which BOTH `DirectChannel::setRange` and `AsyncNetworkDataTap::setRange` interpret as
/// "UNBOUNDED" (`range_a_limit_ == 0` disables the bound) — a node whose initial range is
/// exactly complete would then sieve the WHOLE a-space. So when the trim would zero the count,
/// fall back to re-sieving the last hypercube (`H` a-values, dedup-safe): `eff_hw = orig_count − H`,
/// `count = H`. The invariant `eff_hw + count == orig_count` holds in BOTH branches, so a
/// fully-sieved trimmed leg re-records the absolute high-water as exactly `orig_count`.
///
/// `H` is the hypercube size (== 2^shc_dim); ranges from `computeContiguousRanges` are H-aligned
/// so `orig_count` is 0 or ≥ H. `orig_count == 0` (a zero-weight node with no initial range) is
/// passed through unchanged (count stays 0 — that node already went straight to overflow on a
/// fresh run, so resume keeps that behaviour; no unbounded risk because the count was already 0).
inline ResumeTrim computeResumeTrim(uint64_t orig_start, uint64_t orig_count,
                                    uint64_t hw, uint64_t H) {
    if (hw > orig_count) hw = orig_count;            // clamp (defensive)
    uint64_t eff_hw = hw;
    uint64_t count  = orig_count - eff_hw;
    if (count == 0 && orig_count >= H && H > 0) {    // avoid the 0-count → unbounded trap
        eff_hw = orig_count - H;                     // re-sieve the last hypercube (dedup-safe)
        count  = H;
    }
    return ResumeTrim{ orig_start + eff_hw, count, eff_hw };
}

/// N2 topology/geometry guard (plan §2.6, S4 audit carry). A cluster checkpoint's per-node
/// high-water array is indexed by node_id, so the trim is valid ONLY under the SAME node
/// topology AND the SAME overflow-pool geometry. Returns true iff the loaded block may be
/// safely consumed:
///   * the loaded per-node high-water count equals the current run's node count (no
///     index→range mis-map across a changed worker/node count), AND
///   * the loaded `completed_prefix_cursor` lies within the current overflow pool
///     `[overflow_start, overflow_end]` (a different target/weights → different overflow_start
///     would otherwise make `setCursor()` restore a meaningless cursor, or trip its
///     `cursor <= end_` assert).
/// A false return means: REJECT the resume and start fresh (the documented safe fallback — a
/// fresh run re-sieves everything, never skips). NB this catches topology AND the common
/// geometry shifts; it cannot catch a same-node-count, same-overflow-start run whose target
/// changed without moving overflow_start (an unusual operator error) — the trailer-N check and
/// prefix-bounds give partial cover; documented in the S4 report.
inline bool clusterResumeTopologyOk(uint64_t loaded_node_count, uint64_t current_node_count,
                                    uint64_t completed_prefix_cursor,
                                    uint64_t overflow_start, uint64_t overflow_end) {
    if (loaded_node_count != current_node_count) return false;
    if (completed_prefix_cursor < overflow_start)  return false;
    if (completed_prefix_cursor > overflow_end)    return false;
    return true;
}

} // namespace mpqs::ckpt
