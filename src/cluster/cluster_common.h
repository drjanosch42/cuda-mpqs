// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file cluster_common.h
/// @brief Wire protocol definitions for the MPQS LAN cluster sieve.
///
/// Message types, frame header, payload structs, and protocol constants.
/// Shared between coordinator and worker.

#include <cstdint>
#include <cstring>
#include <bit>

namespace mpqs::cluster {

// All target platforms (x86, ARM Jetson) are little-endian
static_assert(std::endian::native == std::endian::little,
              "MPQS cluster protocol assumes little-endian byte order");

/// Wire protocol magic bytes ("MQ")
static constexpr uint16_t kProtocolMagic = 0x4D51;

/// Protocol version — defined for future handshake use but not currently
/// transmitted in FrameHeader (version negotiation is deferred).
static constexpr uint8_t kProtocolVersion = 1;

/// Frame header: 11 bytes on the wire.
/// Layout: [magic:2B][msg_type:1B][seq_no:4B][payload_len:4B]
struct FrameHeader {
    uint16_t magic;       ///< kProtocolMagic
    uint8_t  msg_type;    ///< MsgType enum value
    uint32_t seq_no;      ///< Monotonic per-connection sequence number
    uint32_t payload_len; ///< Byte length of payload (excludes header and CRC)
} __attribute__((packed));

static_assert(sizeof(FrameHeader) == 11);
static constexpr size_t kCRC32Size = 4;

/// Message types (plan Section 3.3).
enum class MsgType : uint8_t {
    // Connection
    HELLO           = 0x01,  ///< Worker -> Coordinator: registration
    HELLO_ACK       = 0x02,  ///< Coordinator -> Worker: accepted + worker_id

    // Work distribution
    WORK_ASSIGN     = 0x10,  ///< Coordinator -> Worker: N + factor base + params + poly range
    WORK_MORE       = 0x11,  ///< Coordinator -> Worker: additional poly range (Stage 2)
    WORK_REQUEST    = 0x12,  ///< Worker -> Coordinator: request more work (Stage 2)

    // S5: Dynamic chunked distribution
    CHUNK_ASSIGN    = 0x13,  ///< Coordinator -> Worker: chunk of a-values
    CHUNK_COMPLETE  = 0x14,  ///< Worker -> Coordinator: chunk done + stats

    // S8: Straggler recall
    CHUNK_RECALL    = 0x17,  ///< Coordinator -> Worker: stop current chunk, report partial progress

    // Relation transfer
    RELATION_BATCH  = 0x20,  ///< Worker -> Coordinator: serialized HostRelationBatch
    PARTIAL_BATCH   = 0x21,  ///< Worker -> Coordinator: 1-partial relations for LP matching
    INCREMENTAL_BATCH = 0x22, ///< Worker -> Coordinator: incremental full + partials (DataTap)

    // Telemetry
    HEARTBEAT       = 0x30,  ///< Worker -> Coordinator: alive + stats
    STATUS          = 0x31,  ///< Coordinator -> Worker: global progress

    // Control
    STOP            = 0xF0,  ///< Coordinator -> Worker: terminate sieving
    FLUSH_ACK       = 0xF1,  ///< Worker -> Coordinator: final batch sent
    ERROR           = 0xFF,  ///< Either direction: fatal error
};

/// HELLO payload (worker -> coordinator), 86 bytes.
struct HelloPayload {
    char     gpu_name[64];          ///< Null-padded GPU name
    uint16_t sm_version;            ///< e.g. 87 for SM 8.7
    uint16_t num_sms;
    uint32_t vram_mb;
    float    capacity_estimate;     ///< Relative throughput hint [0.0, 1.0]
    uint16_t clock_mhz;             ///< GPU boost clock in MHz (SM×clock weight input)

    // Resume fields (Stage 3)
    uint8_t  resume_from_checkpoint;  ///< 0=fresh start, 1=resuming from checkpoint
    uint8_t  resume_worker_id;        ///< Previous worker_id (0 if fresh)
    uint64_t resume_relations_sent;   ///< Relations already sent (coordinator can verify)
} __attribute__((packed));
static_assert(sizeof(HelloPayload) == 88);

/// HELLO_ACK payload (coordinator -> worker), 2 bytes.
struct HelloAckPayload {
    uint8_t worker_id;              ///< Assigned ID (1-254)
    uint8_t status;                 ///< 0=accepted, 1=rejected
} __attribute__((packed));

/// HEARTBEAT payload (worker -> coordinator).
struct HeartbeatPayload {
    uint8_t  worker_id;
    uint8_t  _pad[3];              ///< Alignment padding
    uint64_t relations_found;      ///< Cumulative full relations
    uint64_t partials_found;       ///< Cumulative 1-partials (0 in Stage 1)
    uint64_t batches_completed;
    uint8_t  polys_exhausted;      ///< 1 if worker's poly range is done
    uint8_t  _pad2[1];
    uint16_t gpu_temp_celsius;
    uint64_t timestamp_ms;         ///< Monotonic clock
} __attribute__((packed));

/// STATUS payload (coordinator -> worker).
struct StatusPayload {
    uint64_t total_relations;      ///< Global count (all nodes)
    uint64_t target_relations;
    uint64_t total_partials;       ///< 0 in Stage 1
    uint64_t lp_matches;           ///< 0 in Stage 1
    float    eta_seconds;          ///< -1 if unknown
} __attribute__((packed));

/// WORK_MORE payload (coordinator -> worker), 16 bytes.
struct WorkMorePayload {
    uint64_t poly_range_start;
    uint64_t poly_range_count;
} __attribute__((packed));
static_assert(sizeof(WorkMorePayload) == 16);

/// CHUNK_ASSIGN payload (coordinator -> worker), 24 bytes.
struct ChunkAssignPayload {
    uint32_t chunk_id;           ///< Unique chunk identifier
    uint64_t poly_range_start;   ///< First a-value index
    uint64_t poly_range_count;   ///< Number of a-values
    uint32_t flags;              ///< Bit 0: is_final_chunk, Bit 1: is_initial_chunk

    static constexpr uint32_t FLAG_FINAL    = 0x1;
    static constexpr uint32_t FLAG_INITIAL  = 0x2;
    static constexpr uint32_t FLAG_OVERFLOW = 0x4;  ///< Chunk from overflow pool (not contiguous range)
} __attribute__((packed));
static_assert(sizeof(ChunkAssignPayload) == 24);

/// CHUNK_COMPLETE payload (worker -> coordinator), 32 bytes.
struct ChunkCompletePayload {
    uint32_t chunk_id;           ///< Matches CHUNK_ASSIGN
    uint32_t elapsed_ms;         ///< Wall time for this chunk
    uint64_t relations_found;    ///< Full relations from this chunk
    uint64_t partials_found;     ///< 1-partials from this chunk
    uint64_t a_values_consumed;  ///< Actual a-values consumed (<= assigned if STOP received)
} __attribute__((packed));
static_assert(sizeof(ChunkCompletePayload) == 32);

/// CHUNK_RECALL payload (coordinator -> worker), 4 bytes.
/// Instructs worker to stop its current chunk at the next sieve boundary.
/// Worker responds with CHUNK_COMPLETE carrying a_values_consumed < assigned.
struct ChunkRecallPayload {
    uint32_t chunk_id;  ///< Chunk being recalled (must match worker's current chunk)
} __attribute__((packed));
static_assert(sizeof(ChunkRecallPayload) == 4);

/// STOP payload (coordinator -> worker), 1 byte.
struct StopPayload {
    uint8_t reason;                ///< 0=target_reached, 1=error, 2=user_abort
} __attribute__((packed));

// Protocol constants
static constexpr uint16_t kDefaultPort = 9100;
static constexpr uint32_t kDefaultInitTimeoutMs = 300000;  ///< 300s init window (Jetson cold JIT ~80s)
static constexpr uint32_t kConnectRetryIntervalMs = 5000;  ///< Worker retry interval during init window
static constexpr uint32_t kHeartbeatIntervalMs = 5000;
// 120s: accommodates Jetson workers with cuda_graph_unroll=8 (~30s/replay).
// Initial NetworkDataTap heartbeat + graph_capture(~15s) + first_replay(~16s)
// ≈ 31s to second heartbeat — exceeds the old 30s limit by ~1s.
static constexpr uint32_t kFlushTimeoutMs = 120000;
static constexpr uint32_t kBatchSendThreshold = 1000;   ///< Send every N relations
static constexpr uint32_t kBatchSendCeilingMs = 10000;   ///< Or every 10 seconds

/// CRC32 computation (table-driven, polynomial 0xEDB88320). Thread-safe.
uint32_t computeCRC32(const void* data, size_t len);

} // namespace mpqs::cluster
