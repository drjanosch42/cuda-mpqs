// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file comm_backend.h
/// @brief Abstract communication backend for cluster sieve.
///
/// Decouples coordinator/worker logic from transport implementation.
/// Concrete implementations: TCPBackend (S4), future MPIBackend.

#include "cluster_common.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mpqs::cluster {

/// Received message from the network layer.
struct RecvMessage {
    uint8_t   sender_id = 0;    ///< Peer ID (0 = coordinator, 1-254 = workers)
    MsgType   type = MsgType::ERROR;
    std::vector<uint8_t> payload;
    bool      valid = false;    ///< True if a message was successfully received
};

/// Why a (blocking) receive returned without a message.
/// Lets callers distinguish a genuine poll timeout (keep waiting) from a
/// closed/errored socket (peer gone — stop waiting). Collapsing both to a bare
/// `false` previously caused a tight spin-and-log-flood in the worker
/// chunk-wait loop when the coordinator closed the connection (recv()==0
/// returns immediately, not after the full timeout).
enum class RecvStatus {
    GOT_MSG,   ///< A complete message was received (return value true).
    TIMEOUT,   ///< Poll interval elapsed with no message; socket still alive.
    CLOSED,    ///< Socket closed (EOF) or errored — peer is gone.
};

/// Abstract communication backend for cluster sieve.
///
/// Encapsulates all transport-level concerns: connection management, framing,
/// handshake (HELLO/HELLO_ACK), and multiplexed I/O. The coordinator and worker
/// interact exclusively through this interface — zero direct socket usage.
///
/// Lifecycle: setLocalInfo() → initialize() → send/recv/broadcast → finalize().
class CommBackend {
public:
    virtual ~CommBackend() = default;

    // --- Lifecycle ---

    /// Set local node handshake info (call before initialize, worker mode).
    /// The coordinator receives peer info during initialize() via the handshake.
    virtual void setLocalInfo(const HelloPayload& hello) { (void)hello; }

    /// Initialize the backend.
    /// Coordinator: listen, accept expected_workers connections, HELLO/HELLO_ACK handshake.
    /// Worker: connect to coordinator, HELLO/HELLO_ACK handshake.
    /// @return true on success
    virtual bool initialize() = 0;

    /// Shut down the backend. Close all connections. Idempotent.
    virtual void finalize() = 0;

    // --- Point-to-point ---

    /// Send a message to a specific peer.
    /// Coordinator: sends to worker target_id. Worker: sends to coordinator (target_id ignored).
    /// @return true on success
    virtual bool send(uint8_t target_id, MsgType type,
                      const void* data, uint32_t len) = 0;

    /// Non-blocking receive from any peer.
    /// @return true if a complete message was received
    virtual bool recv(RecvMessage& out) = 0;

    /// Blocking receive with timeout (milliseconds). 0 = indefinite wait.
    /// @param out_status  Optional: receives why the call returned (GOT_MSG /
    ///                    TIMEOUT / CLOSED). Lets callers tell a benign poll
    ///                    timeout from a dead peer. May be nullptr.
    /// @return true if a complete message was received before timeout
    virtual bool recvBlocking(RecvMessage& out, uint32_t timeout_ms = 0,
                              RecvStatus* out_status = nullptr) = 0;

    // --- Collective ---

    /// Broadcast to all connected peers.
    /// Coordinator: sends to all workers. Worker: no-op.
    virtual void broadcast(MsgType type, const void* data, uint32_t len) = 0;

    /// Barrier: block until all peers have reached this point.
    /// Coordinator: broadcast STOP, wait for all FLUSH_ACKs.
    /// Worker: send FLUSH_ACK, drain remaining messages.
    virtual void barrier() = 0;

    // --- Info ---

    /// Number of connected peers.
    virtual uint32_t peerCount() const = 0;

    /// This node's ID (0 = coordinator, 1+ = worker).
    virtual uint8_t selfId() const = 0;

    /// Get peer's handshake info (valid after initialize).
    /// Coordinator: returns HelloPayload from worker peer_id.
    virtual HelloPayload peerInfo(uint8_t peer_id) const { (void)peer_id; return {}; }

    /// Check if a specific peer is still connected.
    virtual bool isPeerConnected(uint8_t peer_id) const { (void)peer_id; return true; }

    /// Explicitly disconnect a specific peer (close connection, remove from poll set).
    /// Coordinator only. Default no-op — override in concrete backends.
    virtual void disconnectPeer(uint8_t peer_id) { (void)peer_id; }

    bool isCoordinator() const { return selfId() == 0; }
};

/// Factory: create a CommBackend by transport name.
/// @param transport  "tcp" (default). Future: "mpi".
/// @param is_coordinator  true for coordinator, false for worker
/// @param host       Coordinator host (worker only, ignored for coordinator)
/// @param port       TCP port
/// @param expected_workers  Number of workers to accept (coordinator only)
/// @param init_timeout_ms  Initialization window: worker retries + coordinator accept timeout
std::unique_ptr<CommBackend> createCommBackend(
    const std::string& transport,
    bool is_coordinator,
    const std::string& host,
    uint16_t port,
    uint32_t expected_workers,
    uint32_t init_timeout_ms = kDefaultInitTimeoutMs);

} // namespace mpqs::cluster
