// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file tcp_backend.h
/// @brief TCP implementation of CommBackend: epoll-based (coordinator) / single-socket (worker).
///
/// Wraps TcpSocket for all transport concerns. Coordinator uses epoll for multiplexed
/// multi-client I/O. Worker uses a single blocking connection to the coordinator.

#include "comm_backend.h"
#include "tcp_transport.h"

#include <vector>

namespace mpqs::cluster {

/// TCP implementation of CommBackend.
/// Coordinator mode: listen + epoll for multi-client I/O, HELLO/HELLO_ACK handshake.
/// Worker mode: single connection to coordinator, HELLO/HELLO_ACK handshake.
class TCPBackend : public CommBackend {
public:
    TCPBackend(bool is_coordinator, const std::string& host,
               uint16_t port, uint32_t expected_workers,
               uint32_t init_timeout_ms = kDefaultInitTimeoutMs);
    ~TCPBackend() override;

    void setLocalInfo(const HelloPayload& hello) override;
    bool initialize() override;
    void finalize() override;
    bool send(uint8_t target_id, MsgType type, const void* data, uint32_t len) override;
    bool recv(RecvMessage& out) override;
    bool recvBlocking(RecvMessage& out, uint32_t timeout_ms) override;
    void broadcast(MsgType type, const void* data, uint32_t len) override;
    void barrier() override;
    uint32_t peerCount() const override;
    uint8_t selfId() const override;
    HelloPayload peerInfo(uint8_t peer_id) const override;
    bool isPeerConnected(uint8_t peer_id) const override;
    void disconnectPeer(uint8_t peer_id) override;

private:
    bool is_coordinator_;
    std::string host_;
    uint16_t port_;
    uint32_t expected_workers_;
    uint32_t init_timeout_ms_;
    uint8_t self_id_ = 0;

    // Coordinator state
    TcpSocket listen_sock_;
    int epoll_fd_ = -1;
    struct PeerSocket {
        TcpSocket sock;
        uint8_t   peer_id = 0;
        HelloPayload info{};
        bool      connected = true;
    };
    std::vector<PeerSocket> peers_;

    // Worker state
    TcpSocket coord_sock_;
    HelloPayload local_hello_{};

    // Helpers
    PeerSocket* findPeer(uint8_t id);
    bool recvFromEpoll(RecvMessage& out, int timeout_ms);
};

} // namespace mpqs::cluster
