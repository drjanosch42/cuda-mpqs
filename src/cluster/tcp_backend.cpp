// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/// @file tcp_backend.cpp
/// @brief TCP implementation of CommBackend: epoll (coordinator) / single-socket (worker).

#include "tcp_backend.h"
#include "hpc_logger.h"

#include <sys/epoll.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <thread>

namespace mpqs::cluster {

// ============================================================================
// Construction / Destruction
// ============================================================================

TCPBackend::TCPBackend(bool is_coordinator, const std::string& host,
                       uint16_t port, uint32_t expected_workers,
                       uint32_t init_timeout_ms)
    : is_coordinator_(is_coordinator), host_(host), port_(port),
      expected_workers_(expected_workers),
      init_timeout_ms_(init_timeout_ms) {}

TCPBackend::~TCPBackend() {
    finalize();
}

// ============================================================================
// Lifecycle
// ============================================================================

void TCPBackend::setLocalInfo(const HelloPayload& hello) {
    local_hello_ = hello;
}

bool TCPBackend::initialize() {
    using clock = std::chrono::steady_clock;

    if (is_coordinator_) {
        // --- Coordinator: listen, accept, HELLO/HELLO_ACK handshake ---
        if (!listen_sock_.listen(port_)) {
            LOG(LOG_ERROR_CRITICAL) << "[Cluster] Failed to listen on port " << port_;
            return false;
        }
        LOG(LOG_INFO) << "[Cluster] Coordinator listening on port " << port_;

        epoll_fd_ = epoll_create1(0);
        if (epoll_fd_ < 0) {
            LOG(LOG_ERROR_CRITICAL) << "[Cluster] epoll_create1 failed: " << strerror(errno);
            return false;
        }

        // Register listen socket for accept events
        struct epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.fd = listen_sock_.fd();
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_sock_.fd(), &ev);

        // Accept phase — wait for expected_workers within init timeout
        LOG(LOG_INFO) << "[Cluster] Waiting for " << expected_workers_ << " workers ("
                      << (init_timeout_ms_ / 1000) << "s timeout)...";
        auto deadline = clock::now() + std::chrono::milliseconds(init_timeout_ms_);

        while (peers_.size() < expected_workers_) {
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - clock::now()).count();
            if (remaining <= 0) {
                LOG(LOG_WARNING) << "[Cluster] Accept timeout. Got " << peers_.size()
                                 << "/" << expected_workers_ << " workers. Proceeding.";
                break;
            }

            struct epoll_event events[4];
            int n = epoll_wait(epoll_fd_, events, 4,
                               static_cast<int>(std::min<decltype(remaining)>(remaining, 1000)));

            for (int i = 0; i < n; i++) {
                if (events[i].data.fd != listen_sock_.fd()) continue;

                TcpSocket client = listen_sock_.accept();
                if (!client.isValid()) continue;

                // Receive HELLO (blocking — fresh connection)
                MsgType type;
                std::vector<uint8_t> payload;
                client.setRecvTimeout(30000);  // 30s — SSH relay can buffer HELLO for several seconds
                if (!client.recvMsg(type, payload) || type != MsgType::HELLO) {
                    LOG(LOG_WARNING) << "[Cluster] Expected HELLO, closing connection";
                    client.close();
                    continue;
                }
                HelloPayload hello{};
                memcpy(&hello, payload.data(), std::min(payload.size(), sizeof(hello)));

                // Assign worker_id and send HELLO_ACK
                uint8_t wid = static_cast<uint8_t>(peers_.size() + 1);
                HelloAckPayload ack{wid, 0};
                client.sendMsg(MsgType::HELLO_ACK, &ack, sizeof(ack));

                LOG(LOG_INFO) << "[Cluster] Worker " << static_cast<int>(wid) << " connected: "
                              << hello.gpu_name << " (SM " << hello.sm_version / 10
                              << "." << hello.sm_version % 10 << ", " << hello.num_sms
                              << " SMs, capacity=" << std::fixed << std::setprecision(2)
                              << hello.capacity_estimate << ")";
                if (hello.resume_from_checkpoint) {
                    LOG(LOG_INFO) << "[Cluster] Worker resuming: prior_id="
                                  << static_cast<int>(hello.resume_worker_id)
                                  << ", rels_sent=" << hello.resume_relations_sent;
                }

                peers_.push_back(PeerSocket{std::move(client), wid, hello, true});
            }
        }

        // Remove listen socket from epoll (done accepting)
        epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, listen_sock_.fd(), nullptr);

        // Set worker sockets to non-blocking and register in epoll
        for (auto& peer : peers_) {
            int flags = fcntl(peer.sock.fd(), F_GETFL, 0);
            fcntl(peer.sock.fd(), F_SETFL, flags | O_NONBLOCK);
            struct epoll_event cev{};
            cev.events = EPOLLIN;
            cev.data.fd = peer.sock.fd();
            epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, peer.sock.fd(), &cev);
        }

        self_id_ = 0;
        return true;

    } else {
        // --- Worker: connect with retry within init timeout window ---
        auto deadline = clock::now() + std::chrono::milliseconds(init_timeout_ms_);
        uint32_t attempt = 0;

        while (true) {
            attempt++;
            if (coord_sock_.connect(host_, port_)) {
                LOG(LOG_INFO) << "[Cluster/Worker] Connected to " << host_ << ":" << port_
                              << " (attempt " << attempt << ")";
                break;
            }

            auto now = clock::now();
            if (now >= deadline) {
                LOG(LOG_ERROR_CRITICAL) << "[Cluster/Worker] Failed to connect to "
                                        << host_ << ":" << port_
                                        << " after " << attempt << " attempts ("
                                        << (init_timeout_ms_ / 1000) << "s timeout)";
                return false;
            }

            auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now).count();
            uint32_t sleep_ms = std::min(static_cast<uint64_t>(kConnectRetryIntervalMs),
                                         static_cast<uint64_t>(remaining_ms));

            LOG(LOG_DEBUG_1) << "[Cluster/Worker] Connection attempt " << attempt
                             << " failed (retrying in " << sleep_ms << "ms, "
                             << (remaining_ms / 1000) << "s remaining)";

            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }

        // Send HELLO
        coord_sock_.sendMsg(MsgType::HELLO, &local_hello_, sizeof(local_hello_));

        // Receive HELLO_ACK
        MsgType type;
        std::vector<uint8_t> payload;
        if (!coord_sock_.recvMsg(type, payload) || type != MsgType::HELLO_ACK) {
            LOG(LOG_ERROR_CRITICAL) << "[Cluster/Worker] Expected HELLO_ACK";
            return false;
        }
        HelloAckPayload ack{};
        memcpy(&ack, payload.data(), std::min(payload.size(), sizeof(ack)));
        if (ack.status != 0) {
            LOG(LOG_ERROR_CRITICAL) << "[Cluster/Worker] Rejected by coordinator";
            return false;
        }
        self_id_ = ack.worker_id;
        LOG(LOG_INFO) << "[Cluster/Worker] Assigned worker_id=" << static_cast<int>(self_id_);
        return true;
    }
}

void TCPBackend::finalize() {
    for (auto& peer : peers_) peer.sock.close();
    listen_sock_.close();
    coord_sock_.close();
    if (epoll_fd_ >= 0) { ::close(epoll_fd_); epoll_fd_ = -1; }
}

// ============================================================================
// Point-to-point
// ============================================================================

bool TCPBackend::send(uint8_t target_id, MsgType type, const void* data, uint32_t len) {
    if (is_coordinator_) {
        PeerSocket* peer = findPeer(target_id);
        if (!peer || !peer->connected) return false;
        return peer->sock.sendMsg(type, data, len);
    } else {
        return coord_sock_.sendMsg(type, data, len);
    }
}

bool TCPBackend::recv(RecvMessage& out) {
    if (is_coordinator_) {
        return recvFromEpoll(out, 0);
    } else {
        // Non-blocking check: 1ms timeout
        coord_sock_.setRecvTimeout(1);
        MsgType type;
        std::vector<uint8_t> payload;
        bool got = coord_sock_.recvMsg(type, payload);
        coord_sock_.setRecvTimeout(0);
        if (got) {
            out.sender_id = 0;
            out.type = type;
            out.payload = std::move(payload);
            out.valid = true;
            return true;
        }
        return false;
    }
}

bool TCPBackend::recvBlocking(RecvMessage& out, uint32_t timeout_ms) {
    if (is_coordinator_) {
        return recvFromEpoll(out, static_cast<int>(timeout_ms));
    } else {
        coord_sock_.setRecvTimeout(timeout_ms);
        MsgType type;
        std::vector<uint8_t> payload;
        bool got = coord_sock_.recvMsg(type, payload);
        coord_sock_.setRecvTimeout(0);
        if (got) {
            out.sender_id = 0;
            out.type = type;
            out.payload = std::move(payload);
            out.valid = true;
            return true;
        }
        return false;
    }
}

// ============================================================================
// Collective
// ============================================================================

void TCPBackend::broadcast(MsgType type, const void* data, uint32_t len) {
    if (!is_coordinator_) return;
    for (auto& peer : peers_) {
        if (peer.connected)
            peer.sock.sendMsg(type, data, len);
    }
}

void TCPBackend::barrier() {
    if (is_coordinator_) {
        // Send STOP to all, wait for FLUSH_ACKs
        StopPayload stop{0};
        broadcast(MsgType::STOP, &stop, sizeof(stop));

        uint32_t flushed = 0;
        uint32_t total = static_cast<uint32_t>(peers_.size());
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);

        while (flushed < total) {
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            if (remaining <= 0) break;

            RecvMessage msg;
            if (recvFromEpoll(msg, static_cast<int>(std::min<decltype(remaining)>(remaining, 1000)))) {
                if (msg.type == MsgType::FLUSH_ACK) flushed++;
            }
        }
    } else {
        // Worker: send FLUSH_ACK, drain remaining messages
        coord_sock_.sendMsg(MsgType::FLUSH_ACK, nullptr, 0);
        coord_sock_.setRecvTimeout(5000);
        MsgType type;
        std::vector<uint8_t> payload;
        coord_sock_.recvMsg(type, payload);  // Drain
        coord_sock_.setRecvTimeout(0);
    }
}

// ============================================================================
// Info
// ============================================================================

uint32_t TCPBackend::peerCount() const {
    return static_cast<uint32_t>(peers_.size());
}

uint8_t TCPBackend::selfId() const {
    return self_id_;
}

HelloPayload TCPBackend::peerInfo(uint8_t peer_id) const {
    for (const auto& peer : peers_) {
        if (peer.peer_id == peer_id) return peer.info;
    }
    return {};
}

bool TCPBackend::isPeerConnected(uint8_t peer_id) const {
    for (const auto& peer : peers_) {
        if (peer.peer_id == peer_id) return peer.connected;
    }
    return false;
}

void TCPBackend::disconnectPeer(uint8_t peer_id) {
    if (!is_coordinator_) return;
    PeerSocket* peer = findPeer(peer_id);
    if (!peer || !peer->connected) return;

    // Remove from epoll before closing (safe even if already removed)
    if (epoll_fd_ >= 0 && peer->sock.fd() >= 0) {
        epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, peer->sock.fd(), nullptr);
    }
    peer->sock.close();
    peer->connected = false;

    LOG(LOG_INFO) << "[Cluster/TCP] Disconnected peer " << static_cast<int>(peer_id);
}

// ============================================================================
// Private helpers
// ============================================================================

TCPBackend::PeerSocket* TCPBackend::findPeer(uint8_t id) {
    for (auto& peer : peers_) {
        if (peer.peer_id == id) return &peer;
    }
    return nullptr;
}

bool TCPBackend::recvFromEpoll(RecvMessage& out, int timeout_ms) {
    // First check all peers for buffered data (complete frames in TcpSocket::recv_buf_).
    // This handles the case where a previous read pulled multiple frames into the
    // userspace buffer but we only returned one — epoll won't re-report these.
    for (auto& peer : peers_) {
        if (!peer.connected) continue;
        MsgType type;
        std::vector<uint8_t> payload;
        if (peer.sock.recvMsg(type, payload)) {
            out.sender_id = peer.peer_id;
            out.type = type;
            out.payload = std::move(payload);
            out.valid = true;
            return true;
        }
    }

    // No buffered data — wait for new data via epoll
    struct epoll_event events[16];
    int n = epoll_wait(epoll_fd_, events, 16, timeout_ms);

    for (int i = 0; i < n; i++) {
        int fd = events[i].data.fd;

        // Find peer by fd
        PeerSocket* peer = nullptr;
        for (auto& p : peers_) {
            if (p.connected && p.sock.fd() == fd) { peer = &p; break; }
        }
        if (!peer) continue;

        // Handle errors / hangup
        if (events[i].events & (EPOLLERR | EPOLLHUP)) {
            LOG(LOG_WARNING) << "[Cluster] Worker " << static_cast<int>(peer->peer_id)
                             << " disconnected (epoll error/hangup)";
            out.sender_id = peer->peer_id;
            out.type = MsgType::ERROR;
            out.valid = true;
            peer->connected = false;
            epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
            return true;
        }

        // Try to read a message
        MsgType type;
        std::vector<uint8_t> payload;
        if (peer->sock.recvMsg(type, payload)) {
            out.sender_id = peer->peer_id;
            out.type = type;
            out.payload = std::move(payload);
            out.valid = true;
            return true;
        }

        // recvMsg failed — check if connection lost
        if (!peer->sock.isValid()) {
            LOG(LOG_WARNING) << "[Cluster] Worker " << static_cast<int>(peer->peer_id)
                             << " connection lost";
            out.sender_id = peer->peer_id;
            out.type = MsgType::ERROR;
            out.valid = true;
            peer->connected = false;
            epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
            return true;
        }
    }

    return false;
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<CommBackend> createCommBackend(
    const std::string& transport,
    bool is_coordinator,
    const std::string& host,
    uint16_t port,
    uint32_t expected_workers,
    uint32_t init_timeout_ms)
{
    if (transport == "tcp" || transport.empty()) {
        return std::make_unique<TCPBackend>(is_coordinator, host, port, expected_workers,
                                            init_timeout_ms);
    }
    LOG(LOG_ERROR_CRITICAL) << "[Cluster] Unknown transport: " << transport;
    return nullptr;
}

} // namespace mpqs::cluster
