// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file tcp_transport.h
/// @brief RAII TCP socket wrapper with length-prefixed framing + CRC32.

#include "cluster_common.h"
#include "comm_backend.h"  // RecvStatus
#include <string>
#include <vector>
#include <cstdint>

namespace mpqs::cluster {

/// RAII TCP socket wrapper with framed send/recv.
/// Used by both coordinator (server side) and worker (client side).
class TcpSocket {
public:
    TcpSocket();                           ///< Create unconnected socket
    explicit TcpSocket(int fd);            ///< Wrap existing fd (from accept)
    ~TcpSocket();

    TcpSocket(TcpSocket&& o) noexcept;
    TcpSocket& operator=(TcpSocket&& o) noexcept;
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    /// Server: bind + listen on port. Returns true on success.
    bool listen(uint16_t port, int backlog = 4);

    /// Server: accept a single connection (blocking). Returns new TcpSocket.
    TcpSocket accept();

    /// Client: connect to host:port. Returns true on success.
    bool connect(const std::string& host, uint16_t port);

    /// Send a framed message: FrameHeader + payload + CRC32.
    bool sendMsg(MsgType type, const void* payload, uint32_t payload_len);

    /// Receive a framed message. Blocks until complete frame or timeout/error.
    /// Returns false on connection close, error, or CRC mismatch.
    bool recvMsg(MsgType& out_type, std::vector<uint8_t>& out_payload);

    /// Same as recvMsg, but reports *why* it returned via @p out_status so the
    /// caller can tell a benign timeout (EAGAIN/EWOULDBLOCK after SO_RCVTIMEO)
    /// from a dead socket (recv()==0 EOF, or recv()<0 with a real error). This
    /// is what lets the worker chunk-wait loop exit promptly on coordinator
    /// disconnect instead of spinning. @p out_status may be nullptr.
    bool recvMsg(MsgType& out_type, std::vector<uint8_t>& out_payload,
                 RecvStatus* out_status);

    /// Set TCP_NODELAY (disable Nagle's algorithm).
    void setNoDelay(bool enable);

    /// Set receive timeout in milliseconds (0 = no timeout / blocking).
    void setRecvTimeout(uint32_t ms);

    bool isValid() const { return fd_ >= 0; }
    int fd() const { return fd_; }
    void close();

    /// Get peer address as "ip:port" string (for logging).
    std::string peerAddr() const;

private:
    int fd_ = -1;
    uint32_t send_seq_ = 0;
    std::vector<uint8_t> recv_buf_;  ///< Partial frame accumulator for non-blocking mode

    bool sendExact(const void* buf, size_t len);
};

} // namespace mpqs::cluster
