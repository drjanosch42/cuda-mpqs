// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/// @file tcp_transport.cpp
/// @brief POSIX TCP socket implementation with length-prefixed framing and CRC32.

#include "tcp_transport.h"
#include "hpc_logger.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <mutex>
#include <cerrno>
#include <cstring>

namespace mpqs::cluster {

// =============================================================================
// CRC32 (table-driven, polynomial 0xEDB88320)
// =============================================================================

static uint32_t crc32_table[256];
static std::once_flag crc32_init_flag;

static void initCRC32Table() {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320u : 0u);
        crc32_table[i] = crc;
    }
}

uint32_t computeCRC32(const void* data, size_t len) {
    std::call_once(crc32_init_flag, initCRC32Table);
    const auto* p = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++)
        crc = (crc >> 8) ^ crc32_table[(crc ^ p[i]) & 0xFF];
    return crc ^ 0xFFFFFFFF;
}

// =============================================================================
// TcpSocket
// =============================================================================

TcpSocket::TcpSocket() = default;

TcpSocket::TcpSocket(int fd) : fd_(fd) {}

TcpSocket::~TcpSocket() { close(); }

TcpSocket::TcpSocket(TcpSocket&& o) noexcept
    : fd_(o.fd_), send_seq_(o.send_seq_), recv_buf_(std::move(o.recv_buf_)) {
    o.fd_ = -1;
    o.send_seq_ = 0;
}

TcpSocket& TcpSocket::operator=(TcpSocket&& o) noexcept {
    if (this != &o) {
        close();
        fd_ = o.fd_;
        send_seq_ = o.send_seq_;
        recv_buf_ = std::move(o.recv_buf_);
        o.fd_ = -1;
        o.send_seq_ = 0;
    }
    return *this;
}

void TcpSocket::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool TcpSocket::listen(uint16_t port, int backlog) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        LOG(LOG_ERROR_CRITICAL) << "[TCP] socket() failed: " << strerror(errno);
        return false;
    }

    int opt = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        LOG(LOG_ERROR_CRITICAL) << "[TCP] bind() failed on port " << port << ": " << strerror(errno);
        close();
        return false;
    }

    if (::listen(fd_, backlog) < 0) {
        LOG(LOG_ERROR_CRITICAL) << "[TCP] listen() failed: " << strerror(errno);
        close();
        return false;
    }

    LOG(LOG_INFO) << "[TCP] Listening on port " << port;
    return true;
}

TcpSocket TcpSocket::accept() {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = ::accept(fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
        LOG(LOG_ERROR_CRITICAL) << "[TCP] accept() failed: " << strerror(errno);
        return TcpSocket();
    }

    TcpSocket client(client_fd);
    client.setNoDelay(true);
    return client;
}

bool TcpSocket::connect(const std::string& host, uint16_t port) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        LOG(LOG_ERROR_CRITICAL) << "[TCP] socket() failed: " << strerror(errno);
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    // Try numeric first, then DNS
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        struct addrinfo hints{}, *result = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        int rc = getaddrinfo(host.c_str(), nullptr, &hints, &result);
        if (rc != 0 || !result) {
            LOG(LOG_ERROR_CRITICAL) << "[TCP] Cannot resolve host '" << host << "': " << gai_strerror(rc);
            close();
            return false;
        }
        addr.sin_addr = reinterpret_cast<sockaddr_in*>(result->ai_addr)->sin_addr;
        freeaddrinfo(result);
    }

    if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        LOG(LOG_WARNING) << "[TCP] connect() to " << host << ":" << port
                        << " failed: " << strerror(errno);
        close();
        return false;
    }

    setNoDelay(true);
    return true;
}

bool TcpSocket::sendMsg(MsgType type, const void* payload, uint32_t payload_len) {
    FrameHeader hdr;
    hdr.magic = kProtocolMagic;
    hdr.msg_type = static_cast<uint8_t>(type);
    hdr.seq_no = send_seq_++;
    hdr.payload_len = payload_len;

    // Compute CRC32 over header + payload
    // Use a temp buffer to avoid two-pass CRC on separate memory regions
    const size_t frame_size = sizeof(FrameHeader) + payload_len;
    std::vector<uint8_t> frame_buf(frame_size);
    memcpy(frame_buf.data(), &hdr, sizeof(FrameHeader));
    if (payload_len > 0 && payload)
        memcpy(frame_buf.data() + sizeof(FrameHeader), payload, payload_len);
    uint32_t crc = computeCRC32(frame_buf.data(), frame_size);

    // Send: header + payload + CRC
    if (!sendExact(frame_buf.data(), frame_size)) return false;
    if (!sendExact(&crc, sizeof(crc))) return false;
    return true;
}

bool TcpSocket::recvMsg(MsgType& out_type, std::vector<uint8_t>& out_payload) {
    // Buffered recv: accumulates partial data in recv_buf_ across calls.
    // Works correctly for both blocking and non-blocking sockets.
    while (true) {
        // Check if recv_buf_ contains a complete frame
        if (recv_buf_.size() >= sizeof(FrameHeader)) {
            FrameHeader hdr;
            memcpy(&hdr, recv_buf_.data(), sizeof(hdr));

            if (hdr.magic != kProtocolMagic) {
                LOG(LOG_WARNING) << "[TCP] Bad magic: 0x" << std::hex << hdr.magic
                                 << " (expected 0x" << kProtocolMagic << ")";
                recv_buf_.clear();
                return false;
            }

            // Guard against malformed frames: reject payload_len values that would
            // require allocating more than kMaxPayloadBytes of recv buffer.
            static constexpr uint32_t kMaxPayloadBytes = 64u * 1024u * 1024u; // 64 MiB
            if (hdr.payload_len > kMaxPayloadBytes) {
                LOG(LOG_WARNING) << "[TCP] Rejecting frame: payload_len=" << hdr.payload_len
                                 << " exceeds maximum (" << kMaxPayloadBytes << " bytes)";
                recv_buf_.clear();
                return false;
            }

            size_t frame_total = sizeof(FrameHeader) + hdr.payload_len + kCRC32Size;
            if (recv_buf_.size() >= frame_total) {
                // Complete frame available — validate CRC
                size_t hdr_plus_payload = sizeof(FrameHeader) + hdr.payload_len;
                uint32_t wire_crc;
                memcpy(&wire_crc, recv_buf_.data() + hdr_plus_payload, sizeof(wire_crc));
                uint32_t computed_crc = computeCRC32(recv_buf_.data(), hdr_plus_payload);

                if (wire_crc != computed_crc) {
                    LOG(LOG_WARNING) << "[TCP] CRC mismatch: wire=0x" << std::hex << wire_crc
                                     << " computed=0x" << computed_crc;
                    recv_buf_.erase(recv_buf_.begin(), recv_buf_.begin() + frame_total);
                    return false;
                }

                out_type = static_cast<MsgType>(hdr.msg_type);
                out_payload.assign(recv_buf_.begin() + sizeof(FrameHeader),
                                   recv_buf_.begin() + hdr_plus_payload);
                recv_buf_.erase(recv_buf_.begin(), recv_buf_.begin() + frame_total);
                return true;
            }
        }

        // Need more data — read from socket
        uint8_t tmp[8192];
        ssize_t n = ::recv(fd_, tmp, sizeof(tmp), 0);
        if (n > 0) {
            recv_buf_.insert(recv_buf_.end(), tmp, tmp + n);
        } else if (n == 0) {
            return false;  // Connection closed
        } else {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) return false;  // Non-blocking: no data yet
            LOG(LOG_WARNING) << "[TCP] recv() error: " << strerror(errno);
            return false;
        }
    }
}

void TcpSocket::setNoDelay(bool enable) {
    int flag = enable ? 1 : 0;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
}

void TcpSocket::setRecvTimeout(uint32_t ms) {
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

std::string TcpSocket::peerAddr() const {
    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    if (getpeername(fd_, reinterpret_cast<sockaddr*>(&addr), &len) < 0)
        return "<unknown>";
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
    return std::string(ip) + ":" + std::to_string(ntohs(addr.sin_port));
}

bool TcpSocket::sendExact(const void* buf, size_t len) {
    const auto* p = static_cast<const uint8_t*>(buf);
    size_t sent = 0;
    int eagain_retries = 0;
    while (sent < len) {
        ssize_t n = ::send(fd_, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Non-blocking socket: send buffer full, retry after brief pause
                if (++eagain_retries > 1000) {
                    LOG(LOG_WARNING) << "[TCP] send() stuck after " << eagain_retries << " EAGAIN retries";
                    return false;
                }
                usleep(100);  // 100us backoff
                continue;
            }
            LOG(LOG_WARNING) << "[TCP] send() error: " << strerror(errno);
            return false;
        }
        if (n == 0) return false;
        sent += static_cast<size_t>(n);
        eagain_retries = 0;
    }
    return true;
}


} // namespace mpqs::cluster
