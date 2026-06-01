// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "hpc_logger.h"
#include <iostream>

// Define the thread-local storage
thread_local ThreadContext g_log_context;

// --- Helper: map urgency to prefix label ---
static const char* levelLabel(int urgency) {
    switch (urgency) {
        case LOG_RESULT:         return "[RESULT]";
        case LOG_ERROR_CRITICAL: return "[CRIT]";
        // LOG_WARNING and LOG_ERROR_MAJOR are both -2
        case LOG_WARNING:        return "[WARN]";
        case LOG_INFO:           return "[INFO]";
        case LOG_STATS:          return "[STATS]";
        case LOG_DEBUG_1:        return "[DBG1]";
        case LOG_DEBUG_2:        return "[DBG2]";
        case LOG_DEBUG_3:        return "[DBG3]";
        default:
            return (urgency < 0) ? "[ERR]" : "[DBG]";
    }
}

// --- Helper: render per-sink prefix string ---
static std::string renderPrefix(
    const SinkConfig& sc,
    const std::tm& tm_buf,
    long ms_count,
    int urgency,
    int mpi_rank)
{
    std::ostringstream p;

    if (sc.csv_format) {
        // CSV: datetime,rank,stage_id,stage_name,module,submodule,severity
        p << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
          << "." << std::setfill('0') << std::setw(3) << ms_count << ","
          << mpi_rank << ","
          << g_log_context.stage_id << ","
          << (g_log_context.stage_name[0] ? g_log_context.stage_name : "") << ","
          << (g_log_context.module[0] ? g_log_context.module : "") << ","
          << (g_log_context.submodule[0] ? g_log_context.submodule : "") << ","
          << urgency << ",";
        return p.str();
    }

    // Human-readable prefix with bracket fields

    // Field 1: date+time
    if (sc.show_time || sc.show_date) {
        p << "[";
        if (sc.show_date && sc.show_time)
            p << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        else if (sc.show_date)
            p << std::put_time(&tm_buf, "%Y-%m-%d");
        else
            p << std::put_time(&tm_buf, "%H:%M:%S");
        if (sc.show_time)
            p << "." << std::setfill('0') << std::setw(3) << ms_count;
        p << "] ";
    }

    // Field 2: rank
    if (sc.show_rank) {
        p << "[Rank " << mpi_rank << "] ";
    }

    // Field 3: stage
    if (sc.show_stage && g_log_context.stage_id > 0) {
        p << "[Stage " << g_log_context.stage_id;
        if (g_log_context.stage_name[0])
            p << ": " << g_log_context.stage_name;
        p << "] ";
    } else if (sc.show_stage && g_log_context.stage_id == 0 && g_log_context.stage_name[0]) {
        p << "[" << g_log_context.stage_name << "] ";
    }

    // Field 4: module
    if (sc.show_module && g_log_context.module[0]) {
        p << "[" << g_log_context.module << "] ";
    }

    // Field 5: submodule
    if (sc.show_submodule && g_log_context.submodule[0]) {
        p << "[" << g_log_context.submodule << "] ";
    }

    // Field 6: level
    if (sc.show_level) {
        p << levelLabel(urgency) << " ";
    }

    return p.str();
}

// --- Helper: word-wrap message with prefix-width continuation indent ---
static std::string wrapMessage(
    const std::string& prefix,
    const std::string& message,
    int wrap_width)
{
    if (wrap_width <= 0) return prefix + message;

    const int prefix_len = static_cast<int>(prefix.size());
    const int available  = wrap_width - prefix_len;

    if (available <= 0 || static_cast<int>(message.size()) <= available) {
        return prefix + message;
    }

    // Build the continuation indent (prefix_len spaces)
    const std::string indent(prefix_len, ' ');

    std::string result;
    result.reserve(message.size() + prefix.size() + 64);

    // Split message at embedded newlines first; then word-wrap each segment
    size_t seg_start = 0;
    bool first_segment = true;
    while (seg_start <= message.size()) {
        size_t nl = message.find('\n', seg_start);
        if (nl == std::string::npos) nl = message.size();
        std::string seg = message.substr(seg_start, nl - seg_start);

        // Word-wrap this segment
        while (static_cast<int>(seg.size()) > available) {
            // Find last space before the available boundary
            size_t cut = seg.rfind(' ', available);
            if (cut == std::string::npos) {
                // No space found: hard break at available
                cut = available;
            }
            if (first_segment) {
                result += prefix + seg.substr(0, cut) + "\n";
                first_segment = false;
            } else {
                result += indent + seg.substr(0, cut) + "\n";
            }
            seg = seg.substr(cut + (seg[cut] == ' ' ? 1 : 0));
        }
        // Remaining part of segment
        if (first_segment) {
            result += prefix + seg;
            first_segment = false;
        } else {
            result += indent + seg;
        }

        if (nl < message.size()) {
            result += "\n";
            seg_start = nl + 1;
        } else {
            break;
        }
    }

    return result;
}

// --- HPCLogger implementation ---

void HPCLogger::Init(const LogConfig& config) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    config_ = config;
    active_sinks_.clear();

    if (!config_.sinks.empty()) {
        // New-style: use sinks vector directly
        for (const auto& sc : config_.sinks) {
            ActiveSink as;
            as.config = sc;
            if (sc.type == SinkConfig::FILE || sc.type == SinkConfig::ERROR_FILE) {
                as.file.open(sc.path, std::ios::out | std::ios::app);
                if (!as.file.is_open()) {
                    std::cerr << "[HPCLogger] WARNING: Cannot open log file: " << sc.path << std::endl;
                    continue;
                }
            }
            active_sinks_.push_back(std::move(as));
        }
    } else {
        // Backward-compat: construct default sink from flat fields
        if (config_.enable_cout) {
            ActiveSink console_sink;
            console_sink.config.type         = SinkConfig::CONSOLE;
            console_sink.config.min_severity = config_.min_severity_cout;
            console_sink.config.max_severity = LOG_RESULT;
            console_sink.config.show_rank    = false;  // standalone default: suppress rank
            console_sink.config.show_time    = true;
            console_sink.config.show_stage   = true;
            console_sink.config.show_module  = true;
            console_sink.config.show_submodule = true;
            console_sink.config.show_level   = true;
            active_sinks_.push_back(std::move(console_sink));
        }

        if (config_.enable_file) {
            ActiveSink file_sink;
            file_sink.config.type         = SinkConfig::FILE;
            file_sink.config.path         = config_.file_path;
            file_sink.config.min_severity = config_.min_severity_file;
            file_sink.config.max_severity = LOG_RESULT;
            file_sink.config.csv_format   = config_.csv_format;
            file_sink.config.show_date    = true;
            file_sink.config.show_rank    = true;
            file_sink.config.show_stage   = true;
            file_sink.config.show_level   = true;
            file_sink.file.open(config_.file_path, std::ios::out | std::ios::app);
            if (!file_sink.file.is_open()) {
                std::cerr << "[HPCLogger] WARNING: Cannot open log file: " << config_.file_path << std::endl;
            } else {
                active_sinks_.push_back(std::move(file_sink));
            }
        }
    }
}

HPCLogger::~HPCLogger() {
    for (auto& s : active_sinks_) {
        if (s.file.is_open()) {
            s.file.close();
        }
    }
}

void HPCLogger::SetRank(int rank) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    config_.mpi_rank = rank;
}

bool HPCLogger::ShouldLog(int urgency) const {
    // Lock-free: active_sinks_ is read-only after Init()
    for (const auto& s : active_sinks_) {
        if (urgency >= s.config.max_severity && urgency <= s.config.min_severity) {
            return true;
        }
    }
    return false;
}

void HPCLogger::Log(int urgency, const std::string& message) {
    // 1. Capture timestamp (before acquiring lock)
    auto now   = std::chrono::system_clock::now();
    auto ms    = std::chrono::duration_cast<std::chrono::milliseconds>(
                     now.time_since_epoch()) % 1000;
    auto t_c   = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t_c);
#else
    localtime_r(&t_c, &tm_buf);
#endif

    // 2. Serialize writes
    std::lock_guard<std::mutex> lock(write_mutex_);
    const int rank = config_.mpi_rank;

    for (auto& sink : active_sinks_) {
        // Severity filter
        if (urgency < sink.config.max_severity || urgency > sink.config.min_severity)
            continue;

        // Render prefix for this sink
        std::string prefix = renderPrefix(sink.config, tm_buf, ms.count(), urgency, rank);

        // Apply line wrapping (wrap_width is treated as content width; prefix is added on top)
        std::string output = wrapMessage(prefix, message, sink.config.wrap_width + (int)prefix.size());

        // Write to destination
        if (sink.config.type == SinkConfig::CONSOLE) {
            if (urgency < LOG_INFO) {
                std::cerr << output << "\n";
            } else {
                std::cout << output << "\n";
            }
        } else {
            // FILE or ERROR_FILE
            sink.file << output << "\n";
        }
    }
}
