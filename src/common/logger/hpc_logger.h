// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <cmath>

// --- Verbosity Definitions ---
// Negative for errors/warnings/info, zero for stats, positive for debug
constexpr int LOG_RESULT         = -4;  // Factorization result only (--mute)
constexpr int LOG_ERROR_CRITICAL = -3;  // Fatal errors, unrecoverable CUDA failures
constexpr int LOG_ERROR          = -3;  // Alias for LOG_ERROR_CRITICAL
constexpr int LOG_WARNING        = -2;  // Warnings, non-fatal errors, degraded operation
constexpr int LOG_INFO           = -1;  // Stage transitions, key milestones (DEFAULT stdout)
constexpr int LOG_STATS          =  0;  // Statistics summaries, buffer telemetry, throughput (--verbose)
constexpr int LOG_DEBUG_1        =  1;  // Per-batch telemetry, downloads, internal state (--debug)
constexpr int LOG_DEBUG_2        =  2;  // Per-kernel details, allocation internals
constexpr int LOG_DEBUG_3        =  3;  // Developer trace (0 active call sites; retained for forward compat)
// Backward-compat alias (previously a separate level at -2, now same as LOG_WARNING)
constexpr int LOG_ERROR_MAJOR    = LOG_WARNING;

// --- Stage ID Constants ---
constexpr int LOG_STAGE_ORCHESTRATOR_INITIALIZATION  = 0;
constexpr int LOG_STAGE_PARAM_TUNING              = 500;
constexpr int LOG_STAGE_SIEVE                     = 1000;
constexpr int LOG_STAGE_SIEVE_SIEVING             = 1500;
constexpr int LOG_STAGE_RELATION_POSTPROCESSING   = 2000;
constexpr int LOG_STAGE_MULTIPRIMES               = 3000;
constexpr int LOG_STAGE_MATRIX_PREPROCESSING      = 4000;
constexpr int LOG_STAGE_BW_INITIALIZATION          = 5000;
constexpr int LOG_STAGE_BW_SELFVERIFICATION       = 5400;
constexpr int LOG_STAGE_BW_AUTOTUNE               = 5700;
constexpr int LOG_STAGE_BW_STAGE1                 = 6000;
constexpr int LOG_STAGE_BW_STAGE2                 = 7000;
constexpr int LOG_STAGE_BW_STAGE3                 = 8000;
constexpr int LOG_STAGE_BW_POSTPROCESSING         = 8800;
constexpr int LOG_STAGE_SQRT                      = 9000;
constexpr int LOG_STAGE_AUTOTUNE                  = 550;

// --- SinkConfig: per-destination configuration ---
struct SinkConfig {
    enum Type { CONSOLE, FILE, ERROR_FILE };
    Type type = CONSOLE;
    std::string path;              // For FILE/ERROR_FILE types

    // Severity filtering: a message passes if max_severity <= urgency <= min_severity
    int  min_severity = LOG_INFO;  // Most verbose level accepted (default: INFO)
    int  max_severity = LOG_RESULT; // Most urgent level accepted (default: accept all)

    // Prefix toggles (independent per sink)
    bool show_date      = false;
    bool show_time      = true;
    bool show_rank      = false;
    bool show_stage     = true;
    bool show_module    = true;
    bool show_submodule = true;
    bool show_level     = true;

    // Formatting
    int  wrap_width     = 0;       // 0 = no wrapping; >0 = wrap at this column
    bool csv_format     = false;
};

// --- Configuration Struct ---
struct LogConfig {
    std::vector<SinkConfig> sinks;  // Empty vector = default console sink at LOG_INFO
    int  mpi_rank = 0;
};

// --- Thread Local Context ---
// Per-thread logging context: stage, module, submodule
struct ThreadContext {
    int         stage_id    = 0;
    const char* stage_name  = "";   // Human-readable: "Sieve", "LinAlg", "Sqrt", etc.
    const char* module      = "";   // E.g. "PostProc", "LargePrime", "Autotune"
    const char* submodule   = "";   // E.g. "Config", "GPU-GCD", "BatchLoop"

    // Backward-compat alias: existing code uses g_log_context.algorithm_stage_id
    int& algorithm_stage_id = stage_id;
};
extern thread_local ThreadContext g_log_context;

// --- RAII guard for scoped module context ---
// Saves current module+submodule, sets new module (clearing submodule),
// restores on destruction.
class LogModuleGuard {
public:
    explicit LogModuleGuard(const char* mod)
        : prev_module_(g_log_context.module)
        , prev_submodule_(g_log_context.submodule)
    {
        g_log_context.module    = mod;
        g_log_context.submodule = "";
    }
    ~LogModuleGuard() {
        g_log_context.module    = prev_module_;
        g_log_context.submodule = prev_submodule_;
    }
    LogModuleGuard(const LogModuleGuard&) = delete;
    LogModuleGuard& operator=(const LogModuleGuard&) = delete;
private:
    const char* prev_module_;
    const char* prev_submodule_;
};

// --- The Core Logger Class (Singleton) ---
class HPCLogger {
public:
    static HPCLogger& Get() {
        static HPCLogger instance;
        return instance;
    }

    void Init(const LogConfig& config);
    void SetRank(int rank);           // Mutex-protected
    void Log(int urgency, const std::string& message);
    bool ShouldLog(int urgency) const; // Lock-free; reads active_sinks_

private:
    HPCLogger() = default;
    ~HPCLogger();

    LogConfig config_;

    struct ActiveSink {
        SinkConfig config;
        std::ofstream file;   // For FILE/ERROR_FILE types; empty for CONSOLE
    };
    std::vector<ActiveSink> active_sinks_; // Populated at Init(); read-only after
    std::mutex write_mutex_;               // Protects all writes
};

// --- The Temporary Stream Proxy ---
// This enables syntax: LOG(INFO) << "Matrix size: " << N;
class LogMessage {
public:
    LogMessage(int urgency) : urgency_(urgency) {}

    // Destructor triggers the actual logging
    ~LogMessage() {
        HPCLogger::Get().Log(urgency_, buffer_.str());
    }

    // Generic stream operator
    template <typename T>
    LogMessage& operator<<(const T& val) {
        buffer_ << val;
        return *this;
    }

    // Handle std::endl and other manipulators
    LogMessage& operator<<([[maybe_unused]] std::ostream& (*manip)(std::ostream&)) {
        return *this;
    }

private:
    int urgency_;
    std::ostringstream buffer_;
};

// --- Macros for Clean Syntax ---

// Variadic LOG_SET_STAGE: handles both 1-arg (compat) and 2-arg (new) forms.
// LOG_SET_STAGE(500)           -> stage_id=500, stage_name=""
// LOG_SET_STAGE(500, "Tuning") -> stage_id=500, stage_name="Tuning"
#define LOG_SET_STAGE(...) LOG_SET_STAGE_IMPL_(__VA_ARGS__, "", _unused)
#define LOG_SET_STAGE_IMPL_(id, name, ...) do { \
    g_log_context.stage_id = (id); \
    g_log_context.stage_name = (name); \
} while(0)

// Increments the thread-local stage ID
#define LOG_INCREMENT_STAGE(increment) (g_log_context.stage_id += increment)

// Module/submodule context
#define LOG_SET_MODULE(mod)       (g_log_context.module = (mod))
#define LOG_SET_SUBMODULE(sub)    (g_log_context.submodule = (sub))

// Token-pasting helper: two-level expansion ensures __LINE__ expands
// before concatenation (standard C preprocessor requirement).
#define LOG_PASTE_(a, b) a ## b
#define LOG_PASTE2_(a, b) LOG_PASTE_(a, b)

// Scoped module context: saves/restores module+submodule on scope exit
#define LOG_SCOPED_MODULE(mod) \
    LogModuleGuard LOG_PASTE2_(_log_module_guard_, __LINE__)((mod))

// The main logging macro.
#define LOG(level) \
    if (HPCLogger::Get().ShouldLog(level)) \
        LogMessage(level)

// Helper for conditional logging
#define LOG_IF(level, condition) \
    if ((condition) && HPCLogger::Get().ShouldLog(level)) \
        LogMessage(level)

// --- Helpers for clean outputs ---
/**
 * Converts a millisecond value into a human-readable string: "Hh Mm Ss mms"
 * If values are zero, they are omitted for clarity unless the total is < 1s.
 */
inline std::string FormatDuration(double total_ms) {
    using namespace std::chrono;

    // 1. Convert to an integral duration for the units that don't allow fractions
    // We use milliseconds (long long) to allow the % operator to work.
    auto ms_integral = milliseconds(static_cast<long long>(total_ms));

    // 2. Use the modulo operator safely on the integral type
    auto h = duration_cast<hours>(ms_integral);
    auto m = duration_cast<minutes>(ms_integral % hours(1));
    auto s = duration_cast<seconds>(ms_integral % minutes(1));

    // 3. Use fmod on the original double to get the high-precision remainder
    double final_ms = std::fmod(total_ms, 1000.0);

    std::stringstream ss;
    bool printed = false;

    if (h.count() > 0) {
        ss << h.count() << "h ";
        printed = true;
    }
    if (m.count() > 0 || printed) {
        ss << m.count() << "m ";
        printed = true;
    }
    if (s.count() > 0 || printed) {
        ss << s.count() << "s ";
    }

    ss << std::fixed << std::setprecision(2) << final_ms << "ms";

    return ss.str();
}
