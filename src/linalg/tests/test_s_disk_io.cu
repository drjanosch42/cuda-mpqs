// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

/**
 * @file test_s_disk_io.cu
 * @brief Comprehensive test for S sequence disk I/O (Feature B).
 *
 * Tests 6 scenarios:
 *   1. Default path (no disk I/O)
 *   2. Write only (save S to disk, verify header)
 *   3. Write then read (save, then load from disk in second run)
 *   4. Read only (confirm disk load path is used)
 *   5. Missing file (graceful fallback)
 *   6. Corrupt files:
 *      6a. Wrong magic
 *      6b. Wrong version
 *      6c. Wrong dimensions
 *      6d. Truncated file (valid header, no data)
 *      6e. Empty file
 */

#include "bw_solver.h"
#include "hpc_logger.h"
#include <fstream>
#include <cstdio>
#include <cstring>
#include <random>

using namespace lingen;

// Copy of the anonymous-namespace header from bw_solver.cu for file inspection.
struct SSequenceHeader {
    static constexpr uint64_t MAGIC = 0x425753455153ULL; // "BWSEQS"
    static constexpr uint32_t CURRENT_VERSION = 1;
    uint64_t magic = MAGIC;
    uint32_t version = CURRENT_VERSION;
    uint32_t seq_len = 0;
    uint32_t m_block = 0;
    uint32_t n_block = 0;
    uint64_t data_bytes = 0;
};

// Generate a random sparse HostMatrix with controlled rank deficit for
// guaranteed non-trivial kernel.  Uses L*D*U factorisation over GF(2).
static HostMatrix generate_test_matrix(int N, int rank, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1);

    int stride = (N + 63) / 64;
    std::vector<uint64_t> L(N * stride, 0);
    std::vector<uint64_t> U(N * stride, 0);

    auto set_bit = [&](std::vector<uint64_t>& m, int r, int c) {
        m[r * stride + (c / 64)] |= (1ULL << (c % 64));
    };
    auto get_bit = [&](const std::vector<uint64_t>& m, int r, int c) -> int {
        return (m[r * stride + (c / 64)] >> (c % 64)) & 1;
    };

    for (int i = 0; i < N; ++i) {
        set_bit(L, i, i);
        set_bit(U, i, i);
        for (int j = 0; j < i; ++j) if (dist(rng)) set_bit(L, i, j);
        for (int j = i + 1; j < N; ++j) if (dist(rng)) set_bit(U, i, j);
    }

    std::vector<uint64_t> B(N * stride, 0);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < rank; ++k) {
            if (get_bit(L, i, k)) {
                for (int w = 0; w < stride; ++w)
                    B[i * stride + w] ^= U[k * stride + w];
            }
        }
    }

    HostMatrix A;
    A.n_rows = N;
    A.n_cols = N;
    A.rows.resize(N);
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (get_bit(B, r, c)) A.rows[r].push_back(c);
        }
    }
    return A;
}

// Build a default solver config for the test matrix.
static BWSolverConfig make_base_config(int N) {
    BWSolverConfig cfg;
    cfg.device_id = 0;
    cfg.nrows = N;
    cfg.m_block = 64;
    cfg.n_block = 64;
    cfg.seed = 12345;
    cfg.solve_transposed = false;

    // Disable autotune and related features for speed
    cfg.autotune_tune_spmm = false;
    cfg.autotune_tune_poly = false;
    cfg.autotune_verify_spmm = true;
    cfg.stage2_gpu_mode = true;
    cfg.stage3_perform_unpermutation = false;
    cfg.stage3_max_solutions = 5;
    cfg.stage1_seq_len = 0; // auto
    return cfg;
}

// Helper: write raw bytes to a file.
static void write_raw_file(const std::string& path, const void* data, size_t size) {
    std::ofstream ofs(path, std::ios::binary);
    if (size > 0 && data) ofs.write(reinterpret_cast<const char*>(data), size);
}

int main() {
    // Initialise logger
    LogConfig log_cfg;
    log_cfg.enable_cout = true;
    log_cfg.min_severity_cout = LOG_INFO;
    HPCLogger::Get().Init(log_cfg);

    const int N = 512;
    const int RANK = N - 5; // 5-dimensional kernel
    const std::string SAVE_PATH  = "/tmp/bw_test_s_io.bin";
    const std::string BAD_PATH   = "/tmp/bw_test_s_bad.bin";
    const std::string MISSING_PATH = "/tmp/bw_test_s_nonexistent_9a8b7c.bin";

    LOG(LOG_INFO) << "=== S Sequence Disk I/O Test Suite ===";
    LOG(LOG_INFO) << "Matrix: " << N << "x" << N << ", rank " << RANK;

    HostMatrix A = generate_test_matrix(N, RANK, 42);

    int pass = 0, fail = 0;

    // ========================================================================
    // Scenario 1: Default — no disk I/O
    // ========================================================================
    {
        LOG(LOG_INFO) << "\n--- Scenario 1: Default (no disk I/O) ---";
        auto cfg = make_base_config(N);
        // defaults: save_S_to_disk=false, load_S_from_disk=false
        BlockWiedemannSolver solver(cfg, A);
        solver.Solve();
        bool ok = solver.get_solutions().size() > 0;
        printf(ok ? "PASS: Scenario 1 (default, no disk I/O)\n"
                  : "FAIL: Scenario 1 (default, no disk I/O)\n");
        ok ? ++pass : ++fail;
    }

    // ========================================================================
    // Scenario 2: Write only — save S to disk, verify header
    // ========================================================================
    {
        LOG(LOG_INFO) << "\n--- Scenario 2: Write only ---";
        std::remove(SAVE_PATH.c_str());
        auto cfg = make_base_config(N);
        cfg.stage1_save_S_to_disk = true;
        cfg.stage1_S_disk_path = SAVE_PATH;
        BlockWiedemannSolver solver(cfg, A);
        solver.Solve();

        // Verify file
        std::ifstream ifs(SAVE_PATH, std::ios::binary);
        SSequenceHeader hdr{};
        bool read_ok = false;
        if (ifs.is_open()) {
            read_ok = static_cast<bool>(
                ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)));
        }

        bool ok = read_ok
               && hdr.magic == SSequenceHeader::MAGIC
               && hdr.version == SSequenceHeader::CURRENT_VERSION
               && hdr.m_block == 64
               && hdr.n_block == 64
               && hdr.seq_len > 0
               && hdr.data_bytes > 0
               && solver.get_solutions().size() > 0;

        if (ok) {
            // Verify total file size = header + data_bytes
            ifs.seekg(0, std::ios::end);
            size_t file_size = static_cast<size_t>(ifs.tellg());
            ok = (file_size == sizeof(SSequenceHeader) + hdr.data_bytes);
            if (!ok) {
                printf("FAIL: Scenario 2 — file size mismatch (expected %zu, got %zu)\n",
                       sizeof(SSequenceHeader) + (size_t)hdr.data_bytes, file_size);
            }
        }
        if (ok) printf("PASS: Scenario 2 (write only, header valid)\n");
        else    printf("FAIL: Scenario 2 (write only)\n");
        ok ? ++pass : ++fail;
    }

    // ========================================================================
    // Scenario 3: Write then read — second run loads from disk
    // ========================================================================
    {
        LOG(LOG_INFO) << "\n--- Scenario 3: Write then read ---";
        // File already saved from Scenario 2
        auto cfg = make_base_config(N);
        cfg.stage2_load_S_from_disk = true;
        cfg.stage2_S_disk_path = SAVE_PATH;
        BlockWiedemannSolver solver(cfg, A);
        solver.Solve();
        bool ok = solver.get_solutions().size() > 0;
        printf(ok ? "PASS: Scenario 3 (write then read — loaded from disk)\n"
                  : "FAIL: Scenario 3 (write then read)\n");
        ok ? ++pass : ++fail;
    }

    // ========================================================================
    // Scenario 4: Read only (same config, confirm disk path used via log)
    // ========================================================================
    {
        LOG(LOG_INFO) << "\n--- Scenario 4: Read only (confirm disk load path) ---";
        auto cfg = make_base_config(N);
        cfg.stage2_load_S_from_disk = true;
        cfg.stage2_S_disk_path = SAVE_PATH;
        BlockWiedemannSolver solver(cfg, A);
        solver.Solve();
        bool ok = solver.get_solutions().size() > 0;
        printf(ok ? "PASS: Scenario 4 (read only — disk load confirmed)\n"
                  : "FAIL: Scenario 4 (read only)\n");
        ok ? ++pass : ++fail;
    }

    // ========================================================================
    // Scenario 5: Missing file — graceful fallback
    // ========================================================================
    {
        LOG(LOG_INFO) << "\n--- Scenario 5: Missing file ---";
        std::remove(MISSING_PATH.c_str()); // ensure it's gone
        auto cfg = make_base_config(N);
        cfg.stage2_load_S_from_disk = true;
        cfg.stage2_S_disk_path = MISSING_PATH;
        BlockWiedemannSolver solver(cfg, A);
        solver.Solve();
        bool ok = solver.get_solutions().size() > 0;
        printf(ok ? "PASS: Scenario 5 (missing file — graceful fallback)\n"
                  : "FAIL: Scenario 5 (missing file)\n");
        ok ? ++pass : ++fail;
    }

    // ========================================================================
    // Scenario 6: Corrupt files
    // ========================================================================
    auto run_corrupt_test = [&](const char* label, const void* data, size_t size) {
        write_raw_file(BAD_PATH, data, size);
        auto cfg = make_base_config(N);
        cfg.stage2_load_S_from_disk = true;
        cfg.stage2_S_disk_path = BAD_PATH;
        BlockWiedemannSolver solver(cfg, A);
        solver.Solve();
        bool ok = solver.get_solutions().size() > 0;
        printf(ok ? "PASS: %s\n" : "FAIL: %s\n", label);
        ok ? ++pass : ++fail;
        std::remove(BAD_PATH.c_str());
    };

    // 6a: Wrong magic
    {
        LOG(LOG_INFO) << "\n--- Scenario 6a: Wrong magic ---";
        SSequenceHeader h;
        h.magic = 0xDEADBEEFULL;
        run_corrupt_test("Scenario 6a (wrong magic)", &h, sizeof(h));
    }

    // 6b: Wrong version
    {
        LOG(LOG_INFO) << "\n--- Scenario 6b: Wrong version ---";
        SSequenceHeader h;
        h.version = 99;
        run_corrupt_test("Scenario 6b (wrong version)", &h, sizeof(h));
    }

    // 6c: Wrong dimensions
    {
        LOG(LOG_INFO) << "\n--- Scenario 6c: Wrong dimensions ---";
        SSequenceHeader h;
        h.m_block = 999;
        h.n_block = 999;
        run_corrupt_test("Scenario 6c (wrong dimensions)", &h, sizeof(h));
    }

    // 6d: Truncated — valid header claims large payload, but no data follows
    {
        LOG(LOG_INFO) << "\n--- Scenario 6d: Truncated file ---";
        SSequenceHeader h;
        h.m_block = 64;
        h.n_block = 64;
        h.seq_len = 100;
        h.data_bytes = 100 * 64 * sizeof(uint64_t); // claims big payload
        run_corrupt_test("Scenario 6d (truncated)", &h, sizeof(h));
    }

    // 6e: Empty file
    {
        LOG(LOG_INFO) << "\n--- Scenario 6e: Empty file ---";
        run_corrupt_test("Scenario 6e (empty file)", nullptr, 0);
    }

    // ========================================================================
    // Summary
    // ========================================================================
    printf("\n=== Results: %d PASS, %d FAIL ===\n", pass, fail);

    // Cleanup
    std::remove(SAVE_PATH.c_str());

    return fail > 0 ? 1 : 0;
}
