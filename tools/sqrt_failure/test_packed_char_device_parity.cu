// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Stage 6 DEVICE-LEVEL parity test: drives the REAL M9v2 packed GPU preprocessing
// pipeline (gpuPreprocessMatrix_packed, CharMode::BRANCH) on a controlled relation set
// and asserts the device-composed/gathered/appended 32 branch char columns match the
// CPU oracle BIT-FOR-BIT.
//
// Why this is a genuine GPU==CPU parity check:
//   - per-relation char vectors (seeds) are uploaded into the device batch;
//     gpuBuildPackedMatrix (E4) seeds DevicePackedCSR::d_char_bits from them;
//   - gpuRemoveSingletons_packed RELOCATES rows carrying d_char_bits;
//   - execute_merges_kernel XOR-accumulates d_ws_char_bits in lockstep with the
//     Montgomery sqrt_Q product via the ROW_WS_BIT selector;
//   - the final gather at preprocess.cpp selects per alive row and unpacks 32 columns.
//
// Oracle — the GF(2) GLOBAL XOR INVARIANT (grouping-independent, exact):
//   Each original relation ends in exactly ONE reduced row; merges XOR the constituent
//   char vectors. Over a SINGLETON-FREE matrix (every column weight >= 2 → no row is
//   ever dropped by singleton removal), every original survives into exactly one
//   reduced row, so
//       XOR over reduced rows of (device-emitted char vector)
//         ==  XOR over ALL original relations of (seed)          (bit-for-bit).
//   We build a ring matrix (relation i owns columns {c_i, c_{i+1 mod N}}, every column
//   weight 2) that is singleton-free and forces real weight-2 merges, exercising the
//   device XOR-merge + relocation + gather + append on a real GPU run. The emitted char
//   columns are read back from the appended last-32 columns of the reduced CSR.
//
//   NORM guard: a second run with all-0 seeds must emit all-0 char columns (no-op).
//
// Launches CUDA kernels — skips cleanly (exit 0) when no CUDA device is present.
// Exit code 0 iff the global XOR invariant holds (and the norm run emits all 0).

#include "preprocess.h"           // gpuPreprocessMatrix_packed, PreprocessResultV2
#include "character_columns.h"    // CharMode
#include "gpu_char_cols.cuh"      // branchCharBit
#include "mpqs_soa.h"             // HostRelationBatch, RelationBatch
#include "matrix_constructor.h"   // HostMatrixCSR
#include "uint512.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using mpqs::uint512;
using mpqs::matrix::CharMode;
using mpqs::matrix::HostMatrixCSR;
using mpqs::matrix::PreprocessResultV2;
using mpqs::matrix::gpuPreprocessMatrix_packed;
using mpqs::structures::HostRelationBatch;
using mpqs::structures::RelationBatch;

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond, msg) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) { ++g_failures;                                         \
        std::printf("  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
    }                                                                    \
} while (0)

static constexpr uint64_t kLp1Bound = 100000000000ull;  // 1e11
static constexpr uint32_t kR = 32;                       // branch char vector width

// Build a SINGLETON-FREE smooth-only ring matrix.
//   n relations, fb_size FB primes.  Relation i owns FB prime indices {i, (i+1)%n}
//   (matrix columns i+2 and ((i+1)%n)+2), each with odd exponent 1 → GF(2)-live.
//   Every FB column 0..n-1 then has weight exactly 2 (one from relation c-1, one from
//   relation c) → no weight-1 singletons; weight-2 merges drive the reduction.
//   seeds[i] is relation i's branch char vector (random, or 0 under norm).
struct Built {
    HostRelationBatch batch;
    uint32_t seed_xor = 0;   // XOR of all original seeds (the GF(2) global oracle)
};

static Built buildRing(uint32_t n, uint32_t fb_size, bool norm_mode, std::mt19937_64& rng) {
    Built b;
    b.batch.num_relations = n;
    b.batch.sqrt_Q.resize(n);
    b.batch.signs.assign(n, 1u);
    b.batch.val_2_exps.assign(n, 0);
    b.batch.large_primes.assign(n, (unsigned __int128)1);
    b.batch.char_bits.resize(n);
    b.batch.factor_offsets.resize(n + 1);
    b.batch.factor_indices.reserve(2u * n);
    b.batch.factor_counts.reserve(2u * n);

    std::uniform_int_distribution<uint32_t> seed_dist(0, 0xFFFFFFFFu);
    uint64_t off = 0;
    for (uint32_t i = 0; i < n; ++i) {
        b.batch.sqrt_Q[i] = uint512((uint64_t)(0x100000001ull * (i + 1)));
        uint32_t seed = norm_mode ? 0u : seed_dist(rng);
        b.batch.char_bits[i] = seed;
        b.seed_xor ^= seed;

        b.batch.factor_offsets[i] = off;
        uint32_t a = i, c = (i + 1u) % n;             // two distinct FB prime indices
        uint32_t lo = std::min(a, c), hi = std::max(a, c);
        b.batch.factor_indices.push_back(lo); b.batch.factor_counts.push_back(1u); ++off;
        b.batch.factor_indices.push_back(hi); b.batch.factor_counts.push_back(1u); ++off;
    }
    b.batch.factor_offsets[n] = off;
    b.batch.num_factors = off;
    (void)fb_size;
    return b;
}

static void runRing(uint32_t n, uint32_t fb_size, bool norm_mode) {
    int before = g_failures;
    std::mt19937_64 rng(0xD00DFEEDull ^ (norm_mode ? 0x55u : 0xAAu) ^ (uint64_t)n);
    uint512 N((uint64_t)0xFEEDFACECAFEBEEFull);
    if ((N % 2u) == 0) N = N + uint512((uint32_t)1);

    Built b = buildRing(n, fb_size, norm_mode, rng);

    RelationBatch smooth_dev; smooth_dev.initiate(0);
    smooth_dev.uploadFromHost(b.batch);
    auto smooth_view = smooth_dev.get_view();

    RelationBatch partial_dev; partial_dev.initiate(0);
    auto partial_view = partial_dev.get_view();   // empty view (0 partials)

    std::vector<uint32_t> fb(fb_size, 3u);

    PreprocessResultV2 v2 = gpuPreprocessMatrix_packed(
        smooth_view, n,
        partial_view, 0,
        fb_size, N, fb,
        /*k_max=*/10, /*max_weight=*/200,
        /*truncation_factor=*/0.0,   // keep all reduced rows (global XOR needs all rows)
        /*compact_cycles=*/0,        // single deterministic merge pass
        /*truncation_excess=*/200,
        /*gf2_floor_factor=*/0.0,    // never stop on diversity floor
        /*gf2_min_floor=*/0,
        CharMode::BRANCH, kLp1Bound);

    const HostMatrixCSR& csr = v2.reduced;
    if (csr.n_cols < kR) {
        CHECK(false, "device: reduced matrix has fewer than kR columns");
        std::printf("[FAIL] ring n=%u%s: csr %u x %u (n_cols<kR)\n",
                    n, norm_mode ? " NORM" : "", csr.n_rows, csr.n_cols);
        return;
    }
    const uint32_t char_base = csr.n_cols - kR;

    // XOR over all reduced rows of the emitted char vector.
    uint32_t emitted_xor = 0u;
    int char_entries = 0;
    for (uint32_t r = 0; r < csr.n_rows; ++r) {
        uint32_t row_cb = 0u;
        for (uint32_t idx = csr.row_offsets[r]; idx < csr.row_offsets[r + 1]; ++idx) {
            uint32_t c = csr.col_indices[idx];
            if (c >= char_base) { row_cb |= (1u << (c - char_base)); ++char_entries; }
        }
        emitted_xor ^= row_cb;
    }

    if (norm_mode) {
        CHECK(char_entries == 0, "device[norm]: branch char columns all 0 (no-op)");
        CHECK(emitted_xor == 0u, "device[norm]: emitted XOR == 0");
        std::printf("[%s] ring n=%u NORM: csr %u x %u, char_entries=%d emitted_xor=0x%08x\n",
                    (g_failures == before) ? "PASS" : "FAIL", n, csr.n_rows, csr.n_cols,
                    char_entries, emitted_xor);
    } else {
        // GLOBAL XOR INVARIANT (the bit-for-bit GPU==CPU oracle).
        CHECK(csr.n_rows > 0, "device[branch]: reduced matrix has rows");
        CHECK(emitted_xor == b.seed_xor,
              "device[branch]: XOR of emitted char cols over all reduced rows == XOR of all seeds");
        std::printf("[%s] ring n=%u BRANCH: csr %u x %u; emitted_xor=0x%08x seed_xor=0x%08x %s\n",
                    (g_failures == before) ? "PASS" : "FAIL", n, csr.n_rows, csr.n_cols,
                    emitted_xor, b.seed_xor,
                    (emitted_xor == b.seed_xor) ? "(MATCH)" : "(MISMATCH)");
    }
}

int main() {
    std::printf("=== test_packed_char_device_parity (Stage 6 device GPU==CPU) ===\n");

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("SKIP: no CUDA device available.\n");
        return 0;
    }

    // Several ring sizes → varied merge depths/groupings; the global XOR invariant must
    // hold for every one (BRANCH) and char cols must vanish (NORM).
    for (uint32_t n : {64u, 256u, 1024u}) {
        runRing(n, /*fb_size=*/n + 8u, /*norm=*/false);
        runRing(n, /*fb_size=*/n + 8u, /*norm=*/true);
    }

    std::printf("---------------------------------------------------\n");
    std::printf("checks run: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures == 0) { std::printf("RESULT: PASS (0 failures)\n"); return 0; }
    std::printf("RESULT: FAIL (%d failures)\n", g_failures);
    return 1;
}
