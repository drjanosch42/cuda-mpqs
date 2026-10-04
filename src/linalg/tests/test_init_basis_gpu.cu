// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

/**
 * @file test_init_basis_gpu.cu
 * @brief Bit-exactness test: device initialization basis vs the CPU reference.
 *
 * For every case, S ∈ (GF(2)^{m×n})^len is generated on the host, uploaded, and
 * both paths are run; the test requires equality of
 *   t0, rank, status, the pairs (i_k, j_k) in order, the pivot mask (rank failure),
 *   the F_init bytes (t0+1 coefficients, plus an untouched zero guard coefficient),
 *   and gamma.
 *
 * Cases (seeded, deterministic):
 *   1. random S for (m, n) ∈ {64,128,256}² ∪ {(256,64), (64,256), (512,512)} and
 *      non-multiple-of-64 shapes, len = 8;
 *   2. rank-deficient S_0 (duplicated columns, rank m/2, rank 1) ⇒ t0 = 2;
 *   3. t0 > 2: every S_i of rank m/4 with fresh directions; n = 64, m = 256;
 *   4. scattered all-zero columns and all-zero coefficients;
 *   5. columns equal to XORs of 2-3 earlier accepted columns across coefficients;
 *   6. rank never reached: all-zero S (rank 0) and a rank m-1 hyperplane;
 *   7. the last independent direction appears only at (len-1, n-1);
 *   8. 300 fuzz instances with random structured rank deficiency.
 * Exit code 0 iff every case matches.
 */

#include "lingen/stage2/init_basis.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace lingen::stage2;
using Vec = std::vector<uint64_t>;                 // m-bit column vector, ⌈m/64⌉ words
using Seq = std::vector<std::vector<uint64_t>>;    // S_0..S_{len-1}, m·⌈n/64⌉ words each

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(2); } } while (0)

// -----------------------------------------------------------------------------
// GF(2) helpers
// -----------------------------------------------------------------------------

struct Shape { int m, n, len; };

static int mw_of(int m) { return (m + 63) / 64; }

static Vec rand_vec(std::mt19937_64& rng, int m) {
    Vec v(mw_of(m));
    for (auto& w : v) w = rng();
    if (m & 63) v.back() &= (1ULL << (m & 63)) - 1;
    return v;
}

static Vec zero_vec(int m) { return Vec(mw_of(m), 0); }

static void xor_into(Vec& a, const Vec& b) { for (size_t w = 0; w < a.size(); ++w) a[w] ^= b[w]; }

/// Random combination of the vectors in 'pool' (each included with probability 1/2).
static Vec rand_comb(std::mt19937_64& rng, const std::vector<Vec>& pool, int m) {
    Vec v = zero_vec(m);
    for (const auto& p : pool) if (rng() & 1) xor_into(v, p);
    return v;
}

static Seq empty_seq(const Shape& s) {
    const int nw = (s.n + 63) / 64;
    return Seq(s.len, std::vector<uint64_t>((size_t)s.m * nw, 0));
}

/// S_i[:, j] = v.
static void set_col(Seq& S, const Shape& s, int i, int j, const Vec& v) {
    const int nw = (s.n + 63) / 64;
    for (int r = 0; r < s.m; ++r) {
        uint64_t& word = S[i][(size_t)r * nw + (j >> 6)];
        const uint64_t bit = 1ULL << (j & 63);
        if ((v[r >> 6] >> (r & 63)) & 1ULL) word |= bit; else word &= ~bit;
    }
}

static Vec get_col(const Seq& S, const Shape& s, int i, int j) {
    const int nw = (s.n + 63) / 64;
    Vec v = zero_vec(s.m);
    for (int r = 0; r < s.m; ++r)
        if ((S[i][(size_t)r * nw + (j >> 6)] >> (j & 63)) & 1ULL) v[r >> 6] |= 1ULL << (r & 63);
    return v;
}

/// Fills every column via gen(i, j).
static Seq make_seq(const Shape& s, const std::function<Vec(int, int)>& gen) {
    Seq S = empty_seq(s);
    for (int i = 0; i < s.len; ++i)
        for (int j = 0; j < s.n; ++j) set_col(S, s, i, j, gen(i, j));
    return S;
}

// -----------------------------------------------------------------------------
// Comparison of the two paths
// -----------------------------------------------------------------------------

static int g_cases = 0, g_fail = 0;
static int g_last_status = -1, g_last_t0 = -1;   // of the most recent case (fuzz statistics)

static bool run_case(const std::string& name, const Shape& s, const Seq& S, bool quiet = false) {
    ++g_cases;
    const int m = s.m, n = s.n, len = s.len, dim = m + n;
    const int mw = mw_of(m);
    const int stride_F = (dim + 63) / 64;
    const size_t mat_stride = (size_t)dim * stride_F;

    // CPU reference
    InitBasisResult cpu = find_init_basis_cpu(S, m, n);
    std::vector<std::vector<uint64_t>> F_cpu;
    std::vector<int> gamma_cpu;
    build_f_init_cpu(cpu, m, n, F_cpu, gamma_cpu);

    // Device
    const size_t s_words = (size_t)len * m * ((n + 63) / 64);
    uint64_t* d_S = nullptr;
    int *d_meta = nullptr, *d_pairs = nullptr, *d_Gamma = nullptr;
    uint64_t *d_piv = nullptr, *d_F = nullptr;
    CK(cudaMalloc(&d_S, s_words * sizeof(uint64_t) + 8));
    {
        Vec flat;
        flat.reserve(s_words);
        for (const auto& c : S) flat.insert(flat.end(), c.begin(), c.end());
        if (!flat.empty()) CK(cudaMemcpy(d_S, flat.data(), s_words * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&d_meta, INIT_META_WORDS * sizeof(int)));
    CK(cudaMalloc(&d_pairs, (size_t)2 * m * sizeof(int)));
    CK(cudaMalloc(&d_piv, (size_t)mw * sizeof(uint64_t)));
    CK(cudaMemset(d_meta, 0xff, INIT_META_WORDS * sizeof(int)));
    CK(cudaMemset(d_piv, 0, (size_t)mw * sizeof(uint64_t)));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, 0));
    launch_find_init_basis_gpu(d_S, len, m, n, d_meta, d_pairs, d_piv, 0);
    CK(cudaEventRecord(e1, 0));
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));

    int meta[INIT_META_WORDS];
    CK(cudaMemcpy(meta, d_meta, sizeof(meta), cudaMemcpyDeviceToHost));
    const int t0 = meta[INIT_META_T0], rank = meta[INIT_META_RANK], status = meta[INIT_META_STATUS];

    bool ok = true;
    std::string why;
    auto fail = [&](const std::string& w) { if (ok) why = w; ok = false; };

    if (t0 != cpu.t0) fail("t0 " + std::to_string(t0) + " vs " + std::to_string(cpu.t0));
    if (rank != cpu.rank) fail("rank " + std::to_string(rank) + " vs " + std::to_string(cpu.rank));
    if (status != (cpu.rank == m ? 0 : 1)) fail("status " + std::to_string(status));

    if (ok) {
        if (status == 0) {
            std::vector<int> pairs(2 * m);
            CK(cudaMemcpy(pairs.data(), d_pairs, pairs.size() * sizeof(int), cudaMemcpyDeviceToHost));
            if ((int)cpu.pairs.size() != m) fail("cpu pair count");
            for (int k = 0; ok && k < m; ++k)
                if (pairs[2 * k] != cpu.pairs[k].first || pairs[2 * k + 1] != cpu.pairs[k].second)
                    fail("pair " + std::to_string(k));
        } else {
            Vec piv(mw);
            CK(cudaMemcpy(piv.data(), d_piv, mw * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            if (piv != cpu.pivot_mask) fail("pivot mask");
            if (!cpu.pairs.empty()) fail("cpu pairs not cleared on failure");
        }
    }

    if (ok) {
        // F_init with one extra guard coefficient that must stay zero; gamma poisoned first.
        const size_t f_words = (size_t)(t0 + 2) * mat_stride;
        CK(cudaMalloc(&d_F, f_words * sizeof(uint64_t)));
        CK(cudaMalloc(&d_Gamma, dim * sizeof(int)));
        CK(cudaMemset(d_F, 0, f_words * sizeof(uint64_t)));
        CK(cudaMemset(d_Gamma, 0xa5, dim * sizeof(int)));
        launch_build_f_init_gpu(d_F, d_Gamma, d_meta, d_pairs, m, n, 0);
        CK(cudaDeviceSynchronize());
        Vec F_gpu(f_words);
        std::vector<int> gamma_gpu(dim);
        CK(cudaMemcpy(F_gpu.data(), d_F, f_words * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gamma_gpu.data(), d_Gamma, dim * sizeof(int), cudaMemcpyDeviceToHost));

        Vec F_flat;
        for (const auto& c : F_cpu) F_flat.insert(F_flat.end(), c.begin(), c.end());
        if (F_flat.size() != (size_t)(t0 + 1) * mat_stride) fail("cpu F size");
        else if (std::memcmp(F_flat.data(), F_gpu.data(), F_flat.size() * sizeof(uint64_t)) != 0) fail("F_init bytes");
        for (size_t w = F_flat.size(); ok && w < f_words; ++w) if (F_gpu[w]) fail("F guard coefficient written");
        if (ok && gamma_gpu != gamma_cpu) fail("gamma");
    }

    CK(cudaFree(d_S));
    CK(cudaFree(d_meta));
    CK(cudaFree(d_pairs));
    CK(cudaFree(d_piv));
    if (d_F) CK(cudaFree(d_F));
    if (d_Gamma) CK(cudaFree(d_Gamma));

    g_last_status = status;
    g_last_t0 = t0;
    if (!ok) ++g_fail;
    if (!ok || !quiet)
        std::printf("[%s] %-44s m=%3d n=%3d len=%4d -> t0=%4d rank=%3d status=%d  (%.3f ms)%s%s\n",
                    ok ? "PASS" : "FAIL", name.c_str(), m, n, len, t0, rank, status, ms,
                    ok ? "" : "  -- ", ok ? "" : why.c_str());
    return ok;
}

// -----------------------------------------------------------------------------
// Case generators
// -----------------------------------------------------------------------------

int main() {
    // No logger initialisation: the functions under test do not log on valid input
    // (and the logger configuration API differs between standalone and parent builds).
    std::mt19937_64 rng(0x5eedba5e12345ULL);

    // 1. Random S.
    {
        std::vector<std::pair<int, int>> shapes;
        for (int m : {64, 128, 256}) for (int n : {64, 128, 256}) shapes.push_back({m, n});
        shapes.push_back({256, 64});
        shapes.push_back({64, 256});
        shapes.push_back({512, 512});
        shapes.push_back({1, 1});
        shapes.push_back({3, 5});
        shapes.push_back({96, 40});
        shapes.push_back({200, 130});
        shapes.push_back({511, 257});
        for (const auto& mn : shapes) {
            const int m = mn.first, n = mn.second;
            Shape s{m, n, 8};
            Seq S = make_seq(s, [&](int, int) { return rand_vec(rng, m); });
            run_case("1 random", s, S);
        }
    }

    // 2. Rank-deficient S_0 ⇒ t0 = 2.
    for (int m : {64, 128, 256}) {
        const int n = m;
        Shape s{m, n, 8};
        // (a) S_0 columns duplicated from 8 random vectors
        {
            std::vector<Vec> few;
            for (int k = 0; k < 8; ++k) few.push_back(rand_vec(rng, m));
            Seq S = make_seq(s, [&](int i, int j) { return i == 0 ? few[j % 8] : rand_vec(rng, m); });
            run_case("2a S_0 duplicated columns", s, S);
        }
        // (b) S_0 of rank m/2
        {
            std::vector<Vec> half;
            for (int k = 0; k < m / 2; ++k) half.push_back(rand_vec(rng, m));
            Seq S = make_seq(s, [&](int i, int) { return i == 0 ? rand_comb(rng, half, m) : rand_vec(rng, m); });
            run_case("2b S_0 rank m/2", s, S);
        }
        // (c) S_0 of rank 1
        {
            Vec u = rand_vec(rng, m);
            Seq S = make_seq(s, [&](int i, int) { return i == 0 ? ((rng() & 1) ? u : zero_vec(m)) : rand_vec(rng, m); });
            run_case("2c S_0 rank 1", s, S);
        }
    }

    // 3. t0 > 2: every S_i of rank m/4 with fresh directions; and n = 64, m = 256.
    for (const auto& mn : std::vector<std::pair<int, int>>{{256, 256}, {128, 128}, {256, 64}, {512, 64}}) {
        const int m = mn.first, n = mn.second;
        Shape s{m, n, 16};
        std::vector<std::vector<Vec>> dirs(s.len);
        for (int i = 0; i < s.len; ++i)
            for (int k = 0; k < m / 4; ++k) dirs[i].push_back(rand_vec(rng, m));
        Seq S = make_seq(s, [&](int i, int) { return rand_comb(rng, dirs[i], m); });
        run_case("3 rank m/4 per coefficient", s, S);
    }

    // 4. Scattered all-zero columns and all-zero coefficients.
    for (int m : {64, 256}) {
        Shape s{m, m, 12};
        Seq S = make_seq(s, [&](int i, int) {
            if (i == 0 || i == 2 || i == 3) return zero_vec(m);
            return (rng() % 3 == 0) ? zero_vec(m) : rand_vec(rng, m);
        });
        run_case("4 zero columns / coefficients", s, S);
    }

    // 5. Columns equal to XORs of 2-3 earlier accepted columns, across boundaries.
    for (int m : {64, 128, 256}) {
        Shape s{m, m / 2, 10};
        std::vector<Vec> hist;
        Seq S = make_seq(s, [&](int, int) {
            Vec v;
            if (hist.size() >= 3 && (rng() % 2)) {
                v = zero_vec(m);
                const int t = 2 + (int)(rng() % 2);
                for (int k = 0; k < t; ++k) xor_into(v, hist[rng() % hist.size()]);
            } else {
                v = rand_vec(rng, m);
            }
            hist.push_back(v);
            return v;
        });
        run_case("5 XOR of earlier columns", s, S);
    }

    // 6. Rank never reached.
    for (int m : {64, 256}) {
        Shape s{m, m, 32};
        run_case("6a all-zero S (rank 0)", s, empty_seq(s));

        // Hyperplane {x : f·x = 0}: x ← x ⊕ u whenever f·x = 1 (f·u = 1).
        Vec f = rand_vec(rng, m);
        f[0] |= 1;
        auto dotf = [&](const Vec& x) { int p = 0; for (size_t w = 0; w < x.size(); ++w) p ^= __builtin_parityll(x[w] & f[w]); return p; };
        Vec u = zero_vec(m); u[0] = 1;                          // f·u = f_0 = 1
        auto hyper = [&]() { Vec x = rand_vec(rng, m); if (dotf(x)) xor_into(x, u); return x; };
        Seq S = make_seq(s, [&](int, int) { return hyper(); });
        run_case("6b rank m-1 hyperplane", s, S);

        // 7. The last direction only at (len-1, n-1).
        Seq S7 = make_seq(s, [&](int i, int j) {
            if (i == s.len - 1 && j == s.n - 1) { Vec x = hyper(); xor_into(x, u); return x; }
            return hyper();
        });
        run_case("7 resolves at the last column", s, S7);
    }

    // 8. Fuzz: random shapes, per-coefficient rank drawn from a pool mixing fresh and old directions.
    {
        int passed = 0, n_fail_rank = 0, n_t0_gt2 = 0;
        const int N_FUZZ = 300;
        for (int f = 0; f < N_FUZZ; ++f) {
            const int m = 1 + (int)(rng() % 300);
            const int n = 1 + (int)(rng() % 300);
            const int len = 1 + (int)(rng() % 12);
            Shape s{m, n, len};
            std::vector<Vec> old_dirs;
            std::vector<std::vector<Vec>> pools(len);
            for (int i = 0; i < len; ++i) {
                const int r = (int)(rng() % (std::min(m, n) + 1));
                for (int k = 0; k < r; ++k) {
                    if (!old_dirs.empty() && (rng() % 3 == 0)) pools[i].push_back(old_dirs[rng() % old_dirs.size()]);
                    else { pools[i].push_back(rand_vec(rng, m)); old_dirs.push_back(pools[i].back()); }
                }
            }
            const int zero_pct = (int)(rng() % 40);
            Seq S = make_seq(s, [&](int i, int) {
                if ((int)(rng() % 100) < zero_pct || pools[i].empty()) return zero_vec(m);
                return rand_comb(rng, pools[i], m);
            });
            if (run_case("8 fuzz #" + std::to_string(f), s, S, /*quiet=*/true)) ++passed;
            n_fail_rank += (g_last_status == 1);
            n_t0_gt2 += (g_last_t0 > 2);
        }
        std::printf("[%s] 8 fuzz: %d/%d instances match (%d rank failures, %d with t0 > 2)\n",
                    passed == N_FUZZ ? "PASS" : "FAIL", passed, N_FUZZ, n_fail_rank, n_t0_gt2);
    }

    // Sanity: the column accessor round-trips (guards the generators themselves).
    {
        Shape s{70, 70, 1};
        Seq S = empty_seq(s);
        Vec v = rand_vec(rng, 70);
        set_col(S, s, 0, 69, v);
        if (get_col(S, s, 0, 69) != v) { std::printf("[FAIL] column accessor\n"); ++g_fail; }
    }

    std::printf("test_init_basis_gpu: %d/%d cases match%s\n", g_cases - g_fail, g_cases,
                g_fail ? " -- FAILURES" : "");
    return g_fail ? 1 : 0;
}
