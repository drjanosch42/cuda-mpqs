// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/stage2/init_basis.h"
#include "hpc_logger.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace lingen {
namespace stage2 {

// =================================================================================
// CPU reference
// =================================================================================

/// Bit (r, c) of a rows×cols row-major bit matrix with ⌈cols/64⌉ words per row.
static inline bool get_bit_rm(const uint64_t* data, int cols, int r, int c) {
    int stride_words = (cols + 63) / 64;
    return (data[r * stride_words + (c / 64)] >> (c & 63)) & 1ULL;
}

/// Sets bit (r, c) of a rows×cols row-major bit matrix with ⌈cols/64⌉ words per row.
static inline void set_bit_rm(uint64_t* data, int cols, int r, int c) {
    int stride_words = (cols + 63) / 64;
    data[r * stride_words + (c / 64)] |= (1ULL << (c & 63));
}

InitBasisResult find_init_basis_cpu(const std::vector<std::vector<uint64_t>>& S_host, int m, int n) {
    int len_seq = (int)S_host.size();

    // Basis storage: map pivot_index -> vector (of m bits).
    // Echelon (not reduced) form: basis_vecs[p] has its lowest set bit at p.
    int m_words = (m + 63) / 64;
    std::vector<std::vector<uint64_t>> basis_vecs(m);
    std::vector<bool> pivot_found(m, false);

    InitBasisResult res;
    int rank = 0;

    // Iterate i (time) then j (column) - STRICT Python Order
    for (int i = 0; i < len_seq; ++i) {
        for (int j = 0; j < n; ++j) {
            // Extract column vector v = S[i][:, j]
            std::vector<uint64_t> v(m_words, 0);
            for (int r = 0; r < m; ++r) {
                if (get_bit_rm(S_host[i].data(), n, r, j)) {
                    v[r / 64] |= (1ULL << (r % 64));
                }
            }

            // Reduce v against basis (ascending pivots; afterwards v vanishes on every pivot)
            for (int p = 0; p < m; ++p) {
                if (pivot_found[p]) {
                    if ((v[p / 64] >> (p % 64)) & 1ULL) {
                        for (int w = 0; w < m_words; ++w) v[w] ^= basis_vecs[p][w];
                    }
                }
            }

            // Pivot = lowest set bit of the reduced v (none: v ∈ span, reject)
            int first_bit = -1;
            for (int r = 0; r < m; ++r) {
                if ((v[r / 64] >> (r % 64)) & 1ULL) {
                    first_bit = r;
                    break;
                }
            }

            if (first_bit != -1) {
                pivot_found[first_bit] = true;
                basis_vecs[first_bit] = v;
                res.pairs.push_back({i, j});
                rank++;

                if (rank == m) {
                    res.t0 = i + 1;
                    res.rank = rank;
                    return res;
                }
            }
        }
    }

    // Rank condition failed: degenerate start {t0 = 0, pairs = ∅}.
    res.t0 = 0;
    res.rank = rank;
    res.pairs.clear();
    res.pivot_mask.assign(m_words, 0);
    for (int p = 0; p < m; ++p)
        if (pivot_found[p]) res.pivot_mask[p / 64] |= (1ULL << (p % 64));
    return res;
}

void log_init_rank_failure(int rank, int m, const std::vector<uint64_t>& pivot_mask) {
    std::vector<int> missing;
    for (int p = 0; p < m; ++p)
        if (!((pivot_mask[p / 64] >> (p % 64)) & 1ULL)) missing.push_back(p);

    std::ostringstream oss;
    oss << "[";
    for (size_t k = 0; k < std::min((size_t)32, missing.size()); ++k) {
        oss << missing[k];
        if (k + 1 < missing.size()) oss << ", ";
    }
    if (missing.size() > 32) oss << "...";
    oss << "]";

    LOG(LOG_ERROR_CRITICAL) << "[Basecase] Rank condition failed! Rank=" << rank << " < m=" << m;
    LOG(LOG_ERROR_CRITICAL) << "[Basecase] Missing pivot indices: " << oss.str();
}

void build_f_init_cpu(const InitBasisResult& init, int m, int n,
                      std::vector<std::vector<uint64_t>>& F, std::vector<int>& gamma) {
    int t0 = init.t0;
    int dim = m + n;

    // Storage: dim x dim (logically dim x n), stride (dim + 63)/64 as the solve loop expects.
    int stride_F = (dim + 63) / 64;
    int words_per_mat = dim * stride_F;

    F.assign(t0 + 1, std::vector<uint64_t>(words_per_mat, 0));
    gamma.assign(dim, t0);

    // 1. Top n rows: Identity at degree 0, F[0][j, j] = 1 for j < n
    for (int j = 0; j < n; ++j) {
        set_bit_rm(F[0].data(), dim, j, j);
    }

    // 2. Bottom m rows: monomial x^(t0 - i_k) at (n + k, j_k)
    for (int k = 0; k < m; ++k) {
        if (k >= (int)init.pairs.size()) break; // rank failure: no monomials

        int i_k = init.pairs[k].first;
        int j_k = init.pairs[k].second;
        int deg = t0 - i_k;
        int row_idx = n + k;

        if (deg < 0 || deg > t0) {
            LOG(LOG_ERROR_CRITICAL) << "Logic Error: deg=" << deg << " out of bounds";
        }

        set_bit_rm(F[deg].data(), dim, row_idx, j_k);
    }
}

} // namespace stage2
} // namespace lingen

// =================================================================================
// Device path
// =================================================================================

namespace lingen {
namespace stage2 {
namespace {

constexpr unsigned FULL_MASK = 0xffffffffu;

/**
 * @brief Initialization-basis scan on the device (one warp, bit-identical to the CPU).
 *
 * Launch: <<<1, 32, MW·m·8 B dynamic shared memory>>>, MW ∈ {1, 2, 4, 8} with
 * m ≤ 64·MW. Fully warp-synchronous: every control decision depends only on
 * warp-uniform values (produced by __ballot_sync / __shfl_xor_sync with the full
 * mask), so all 32 lanes take identical branches; no __syncthreads is needed.
 *
 * Data layout.
 *  - Lane ℓ owns the rows p = ℓ + 32k, k < 2·MW (rows p ≥ m are inert).
 *  - Basis B is stored transposed in shared memory, B[w·m + p] = word w of basis
 *    vector b_p; lane-consecutive p are bank-conflict-free. Row p is only ever read
 *    or written by its owner lane.
 *  - piv[MW] (pivot mask), v[MW], r[MW] and rank are warp-uniform registers.
 *
 * Per candidate v = S_i[:, j], in strict lexicographic order (i, j):
 *  1. v is assembled by one ballot per 32-row slab of the owned S words.
 *  2. r = v ⊕ ⊕{ b_p : p pivot, v_p = 1 } (per-lane masked XOR, then a 5-step
 *     XOR butterfly).
 *  3. r = 0 ⇒ reject. Otherwise q = lowest set bit of r (a non-pivot), accept:
 *  4. Gauss-Jordan update: b_p ^= r for every pivot p with (b_p)_q = 1; b_q = r;
 *     piv ∪= {q}; pairs[rank] = (i, j); rank += 1.
 *  5. rank = m ⇒ t0 = i + 1, stop (later columns of S_i are not scanned).
 * If the scan ends with rank < m the status is 1 (t0 = 0) and the pivot mask is
 * exported for the failure log; the full scan keeps the logged rank equal to the CPU's.
 *
 * Invariant (fully reduced basis). For every pivot p: (b_p)_p = 1 and (b_p)_{p'} = 0
 * for every other pivot p'. It holds vacuously at start; in step 4, r vanishes on the
 * old pivots (Lemma, part 1), so b_p ^= r keeps b_p's old-pivot coordinates and clears
 * its q coordinate, and the new b_q = r has (r)_q = 1 and vanishes on the old pivots.
 *
 * Lemma (equivalence with the CPU reference). Let P be the current pivot set and
 * V = span{b_p : p ∈ P} = span of the accepted candidates (identical on CPU and GPU
 * by induction). For a candidate v let r_G be the GPU reduction above and r_C the
 * CPU reduction (for p ascending in P: if v_p = 1 then v ^= c_p, where the CPU basis
 * c_p is in echelon form, lowest set bit p).
 *  1. r_G vanishes on P: for p' ∈ P, (r_G)_{p'} = v_{p'} ⊕ Σ_{p: v_p = 1} (b_p)_{p'}
 *     = v_{p'} ⊕ v_{p'} = 0 by the invariant.
 *  2. r_C vanishes on P: XOR with c_p changes only bits ≥ p; after processing p,
 *     bit p is 0 and is never touched again by later (larger) pivots.
 *  3. Both lie in v + V, so d = r_G ⊕ r_C ∈ V and d vanishes on P. Writing
 *     d = Σ_{p ∈ T} b_p (T ⊆ P, possible since {b_p} is a basis of V), the invariant
 *     gives d_p = [p ∈ T] for p ∈ P, hence T = ∅ and d = 0.
 * Therefore r_G = r_C bit for bit at every step: the accept/reject decision, the
 * pivot q = lowest set bit, the pivot set, rank, pairs (in order) and t0 coincide
 * with the CPU reference, and so do the rank-failure outputs (rank, pivot mask).
 *
 * @param S        len coefficients, S_i at word i·m·nw, entry (r, j) = bit j&63 of
 *                 word i·m·nw + r·nw + (j>>6), nw = ⌈n/64⌉.
 * @param meta     out: {t0, rank, status, 0}.
 * @param pairs    out: pairs[2k], pairs[2k+1] = (i_k, j_k), k < rank.
 * @param pivmask  out (failure only): ⌈m/64⌉ words.
 */
template <int MW>
__global__ void __launch_bounds__(32)
k_find_init_basis(const uint64_t* __restrict__ S, int len, int m, int n,
                  int* __restrict__ meta, int* __restrict__ pairs,
                  uint64_t* __restrict__ pivmask)
{
    constexpr int RMAX = 2 * MW;               // 32-row slabs per lane (m ≤ 32·RMAX)
    extern __shared__ uint64_t B[];            // [MW][m], B[w·m + p]

    const int lane = threadIdx.x;
    const int nw = (n + 63) >> 6;
    const size_t mat_words = (size_t)m * nw;

    uint64_t piv[MW];
#pragma unroll
    for (int w = 0; w < MW; ++w) piv[w] = 0;
    int rank = 0;

    for (int i = 0; i < len; ++i) {
        const uint64_t* Si = S + (size_t)i * mat_words;
        for (int c = 0; c < nw; ++c) {
            // Owned words of word-column c: rows p = lane + 32k.
            uint64_t word[RMAX];
#pragma unroll
            for (int k = 0; k < RMAX; ++k) {
                const int p = lane + 32 * k;
                word[k] = (p < m) ? __ldg(Si + (size_t)p * nw + c) : 0ULL;
            }

            const int jend = min(64, n - 64 * c);
            for (int jj = 0; jj < jend; ++jj) {
                // 1. v = S_i[:, 64c + jj]  (row p ↦ bit p; slab k ↦ half k&1 of word k>>1)
                uint64_t v[MW];
                uint64_t any = 0;
#pragma unroll
                for (int w = 0; w < MW; ++w) {
                    const uint64_t lo = __ballot_sync(FULL_MASK, (word[2 * w]     >> jj) & 1ULL);
                    const uint64_t hi = __ballot_sync(FULL_MASK, (word[2 * w + 1] >> jj) & 1ULL);
                    v[w] = lo | (hi << 32);
                    any |= v[w];
                }
                if (any == 0) continue;

                // 2. r = v ⊕ ⊕{ b_p : p pivot, v_p = 1 }
                uint64_t acc[MW];
#pragma unroll
                for (int w = 0; w < MW; ++w) acc[w] = 0;
#pragma unroll
                for (int k = 0; k < RMAX; ++k) {
                    const int p = lane + 32 * k;       // word p>>6 = k>>1, bit p&63 = lane + 32(k&1)
                    const int bit = lane + 32 * (k & 1);
                    if (p < m && ((piv[k >> 1] & v[k >> 1]) >> bit) & 1ULL) {
#pragma unroll
                        for (int w = 0; w < MW; ++w) acc[w] ^= B[w * m + p];
                    }
                }
                uint64_t r[MW];
                uint64_t rany = 0;
#pragma unroll
                for (int w = 0; w < MW; ++w) {
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        acc[w] ^= __shfl_xor_sync(FULL_MASK, acc[w], off);
                    r[w] = v[w] ^ acc[w];
                    rany |= r[w];
                }
                // 3. v ∈ span ⇒ reject
                if (rany == 0) continue;

                // q = lowest set bit of r (warp-uniform)
                int q = -1;
#pragma unroll
                for (int w = MW - 1; w >= 0; --w)
                    if (r[w]) q = 64 * w + __ffsll((long long)r[w]) - 1;

                // 4. Gauss-Jordan update on the owned pivot rows, then b_q = r.
                const int qw = q >> 6;
                const int qb = q & 63;
#pragma unroll
                for (int k = 0; k < RMAX; ++k) {
                    const int p = lane + 32 * k;
                    const int bit = lane + 32 * (k & 1);
                    if (p < m && (piv[k >> 1] >> bit) & 1ULL) {
                        if ((B[qw * m + p] >> qb) & 1ULL) {
#pragma unroll
                            for (int w = 0; w < MW; ++w) B[w * m + p] ^= r[w];
                        }
                    }
                }
                if (lane == (q & 31)) {
#pragma unroll
                    for (int w = 0; w < MW; ++w) B[w * m + q] = r[w];
                }
#pragma unroll
                for (int w = 0; w < MW; ++w)
                    if (w == qw) piv[w] |= 1ULL << qb;
                if (lane == 0) {
                    pairs[2 * rank]     = i;
                    pairs[2 * rank + 1] = 64 * c + jj;
                }
                ++rank;
                __syncwarp(FULL_MASK);

                // 5. Rank m reached.
                if (rank == m) {
                    if (lane == 0) {
                        meta[INIT_META_T0]     = i + 1;
                        meta[INIT_META_RANK]   = rank;
                        meta[INIT_META_STATUS] = 0;
                        meta[3]                = 0;
                    }
                    return;
                }
            }
        }
    }

    // Rank condition failed.
    if (lane == 0) {
        meta[INIT_META_T0]     = 0;
        meta[INIT_META_RANK]   = rank;
        meta[INIT_META_STATUS] = 1;
        meta[3]                = 0;
        const int mw = (m + 63) >> 6;
#pragma unroll
        for (int w = 0; w < MW; ++w)
            if (w < mw) pivmask[w] = piv[w];
    }
}

/**
 * @brief F_init / gamma on the device (see init_basis.h for the formula).
 * Thread x < dim: row x of F[0] gets bit x (x < n); row n + x of F[t0 - i_x] gets
 * bit j_x (x < m, status 0); gamma[x] = t0. Every written word belongs to exactly one
 * thread (distinct rows), so plain |= is race-free; F must be zero-initialized.
 */
__global__ void k_build_f_init(uint64_t* __restrict__ F0, int* __restrict__ Gamma,
                               const int* __restrict__ meta, const int* __restrict__ pairs,
                               int m, int n)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim = m + n;
    if (x >= dim) return;

    const int t0 = meta[INIT_META_T0];
    const int status = meta[INIT_META_STATUS];
    const int stride_F = (dim + 63) >> 6;
    const size_t mat_stride = (size_t)dim * stride_F;

    if (x < n)
        F0[(size_t)x * stride_F + (x >> 6)] |= 1ULL << (x & 63);
    if (x < m && status == 0) {
        const int i_k = pairs[2 * x];
        const int j_k = pairs[2 * x + 1];
        const int deg = t0 - i_k;              // ∈ [1, t0] since i_k ≤ t0 - 1
        F0[(size_t)deg * mat_stride + (size_t)(n + x) * stride_F + (j_k >> 6)] |= 1ULL << (j_k & 63);
    }
    Gamma[x] = t0;
}

#define CHECK_CUDA_IB(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "[Basecase] CUDA Error: " \
                                << cudaGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err_))); \
    } \
} while (0)

/// Occupancy-validated single-warp launch of k_find_init_basis<MW>.
template <int MW>
void launch_find_mw(const uint64_t* d_S, int len, int m, int n,
                    int* d_meta, int* d_pairs, uint64_t* d_pivmask, cudaStream_t stream)
{
    const size_t smem = (size_t)MW * m * sizeof(uint64_t);
    int blocks_per_sm = 0;
    CHECK_CUDA_IB(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, k_find_init_basis<MW>, 32, smem));
    if (blocks_per_sm < 1)
        throw std::runtime_error("k_find_init_basis: launch configuration infeasible");
    k_find_init_basis<MW><<<1, 32, smem, stream>>>(d_S, len, m, n, d_meta, d_pairs, d_pivmask);
    CHECK_CUDA_IB(cudaGetLastError());
}

} // namespace

bool init_basis_gpu_supported(int m, int n) {
    return m >= 1 && m <= INIT_BASIS_GPU_MAX_M && n >= 1;
}

void launch_find_init_basis_gpu(const uint64_t* d_S, int len, int m, int n,
                                int* d_meta, int* d_pairs, uint64_t* d_pivmask,
                                cudaStream_t stream)
{
    if (!init_basis_gpu_supported(m, n))
        throw std::runtime_error("launch_find_init_basis_gpu: unsupported (m, n)");
    const int mw = (m + 63) / 64;
    if      (mw <= 1) launch_find_mw<1>(d_S, len, m, n, d_meta, d_pairs, d_pivmask, stream);
    else if (mw <= 2) launch_find_mw<2>(d_S, len, m, n, d_meta, d_pairs, d_pivmask, stream);
    else if (mw <= 4) launch_find_mw<4>(d_S, len, m, n, d_meta, d_pairs, d_pivmask, stream);
    else              launch_find_mw<8>(d_S, len, m, n, d_meta, d_pairs, d_pivmask, stream);
}

void launch_build_f_init_gpu(uint64_t* d_F0, int* d_Gamma, const int* d_meta,
                             const int* d_pairs, int m, int n, cudaStream_t stream)
{
    const int dim = m + n;
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    k_build_f_init<<<blocks, threads, 0, stream>>>(d_F0, d_Gamma, d_meta, d_pairs, m, n);
    CHECK_CUDA_IB(cudaGetLastError());
}

} // namespace stage2
} // namespace lingen
