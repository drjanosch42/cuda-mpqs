// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/stage2/basecase_solver_reference.h"
#include "lingen/operations/poly_mat_vec.h"
#include "hpc_logger.h"

#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lingen {
namespace stage2 {

// -----------------------------------------------------------------------------
// Error Handling Macros
// -----------------------------------------------------------------------------

#define CHECK_CUDA_BC_LEGACY(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "[BasecaseLegacy] CUDA Error: " \
                                << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)

// =================================================================================
// BasecaseSolverLegacy (Generic) (ORACLE STATUS)
// =================================================================================

BasecaseSolverLegacy::BasecaseSolverLegacy(int m, int n, int delta) 
    : m_(m), n_(n), dim_(m + n), delta_(delta) 
{
    LOG(LOG_STATS) << "[BasecaseLegacy] Constructed solver for m=" << m_ << ", n=" << n_ << " (dim=" << dim_ << ")";
}

BasecaseSolverLegacy::BasecaseSolverLegacy(int block_dim, int delta)
    : BasecaseSolverLegacy(block_dim, block_dim, delta)
{
    // Delegating constructor
}
  
BasecaseSolverLegacy::~BasecaseSolverLegacy() {}

// =================================================================================
// CUDA Kernels for Optimized Annihilation Check (no oracle status)
// =================================================================================

/**
 * @brief Extracts row 'row_idx' from each matrix in the sequence S and pads it.
 * 
 * S is a sequence of m x n matrices. We want to extract the c-th row of each S_t
 * (which corresponds to the c-th column of S^T).
 * 
 * Input:
 *  - S_data: Flat array of L matrices (m x n).
 *  - m, n: Dimensions of S_t.
 *  - row_idx: The row index to extract (0 <= row_idx < m).
 *  - len_S: Length of sequence.
 * 
 * Output:
 *  - V_out: Flat array of L vectors. Each vector has width N bits.
 *           Stride = N/64 uint64s.
 */
template <int N>
__global__ void k_extract_and_pad_row(
    const uint64_t* __restrict__ S_data,
    uint64_t* __restrict__ V_out,
    int len_S,
    int m,
    int n,
    int row_idx
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len_S) return;

    // Calculate source pointer
    // S_data layout: [S_0] [S_1] ...
    // S_t layout: Row-major m x n (packed).
    // Stride per row: (n + 63) / 64 words.
    int s_row_stride = (n + 63) / 64;
    int s_mat_stride = m * s_row_stride;
    
    const uint64_t* s_mat = S_data + (size_t)t * s_mat_stride;
    const uint64_t* src_row_ptr = s_mat + (size_t)row_idx * s_row_stride;

    // Calculate dest pointer
    // V_out layout: [V_0] [V_1] ...
    // V_t is width N bits. Stride N/64 words.
    int v_stride = N / 64;
    uint64_t* dst_ptr = V_out + (size_t)t * v_stride;

    // Copy bits 0..n-1
    // We copy word by word, masking the last one if needed.
    for (int w = 0; w < v_stride; ++w) {
        uint64_t val = 0;
        if (w < s_row_stride) {
            val = src_row_ptr[w];
            // Mask out bits beyond n in the last word of the source row
            if (w == s_row_stride - 1) {
                int valid_bits = n % 64;
                if (valid_bits != 0) {
                    val &= ((1ULL << valid_bits) - 1);
                }
            }
        }
        dst_ptr[w] = val;
    }
}

/**
 * @brief Checks if vectors in W are zero for t >= start_t.
 * 
 * @param W_data Vector series data (width N bits).
 * @param len_S Total length.
 * @param start_t Start checking from this index.
 * @param d_fail Pointer to failure flag (set to 1 if check fails).
 */
template <int N>
__global__ void k_check_zero_suffix(
    const uint64_t* __restrict__ W_data,
    int len_S,
    int start_t,
    int* d_fail
) {
    int t = start_t + blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len_S) return;

    int w_stride = N / 64;
    const uint64_t* w_vec = W_data + (size_t)t * w_stride;

    bool non_zero = false;
    for (int w = 0; w < w_stride; ++w) {
        if (w_vec[w] != 0) {
            non_zero = true;
            break;
        }
    }

    if (non_zero) {
        // We found a mismatch.
        // Use atomic to safely set flag from multiple threads
        atomicExch(d_fail, 1);
    }
}

// =================================================================================
// Helper: Pad Pi to N x N on Host
// =================================================================================

static std::vector<uint64_t> pad_pi_to_host(const std::vector<uint64_t>& pi_in, int dim, int len_pi, int N) {
    // Input Pi: len_pi matrices. Each dim x dim. Stride (dim+63)/64 words per row.
    // Output Pi: len_pi matrices. Each N x N. Stride N/64 words per row.
    
    size_t in_row_stride = (dim + 63) / 64;
    size_t in_mat_stride = dim * in_row_stride;
    
    size_t out_row_stride = N / 64;
    size_t out_mat_stride = N * out_row_stride;
    
    std::vector<uint64_t> pi_out(len_pi * out_mat_stride, 0);
    
    for (int k = 0; k < len_pi; ++k) {
        const uint64_t* src_mat = pi_in.data() + k * in_mat_stride;
        uint64_t* dst_mat = pi_out.data() + k * out_mat_stride;
        
        for (int r = 0; r < dim; ++r) {
            const uint64_t* src_row = src_mat + r * in_row_stride;
            uint64_t* dst_row = dst_mat + r * out_row_stride;
            
            // Copy row data
            for (size_t w = 0; w < out_row_stride; ++w) {
                if (w < in_row_stride) {
                    dst_row[w] = src_row[w];
                    // Mask last word if needed (to zero out padding bits in the row)
                    if (w == in_row_stride - 1) {
                        int valid = dim % 64;
                        if (valid != 0) dst_row[w] &= ((1ULL << valid) - 1);
                    }
                } else {
                    dst_row[w] = 0;
                }
            }
        }
    }
    return pi_out;
}

// -----------------------------------------------------------------------------
// Internal Verification
// -----------------------------------------------------------------------------
 
bool BasecaseSolverLegacy::check_annihilation(const uint64_t* d_S, int len_S, cudaStream_t stream) {
    LOG(LOG_STATS) << "[BasecaseLegacy] Optimized GPU Verification...";
    
    // 1. Determine Padding Dimension N
    int N = 64;
    while (N < dim_) N *= 2;
    if (N > 512) {
        LOG(LOG_ERROR_CRITICAL) << "[BasecaseLegacy] Dimension " << dim_ << " too large for optimized check (max 512).";
        return false;
    }
    
    // 2. Prepare Pi on Device (Padded)
    int pi_len = (int)pi_data_.size() / (dim_ * ((dim_+63)/64)); // Total coefficients
    int deg_pi = pi_len - 1;
    
    auto pi_padded_host = pad_pi_to_host(pi_data_, dim_, pi_len, N);
    
    uint64_t* d_Pi = nullptr;
    CHECK_CUDA_BC_LEGACY(cudaMalloc(&d_Pi, pi_padded_host.size() * 8));
    CHECK_CUDA_BC_LEGACY(cudaMemcpyAsync(d_Pi, pi_padded_host.data(), pi_padded_host.size() * 8, cudaMemcpyHostToDevice, stream));
    
    // 3. Allocate Buffers
    // V and W are vector series of length len_S, width N.
    size_t vec_series_bytes = (size_t)len_S * (N / 64) * 8;
    uint64_t *d_V = nullptr, *d_W = nullptr;
    int *d_fail = nullptr;
    
    CHECK_CUDA_BC_LEGACY(cudaMalloc(&d_V, vec_series_bytes));
    CHECK_CUDA_BC_LEGACY(cudaMalloc(&d_W, vec_series_bytes));
    CHECK_CUDA_BC_LEGACY(cudaMalloc(&d_fail, sizeof(int)));
    
    // 4. Iterate over columns of S^T (rows of S)
    // We check R(x) = Pi(x) * S^T(x).
    
    bool overall_pass = true;
    
    for (int c = 0; c < m_; ++c) {
        // Extract row c from S into V
        int threads = 256;
        int blocks = (len_S + threads - 1) / threads;
        
        if (N == 64)       k_extract_and_pad_row<64><<<blocks, threads, 0, stream>>>(d_S, d_V, len_S, m_, n_, c);
        else if (N == 128) k_extract_and_pad_row<128><<<blocks, threads, 0, stream>>>(d_S, d_V, len_S, m_, n_, c);
        else if (N == 256) k_extract_and_pad_row<256><<<blocks, threads, 0, stream>>>(d_S, d_V, len_S, m_, n_, c);
        else if (N == 512) k_extract_and_pad_row<512><<<blocks, threads, 0, stream>>>(d_S, d_V, len_S, m_, n_, c);
        
        // Compute W = Pi * V (Left Apply)
        // W[t] = sum Pi[k] * V[t-k]
        if (N == 64) {
             PolyMatrixView<64> Pi_view((uint64_t*)d_Pi, pi_len);
             PolyMatVec::apply_left<64, 64>(Pi_view, d_V, len_S, d_W, len_S, PolyMatVecBackend::GPU_Kernel, stream);
        } else if (N == 128) {
             PolyMatrixView<128> Pi_view((uint64_t*)d_Pi, pi_len);
             PolyMatVec::apply_left<128, 128>(Pi_view, d_V, len_S, d_W, len_S, PolyMatVecBackend::GPU_Kernel, stream);
        } else if (N == 256) {
             PolyMatrixView<256> Pi_view((uint64_t*)d_Pi, pi_len);
             PolyMatVec::apply_left<256, 256>(Pi_view, d_V, len_S, d_W, len_S, PolyMatVecBackend::GPU_Kernel, stream);
        } else if (N == 512) {
             PolyMatrixView<512> Pi_view((uint64_t*)d_Pi, pi_len);
             PolyMatVec::apply_left<512, 512>(Pi_view, d_V, len_S, d_W, len_S, PolyMatVecBackend::GPU_Kernel, stream);
        }
        
        // Check Zero
        CHECK_CUDA_BC_LEGACY(cudaMemsetAsync(d_fail, 0, sizeof(int), stream));
        
        if (len_S > deg_pi) {
            int check_len = len_S - deg_pi;
            blocks = (check_len + threads - 1) / threads;
            
            if (N == 64)       k_check_zero_suffix<64><<<blocks, threads, 0, stream>>>(d_W, len_S, deg_pi, d_fail);
            else if (N == 128) k_check_zero_suffix<128><<<blocks, threads, 0, stream>>>(d_W, len_S, deg_pi, d_fail);
            else if (N == 256) k_check_zero_suffix<256><<<blocks, threads, 0, stream>>>(d_W, len_S, deg_pi, d_fail);
            else if (N == 512) k_check_zero_suffix<512><<<blocks, threads, 0, stream>>>(d_W, len_S, deg_pi, d_fail);
        }
        
        int h_fail = 0;
        CHECK_CUDA_BC_LEGACY(cudaMemcpyAsync(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_BC_LEGACY(cudaStreamSynchronize(stream));
        
        if (h_fail) {
            LOG(LOG_ERROR_MAJOR) << "[BasecaseLegacy] Optimized Verification FAILED for Sequence Row " << c;
            overall_pass = false;
        }
    }
    
    CHECK_CUDA_BC_LEGACY(cudaFree(d_Pi));
    CHECK_CUDA_BC_LEGACY(cudaFree(d_V));
    CHECK_CUDA_BC_LEGACY(cudaFree(d_W));
    CHECK_CUDA_BC_LEGACY(cudaFree(d_fail));
    
    if (overall_pass) LOG(LOG_INFO) << "[BasecaseLegacy] Optimized Verification: PASS";
    
    return overall_pass;
}
 
bool BasecaseSolverLegacy::check_annihilation_legacy(const std::vector<std::vector<uint64_t>>& S, 
                                        const std::vector<std::vector<uint64_t>>& Pi) {
    int len_S = (int)S.size();
    int deg_Pi = (int)Pi.size() - 1;
    int dim = dim_;
    int n = n_;
    int m = m_;   

    LOG(LOG_STATS) << "[BasecaseLegacy] Verifying Generator Property for t=[" << deg_Pi << ", " << len_S << ")...";

    bool all_ok = true;
    int fail_count = 0;

    for (int t = deg_Pi; t < len_S; ++t) {
        // Delta check: sum_{k} F_k * S_{t-k}^T
        std::vector<uint64_t> delta_check(dim, 0); 

        for (int k = 0; k <= deg_Pi; ++k) {
            int s_idx = t - k;
            if (s_idx < 0) continue;

            for (int r = 0; r < dim; ++r) {
                for (int c = 0; c < m; ++c) {
                    bool dot = false;
                    for (int l = 0; l < n; ++l) {
                         bool f_val = get_bit(Pi[k].data(), dim, dim, r, l);
                         bool s_val = get_bit(S[s_idx].data(), m, n, c, l);
                         if (f_val && s_val) dot = !dot;
                    }
                    if (dot) delta_check[r] ^= (1ULL << c);
                }
            }
        }

        bool t_ok = true;
        for (int r = 0; r < dim; ++r) {
            if (delta_check[r] != 0) {
                t_ok = false;
                if (fail_count == 0) {
                    LOG(LOG_ERROR_MAJOR) << "[BasecaseLegacy] Annihilation FAILED at t=" << t 
                                         << ", Row=" << r << ", Val=" << std::hex << delta_check[r];
                }
            }
        }
        if (!t_ok) {
            all_ok = false;
            fail_count++;
            if (fail_count >= 10) break; 
        }
    }

    if (all_ok) LOG(LOG_INFO) << "[BasecaseLegacy] Annihilation Check: PASS";
    else LOG(LOG_ERROR_MAJOR) << "[BasecaseLegacy] Annihilation Check: FAILED (" << fail_count << " mismatches)";
    
    return all_ok;
}

// -----------------------------------------------------------------------------
// Bit Manipulation Helpers
// -----------------------------------------------------------------------------

bool BasecaseSolverLegacy::get_bit(const uint64_t* data, int rows, int cols, int r, int c) const {
    int stride_words = (cols + 63) / 64;
    int word_idx = r * stride_words + (c / 64);
    int bit_idx = c % 64;
    return (data[word_idx] >> bit_idx) & 1ULL;
}

void BasecaseSolverLegacy::set_bit(uint64_t* data, int rows, int cols, int r, int c, bool val) {
    int stride_words = (cols + 63) / 64;
    int word_idx = r * stride_words + (c / 64);
    int bit_idx = c % 64;
    if (val) data[word_idx] |= (1ULL << bit_idx);
    else     data[word_idx] &= ~(1ULL << bit_idx);
}

void BasecaseSolverLegacy::xor_bit(uint64_t* data, int rows, int cols, int r, int c) {
    int stride_words = (cols + 63) / 64;
    int word_idx = r * stride_words + (c / 64);
    int bit_idx = c % 64;
    data[word_idx] ^= (1ULL << bit_idx);
}

// -----------------------------------------------------------------------------
// Initialization Helpers (Generic m, n)
// -----------------------------------------------------------------------------

BasecaseSolverLegacy::InitResult BasecaseSolverLegacy::find_initialization_basis(const std::vector<std::vector<uint64_t>>& S_host) {
    int m = m_;
    int n = n_;
    int len_seq = (int)S_host.size();
    
    // Basis storage: map pivot_index -> vector (of m bits)
    // For m > 64, we store vectors as std::vector<uint64_t>
    int m_words = (m + 63) / 64;
    std::vector<std::vector<uint64_t>> basis_vecs(m); 
    std::vector<bool> pivot_found(m, false);
    
    std::vector<std::pair<int, int>> pairs;
    int rank = 0;
    int t0 = 0;

    // Iterate i (time) then j (column) - STRICT Python Order
    for (int i = 0; i < len_seq; ++i) {
        for (int j = 0; j < n; ++j) {
            // Extract column vector v = S[i][:, j]
            std::vector<uint64_t> v(m_words, 0);
            for(int r = 0; r < m; ++r) {
                if (get_bit(S_host[i].data(), m, n, r, j)) {
                    v[r / 64] |= (1ULL << (r % 64));
                }
            }

            // Reduce v against basis
            for(int p = 0; p < m; ++p) {
                if (pivot_found[p]) {
                    // Check if v has bit p set
                    if ((v[p / 64] >> (p % 64)) & 1ULL) {
                        // XOR basis vector into v
                        for(int w = 0; w < m_words; ++w) v[w] ^= basis_vecs[p][w];
                    }
                }
            }

            // Check if v is zero
            int first_bit = -1;
            for(int r = 0; r < m; ++r) {
                if ((v[r / 64] >> (r % 64)) & 1ULL) {
                    first_bit = r;
                    break;
                }
            }

            if (first_bit != -1) {
                // New pivot found
                pivot_found[first_bit] = true;
                basis_vecs[first_bit] = v;
                pairs.push_back({i, j});
                rank++;

                if (rank == m) {
                    t0 = i + 1;
                    return {t0, pairs};
                }
            }
        }
    }

    if (rank < m) {
        std::vector<int> missing;
        for(int p=0; p<m; ++p) if(!pivot_found[p]) missing.push_back(p);
        
        std::ostringstream oss;
        oss << "[";
        for(size_t k=0; k<std::min((size_t)32, missing.size()); ++k) {
            oss << missing[k];
            if (k + 1 < missing.size()) oss << ", ";
        }
        if(missing.size() > 32) oss << "...";
        oss << "]";
        
        LOG(LOG_ERROR_CRITICAL) << "[BasecaseLegacy] Rank condition failed! Rank=" << rank << " < m=" << m;
        LOG(LOG_ERROR_CRITICAL) << "[BasecaseLegacy] Missing pivot indices: " << oss.str();
        return {0, {}};
    }
    
    // Should be unreachable due to check above
    return {0, {}};
    
    LOG(LOG_ERROR_CRITICAL) << "[BasecaseLegacy] Rank condition failed! Rank=" << rank << " < m=" << m;
    return {0, {}};
}

BasecaseSolverLegacy::FInitResult BasecaseSolverLegacy::build_f_init(const InitResult& init) {
    int t0 = init.t0;
    int dim = dim_;
    int n = n_;
    int m = m_;
    
    // Allocate F_poly
    // Storage: currently dim x dim for compatibility with legacy solver logic,
    // although logically it is dim x n.
    // Stride must match what `solve` loop expects: `(dim + 63)/64`.
    int stride_F = (dim + 63) / 64; 
    int words_per_mat = dim * stride_F;
    
    std::vector<std::vector<uint64_t>> F(t0 + 1, std::vector<uint64_t>(words_per_mat, 0));
    std::vector<int> gamma(dim, t0);

    // 1. Top n rows: Identity at degree 0
    // F[0][j, j] = 1 for j < n
    for (int j = 0; j < n; ++j) {
        set_bit(F[0].data(), dim, dim, j, j, true);
    }

    // 2. Bottom m rows: Monomials from basis
    // Row index in F: n + k (where k is index in basis_pairs 0..m-1)
    // Col index: j_k
    // Degree: t0 - i_k
    for (int k = 0; k < m; ++k) {
        if (k >= (int)init.basis_pairs.size()) break; // Should not happen if rank=m
        
        int i_k = init.basis_pairs[k].first;
        int j_k = init.basis_pairs[k].second;
        int deg = t0 - i_k;
        int row_idx = n + k;
        
        // Sanity check
        if (deg < 0 || deg > t0) {
            LOG(LOG_ERROR_CRITICAL) << "Logic Error: deg=" << deg << " out of bounds";
        }
        
        set_bit(F[deg].data(), dim, dim, row_idx, j_k, true);
    }
    
    return {F, gamma};
}

// -----------------------------------------------------------------------------
// Discrepancy Computation (Generic)
// -----------------------------------------------------------------------------
std::vector<uint64_t> BasecaseSolverLegacy::compute_discrepancy(
    const std::vector<std::vector<uint64_t>>& F_poly,
    const std::vector<std::vector<uint64_t>>& S_host,
    int t) 
{
    int dim = dim_;
    int m = m_;
    int n = n_;
    
    // Output Delta is dim x m
    int delta_stride = (m + 63) / 64;
    std::vector<uint64_t> delta(dim * delta_stride, 0);
    
    int current_deg = (int)F_poly.size() - 1;

    // F_k stride: dim x dim (but used as dim x n)
    int f_stride = (dim + 63) / 64;
    
    // S_idx stride: m x n
    int s_stride = (n + 63) / 64;
    int n_words = s_stride;

    for (int k = 0; k <= current_deg; ++k) {
        int s_idx = t - k;
        if (s_idx < 0) continue;

        const uint64_t* F_mat = F_poly[k].data();
        const uint64_t* S_mat = S_host[s_idx].data();
        
        // Compute F_k * S_mat^T
        // For each row r of F, each row c of S, compute dot product over n cols
        
        for (int r = 0; r < dim; ++r) {
            const uint64_t* F_row = F_mat + r * f_stride;
            
            for (int c = 0; c < m; ++c) {
                const uint64_t* S_row = S_mat + c * s_stride;
                
                // Dot product F_row . S_row (packed)
                uint64_t acc = 0;
                // n bits might span multiple words
                // We must mask the last word if n % 64 != 0, 
                // but S_row and F_row are presumably clean or we ignore extra bits.
                // Assuming standard packing where unused bits are 0.

		if(n_words == 1) {
		    acc ^= (F_row[0] & S_row[0]);
		} else if(n_words == 2) {
		    acc ^= (F_row[0] & S_row[0]);
		    acc ^= (F_row[1] & S_row[1]);
		} else if(n_words == 4) {
		    acc ^= (F_row[0] & S_row[0]);
		    acc ^= (F_row[1] & S_row[1]);
		    acc ^= (F_row[2] & S_row[2]);
		    acc ^= (F_row[3] & S_row[3]);
		} else if(n_words == 8) {
		    acc ^= (F_row[0] & S_row[0]);
		    acc ^= (F_row[1] & S_row[1]);
		    acc ^= (F_row[2] & S_row[2]);
		    acc ^= (F_row[3] & S_row[3]);
		    acc ^= (F_row[4] & S_row[4]);
		    acc ^= (F_row[5] & S_row[5]);
		    acc ^= (F_row[6] & S_row[6]);
		    acc ^= (F_row[7] & S_row[7]);
		} else {
		    for(int w = 0; w < n_words; ++w) {
		        acc ^= (F_row[w] & S_row[w]);
		    }
		}
                
                // Parity of popcount
                if (__builtin_popcountll(acc) & 1) {
                    // Set bit c in row r of Delta
                    delta[r * delta_stride + (c / 64)] ^= (1ULL << (c % 64));
                }
            }
        }
    }
    return delta;
}

// -----------------------------------------------------------------------------
// Elimination and Update (Generic)
// -----------------------------------------------------------------------------

BasecaseSolverLegacy::StepResult BasecaseSolverLegacy::compute_elimination_step(const std::vector<uint64_t>& delta) {
    int dim = dim_;
    int m = m_;
    
    // Copy Delta
    std::vector<uint64_t> reduced_delta = delta;
    int delta_stride = (m + 63) / 64;

    // Init Tau = Identity
    int tau_stride = (dim + 63) / 64;
    std::vector<uint64_t> tau(dim * tau_stride, 0);
    for(int i=0; i<dim; ++i) set_bit(tau.data(), dim, dim, i, i, true);

    // Stable Sort Permutation by Gamma
    std::vector<int> p_ord(dim);
    std::iota(p_ord.begin(), p_ord.end(), 0);
    std::stable_sort(p_ord.begin(), p_ord.end(), [&](int a, int b){
        return gamma_[a] < gamma_[b];
    });

    // Gaussian Elimination
    // Eliminate columns 0..m-1
    for (int col = 0; col < m; ++col) {
        int pivot_row = -1;
        
        // Find pivot: first row in p_ord with 1 in current col
        for (int idx : p_ord) {
            if (get_bit(reduced_delta.data(), dim, m, idx, col)) {
                pivot_row = idx;
                break;
            }
	}

        if (pivot_row != -1) {
            // Remove pivot from list so it's not chosen again for another column
            auto it = std::remove(p_ord.begin(), p_ord.end(), pivot_row);
            p_ord.erase(it, p_ord.end());
            
            // Eliminate other rows
            for (int r = 0; r < dim; ++r) {
                if (r == pivot_row) continue;
                if (get_bit(reduced_delta.data(), dim, m, r, col)) {
                    // Reduce Delta row r ^= pivot
                    for(int w = 0; w < delta_stride; ++w) {
                        reduced_delta[r * delta_stride + w] ^= reduced_delta[pivot_row * delta_stride + w];
                    }
                    // Reduce Tau row r ^= pivot
                    for(int w = 0; w < tau_stride; ++w) {
                        tau[r * tau_stride + w] ^= tau[pivot_row * tau_stride + w];
                    }
                }
            }
        }
    }
    
    return {tau, reduced_delta};
}

std::vector<bool> BasecaseSolverLegacy::compute_shift_vector(const std::vector<uint64_t>& reduced_delta) {
    int dim = dim_;
    int m = m_;
    int delta_stride = (m + 63) / 64;
    
    std::vector<int> zero_indices;
    std::vector<int> non_zero_indices;
    
    for (int r = 0; r < dim; ++r) {
        bool is_zero = true;
        for (int w = 0; w < delta_stride; ++w) {
            if (reduced_delta[r * delta_stride + w] != 0) {
                is_zero = false;
                break;
            }
        }
        
        if (is_zero) zero_indices.push_back(r);
        else non_zero_indices.push_back(r);
    }
    
    // Sort zero_indices by gamma (Stable)
    std::stable_sort(zero_indices.begin(), zero_indices.end(), [&](int a, int b){
        return gamma_[a] < gamma_[b];
    });
    
    // Select exactly m shifts
    std::vector<bool> will_shift(dim, false);
    int shifts_needed = m;
    
    // Priority 1: Non-zero rows (degree must increase to correct discrepancy)
    for(int idx : non_zero_indices) {
        if (shifts_needed > 0) {
            will_shift[idx] = true;
            shifts_needed--;
        }
    }
    
    // Priority 2: Zero rows with smallest gamma
    for(int idx : zero_indices) {
        if (shifts_needed > 0) {
            will_shift[idx] = true;
            shifts_needed--;
        }
    }
    
    return will_shift;
}

std::vector<std::vector<uint64_t>> BasecaseSolverLegacy::apply_update(
    const std::vector<std::vector<uint64_t>>& F_poly,
    const std::vector<uint64_t>& tau,
    const std::vector<bool>& will_shift
) {
    int dim = dim_;
    int stride_F = (dim + 63) / 64;
    int words_per_mat = dim * stride_F;
    int tau_stride = (dim + 63) / 64;
    
    std::vector<std::vector<uint64_t>> F_next(F_poly.size() + 1);
    for(auto& mat : F_next) mat.resize(words_per_mat, 0);
    
    // F_next = (Shift) * Tau * F
    // F_next[dest] row r = sum_l Tau[r,l] * F[k] row l
    /*
    for(int k=0; k < (int)F_poly.size(); ++k) {
        const uint64_t* F_curr_data = F_poly[k].data();
        
        for(int r=0; r < dim; ++r) {
            // Determine destination degree
            int dest_k = k + (will_shift[r] ? 1 : 0);
            uint64_t* F_dest_row = F_next[dest_k].data() + r * stride_F;
            
            // Dense row-vector multiply: F_dest_row += Tau[r] . F_curr
            // We iterate l over rows of F_curr
            // Check bit l of Tau[r]
            
            const uint64_t* tau_row_ptr = tau.data() + r * tau_stride;
            
            for(int l = 0; l < dim; ++l) {
                if ((tau_row_ptr[l / 64] >> (l % 64)) & 1ULL) {
                    const uint64_t* F_src_row = F_curr_data + l * stride_F;
                    // XOR entire row
                    for(int w = 0; w < stride_F; ++w) {
                        F_dest_row[w] ^= F_src_row[w];
                    }
                }
            }
        }
    }
    */
    // Optimized Algorithm:
    // Iterate rows 'r' of Tau (dest rows).
    // Identify set bits 'l' in Tau[r]. These are the source rows of F to accumulate.
    // Use CTZ to skip zeros.
    
    for (int r = 0; r < dim; ++r) {
        const uint64_t* tau_row_ptr = tau.data() + r * tau_stride;
        
        // Loop over words of tau row 'r'
        for (int w = 0; w < tau_stride; ++w) {
            uint64_t word = tau_row_ptr[w];
            // Iterate set bits in word
            while (word != 0) {
                int bit = __builtin_ctzll(word);
                int l = w * 64 + bit; // Source row index
                
                // Add row l of F_curr to row r of F_next for all degrees k
                // This essentially does: F_next[dest_k][r] ^= F_curr[k][l]
                for(int k = 0; k < (int)F_poly.size(); ++k) {
                    int dest_k = k + (will_shift[r] ? 1 : 0);
                    
                    uint64_t* F_dest_row = F_next[dest_k].data() + r * stride_F;
                    const uint64_t* F_src_row = F_poly[k].data() + l * stride_F;
                    
                    // Vector XOR
                    for(int i = 0; i < stride_F; ++i) {
                        F_dest_row[i] ^= F_src_row[i];
                    }
                }
                
                // Clear LSB
                word &= (word - 1);
            }
        }
    }
    // Trim trailing zeros
    while(F_next.size() > 1) {
        bool all_zero = true;
        const auto& last = F_next.back();
        for(auto w : last) if(w!=0) { all_zero = false; break; }
        if(all_zero) F_next.pop_back(); 
        else break;
    }
    
    return F_next;
}

// -----------------------------------------------------------------------------
// Core Solver Logic
// -----------------------------------------------------------------------------

 void BasecaseSolverLegacy::solve(const uint64_t* d_S, int len, cudaStream_t stream,
			    bool verify_annihilation_gpu, bool verify_annihilation_legacy) { 
    int m = m_;  
    int n = n_;  
    int dim = dim_; // m_ + n_
    
    LOG(LOG_STATS) << "[BasecaseLegacy] Solver initializing with m=" << m << ", n=" << n << ", dim=" << dim;
    
    // Validate inputs against internal assumptions
    if (len <= 0) {
        LOG(LOG_ERROR_CRITICAL) << "[BasecaseLegacy] Invalid sequence length: " << len;
        return;
    }

    LOG(LOG_STATS) << "[BasecaseLegacy] Downloading sequence of length " << len << " from GPU...";
    
    // S_k is m x n. Stride is determined by n.
    size_t words_per_row = (n + 63) / 64;
    size_t words_per_mat = m * words_per_row;
    
    std::vector<std::vector<uint64_t>> S_host(len);
    std::vector<uint64_t> temp_buf(words_per_mat);

    // Download loop
    for (int k = 0; k < len; ++k) {
        const uint64_t* src_ptr = d_S + k * words_per_mat;
        cudaMemcpyAsync(temp_buf.data(), src_ptr, words_per_mat * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); // Sync per block to keep memory usage low on host (optional optimization)
        S_host[k] = temp_buf;
    }

    LOG(LOG_DEBUG_2) << "[BasecaseLegacy] Initializing basis, f...";
    // 1. Run Generic initialization
    auto init_generic = find_initialization_basis(S_host);
    auto finit_generic = build_f_init(init_generic);

    std::vector<std::vector<uint64_t>> F_poly = finit_generic.F_poly;
    gamma_ = finit_generic.gamma;
    int t0 = init_generic.t0;

    LOG(LOG_STATS) << "[BasecaseLegacy] Initialization: t0=" << t0 << ", Basis rank=" << m;
    {
        std::ostringstream oss;
        oss << "[";
        for(size_t k=0; k<init_generic.basis_pairs.size(); ++k) {
            oss << "(" << init_generic.basis_pairs[k].first << ", " << init_generic.basis_pairs[k].second << ")";
            if(k+1 < init_generic.basis_pairs.size()) oss << ", ";
        }
        oss << "]";
        LOG(LOG_DEBUG_2) << "[BasecaseLegacy] Basis Pairs: " << oss.str();
    }
    
    LOG(LOG_STATS) << "[BasecaseLegacy] Initialized F(x) up to degree " << (F_poly.size()-1);


    // --- Iterative BBM ---
    int L = len;
    
    for (int t = t0; t < L; ++t) {
        // 1. Discrepancy (Generic)
        std::vector<uint64_t> delta = compute_discrepancy(F_poly, S_host, t);

        // 2. Elimination (Generic)
        auto step_res = compute_elimination_step(delta);
        
        // 3. Shift Selection (Generic)
        auto will_shift = compute_shift_vector(step_res.reduced_delta);
        
        // 4. Update F (Generic)
        F_poly = apply_update(F_poly, step_res.tau, will_shift);

        // Update Gamma
        for(int r=0; r<dim; ++r) {
            if(will_shift[r]) gamma_[r]++;
        }

        if (t % 500 == 0 || t == t0) {
            LOG(LOG_DEBUG_1) << "[BasecaseLegacy] t=" << t << ", Deg=" << (F_poly.size()-1) << "...";
        }
    }

    LOG(LOG_INFO) << "[BasecaseLegacy] Solver finished. Final Degree=" << (F_poly.size()-1);
    
    // Store result
    pi_data_.clear();
    for(const auto& mat : F_poly) {
        pi_data_.insert(pi_data_.end(), mat.begin(), mat.end());
    }
    // --- Verification ---
    if (verify_annihilation_gpu) {
        if (!check_annihilation(d_S, len, stream)) {
            LOG(LOG_ERROR_CRITICAL) << "[BasecaseLegacy] GPU Annihilation Verification Failed!";
        } else {
            LOG(LOG_DEBUG_1) << "[BasecaseLegacy] GPU Annihilation Verification PASSED.";
	}
    } else {
        LOG(LOG_DEBUG_1) << "[BasecaseLegacy] GPU Annihilation Verification SKIPPED.";
    }
    if (verify_annihilation_legacy) {
        if (!check_annihilation_legacy(S_host, F_poly)) {
             LOG(LOG_ERROR_CRITICAL) << "[BasecaseLegacy] Legacy Annihilation Verification Failed!";
        } else {
	    LOG(LOG_INFO) << "[BasecaseLegacy] Legacy Annihilation Verification PASSED.";
	}
    } else {
        LOG(LOG_DEBUG_2) << "[BasecaseLegacy] Legacy Annihilation Verification SKIPPED.";
    }
}

} // namespace stage2
} // namespace lingen
