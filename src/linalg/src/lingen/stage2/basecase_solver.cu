// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/stage2/basecase_solver.h"
#include "lingen/operations/poly_mat_vec.h"
#include "lingen/stage2/device_poly.h"
#include "lingen/operations/matmul_gf2.h"
#include "lingen/io/hash.h"
#include "hpc_logger.h"

#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <numeric>
#include <set>
#include <sstream>
#include <memory>
#include <cassert>
#include <stdexcept>
#include <string>

namespace lingen {
namespace stage2 {

// -----------------------------------------------------------------------------
// Error Handling Macros
// -----------------------------------------------------------------------------

#define CHECK_CUDA_BC(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "[Basecase] CUDA Error: " \
                                << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)

// =================================================================================
// BasecaseSolver (Generic)
// =================================================================================

BasecaseSolver::BasecaseSolver(const BWStage2Config& config) 
    : config_(config)
{
    dim_ = config_.m_block + config_.n_block;
    
    // Auto-derive delta if 0
    if (config_.delta == 0) {
        delta_ = config_.seq_len / 2;
    } else {
        delta_ = config_.delta;
    }

    LOG(LOG_STATS) << "[Basecase] Constructed solver for m=" << config_.m_block
                  << ", n=" << config_.n_block
                  << " (dim=" << dim_ << "),"
                  << " Mode=" << (config_.gpu_mode ? "GPU" : "CPU_ONLY");
    
    // Logging Verification Settings
    LOG(LOG_DEBUG_1) << "[Basecase] Settings: Delta=" << delta_ 
                     << ", Check GPU=" << (config_.check_annihilation_gpu ? "ON" : "OFF")
                     << ", Check Legacy=" << (config_.check_annihilation_legacy ? "ON" : "OFF")
                     << ", Oracle=" << (config_.internal_oracle_verification ? "ON" : "OFF");
}

BasecaseSolver::~BasecaseSolver() {}

// =================================================================================
// State Hashing
// =================================================================================

uint64_t BasecaseSolver::compute_state_hash() const {
    uint64_t h = 0xcbf29ce484222325ULL;
    
    // Hash Pi (Coefficients)
    if (!pi_data_.empty()) {
        h = fnv1a_hash_bytes(pi_data_.data(), pi_data_.size() * sizeof(uint64_t), h);
    }
    
    // Hash Gamma (Degrees)
    if (!gamma_.empty()) {
        h = fnv1a_hash_bytes(gamma_.data(), gamma_.size() * sizeof(int), h);
    }
    
    return h;
} 

// =================================================================================
// CUDA Kernels for Optimized Annihilation Check
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
 
bool BasecaseSolver::check_annihilation(const uint64_t* d_S, int len_S, cudaStream_t stream) {
    LOG(LOG_STATS) << "[Basecase] Optimized GPU Verification...";
    
    // 1. Determine Padding Dimension N
    int N = 64;
    while (N < dim_) N *= 2;
    if (N > 512) {
        LOG(LOG_ERROR_CRITICAL) << "[Basecase] Dimension " << dim_ << " too large for optimized check (max 512).";
        return false;
    }
    
    // 2. Prepare Pi on Device (Padded)
    int pi_len = (int)pi_data_.size() / (dim_ * ((dim_+63)/64)); // Total coefficients
    int deg_pi = pi_len - 1;
    
    auto pi_padded_host = pad_pi_to_host(pi_data_, dim_, pi_len, N);
    
    uint64_t* d_Pi = nullptr;
    CHECK_CUDA_BC(cudaMalloc(&d_Pi, pi_padded_host.size() * 8));
    CHECK_CUDA_BC(cudaMemcpyAsync(d_Pi, pi_padded_host.data(), pi_padded_host.size() * 8, cudaMemcpyHostToDevice, stream));
    
    // 3. Allocate Buffers
    // V and W are vector series of length len_S, width N.
    size_t vec_series_bytes = (size_t)len_S * (N / 64) * 8;
    uint64_t *d_V = nullptr, *d_W = nullptr;
    int *d_fail = nullptr;
    
    CHECK_CUDA_BC(cudaMalloc(&d_V, vec_series_bytes));
    CHECK_CUDA_BC(cudaMalloc(&d_W, vec_series_bytes));
    CHECK_CUDA_BC(cudaMalloc(&d_fail, sizeof(int)));
    
    // 4. Iterate over columns of S^T (rows of S)
    // We check R(x) = Pi(x) * S^T(x).
    
    bool overall_pass = true;
    
    for (int c = 0; c < config_.m_block; ++c) {
        // Extract row c from S into V
        int threads = 256;
        int blocks = (len_S + threads - 1) / threads;

	int m_ = config_.m_block;
	int n_ = config_.n_block;
        
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
        CHECK_CUDA_BC(cudaMemsetAsync(d_fail, 0, sizeof(int), stream));
        
        if (len_S > deg_pi) {
            int check_len = len_S - deg_pi;
            blocks = (check_len + threads - 1) / threads;
            
            if (N == 64)       k_check_zero_suffix<64><<<blocks, threads, 0, stream>>>(d_W, len_S, deg_pi, d_fail);
            else if (N == 128) k_check_zero_suffix<128><<<blocks, threads, 0, stream>>>(d_W, len_S, deg_pi, d_fail);
            else if (N == 256) k_check_zero_suffix<256><<<blocks, threads, 0, stream>>>(d_W, len_S, deg_pi, d_fail);
            else if (N == 512) k_check_zero_suffix<512><<<blocks, threads, 0, stream>>>(d_W, len_S, deg_pi, d_fail);
        }
        
        int h_fail = 0;
        CHECK_CUDA_BC(cudaMemcpyAsync(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_BC(cudaStreamSynchronize(stream));
        
        if (h_fail) {
            LOG(LOG_ERROR_MAJOR) << "[Basecase] Optimized Verification FAILED for Sequence Row " << c;
            overall_pass = false;
        }
    }
    
    CHECK_CUDA_BC(cudaFree(d_Pi));
    CHECK_CUDA_BC(cudaFree(d_V));
    CHECK_CUDA_BC(cudaFree(d_W));
    CHECK_CUDA_BC(cudaFree(d_fail));
    
    if (overall_pass) LOG(LOG_INFO) << "[Basecase] Optimized Verification: PASS";
    
    return overall_pass;
}
 
bool BasecaseSolver::check_annihilation_legacy(const std::vector<std::vector<uint64_t>>& S, 
                                        const std::vector<std::vector<uint64_t>>& Pi) {
    int len_S = (int)S.size();
    int deg_Pi = (int)Pi.size() - 1;
    int dim = dim_;
    int m = config_.m_block;   
    int n = config_.n_block;

    LOG(LOG_STATS) << "[Basecase] Verifying Generator Property for t=[" << deg_Pi << ", " << len_S << ")...";

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
                    LOG(LOG_ERROR_MAJOR) << "[Basecase] Annihilation FAILED at t=" << t 
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

    if (all_ok) LOG(LOG_INFO) << "[Basecase] Annihilation Check: PASS";
    else LOG(LOG_ERROR_MAJOR) << "[Basecase] Annihilation Check: FAILED (" << fail_count << " mismatches)";
    
    return all_ok;
}

// -----------------------------------------------------------------------------
// Bit Manipulation Helpers
// -----------------------------------------------------------------------------

bool BasecaseSolver::get_bit(const uint64_t* data, int rows, int cols, int r, int c) const {
    int stride_words = (cols + 63) / 64;
    int word_idx = r * stride_words + (c / 64);
    int bit_idx = c & 63; // % 64;
    return (data[word_idx] >> bit_idx) & 1ULL;
}

void BasecaseSolver::set_bit(uint64_t* data, int rows, int cols, int r, int c, bool val) {
    int stride_words = (cols + 63) / 64;
    int word_idx = r * stride_words + (c / 64);
    int bit_idx = c & 63; // % 64;
    if (val) data[word_idx] |= (1ULL << bit_idx);
    else     data[word_idx] &= ~(1ULL << bit_idx);
}

void BasecaseSolver::xor_bit(uint64_t* data, int rows, int cols, int r, int c) {
    int stride_words = (cols + 63) / 64;
    int word_idx = r * stride_words + (c / 64);
    int bit_idx = c & 63; // % 64;
    data[word_idx] ^= (1ULL << bit_idx);
}

// -----------------------------------------------------------------------------
// Initialization Helpers (Generic m, n)
// -----------------------------------------------------------------------------

BasecaseSolver::InitResult BasecaseSolver::find_initialization_basis(const std::vector<std::vector<uint64_t>>& S_host) {
    int m = config_.m_block;
    int n = config_.n_block;
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
        
        LOG(LOG_ERROR_CRITICAL) << "[Basecase] Rank condition failed! Rank=" << rank << " < m=" << m;
        LOG(LOG_ERROR_CRITICAL) << "[Basecase] Missing pivot indices: " << oss.str();
        return {0, {}};
    }
    
    // Should be unreachable due to check above
    return {0, {}};
    
    LOG(LOG_ERROR_CRITICAL) << "[Basecase] Rank condition failed! Rank=" << rank << " < m=" << m;
    return {0, {}};
}

BasecaseSolver::FInitResult BasecaseSolver::build_f_init(const InitResult& init) {
    int t0 = init.t0;
    int dim = dim_;
    int n = config_.n_block;
    int m = config_.m_block;
    
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
std::vector<uint64_t> BasecaseSolver::compute_discrepancy(
    const std::vector<std::vector<uint64_t>>& F_poly,
    const std::vector<std::vector<uint64_t>>& S_host,
    int t) 
{
    int dim = dim_;
    int m = config_.m_block;
    int n = config_.n_block;
    
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

BasecaseSolver::StepResult BasecaseSolver::compute_elimination_step(const std::vector<uint64_t>& delta) {
    int dim = dim_;
    int m = config_.m_block;
    
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

std::vector<bool> BasecaseSolver::compute_shift_vector(const std::vector<uint64_t>& reduced_delta) {
    int dim = dim_;
    int m = config_.m_block;
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

std::vector<std::vector<uint64_t>> BasecaseSolver::apply_update(
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

// =================================================================================
// GPU Kernels: Discrepancy and Elimination
// =================================================================================

namespace kernel {

// Kernel configuration
// TileD: Rows of Delta computed per block (Rows of F)
// TileM: Cols of Delta computed per block (Rows of S)
template<int TileD, int TileM>
__global__ void k_compute_discrepancy(
    PackedBitPolyView F_view,
    const uint64_t* __restrict__ S_data,
    uint64_t* __restrict__ Delta_out,
    int t,
    int m,
    int n,
    size_t S_mat_stride_words,
    size_t Delta_stride_words
) {
    // Dynamic shared memory for tiles
    extern __shared__ uint64_t smem[];
    
    // Derived geometry
    int words_n = (n + 63) / 64;
    
    // Shared Memory pointers
    uint64_t* sm_F = smem; // TileD x words_n
    uint64_t* sm_S = smem + TileD * words_n; // TileM x words_n
    
    // Global Coordinates (output tile)
    int row_start = blockIdx.y * TileD;
    int col_start = blockIdx.x * TileM;
    
    // Thread mapping
    int tx = threadIdx.x; // 0..31 (Maps to TileM columns)
    int ty = threadIdx.y; // 0..TileD-1 (Maps to TileD rows)
    int tid = ty * blockDim.x + tx;
    int n_threads = blockDim.x * blockDim.y;
    
    // Accumulator (1 bit per thread)
    int my_acc_bit = 0;
    
    // Loop over k (0 to degree)
    // We iterate backwards in S (S_{t-k}) and forwards in F (F_k)
    int deg = (int)F_view.length - 1;
    int limit_k = (deg < t) ? deg : t;
    
    for (int k = 0; k <= limit_k; ++k) {
        int s_idx = t - k;
        
        // 1. Cooperative Load F tile (k-th coeff, rows [row_start...])
        // F is logically dim x n (stored dim x cols)
        // Access: F_view.coeff(k).row_ptr(row_start + r)
        PackedBitMatView F_k = F_view.coeff(k);
        
        for (int i = tid; i < TileD * words_n; i += n_threads) {
            int r = i / words_n;
            int w = i % words_n;
            if (row_start + r < F_view.rows) {
                sm_F[i] = F_k.row_ptr(row_start + r)[w];
            } else {
                sm_F[i] = 0;
            }
        }
        
        // 2. Cooperative Load S tile (S_{t-k}, rows [col_start...])
        // S_mat is m x n
        // Access: S_data + s_idx * stride + (col_start + c) * row_stride
        const uint64_t* S_mat_ptr = S_data + (size_t)s_idx * S_mat_stride_words;
        size_t S_row_stride = (n + 63) / 64; // Same as words_n usually
        
        for (int i = tid; i < TileM * words_n; i += n_threads) {
            int c = i / words_n;
            int w = i % words_n;
            if (col_start + c < m) {
                sm_S[i] = S_mat_ptr[(col_start + c) * S_row_stride + w];
            } else {
                sm_S[i] = 0;
            }
        }
        
        __syncthreads();
        
        // 3. Compute Dot Product for my element (ty, tx)
        // Delta[row_start + ty, col_start + tx]
        // dot = F_row . S_row
        if (row_start + ty < F_view.rows && col_start + tx < m) {
            const uint64_t* f_ptr = sm_F + ty * words_n;
            const uint64_t* s_ptr = sm_S + tx * words_n;
            
            uint64_t dot_chunk = 0;
            for(int w = 0; w < words_n; ++w) {
                dot_chunk ^= (f_ptr[w] & s_ptr[w]);
            }
            if (__popcll(dot_chunk) & 1) {
                my_acc_bit ^= 1;
            }
        }
        
        __syncthreads();
    }
    
    // 4. Write result
    // We use warp ballot to coalesce writes for bits in a row.
    // Threads with same ty form a row in the tile.
    // blockDim.x is 32 (TileM). So one warp is exactly one row of the tile (if TileM=32).
    // If TileM=32, threadIdx.x corresponds to column offset 0..31.
    // __ballot_sync gives bits for tx=0..31.
    
    uint32_t row_bits = __ballot_sync(0xFFFFFFFF, my_acc_bit);
    
    // Lane 0 writes the 32-bit chunk
    if (tx == 0) {
        int global_r = row_start + ty;
        int global_c_base = col_start;
        
        if (global_r < F_view.rows && global_c_base < m) {
            // Address of the word containing column global_c_base
            // Delta is row major packed.
            // Pointer to start of row
            uint64_t* delta_row = Delta_out + global_r * Delta_stride_words;
            
            // We write 32 bits starting at bit offset global_c_base
            // global_c_base is aligned to 32 bits (since TileM=32).
            int word_idx = global_c_base / 64;
            int bit_offset = global_c_base % 64;
            
            // We assume d_Delta is zero-initialized and we have exclusive access to this chunk.
            // We can use atomicXor on uint64 to be safe, or direct write if aligned.
            // Since we tile over m, and write distinct bit ranges, direct write is tricky if sharing 64-bit word.
            // atomicXor is safest for generic code, though slower.
            // However, with TileM=32, we update either lower or upper 32 bits of a 64-bit word.
            // Cast to uint32*? 
            // Yes, assuming little endian and 4-byte alignment.
            uint32_t* row_u32 = (uint32_t*)delta_row;
            // bit_offset is 0 or 32.
            // word_idx * 2 + (bit_offset/32)
            int u32_idx = (word_idx * 2) + (bit_offset / 32);
            
            // Masking for edge cases (m not multiple of 32)
            int valid_bits = m - global_c_base;
            if (valid_bits < 32) {
                uint32_t mask = (1U << valid_bits) - 1;
                row_bits &= mask;
            }
            
            // Just overwrite? Discrepancy computed from scratch.
            // We need to write row_bits.
            // Since multiple blocks don't overlap in output, direct write is OK.
            row_u32[u32_idx] = row_bits;
        }
    }
}

// Elimination Kernel
// Single thread block persistent style logic to handle sequential dependency.
// We use Global Memory for Delta/Tau access but cache row in shared for broadcast.
// Assumes dim <= 1024.
// Shared Memory: p_ord[1024], used[1024] (reused s_keys), s_keys[1024](uint64)
// Total shared size: approx 12-16KB + row buffers.
__global__ void k_elimination(
    uint64_t* reduced_delta, // Read-Write (in place modification of copy)
    uint64_t* tau,           // Write (init to I, then modify)
    const int* __restrict__ gamma,
    int m,
    int dim,
    size_t delta_stride, // words
    size_t tau_stride    // words
) {
    extern __shared__ uint64_t smem_elim[];
    // Layout: s_keys[1024], s_p_ord[1024], pivot_buffers[32]
    uint64_t* s_keys = smem_elim;
    int* s_p_ord = (int*)&s_keys[1024]; 
    uint64_t* s_pivot_delta = (uint64_t*)&s_p_ord[1024];
    uint64_t* s_pivot_tau = s_pivot_delta + 16; // Up to 128 bytes each (dim<=1024 -> 16 uint64s)
    
    // Reuse s_keys memory for 'used' array after sort
    bool* s_used = (bool*)s_keys; 

    int tid = threadIdx.x;
    
    // 1. Init Tau = Identity (Global)
    for (int r = tid; r < dim; r += blockDim.x) {
        uint64_t* row = tau + r * tau_stride;
        for (int w = 0; w < tau_stride; ++w) row[w] = 0;
        row[r / 64] |= (1ULL << (r % 64));
    }
    
    // 2. Prepare Sort Keys (Stable Sort by Gamma)
    if (tid < 1024) {
        if (tid < dim) {
            uint64_t g = (uint64_t)gamma[tid];
            s_keys[tid] = (g << 32) | tid;
        } else {
            s_keys[tid] = 0xFFFFFFFFFFFFFFFFULL; // Pad with Max
        }
    }
    __syncthreads();
    
    // 3. Bitonic Sort (size 1024)
    for (int k = 2; k <= 1024; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (tid < 1024) {
                 int ixj = tid ^ j;
                 if (ixj > tid) {
                     uint64_t ki = s_keys[tid];
                     uint64_t kj = s_keys[ixj];
                     bool swap = (tid & k) == 0 ? (ki > kj) : (ki < kj);
                     if (swap) {
                         s_keys[tid] = kj;
                         s_keys[ixj] = ki;
                     }
                 }
            }
            __syncthreads();
        }
    }
    
    // 4. Extract p_ord (Read s_keys)
    if (tid < dim) {
        s_p_ord[tid] = (int)(s_keys[tid] & 0xFFFFFFFF);
        // s_used[tid] = false;
    }
    __syncthreads();

    // 5. Init Used (Overwrite s_keys)
    if (tid < dim) {
        s_used[tid] = false;
    }
     __syncthreads();    
    
    // 5. Pivot Loop
    for (int col = 0; col < m; ++col) {
        // Find Pivot
        int candidate = 0x7FFFFFFF;
        if (tid < dim) {
            if (!s_used[tid]) {
                int r = s_p_ord[tid];
                uint64_t word = reduced_delta[r * delta_stride + (col / 64)];
                if ((word >> (col % 64)) & 1ULL) {
                    candidate = tid; // Index in p_ord
                }
            }
        }
        
        // Block Reduce Min
        // Warp Reduce
        for (int i = 16; i > 0; i >>= 1) {
            int other = __shfl_down_sync(0xFFFFFFFF, candidate, i);
            if (other < candidate) candidate = other;
        }
        // Shared Mem Inter-Warp
        __shared__ int s_warp_mins[32];
        int lane = tid % 32;
        int warp = tid / 32;
        if (lane == 0) s_warp_mins[warp] = candidate;
        __syncthreads();
        
        // Final Reduce (first warp)
        __shared__ int pivot_idx;
        if (warp == 0) {
            candidate = (tid < (blockDim.x/32)) ? s_warp_mins[tid] : 0x7FFFFFFF;
            for (int i = 16; i > 0; i >>= 1) {
                int other = __shfl_down_sync(0xFFFFFFFF, candidate, i);
                if (other < candidate) candidate = other;
            }
            if (tid == 0) pivot_idx = candidate;
        }
        __syncthreads();
        
        if (pivot_idx != 0x7FFFFFFF) {
            __shared__ int p_row;
            if (tid == 0) {
                p_row = s_p_ord[pivot_idx];
                s_used[pivot_idx] = true;
            }
            __syncthreads();
            
            // Load pivot row to shared mem
            int cur_p_row = p_row;
            for (int w = tid; w < delta_stride; w += blockDim.x) s_pivot_delta[w] = reduced_delta[cur_p_row * delta_stride + w];
            for (int w = tid; w < tau_stride; w += blockDim.x)   s_pivot_tau[w] = tau[cur_p_row * tau_stride + w];
            __syncthreads();
            
            // Elimination
            for (int r = tid; r < dim; r += blockDim.x) {
                if (r != cur_p_row) {
                    uint64_t* d_row_ptr = reduced_delta + r * delta_stride;
                    uint64_t dw = d_row_ptr[col / 64];
                    if ((dw >> (col % 64)) & 1ULL) {
                        for (int w = 0; w < delta_stride; ++w) d_row_ptr[w] ^= s_pivot_delta[w];
                        uint64_t* t_row_ptr = tau + r * tau_stride;
                        for (int w = 0; w < tau_stride; ++w) t_row_ptr[w] ^= s_pivot_tau[w];
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ void k_eliminationN(
    uint64_t* reduced_delta, // Read-Write (in place modification of copy)
    uint64_t* tau,           // Write (init to I, then modify)
    const int* __restrict__ gamma,
    int m,
    int dim,
    size_t delta_stride, // words
    size_t tau_stride    // words
) {
    extern __shared__ uint64_t smem_elim[];
    // Layout: s_keys[1024], s_p_ord[1024], pivot_buffers[32]
    uint64_t* s_keys = smem_elim;
    int* s_p_ord = (int*)&s_keys[1024]; 
    uint64_t* s_pivot_delta = (uint64_t*)&s_p_ord[1024];
    uint64_t* s_pivot_tau = s_pivot_delta + 16; // Up to 128 bytes each (dim<=1024 -> 16 uint64s)
    
    // Reuse s_keys memory for 'used' array after sort
    bool* s_used = (bool*)s_keys; 

    int tid = threadIdx.x;
    
    // 1. Init Tau = Identity (Global)
    for (int r = tid; r < dim; r += blockDim.x) {
        uint64_t* row = tau + r * tau_stride;
        for (int w = 0; w < tau_stride; ++w) row[w] = 0;
        row[r / 64] |= (1ULL << (r % 64));
    }
    
    // 2. Prepare Sort Keys (Stable Sort by Gamma)
    int N = blockDim.x;
    if (tid < N) {
        if (tid < dim) {
            uint64_t g = (uint64_t)gamma[tid];
            s_keys[tid] = (g << 32) | tid;
        } else {
            s_keys[tid] = 0xFFFFFFFFFFFFFFFFULL; // Pad with Max
        }
    }
    __syncthreads();
    
    // 3. Bitonic Sort (size 1024)
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (tid < N) {
                 int ixj = tid ^ j;
                 if (ixj > tid) {
                     uint64_t ki = s_keys[tid];
                     uint64_t kj = s_keys[ixj];
                     bool swap = (tid & k) == 0 ? (ki > kj) : (ki < kj);
                     if (swap) {
                         s_keys[tid] = kj;
                         s_keys[ixj] = ki;
                     }
                 }
            }
            __syncthreads();
        }
    }
    
    // 4. Extract p_ord (Read s_keys)
    if (tid < dim) {
        s_p_ord[tid] = (int)(s_keys[tid] & 0xFFFFFFFF);
        // s_used[tid] = false;
    }
    __syncthreads();

    // 5. Init Used (Overwrite s_keys)
    if (tid < dim) {
        s_used[tid] = false;
    }
     __syncthreads();    
    
    // 5. Pivot Loop
    for (int col = 0; col < m; ++col) {
        // Find Pivot
        int candidate = 0x7FFFFFFF;
        if (tid < dim) {
            if (!s_used[tid]) {
                int r = s_p_ord[tid];
                uint64_t word = reduced_delta[r * delta_stride + (col / 64)];
                if ((word >> (col % 64)) & 1ULL) {
                    candidate = tid; // Index in p_ord
                }
            }
        }
        
        // Block Reduce Min
        // Warp Reduce
        for (int i = 16; i > 0; i >>= 1) {
            int other = __shfl_down_sync(0xFFFFFFFF, candidate, i);
            if (other < candidate) candidate = other;
        }
        // Shared Mem Inter-Warp
        __shared__ int s_warp_mins[32];
        int lane = tid % 32;
        int warp = tid / 32;
        if (lane == 0) s_warp_mins[warp] = candidate;
        __syncthreads();
        
        // Final Reduce (first warp)
        __shared__ int pivot_idx;
        if (warp == 0) {
            candidate = (tid < (blockDim.x/32)) ? s_warp_mins[tid] : 0x7FFFFFFF;
            for (int i = 16; i > 0; i >>= 1) {
                int other = __shfl_down_sync(0xFFFFFFFF, candidate, i);
                if (other < candidate) candidate = other;
            }
            if (tid == 0) pivot_idx = candidate;
        }
        __syncthreads();
        
        if (pivot_idx != 0x7FFFFFFF) {
            __shared__ int p_row;
            if (tid == 0) {
                p_row = s_p_ord[pivot_idx];
                s_used[pivot_idx] = true;
            }
            __syncthreads();
            
            // Load pivot row to shared mem
            int cur_p_row = p_row;
            for (int w = tid; w < delta_stride; w += blockDim.x) s_pivot_delta[w] = reduced_delta[cur_p_row * delta_stride + w];
            for (int w = tid; w < tau_stride; w += blockDim.x)   s_pivot_tau[w] = tau[cur_p_row * tau_stride + w];
            __syncthreads();
            
            // Elimination
            for (int r = tid; r < dim; r += blockDim.x) {
                if (r != cur_p_row) {
                    uint64_t* d_row_ptr = reduced_delta + r * delta_stride;
                    uint64_t dw = d_row_ptr[col / 64];
                    if ((dw >> (col % 64)) & 1ULL) {
                        for (int w = 0; w < delta_stride; ++w) d_row_ptr[w] ^= s_pivot_delta[w];
                        uint64_t* t_row_ptr = tau + r * tau_stride;
                        for (int w = 0; w < tau_stride; ++w) t_row_ptr[w] ^= s_pivot_tau[w];
                    }
                }
            }
        }
        __syncthreads();
    }
}

} // namespace kernel

// =================================================================================
// Discrepancy Computation (GPU)
// =================================================================================

// Discrepancy (GPU to GPU Buffer)
void BasecaseSolver::compute_discrepancy_gpu_to_buffer(
    PackedBitPolyView F_view,
    const uint64_t* d_S,
    int t,
    uint64_t* d_Delta_out,
    cudaStream_t stream
) {
    size_t delta_stride_words = (config_.m_block + 63) / 64;
    size_t delta_bytes = dim_ * delta_stride_words * sizeof(uint64_t);
    CHECK_CUDA_BC(cudaMemsetAsync(d_Delta_out, 0, delta_bytes, stream));
    
    dim3 block(32, 16); 
    dim3 grid((config_.m_block + 31) / 32, (dim_ + 15) / 16);
    int words_n = (config_.n_block + 63) / 64;
    size_t smem_size = (16 + 32) * words_n * sizeof(uint64_t);
    size_t S_mat_stride_words = config_.m_block * (size_t)((config_.n_block + 63) / 64);
    
    kernel::k_compute_discrepancy<16, 32><<<grid, block, smem_size, stream>>>(
        F_view, d_S, d_Delta_out, t, config_.m_block, config_.n_block, S_mat_stride_words, delta_stride_words
    );
    CHECK_CUDA_BC(cudaGetLastError());
}

// =================================================================================
// Elimination (GPU)
// =================================================================================

void BasecaseSolver::compute_elimination_step_gpu(
    const uint64_t* d_Delta,
    const int* d_Gamma,
    uint64_t* d_Tau,
    uint64_t* d_ReducedDelta,
    cudaStream_t stream
) {
    size_t delta_stride = (config_.m_block + 63) / 64;
    size_t delta_bytes = dim_ * delta_stride * sizeof(uint64_t);
    CHECK_CUDA_BC(cudaMemcpyAsync(d_ReducedDelta, d_Delta, delta_bytes, cudaMemcpyDeviceToDevice, stream));
    
    size_t tau_stride = (dim_ + 63) / 64;
    size_t smem_size = 1024*8 + 1024*4 + 2048; 
    
    // Cache the occupancy query — result is constant (kernel, block size, smem never change)
    if (cached_elim_max_blocks_per_SM_ < 0) {
        CHECK_CUDA_BC(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &cached_elim_max_blocks_per_SM_,
                kernel::k_elimination,
                1024,
                smem_size
            )
        );
    }
    if(cached_elim_max_blocks_per_SM_ > 0){
        kernel::k_elimination<<<1, 1024, smem_size, stream>>>(
            d_ReducedDelta, d_Tau, d_Gamma, config_.m_block, dim_, delta_stride, tau_stride
        );
    }
    else{//LAUNCH REPLACEMENT KERNEL IF TOO MANY REGISTERS ARE REQUESTED
        kernel::k_eliminationN<<<1, 512, smem_size, stream>>>(
            d_ReducedDelta, d_Tau, d_Gamma, config_.m_block, dim_, delta_stride, tau_stride
        );
    }
    CHECK_CUDA_BC(cudaGetLastError());
}

// =================================================================================
// GPU Kernels: Apply Update (Slow/Deterministic)
// =================================================================================

namespace kernel {

__device__ __forceinline__ int get_lowest_set_bit_index(uint64_t w) {
    // __ffsll returns 1-based index of least significant bit set.
    // e.g., if w=1, returns 1. If w=2, returns 2.
    // We want 0-based index.
    return __ffsll((long long)w) - 1;
}

// Phase A: G_k = Tau * F_k done via MatMul

// Phase B: Scatter G_k[r] to F_next[k+shift[r]][r]
__global__ void k_scatter_update(
    const uint64_t* __restrict__ G_in,
    uint64_t* __restrict__ F_out,
    const int* __restrict__ shifts,
    int len_in,
    int dim,
    int stride_words
) {
    // We iterate over k (0..len_in-1), r (0..dim-1), w (0..stride-1)
    // Grid strategy: Linear ID over total words.
    size_t total_words = (size_t)len_in * dim * stride_words;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_words) return;
    
    // Decode tid -> (k, r, w)
    // Layout of G: [k][r][w]
    int w = tid % stride_words;
    size_t tmp = tid / stride_words;
    int r = tmp % dim;
    int k = tmp / dim;
    
    // Read G[k][r][w]
    uint64_t val = G_in[tid];
    
    // Determine destination
    int s = shifts[r];
    int dest_k = k + s;
    
    // Write F_next[dest_k][r][w]
    // F_next layout: [dest_k][r][w]
    // Stride matches.
    size_t out_idx = (size_t)dest_k * dim * stride_words + r * stride_words + w;
    
    // Non-atomic write: Each (k, r) maps to a unique (dest_k, r) because shifts[r] is constant for all k.
    F_out[out_idx] = val;
}

// Slow fallback kernel (Phase A+B merged) 
/**
 * @brief Computes F_next = Shift * Tau * F_in.
 * 
 * Grid Strategy: 1D linear over total output words.
 * Deterministic: No atomics, each thread writes one word.
 */
__global__ void k_apply_update_simple(
    const uint64_t* __restrict__ F_in,
    uint64_t* __restrict__ F_out,
    const uint64_t* __restrict__ tau,
    const int* __restrict__ shifts,
    int len_in,       // Input degree + 1
    int dim,          // Matrix dimension (rows/cols)
    int stride_words  // Words per row
) {
    // Total output words: (len_in + 1) * dim * stride
    // Output layout: [k][row][word]
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_words = (size_t)(len_in + 1) * dim * stride_words;
    
    if (tid >= total_words) return;
    
    // Map linear ID to (dest_k, r, w)
    // Layout is contiguous F matrices.
    // F[k] is dim x dim packed.
    int w = tid % stride_words;
    size_t tmp = tid / stride_words;
    int r = tmp % dim;
    int dest_k = tmp / dim;
    
    // Determine shift for this row
    int s = shifts[r];
    int src_k = dest_k - s;
    
    uint64_t acc = 0;
    
    if (src_k >= 0 && src_k < len_in) {
        // Pointers
        const uint64_t* F_src_k_base = F_in + (size_t)src_k * dim * stride_words;
        const uint64_t* tau_row = tau + (size_t)r * stride_words;
        
        // Sparse-dense matmul row: acc = sum(tau[r][l] * F[src_k][l][w])
        // Iterate words of tau row
        for (int tw_idx = 0; tw_idx < stride_words; ++tw_idx) {
            uint64_t tw = tau_row[tw_idx];
            while (tw != 0) {
                int bit = get_lowest_set_bit_index(tw);
                int l = tw_idx * 64 + bit; // Source row index
                
                // XOR source word
                // F layout: [k][row][word]
                // F[src_k][l][w] is at F_src_k_base + l*stride + w
                acc ^= F_src_k_base[(size_t)l * stride_words + w];
                
                // Clear bit
                tw &= (tw - 1);
            }
        }
    }
    
    F_out[tid] = acc;
}

} // namespace kernel

// =================================================================================
// GPU Kernels: Compute Shifts and Gamma Update
// =================================================================================

namespace kernel {
// Shift Selection Kernel
// Identifies Zero/Non-Zero rows and selects shifts.
__global__ void k_compute_shifts(
    const uint64_t* __restrict__ reduced_delta,
    const int* __restrict__ gamma,
    int* __restrict__ shifts,
    int m,
    int dim,
    size_t delta_stride
) {
    extern __shared__ uint64_t smem_shifts[];
    // Layout: 
    // s_keys[1024] (8KB)
    // s_is_zero[1024] (bool/int) - overlay with high part of array if needed?
    // We have plenty of shared mem.
    uint64_t* s_keys = smem_shifts;
    int* s_is_zero = (int*)&s_keys[1024]; // 4KB
    
    int tid = threadIdx.x;
    
    // 1. Check IsZero
    bool is_zero = true;
    if (tid < dim) {
        const uint64_t* row = reduced_delta + tid * delta_stride;
        for (int w = 0; w < delta_stride; ++w) {
            if (row[w] != 0) { is_zero = false; break; }
        }
        s_is_zero[tid] = is_zero ? 1 : 0;
        shifts[tid] = is_zero ? 0 : 1; // Initially mark non-zeros
    }
    __syncthreads();
    
    // 2. Count Non-Zeros (Warp Reduce)
    int is_nz = (tid < dim) ? (1 - s_is_zero[tid]) : 0;
    int sum = is_nz;
    for (int i = 16; i > 0; i >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
    
    __shared__ int s_warp_sums[32];
    int lane = tid % 32;
    int warp = tid / 32;
    if (lane == 0) s_warp_sums[warp] = sum;
    __syncthreads();
    
    __shared__ int num_nz;
    if (warp == 0) {
        sum = (tid < (blockDim.x/32)) ? s_warp_sums[tid] : 0;
        for (int i = 16; i > 0; i >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
        if (tid == 0) num_nz = sum;
    }
    __syncthreads();
    
    int total_nz = num_nz;
    
    // 3. Select Shifts
    if (total_nz <= m) {
        // We need m - total_nz zeros
        int rem = m - total_nz;
        
        // Prepare keys for zeros: (gamma << 32) | idx
        // For non-zeros: MAX (so they sort to end)
        if (tid < 1024) {
             if (tid < dim && s_is_zero[tid]) {
                 uint64_t g = (uint64_t)gamma[tid];
                 s_keys[tid] = (g << 32) | tid;
             } else {
                 s_keys[tid] = 0xFFFFFFFFFFFFFFFFULL;
             }
        }
        __syncthreads();
        
        // Bitonic Sort
        for (int k = 2; k <= 1024; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                if (tid < 1024) {
                     int ixj = tid ^ j;
                     if (ixj > tid) {
                         uint64_t ki = s_keys[tid];
                         uint64_t kj = s_keys[ixj];
                         bool swap = (tid & k) == 0 ? (ki > kj) : (ki < kj);
                         if (swap) {
                             s_keys[tid] = kj;
                             s_keys[ixj] = ki;
                         }
                     }
                }
                __syncthreads();
            }
        }
        
        // First 'rem' elements are the zeros we want
        if (tid < rem) {
            uint64_t key = s_keys[tid];
            int idx = (int)(key & 0xFFFFFFFF);
            if (idx < dim) shifts[idx] = 1; 
        }
    } else {
        // total_nz > m
        // We marked too many (all non-zeros).
        // Unmark those beyond first m.
        // Requires prefix sum or serial scan.
        if (tid == 0) {
            int cnt = 0;
            for (int i = 0; i < dim; ++i) {
                if (shifts[i]) {
                    if (cnt < m) {
                        cnt++;
                    } else {
                        shifts[i] = 0;
                    }
                }
            }
        }
    }
}

__global__ void k_compute_shiftsN(
    const uint64_t* __restrict__ reduced_delta,
    const int* __restrict__ gamma,
    int* __restrict__ shifts,
    int m,
    int dim,
    size_t delta_stride
) {
    extern __shared__ uint64_t smem_shifts[];
    // Layout:
    // s_keys[1024] (8KB)
    // s_is_zero[1024] (bool/int) - overlay with high part of array if needed?
    // We have plenty of shared mem.
    uint64_t* s_keys = smem_shifts;
    int* s_is_zero = (int*)&s_keys[1024]; // 4KB

    int tid = threadIdx.x;
    int N = blockDim.x;  // 512 when launched as fallback (was hardcoded 1024)

    // 1. Check IsZero
    bool is_zero = true;
    if (tid < dim) {
        const uint64_t* row = reduced_delta + tid * delta_stride;
        for (int w = 0; w < delta_stride; ++w) {
            if (row[w] != 0) { is_zero = false; break; }
        }
        s_is_zero[tid] = is_zero ? 1 : 0;
        shifts[tid] = is_zero ? 0 : 1; // Initially mark non-zeros
    }
    __syncthreads();

    // 2. Count Non-Zeros (Warp Reduce)
    int is_nz = (tid < dim) ? (1 - s_is_zero[tid]) : 0;
    int sum = is_nz;
    for (int i = 16; i > 0; i >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, i);

    __shared__ int s_warp_sums[32];
    int lane = tid % 32;
    int warp = tid / 32;
    if (lane == 0) s_warp_sums[warp] = sum;
    __syncthreads();

    __shared__ int num_nz;
    if (warp == 0) {
        sum = (tid < (blockDim.x/32)) ? s_warp_sums[tid] : 0;
        for (int i = 16; i > 0; i >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
        if (tid == 0) num_nz = sum;
    }
    __syncthreads();

    int total_nz = num_nz;

    // 3. Select Shifts
    if (total_nz <= m) {
        // We need m - total_nz zeros
        int rem = m - total_nz;

        // Prepare keys for zeros: (gamma << 32) | idx
        // For non-zeros: MAX (so they sort to end)
        if (tid < N) {
             if (tid < dim && s_is_zero[tid]) {
                 uint64_t g = (uint64_t)gamma[tid];
                 s_keys[tid] = (g << 32) | tid;
             } else {
                 s_keys[tid] = 0xFFFFFFFFFFFFFFFFULL;
             }
        }
        __syncthreads();

        // Bitonic Sort
        for (int k = 2; k <= N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                if (tid < N) {
                     int ixj = tid ^ j;
                     if (ixj > tid) {
                         uint64_t ki = s_keys[tid];
                         uint64_t kj = s_keys[ixj];
                         bool swap = (tid & k) == 0 ? (ki > kj) : (ki < kj);
                         if (swap) {
                             s_keys[tid] = kj;
                             s_keys[ixj] = ki;
                         }
                     }
                }
                __syncthreads();
            }
        }

        // First 'rem' elements are the zeros we want
        if (tid < rem) {
            uint64_t key = s_keys[tid];
            int idx = (int)(key & 0xFFFFFFFF);
            if (idx < dim) shifts[idx] = 1;
        }
    } else {
        // total_nz > m
        // We marked too many (all non-zeros).
        // Unmark those beyond first m.
        // Requires prefix sum or serial scan.
        if (tid == 0) {
            int cnt = 0;
            for (int i = 0; i < dim; ++i) {
                if (shifts[i]) {
                    if (cnt < m) {
                        cnt++;
                    } else {
                        shifts[i] = 0;
                    }
                }
            }
        }
    }
}

// Gamma Update Kernel
__global__ void k_update_gamma(
    int* gamma,
    const int* shifts,
    int dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dim) {
        if (shifts[tid]) gamma[tid]++;
    }
}

} // namespace kernel

// =================================================================================
// Compute Shifts and Gamma Update (GPU)
// =================================================================================

void BasecaseSolver::compute_shift_vector_gpu(
    const uint64_t* d_ReducedDelta,
    const int* d_Gamma,
    int* d_Shifts,
    cudaStream_t stream
) {
    size_t delta_stride = (config_.m_block + 63) / 64;
    size_t smem_size = 1024*8 + 1024*4; // Keys + IsZero (same for both variants)

    // Cache the occupancy query — result is constant (kernel, block size, smem never change)
    if (cached_shifts_max_blocks_per_SM_ < 0) {
        CHECK_CUDA_BC(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &cached_shifts_max_blocks_per_SM_,
                kernel::k_compute_shifts,
                1024,
                smem_size
            )
        );
    }
    if (cached_shifts_max_blocks_per_SM_ > 0) {
        kernel::k_compute_shifts<<<1, 1024, smem_size, stream>>>(
            d_ReducedDelta, d_Gamma, d_Shifts, config_.m_block, dim_, delta_stride
        );
    } else {
        LOG(LOG_DEBUG_1) << "[Basecase] k_compute_shifts@1024 has zero occupancy; using 512-thread fallback";
        kernel::k_compute_shiftsN<<<1, 512, smem_size, stream>>>(
            d_ReducedDelta, d_Gamma, d_Shifts, config_.m_block, dim_, delta_stride
        );
    }
    CHECK_CUDA_BC(cudaGetLastError());
}

void BasecaseSolver::update_gamma_gpu(
    int* d_Gamma,
    const int* d_Shifts,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (dim_ + threads - 1) / threads;
    kernel::k_update_gamma<<<blocks, threads, 0, stream>>>(d_Gamma, d_Shifts, dim_);
    CHECK_CUDA_BC(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Helper: Unpack Flat Poly to Vector<Vector>
// -----------------------------------------------------------------------------
static std::vector<std::vector<uint64_t>> unpack_flat_poly_to_vector(
    const std::vector<uint64_t>& flat,
    int len,
    int dim
) {
    int stride_words = (dim + 63) / 64;
    size_t mat_size = (size_t)dim * stride_words;
    
    std::vector<std::vector<uint64_t>> poly(len);
    
    for(int k=0; k<len; ++k) {
        size_t offset = k * mat_size;
        // The last matrix might be partially filled if trailing zero trimming happens externally,
        // but here we assume 'flat' corresponds exactly to len * mat_size.
        if (offset + mat_size <= flat.size()) {
            poly[k].assign(flat.begin() + offset, flat.begin() + offset + mat_size);
        } else {
            poly[k].assign(mat_size, 0); // Padding if missing
        }
    }
    return poly;
}

// -----------------------------------------------------------------------------
// Core Solver Logic
// -----------------------------------------------------------------------------

void BasecaseSolver::solve(const uint64_t* d_S, cudaStream_t stream) {
    int m = config_.m_block;  
    int n = config_.n_block;  
    int dim = dim_; // m + n
    int len = config_.seq_len;
    
    LOG(LOG_STATS) << "[Basecase] Solver initializing with m=" << m << ", n=" << n << ", dim=" << dim;
    if(config_.internal_oracle_verification)
        LOG(LOG_STATS) << "[Basecase] [OracleCheck] Solver Oracle Check is enabled.";
    else
        LOG(LOG_DEBUG_2) << "[Basecase] [OracleCheck] Solver Oracle Check is disabled.";
    
    // Validate inputs against internal assumptions
    if (len <= 0) {
        LOG(LOG_ERROR_CRITICAL) << "[Basecase] Invalid sequence length: " << len;
        return;
    }

    LOG(LOG_DEBUG_1) << "[Basecase] Downloading sequence of length " << len << " from GPU...";
    
    // Download sequence
    // S_k is m x n. Stride is determined by n.
    size_t words_per_row = (n + 63) / 64;
    size_t words_per_mat = m * words_per_row;
    std::vector<std::vector<uint64_t>> S_host(len);
    std::vector<uint64_t> temp_buf(words_per_mat);

    // Download loop
    for (int k = 0; k < len; ++k) {
        const uint64_t* src_ptr = d_S + k * words_per_mat;
        cudaMemcpyAsync(temp_buf.data(), src_ptr, words_per_mat * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); 
        S_host[k] = temp_buf;
    }

    /* 
    // 1. Allocate a single large buffer on the host (preferably pinned)
    std::vector<uint64_t> large_host_buf(len * words_per_mat);

    // 2. Transfer everything in one call
    CHECK_CUDA_BC(cudaMemcpyAsync(large_host_buf.data(), d_S, 
				  len * words_per_mat * sizeof(uint64_t), 
				  cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_BC(cudaStreamSynchronize(stream));

    // 3. Distribute to S_host structure (pure CPU operation)
    for (int k = 0; k < len; ++k) {
        std::copy(large_host_buf.begin() + k * words_per_mat, 
                  large_host_buf.begin() + (k + 1) * words_per_mat, 
                  S_host[k].begin());
    }*/
    
    LOG(LOG_DEBUG_2) << "[Basecase] Initializing basis, f...";
    auto init_generic = find_initialization_basis(S_host);
    auto finit_generic = build_f_init(init_generic);

    std::vector<std::vector<uint64_t>> F_poly = finit_generic.F_poly;
    gamma_ = finit_generic.gamma;
    int t0 = init_generic.t0;

    LOG(LOG_STATS) << "[Basecase] Initialization: t0=" << t0 << ", Basis rank=" << m;
    
    // --- Hybrid GPU Setup ---
    // Double-buffer F and preallocated G for sync-free hot loop.
    // Eliminates all cudaMalloc/cudaFree/cudaMemset sync points from the iteration.
    uint64_t* d_F_buf[2] = {nullptr, nullptr};
    uint64_t* d_G = nullptr;
    int cur_buf = 0;
    int F_len = 0;
    int stride_F = 0;
    size_t mat_stride = 0;

    // Persistent Buffers for Hybrid Mode
    uint64_t *d_Delta = nullptr, *d_Tau = nullptr, *d_ReducedDelta = nullptr;
    int *d_Gamma = nullptr, *d_Shifts = nullptr;

    if (config_.gpu_mode) {

        LOG(LOG_STATS) << "[Basecase] Initializing GPU Polymer...";

        F_len = (int)F_poly.size();
        stride_F = (dim + 63) / 64;
        mat_stride = (size_t)dim * stride_F;  // words per coefficient matrix

        // Preallocate F as a double buffer at maximum size.
        // F grows by 1 coefficient per iteration; upper bound is initial + iterations + 1.
        int max_F_len = (len - t0) + F_len + 1;
        size_t max_F_bytes = (size_t)max_F_len * mat_stride * sizeof(uint64_t);

        CHECK_CUDA_BC(cudaMalloc(&d_F_buf[0], max_F_bytes));
        CHECK_CUDA_BC(cudaMalloc(&d_F_buf[1], max_F_bytes));
        CHECK_CUDA_BC(cudaMemset(d_F_buf[0], 0, max_F_bytes));
        CHECK_CUDA_BC(cudaMemset(d_F_buf[1], 0, max_F_bytes));

        // Upload initial F into buffer 0
        std::vector<uint64_t> flat_host;
        for(const auto& mat : F_poly) flat_host.insert(flat_host.end(), mat.begin(), mat.end());
        CHECK_CUDA_BC(cudaMemcpyAsync(d_F_buf[0], flat_host.data(),
            flat_host.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

        // Preallocate G buffer (reused every iteration for Tau * F product)
        size_t max_G_bytes = (size_t)(max_F_len - 1) * mat_stride * sizeof(uint64_t);
        CHECK_CUDA_BC(cudaMalloc(&d_G, max_G_bytes));

        // Allocate GPU buffers for Steps 1 & 2
	size_t delta_bytes = dim * ((m + 63)/64) * 8;
	size_t tau_bytes   = dim * ((dim + 63)/64) * 8;
        CHECK_CUDA_BC(cudaMalloc(&d_Delta, delta_bytes));
        CHECK_CUDA_BC(cudaMalloc(&d_Tau, tau_bytes));
        CHECK_CUDA_BC(cudaMalloc(&d_ReducedDelta, delta_bytes));
        CHECK_CUDA_BC(cudaMalloc(&d_Gamma, dim * sizeof(int)));
        CHECK_CUDA_BC(cudaMalloc(&d_Shifts, dim * sizeof(int)));

        // Init Gamma on Device
        CHECK_CUDA_BC(cudaMemcpyAsync(d_Gamma, gamma_.data(), dim * sizeof(int), cudaMemcpyHostToDevice, stream));

        LOG(LOG_DEBUG_1) << "[Basecase] Preallocated double-buffer F ("
                         << (2 * max_F_bytes / 1024) << " KiB) + G ("
                         << (max_G_bytes / 1024) << " KiB)";

        if (config_.use_cuda_graph) {
            LOG(LOG_DEBUG_1) << "[Basecase] CUDA graph requested but not applicable to Stage 2 "
                             << "(variable kernel configs per iteration). Running standard async path.";
        }
    }

    // =========================================================================
    // CUDA Graph Analysis (2026-03-27)
    //
    // This loop is NOT amenable to CUDA graph capture:
    //  - F_len grows by 1 each iteration → matmul batch count, memset size,
    //    and scatter grid dims all change.
    //  - Discrepancy kernel reads a growing polynomial (F_view changes).
    //  - Only Steps 2b (Elimination) and 3 (Shifts) have fixed configs,
    //    but they're single-block persistent kernels with negligible launch
    //    overhead — graphing 2 of 9 operations provides no measurable benefit.
    //  - The loop is already fully async (zero mandatory sync points) thanks
    //    to double-buffer preallocation of F, G, Delta, Tau, etc.
    //
    // BWStage2Config::use_cuda_graph is accepted but has no effect.
    // See docs/cuda_graph_stage2_plan.md for full analysis.
    // =========================================================================

    // --- Iterative BBM ---
    int L = len;

    for (int t = t0; t < L; ++t) {
        StepResult step_res_host; // Only used if CPU path or Oracle Check
        std::vector<uint64_t> delta_ref; 
        std::vector<bool> will_shift_ref;      

        if (config_.gpu_mode) {
            // 1. Discrepancy (GPU)
	    PackedBitPolyView F_view(d_F_buf[cur_buf], F_len, dim, dim);
	    compute_discrepancy_gpu_to_buffer(F_view, d_S, t, d_Delta, stream);
            
            if (config_.internal_oracle_verification) {
     
                // Validate Delta
                delta_ref = compute_discrepancy(F_poly, S_host, t);
                std::vector<uint64_t> delta_gpu(dim * ((m+63)/64));
                cudaMemcpy(delta_gpu.data(), d_Delta, delta_gpu.size()*8, cudaMemcpyDeviceToHost);
                if (std::memcmp(delta_gpu.data(), delta_ref.data(), delta_gpu.size()*8) != 0) {
                    LOG(LOG_ERROR_CRITICAL) << "[Basecase] [OracleCheck] Delta at t=" << t << " MISMATCH!";
                    throw std::runtime_error("Basecase OracleCheck: Delta mismatch");
                } else {
		    LOG(LOG_DEBUG_3) << "[Basecase] [OracleCheck] Delta at t=" << t << " MATCH.";
		}
		
            }

            // 2. Elimination (GPU)
            compute_elimination_step_gpu(d_Delta, d_Gamma, d_Tau, d_ReducedDelta, stream);

            if (config_.internal_oracle_verification) {
                // Validate Elimination
                // We use delta_ref computed above, or download d_Delta if logic differs
                if (delta_ref.empty()) delta_ref = compute_discrepancy(F_poly, S_host, t); // should match
                auto step_ref = compute_elimination_step(delta_ref);

                std::vector<uint64_t> tau_gpu(dim * ((dim+63)/64));
                std::vector<uint64_t> red_gpu(dim * ((m+63)/64));
                CHECK_CUDA_BC(cudaMemcpy(tau_gpu.data(), d_Tau, tau_gpu.size()*8, cudaMemcpyDeviceToHost));
                CHECK_CUDA_BC(cudaMemcpy(red_gpu.data(), d_ReducedDelta, red_gpu.size()*8, cudaMemcpyDeviceToHost));	
                
                if (std::memcmp(tau_gpu.data(), step_ref.tau.data(), tau_gpu.size()*8) != 0) {
                    LOG(LOG_ERROR_CRITICAL) << "[Basecase] [OracleCheck] Tau at t=" << t << " MISMATCH!";		    
                    LOG(LOG_ERROR_CRITICAL) << "[Basecase] [OracleCheck] Tau size = " << tau_gpu.size();
                    LOG(LOG_ERROR_CRITICAL) << "[Basecase] [OracleCheck] Printing first 10 differences:";
		    int diff_count = 0;
		    for(size_t k = 0; (k < tau_gpu.size()) && (diff_count < 10); ++k) {
		        if(tau_gpu[k] != step_ref.tau[k]) {
			    LOG(LOG_ERROR_CRITICAL) << "[Basecase] [OracleCheck] Tau mismatch at position = " << k << ": " << std::hex << tau_gpu[k] << " GPU vs " << std::hex << step_ref.tau[k] << std::dec << " CPU oracle";
			    diff_count++;
		        }
		    }
		    throw std::runtime_error("Basecase OracleCheck: Tau mismatch");
                } else {
		    LOG(LOG_DEBUG_3) << "[Basecase] [OracleCheck] Tau at t=" << t << " MATCH.";
		}
                if (std::memcmp(red_gpu.data(), step_ref.reduced_delta.data(), red_gpu.size()*8) != 0) {
                    LOG(LOG_ERROR_CRITICAL) << "[Basecase] [OracleCheck] ReducedDelta at t=" << t << " MISMATCH!";
                    throw std::runtime_error("Basecase OracleCheck: ReducedDelta mismatch");
                } else {
		    LOG(LOG_DEBUG_3) << "[Basecase] [OracleCheck] ReducedDelta at t=" << t << " MATCH.";
		}
                // Prepare CPU data for next step checks
                step_res_host = step_ref;		
            }   

	    // 3. Shift Selection (GPU)
            compute_shift_vector_gpu(d_ReducedDelta, d_Gamma, d_Shifts, stream);
	    
            if (config_.internal_oracle_verification) {
                will_shift_ref = compute_shift_vector(step_res_host.reduced_delta);
                std::vector<int> shifts_gpu(dim);
                cudaMemcpy(shifts_gpu.data(), d_Shifts, dim*4, cudaMemcpyDeviceToHost);
                
                bool match = true;
                for(int i=0; i<dim; ++i) {
                    if ((shifts_gpu[i] != 0) != will_shift_ref[i]) match = false;
                }
                
                if (!match) {
                    LOG(LOG_ERROR_CRITICAL) << "[Basecase] [OracleCheck] Willshift at t=" << t << " MISMATCH!";
                    throw std::runtime_error("Basecase OracleCheck: Willshift mismatch");
                } else {
                    LOG(LOG_DEBUG_3) << "[Basecase] [OracleCheck] Willshift at t=" << t << " MATCH.";
                }
            }

            // 4. Update F (GPU) — double-buffer, zero malloc/free in loop
            {
            int len_in = F_len;
            int len_out = F_len + 1;
            int next_buf = 1 - cur_buf;

            // Zero the destination region (async on stream, no implicit sync)
            size_t out_bytes = (size_t)len_out * mat_stride * sizeof(uint64_t);
            CHECK_CUDA_BC(cudaMemsetAsync(d_F_buf[next_buf], 0, out_bytes, stream));

            // G = Tau * F[cur_buf]
            matmul_gf2_gpu_broadcast_A(dim, d_Tau, d_F_buf[cur_buf], d_G, len_in, stream);

            // Scatter G → F[next_buf] with per-row shifts
            size_t total_items = (size_t)len_in * dim * stride_F;
            int threads = 256;
            int blocks = (total_items + threads - 1) / threads;
            kernel::k_scatter_update<<<blocks, threads, 0, stream>>>(
                 d_G, d_F_buf[next_buf], d_Shifts, len_in, dim, stride_F
            );

            // Swap: next becomes current, F grows by one coefficient
            cur_buf = next_buf;
            F_len = len_out;
            }
             
            if (config_.internal_oracle_verification) {
                  size_t download_words = (size_t)F_len * mat_stride;
                  std::vector<uint64_t> flat_out(download_words);
                  CHECK_CUDA_BC(cudaMemcpyAsync(flat_out.data(), d_F_buf[cur_buf],
                      download_words * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
                  CHECK_CUDA_BC(cudaStreamSynchronize(stream));
                  auto F_poly_gpu = unpack_flat_poly_to_vector(flat_out, F_len, dim);
                  while(F_poly_gpu.size() > 1) {
                        bool all_zero = true;
                        const auto& last = F_poly_gpu.back();
                        for(auto w : last) if(w!=0) { all_zero = false; break; }
                        if(all_zero) F_poly_gpu.pop_back(); 
                        else break;
                  }
                  
                  auto F_poly_ref = apply_update(F_poly, step_res_host.tau, will_shift_ref);
                  if (F_poly_gpu.size() != F_poly_ref.size()) {
		      LOG(LOG_ERROR_CRITICAL) << "[Basecase] [OracleCheck] Update at t=" << t << " MISMATCH!";
		      throw std::runtime_error("Basecase OracleCheck: Update mismatch");
		  } else {
		     LOG(LOG_DEBUG_3) << "[Basecase] [OracleCheck] Update at t=" << t << " MATCH.";
		  }		  
                  F_poly = F_poly_gpu;
             } else {
                  F_poly.resize(F_len);
             }
             
             // 5. Update Gamma (GPU)
             update_gamma_gpu(d_Gamma, d_Shifts, stream);
             
             // Sync Gamma to Host if needed for check
             if (config_.internal_oracle_verification || config_.reference_check) {
                 CHECK_CUDA_BC(cudaMemcpyAsync(gamma_.data(), d_Gamma, dim * sizeof(int), cudaMemcpyDeviceToHost, stream));
                 CHECK_CUDA_BC(cudaStreamSynchronize(stream));
             }
#ifdef BASECASE_SOLVER_CUDA_STREAM_SYNC
	     else if(!(t % 10)) {
	         // We periodically sync the stream to limit total number
	         // of sequential kernel launches within this loop
	         CHECK_CUDA_BC(cudaStreamSynchronize(stream));
	     }
#endif
        } else {
            // CPU ONLY Path
            // 1. Discrepancy (CPU)
            auto delta = compute_discrepancy(F_poly, S_host, t);
            // 2. Elimination (CPU)
            step_res_host = compute_elimination_step(delta);
	    // 3. Shift Selection (CPU)
	    will_shift_ref = compute_shift_vector(step_res_host.reduced_delta);
	    // 4. Apply Update to F (CPU)
	    F_poly = apply_update(F_poly, step_res_host.tau, will_shift_ref);
	    // 5. Update Gamma
	    for(int r=0; r<dim; ++r) {
	        if(will_shift_ref[r]) gamma_[r]++;
	    }
        }


        if (config_.reference_check || (t % config_.reference_check_interval == 0 || t == t0)) {
            if (config_.reference_check) {
                // gamma_ is synced above if GPU mode
	        uint64_t state_h = compute_state_hash();
                LOG(LOG_DEBUG_1) << "[Basecase] t=" << t << ", Deg=" << (F_poly.size()-1)
                                 << ", Hash=" << std::hex << state_h << std::dec;
            } else if (t % 1000 == 0) {
                LOG(LOG_DEBUG_1) << "[Basecase] t=" << t << ", Deg=" << (F_poly.size()-1) << "...";
            }
        }
        LOG_IF(LOG_STATS, t % 100 == 0)
            << "[Basecase] t=" << t << ", Deg=" << (F_poly.size()-1) << "...";
    }

    LOG(LOG_INFO) << "[Basecase] Solver finished. Final Degree=" << (F_poly.size()-1);

    if (config_.gpu_mode) {
        LOG(LOG_DEBUG_1) << "[Basecase] Downloading final result from GPU...";
        size_t download_words = (size_t)F_len * mat_stride;
        std::vector<uint64_t> flat_out(download_words);
        CHECK_CUDA_BC(cudaMemcpyAsync(flat_out.data(), d_F_buf[cur_buf],
            download_words * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

        // Sync final Gamma for hash
        CHECK_CUDA_BC(cudaMemcpyAsync(gamma_.data(), d_Gamma, dim * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_BC(cudaStreamSynchronize(stream));

        F_poly = unpack_flat_poly_to_vector(flat_out, F_len, dim);
        while(F_poly.size() > 1) {
             bool all_zero = true;
             const auto& last = F_poly.back();
             for(auto w : last)
	         if(w != 0) {
		     all_zero = false; break;
		 }
             if(all_zero) F_poly.pop_back();
             else break;
        }

        // Clean up — double buffers, G, and persistent buffers
        CHECK_CUDA_BC(cudaFree(d_F_buf[0]));
        CHECK_CUDA_BC(cudaFree(d_F_buf[1]));
        CHECK_CUDA_BC(cudaFree(d_G));
        CHECK_CUDA_BC(cudaFree(d_Delta));
        CHECK_CUDA_BC(cudaFree(d_Tau));
        CHECK_CUDA_BC(cudaFree(d_ReducedDelta));
        CHECK_CUDA_BC(cudaFree(d_Gamma));
        CHECK_CUDA_BC(cudaFree(d_Shifts));
    }    
    
    // Store result
    pi_data_.clear();
    for(const auto& mat : F_poly) {
        pi_data_.insert(pi_data_.end(), mat.begin(), mat.end());
    }
    
    uint64_t state_hash = compute_state_hash();
    LOG(LOG_DEBUG_1) << "[Basecase] Final State Hash (Pi|Gamma): " << std::hex << state_hash << std::dec;

    // --- Verification ---
    if (config_.check_annihilation_gpu) {
        if (!check_annihilation(d_S, len, stream)) {
            LOG(LOG_ERROR_CRITICAL) << "[Basecase] GPU Annihilation Verification Failed!";
        } else {
            LOG(LOG_DEBUG_1) << "[Basecase] GPU Annihilation Verification PASSED.";
	}
    } else {
        LOG(LOG_DEBUG_1) << "[Basecase] GPU Annihilation Verification SKIPPED.";
    }
    if (config_.check_annihilation_legacy) {
        if (!check_annihilation_legacy(S_host, F_poly)) {
             LOG(LOG_ERROR_CRITICAL) << "[Basecase] Legacy Annihilation Verification Failed!";
        } else {
	    LOG(LOG_INFO) << "[Basecase] Legacy Annihilation Verification PASSED.";
	}
    }
}

} // namespace stage2
} // namespace lingen
