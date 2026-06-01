// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "preprocessing.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "hpc_logger.h"

inline int calculate_optimal_k(const std::vector<idx_t>& gaps) {
    if (gaps.empty()) return 0;
    double sum = 0;
    for (auto g : gaps) sum += g;
    double mean = sum / gaps.size();
    if (mean <= 1.0) return 0;
    return std::max(0, (int)std::floor(std::log2(mean)));
}

void MatrixPreprocessor::generate_tiled_coo(
    const HostMatrixView& raw_matrix,
    const std::vector<row_idx_t>& permutation,
    row_idx_t start_row,
    row_idx_t end_row,
    idx_t dense_col_limit,
    std::vector<uint32_t>& out_coords,
    std::vector<uint32_t>& out_ptrs,
    uint32_t block_size 
) {
    out_coords.clear();
    out_ptrs.clear();
    row_idx_t num_rows = end_row - start_row;
    row_idx_t num_blocks = (num_rows + block_size - 1) / block_size;
    out_ptrs.push_back(0);

    for (row_idx_t b = 0; b < num_blocks; ++b) {
        row_idx_t block_start = start_row + b * block_size;
        row_idx_t block_end = std::min(block_start + block_size, end_row);
        std::vector<uint32_t> block_coords;
        block_coords.reserve((block_end - block_start) * 16); 

        for (row_idx_t r = block_start; r < block_end; ++r) {
            row_idx_t sorted_r = permutation[r];
            uint32_t row_offset = r - block_start; 
            for (idx_t col : raw_matrix.get_row(sorted_r)) {
                if (col >= dense_col_limit) {
                    uint32_t packed = (row_offset << 22) | (col & 0x3FFFFF);
                    block_coords.push_back(packed);
                }
            }
        }
        std::sort(block_coords.begin(), block_coords.end(), [](uint32_t a, uint32_t b) {
            return (a & 0x3FFFFF) < (b & 0x3FFFFF);
        });
        out_coords.insert(out_coords.end(), block_coords.begin(), block_coords.end());
        out_ptrs.push_back((uint32_t)out_coords.size());
    }
}

CompressedMatrix MatrixPreprocessor::preprocess(
    const HostMatrixView& raw_matrix, 
    idx_t dense_col_limit,
    row_idx_t heavy_row_limit,
    bool use_ellpack,
    float pfor_threshold,
    uint32_t tiled_row_block_size,
    bool sort_rows // New Parameter
) {
    CompressedMatrix result;
    result.use_ellpack = use_ellpack;
    result.n_dense_cols = dense_col_limit;
    result.n_sparse_cols = raw_matrix.n_cols - dense_col_limit;
    result.total_nnz = 0;
    result.n_heavy_rows = std::min(heavy_row_limit, raw_matrix.n_rows);
    result.tiled_row_block_size = tiled_row_block_size;

    std::vector<row_idx_t> p(raw_matrix.n_rows);
    std::iota(p.begin(), p.end(), 0);
    
    // Only sort if requested
    if (sort_rows) {
        std::sort(p.begin(), p.end(), [&](row_idx_t a, row_idx_t b) {
	    return raw_matrix.get_row(a).size() > raw_matrix.get_row(b).size();
        });
    }
    result.permutation = p;

    // 1. Buffers & Dense/Heavy
    size_t ints_per_row = (dense_col_limit + 31) / 32;
    result.dense_values.resize(raw_matrix.n_rows * ints_per_row, 0);
    size_t sparse_width_ints = (result.n_sparse_cols + 31) / 32;
    
    if (result.n_heavy_rows > 0) {
        result.heavy_bitmaps.resize(result.n_heavy_rows * sparse_width_ints, 0);
        result.heavy_csr_offsets.push_back(0);
        generate_tiled_coo(raw_matrix, p, 0, result.n_heavy_rows, dense_col_limit, 
                          result.heavy_tiled_coords, result.heavy_tiled_ptr, tiled_row_block_size);
    }

    for (row_idx_t r = 0; r < raw_matrix.n_rows; ++r) {
        row_idx_t sorted_r = p[r];
        const auto& row_indices = raw_matrix.get_row(sorted_r);
        if (r < result.n_heavy_rows) {
            for (idx_t col : row_indices) {
                if (col < dense_col_limit) {
                    result.dense_values[r * ints_per_row + col/32] |= (1U << (col%32));
                } else {
                    idx_t rel_col = col - dense_col_limit;
                    result.heavy_bitmaps[r * sparse_width_ints + rel_col/32] |= (1U << (rel_col%32));
                    result.heavy_csr_indices.push_back(col);
                    result.total_nnz++;
                }
            }
            if (r < result.n_heavy_rows) result.heavy_csr_offsets.push_back((uint32_t)result.heavy_csr_indices.size());
        } else {
            for (idx_t col : row_indices) {
                if (col < dense_col_limit) {
                    result.dense_values[r * ints_per_row + col/32] |= (1U << (col%32));
                }
            }
        }
    }

    // 2. Sparse Part 
    row_idx_t first_sparse_row = result.n_heavy_rows;
    row_idx_t n_sparse = raw_matrix.n_rows - first_sparse_row;
    
    if (n_sparse > 0) {
        // A. Tiled Sparse
        generate_tiled_coo(raw_matrix, p, first_sparse_row, raw_matrix.n_rows, dense_col_limit,
                          result.sparse_tiled_coords, result.sparse_tiled_ptr, tiled_row_block_size);

        // B. Delta-16
        result.delta_16_offsets.reserve(raw_matrix.n_rows + 1);
        for(row_idx_t r=0; r<first_sparse_row; ++r) result.delta_16_offsets.push_back(0);
        for (row_idx_t r = first_sparse_row; r < raw_matrix.n_rows; ++r) {
            result.delta_16_offsets.push_back((uint32_t)result.delta_16_stream.size());
            row_idx_t sorted_r = p[r];
            idx_t prev_col = dense_col_limit - 1;
            for (idx_t col : raw_matrix.get_row(sorted_r)) {
                if (col < dense_col_limit) continue;
                uint64_t diff = col - prev_col;
                while (diff > 0xFFFF) {
                    result.delta_16_stream.push_back(0xFFFF);
                    result.delta_16_stream.push_back(0);
                    diff -= 0xFFFF;
                }
                result.delta_16_stream.push_back((uint16_t)diff);
                prev_col = col;
            }
        }
        result.delta_16_offsets.push_back((uint32_t)result.delta_16_stream.size());

        // C. Byte-Aligned PForDelta
        result.pfor_row_block_starts.reserve(raw_matrix.n_rows + 1);
        for(row_idx_t r=0; r<first_sparse_row; ++r) result.pfor_row_block_starts.push_back(0);
        int block_size = 128; 
        for (row_idx_t r = first_sparse_row; r < raw_matrix.n_rows; ++r) {
            result.pfor_row_block_starts.push_back((uint32_t)result.pfor_meta.size());
            row_idx_t sorted_r = p[r];
            std::vector<idx_t> deltas;
            deltas.reserve(raw_matrix.get_row(sorted_r).size());
            idx_t prev_col = dense_col_limit - 1;
            for (idx_t col : raw_matrix.get_row(sorted_r)) {
                if (col < dense_col_limit) continue;
                deltas.push_back(col - prev_col);
                prev_col = col;
            }
            size_t num_items = deltas.size();
            for (size_t i = 0; i < num_items; i += block_size) {
                size_t count = std::min((size_t)block_size, num_items - i);
                idx_t max_d = 0;
                for (size_t j = 0; j < count; ++j) if (deltas[i + j] > max_d) max_d = deltas[i + j];
                int type = 0; 
                size_t bytes_per_item = 1;
                if (max_d > 65535) { type = 2; bytes_per_item = 4; }
                else if (max_d > 255) { type = 1; bytes_per_item = 2; }
                
                size_t current_offset = result.pfor_data.size();
                size_t padding = 0;
                if (bytes_per_item > 1) {
                    size_t align_mask = bytes_per_item - 1;
                    if (current_offset & align_mask) {
                        padding = bytes_per_item - (current_offset & align_mask);
                        for(size_t p=0; p<padding; ++p) result.pfor_data.push_back(0);
                    }
                }
                
                uint16_t meta = (uint16_t)((count << 8) | (type & 0x3));
                result.pfor_meta.push_back(meta);
                result.pfor_block_byte_offsets.push_back((uint32_t)result.pfor_data.size());
                for (size_t j = 0; j < count; ++j) {
                    idx_t val = deltas[i + j];
                    if (type == 0) result.pfor_data.push_back((uint8_t)val);
                    else if (type == 1) { uint16_t v=(uint16_t)val; uint8_t* p=(uint8_t*)&v; result.pfor_data.push_back(p[0]); result.pfor_data.push_back(p[1]); }
                    else { uint8_t* p=(uint8_t*)&val; result.pfor_data.push_back(p[0]); result.pfor_data.push_back(p[1]); result.pfor_data.push_back(p[2]); result.pfor_data.push_back(p[3]); }
                }
            }
        }
        result.pfor_row_block_starts.push_back((uint32_t)result.pfor_meta.size());

        // D. Bit-Exact PForDelta
        result.pfor_be_row_block_starts.reserve(raw_matrix.n_rows + 1);
        for(row_idx_t r=0; r<first_sparse_row; ++r) result.pfor_be_row_block_starts.push_back(0);
        
        for (row_idx_t r = first_sparse_row; r < raw_matrix.n_rows; ++r) {
            result.pfor_be_row_block_starts.push_back((uint32_t)result.pfor_be_meta_bits.size());
            row_idx_t sorted_r = p[r];
            std::vector<idx_t> deltas;
            idx_t prev_col = dense_col_limit - 1;
            for (idx_t col : raw_matrix.get_row(sorted_r)) {
                if (col < dense_col_limit) continue;
                deltas.push_back(col - prev_col);
                prev_col = col;
            }

            size_t num_items = deltas.size();
            for (size_t i = 0; i < num_items; i += block_size) {
                size_t count = std::min((size_t)block_size, num_items - i);
                
                std::vector<idx_t> block_deltas;
                block_deltas.reserve(count);
                for(size_t j=0; j<count; ++j) block_deltas.push_back(deltas[i+j]);
                std::vector<idx_t> sorted_d = block_deltas;
                std::sort(sorted_d.begin(), sorted_d.end());
                
                size_t threshold_idx = (size_t)(count * pfor_threshold);
                if (threshold_idx >= count) threshold_idx = count - 1;
                idx_t max_val = sorted_d[threshold_idx];
                
                int b = 0;
                while (max_val > 0) { max_val >>= 1; b++; }
                
                std::vector<std::pair<uint8_t, idx_t>> exceptions;
                idx_t mask = (b == 32) ? 0xFFFFFFFF : ((1U << b) - 1);
                
                for(size_t j=0; j<count; ++j) {
                    if (block_deltas[j] > mask) {
                        exceptions.push_back({(uint8_t)j, block_deltas[j]});
                    }
                }
                
                uint32_t meta_val = (uint32_t)((count << 16) | (exceptions.size() << 8) | (b & 0xFF));
                result.pfor_be_meta_bits.push_back(meta_val);
                
                result.pfor_be_meta_ex_start.push_back((uint32_t)result.pfor_be_exceptions.size());
                result.pfor_be_meta_data_start.push_back((uint32_t)result.pfor_be_data.size());

                for(auto& ex : exceptions) {
                    result.pfor_be_exception_idx.push_back(ex.first);
                    result.pfor_be_exceptions.push_back(ex.second);
                }

                uint64_t buffer = 0;
                int bits_in_buffer = 0;
                
                for(size_t j=0; j<count; ++j) {
                    idx_t val = block_deltas[j] & mask; 
                    buffer |= ((uint64_t)val << bits_in_buffer);
                    bits_in_buffer += b;
                    
                    while(bits_in_buffer >= 32) {
                        result.pfor_be_data.push_back((uint32_t)(buffer & 0xFFFFFFFF));
                        buffer >>= 32;
                        bits_in_buffer -= 32;
                    }
                }
                if (bits_in_buffer > 0) {
                    result.pfor_be_data.push_back((uint32_t)(buffer & 0xFFFFFFFF));
                }
            }
        }
        
        // Pad PFor-BE data so the kernel can safely overread.
        // kernel_sparse_pfor_bit_exact reads 4 chunks × 32 lanes = 128 elements,
        // each accessing data[d_start + word_idx] and data[d_start + word_idx + 1].
        // Worst case (b=32, i=127): word_idx=127, reads index 128. So up to 129
        // words past d_start are touched. Pad with 132 words for safety margin.
        for(int k=0; k<132; ++k) result.pfor_be_data.push_back(0);

        result.pfor_be_row_block_starts.push_back((uint32_t)result.pfor_be_meta_bits.size());

        // E. Golomb-Rice
        if (!use_ellpack && result.stream_remainders.empty()) {
            size_t num_warps = (raw_matrix.n_rows + 31) / 32;
            result.warp_rice_params.resize(num_warps);
            result.row_rem_offsets.reserve(raw_matrix.n_rows + 1);
            result.row_quo_offsets.reserve(raw_matrix.n_rows + 1);
            for(row_idx_t r=0; r<first_sparse_row; ++r) { result.row_rem_offsets.push_back(0); result.row_quo_offsets.push_back(0); }
            uint32_t q_accumulator = 0; int q_bits_filled = 0;
            for (size_t w = first_sparse_row / 32; w < num_warps; ++w) {
                size_t row_start = w * 32;
                size_t row_end = std::min(row_start + 32, (size_t)raw_matrix.n_rows);
                std::vector<idx_t> warp_gaps;
                for (size_t r = row_start; r < row_end; ++r) {
                    if (r < first_sparse_row) continue;
                    row_idx_t sorted_r = p[r];
                    idx_t prev_col = dense_col_limit - 1;
                    for (idx_t col : raw_matrix.get_row(sorted_r)) { if (col >= dense_col_limit) { warp_gaps.push_back(col - prev_col - 1); prev_col = col; } }
                }
                int k = calculate_optimal_k(warp_gaps); result.warp_rice_params[w] = k;
                for (size_t r = row_start; r < row_end; ++r) {
                    if (r < first_sparse_row) continue;
                    result.row_rem_offsets.push_back((uint32_t)result.stream_remainders.size());
                    result.row_quo_offsets.push_back((uint32_t)result.stream_quotients.size() * 32 + q_bits_filled);
                    row_idx_t sorted_r = p[r]; idx_t prev_col = dense_col_limit - 1;
                    for (idx_t col : raw_matrix.get_row(sorted_r)) {
                        if (col >= dense_col_limit) {
                            result.total_nnz++;
                            idx_t delta = col - prev_col - 1;
                            uint32_t remainder = delta & ((1 << k) - 1);
                            uint32_t quotient = delta >> k;
                            result.stream_remainders.push_back(remainder);
                            for (uint32_t i = 0; i < quotient; ++i) { q_accumulator |= (1U << q_bits_filled); q_bits_filled++; if (q_bits_filled == 32) { result.stream_quotients.push_back(q_accumulator); q_accumulator = 0; q_bits_filled = 0; } }
                            q_bits_filled++; if (q_bits_filled == 32) { result.stream_quotients.push_back(q_accumulator); q_accumulator = 0; q_bits_filled = 0; }
                            prev_col = col;
                        }
                    }
                }
            }
            result.row_rem_offsets.push_back((uint32_t)result.stream_remainders.size());
            result.row_quo_offsets.push_back((uint32_t)result.stream_quotients.size() * 32 + q_bits_filled);
            if (q_bits_filled > 0) result.stream_quotients.push_back(q_accumulator);
        }
    }
    return result;
}

std::vector<HostMatrix> MatrixPreprocessor::slice_matrix_vertical(
    const HostMatrix& matrix,
    idx_t dense_col_limit,
    uint32_t row_block_size,
    uint32_t max_nnz_per_block
) {
    LOG(LOG_INFO) << "[Slicing] Vertically slicing matrix (Max NNZ/Block=" << max_nnz_per_block << ")...";
    
    std::vector<HostMatrix> slices;
    
    // Create first slice
    HostMatrix current_slice;
    current_slice.n_rows = matrix.n_rows;
    current_slice.n_cols = matrix.n_cols; 
    current_slice.rows.resize(matrix.n_rows);
    
    size_t num_row_blocks = (matrix.n_rows + row_block_size - 1) / row_block_size;
    
    std::vector<uint32_t> cuts;
    cuts.push_back(dense_col_limit); 
    
    idx_t bucket_width = 1024; 
    idx_t n_sparse = matrix.n_cols - dense_col_limit;
    if (n_sparse <= 0) return {matrix}; 
    
    size_t num_buckets = (n_sparse + bucket_width - 1) / bucket_width;
    std::vector<uint32_t> counts(num_row_blocks * num_buckets, 0);
    
    #pragma omp parallel for schedule(dynamic, 128)
    for(row_idx_t r = 0; r < matrix.n_rows; ++r) {
        uint32_t b_id = r / row_block_size;
        for(idx_t c : matrix.rows[r]) {
            if (c >= dense_col_limit) {
                idx_t rel_col = c - dense_col_limit;
                idx_t bucket = rel_col / bucket_width;
                if(bucket < num_buckets) {
                    #pragma omp atomic
                    counts[b_id * num_buckets + bucket]++;
                }
            }
        }
    }
    
    std::vector<uint32_t> current_block_sums(num_row_blocks, 0);
    
    for (size_t bucket = 0; bucket < num_buckets; ++bucket) {
        bool overflow = false;
        for (size_t b = 0; b < num_row_blocks; ++b) {
            uint32_t n = counts[b * num_buckets + bucket];
            if (current_block_sums[b] + n > max_nnz_per_block) {
                overflow = true;
                break;
            }
            current_block_sums[b] += n;
        }
        
        if (overflow) {
            idx_t cut_col = dense_col_limit + bucket * bucket_width;
            cuts.push_back(cut_col);
            
            std::fill(current_block_sums.begin(), current_block_sums.end(), 0);
            for (size_t b = 0; b < num_row_blocks; ++b) {
                current_block_sums[b] = counts[b * num_buckets + bucket];
            }
        }
    }
    cuts.push_back(matrix.n_cols); 
    
    LOG(LOG_INFO) << "[Slicing] Generated " << cuts.size() - 1 << " vertical slices.";
    
    slices.resize(cuts.size() - 1);
    
    #pragma omp parallel for
    for (size_t s = 0; s < slices.size(); ++s) {
        idx_t start_col = cuts[s];
        idx_t end_col = cuts[s+1];
        
        slices[s].n_rows = matrix.n_rows;
        slices[s].n_cols = matrix.n_cols;
        slices[s].rows.resize(matrix.n_rows);
        
        for (row_idx_t r = 0; r < matrix.n_rows; ++r) {
            for (idx_t c : matrix.rows[r]) {
                if (c >= start_col && c < end_col) {
                    slices[s].rows[r].push_back(c);
                }
            }
        }
    }
    
    #pragma omp parallel for
    for (row_idx_t r = 0; r < matrix.n_rows; ++r) {
        for (idx_t c : matrix.rows[r]) {
            if (c < dense_col_limit) {
                slices[0].rows[r].push_back(c);
            }
        }
        std::sort(slices[0].rows[r].begin(), slices[0].rows[r].end());
    }
    
    return slices;
}

// =================================================================================
// M4RM Preprocessing Implementation
// =================================================================================

M4RMContext MatrixPreprocessor::preprocess_m4rm(const HostMatrixView& AT, int num_rows) {
    M4RMContext ctx;
    prepare_m4rm_streams(AT, num_rows, ctx);
    return ctx;
}
