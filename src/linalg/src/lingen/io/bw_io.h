// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include "lingen/types.h"
#include "hpc_logger.h"

namespace lingen {
namespace io {

// Magic headers to ensure data integrity
constexpr uint32_t MAGIC_MAT_A = 0x4154414D; // "MATA"
constexpr uint32_t MAGIC_VEC   = 0x54434556; // "VECT"
constexpr uint32_t MAGIC_SEQ   = 0x55514553; // "SEQU"
constexpr uint32_t MAGIC_SEQ2  = 0x32514553; // "SEQ2"  
constexpr uint32_t MAGIC_POLY  = 0x594C4F50; // "POLY"
constexpr uint32_t MAGIC_SOL   = 0x4E4C4F53; // "SOLN"

class BWIOSystem {
public:
    explicit BWIOSystem(std::string prefix) : prefix_(std::move(prefix)) {}

    std::string get_prefix() const { return prefix_; }
    
    // Check if a file exists to allow resuming
    bool exists(const std::string& suffix) const {
        struct stat buffer;
        std::string fname = prefix_ + suffix;
        return (stat(fname.c_str(), &buffer) == 0);
    }

    // --- Core Export Functions ---

    void save_matrix_A(const HostMatrix& A) {
        std::string fname = prefix_ + "_A.bin";
        std::ofstream f(fname, std::ios::binary);
        if(!f) { LOG(LOG_ERROR_MAJOR) << "[BW IO] Failed to open " << fname; return; }

        uint32_t r = A.n_rows;
        uint32_t c = A.n_cols;
        
        // Header
        write_u32(f, MAGIC_MAT_A);
        write_u32(f, r);
        write_u32(f, c);

        // Dense dump (uint8)
        std::vector<uint8_t> row_buf(c, 0);
        for(const auto& row_indices : A.rows) {
            std::fill(row_buf.begin(), row_buf.end(), 0);
            for(auto col_idx : row_indices) if(col_idx < c) row_buf[col_idx] ^= 1;
            f.write((char*)row_buf.data(), c);
        }
        LOG(LOG_INFO) << "[BW IO] Saved Matrix A to " << fname;
    }

    void save_vector(const std::string& suffix, const std::vector<uint64_t>& vec, int rows, int N_block) {
        std::string fname = prefix_ + suffix;
        std::ofstream f(fname, std::ios::binary);
        
        write_u32(f, MAGIC_VEC);
        write_u32(f, (uint32_t)rows);
        write_u32(f, (uint32_t)N_block);
        f.write((char*)vec.data(), vec.size() * sizeof(uint64_t));
        LOG(LOG_INFO) << "[BW IO] Saved Vector " << suffix << " (" << (vec.size()*8)/1024 << " KB)";
    }

    // Supports rectangular matrices with SEQ2 magic
    void save_sequence(const std::vector<uint64_t>& S, int len, int rows, int cols) {
        std::string fname = prefix_ + "_S.bin";
        std::ofstream f(fname, std::ios::binary);

        write_u32(f, MAGIC_SEQ2);
        write_u32(f, (uint32_t)len);
        write_u32(f, (uint32_t)rows);
        write_u32(f, (uint32_t)cols);
        
        f.write((char*)S.data(), S.size() * sizeof(uint64_t));
        LOG(LOG_INFO) << "[BW IO] Saved Sequence S (" << len << " terms) [" << rows << "x" << cols << "]";
    }

    // --- Core Import Functions ---

    bool load_sequence(std::vector<uint64_t>& S, int& len_out, int m_expected, int n_expected) {
        std::string fname = prefix_ + "_S.bin";
        std::ifstream f(fname, std::ios::binary | std::ios::ate);
        if (!f) return false;

        // std::streamsize size = f.tellg();
        f.seekg(0, std::ios::beg);

        uint32_t magic;
        f.read((char*)&magic, 4);

        int m = 0, n = 0, len = 0;

        if (magic == MAGIC_SEQ2) {
            uint32_t u_len, u_m, u_n;
            f.read((char*)&u_len, 4);
            f.read((char*)&u_m, 4);
            f.read((char*)&u_n, 4);
            len = (int)u_len;
            m = (int)u_m;
            n = (int)u_n;
        } else if (magic == MAGIC_SEQ) {
            // Legacy Support
            uint32_t u_len, u_blk;
            f.read((char*)&u_len, 4);
            f.read((char*)&u_blk, 4);
            len = (int)u_len;
            m = (int)u_blk;
            n = (int)u_blk;
        } else {
            LOG(LOG_ERROR_MAJOR) << "[BW IO] Bad magic in " << fname;
            return false;
        }

        if (m != m_expected || n != n_expected) {
            LOG(LOG_ERROR_MAJOR) << "[BW IO] Dimensions mismatch in " << fname 
                                 << ". Expected " << m_expected << "x" << n_expected 
                                 << " got " << m << "x" << n;
            return false;
        }

        len_out = len;
        
        size_t words_per_row = (n + 63) / 64;
        size_t mat_stride = m * words_per_row;
        size_t total_words = (size_t)len * mat_stride;
        
        S.resize(total_words);
        f.read((char*)S.data(), total_words * sizeof(uint64_t));
        
        LOG(LOG_INFO) << "[BW IO] Loaded Sequence S (" << len << " terms) from " << fname;
        return true;
    }  

    // Save matrix polynomial
    void save_polynomial(const std::vector<uint64_t>& Pi, int len, int rows, int cols) {
        std::string fname = prefix_ + "_Pi.bin";
        std::ofstream f(fname, std::ios::binary);
        write_u32(f, MAGIC_POLY);
        write_u32(f, (uint32_t)len);
        write_u32(f, (uint32_t)rows); 
        write_u32(f, (uint32_t)cols); 
        f.write((char*)Pi.data(), Pi.size() * sizeof(uint64_t));
        LOG(LOG_INFO) << "[BW IO] Saved Polynomial Pi (Degree " << len-1 << ") [" << rows << "x" << cols << "]";
    }  

    void save_solutions(const std::vector<std::vector<uint64_t>>& solutions, int rows) {
        if (solutions.empty()) return;
        save_single_solution("_W.bin", solutions[0], rows);
        for(size_t i=0; i<solutions.size(); ++i) {
            std::string suffix = "_sol_" + std::to_string(i) + ".bin";
            save_single_solution(suffix, solutions[i], rows);
        }
        LOG(LOG_INFO) << "[BW IO] Saved " << solutions.size() << " solution vectors.";
    }

private:
    std::string prefix_;

    void write_u32(std::ofstream& f, uint32_t val) {
        f.write((char*)&val, sizeof(uint32_t));
    }

    void save_single_solution(const std::string& suffix, const std::vector<uint64_t>& w, int rows) {
        std::string fname = prefix_ + suffix;
        std::ofstream f(fname, std::ios::binary);
        write_u32(f, MAGIC_SOL);
        write_u32(f, (uint32_t)rows);
        write_u32(f, 1); 
        f.write((char*)w.data(), w.size() * sizeof(uint64_t));
    }
};

} // namespace io
} // namespace lingen
