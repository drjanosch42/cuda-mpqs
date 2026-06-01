// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <iostream>
#include "common.h" 
#include "hpc_logger.h"

class ValidationExporter {
public:
    static void export_problem(const std::string& prefix, 
                               const HostMatrix& A, 
                               const std::vector<uint64_t>& hX, 
                               const std::vector<uint64_t>& hY,
                               int N, int nrows_logical, int ncols_logical) 
    {
        std::string fn_A = prefix + "_A.bin";
        std::string fn_X = prefix + "_X.bin";
        std::string fn_Y = prefix + "_Y.bin";

        LOG(LOG_INFO) << "[Validation] Exporting problem instance to " << prefix << "_*.bin";

        // 1. Export A (Dense Format for simplicity in Python loading)
        // Format: [Magic:4][Rows:4][Cols:4][Data:Rows*Cols (uint8)]
        std::ofstream fA(fn_A, std::ios::binary);
        const char magicA[4] = {'M', 'A', 'T', 'X'};
        uint32_t r = nrows_logical;
        uint32_t c = ncols_logical;
        
        fA.write(magicA, 4);
        fA.write((char*)&r, 4);
        fA.write((char*)&c, 4);

        std::vector<uint8_t> dense_row(c, 0);
        for(const auto& row_indices : A.rows) {
            std::fill(dense_row.begin(), dense_row.end(), 0);
            for(auto col_idx : row_indices) {
                // FIX: Use XOR accumulation to handle duplicates (1+1=0) correctly
                if(col_idx < c) dense_row[col_idx] ^= 1;
            }
            fA.write((char*)dense_row.data(), c);
        }
        fA.close();

        // 2. Export X and Y (Raw uint64 array)
        // Format: [Magic:4][Rows:4][N_bits:4][Data...]
        auto write_vec = [&](std::string fname, const std::vector<uint64_t>& vec) {
            std::ofstream fv(fname, std::ios::binary);
            const char magicV[4] = {'V', 'E', 'C', 'T'};
            uint32_t v_n = N;
            fv.write(magicV, 4);
            fv.write((char*)&r, 4); // rows
            fv.write((char*)&v_n, 4); // N
            fv.write((char*)vec.data(), vec.size() * sizeof(uint64_t));
            fv.close();
        };

        write_vec(fn_X, hX);
        write_vec(fn_Y, hY);
    }
    
    static void export_sequence(const std::string& filename, const std::vector<uint64_t>& S, int terms, int N) {
        // Format: [Magic:4][Terms:4][N:4][Data...]
        std::ofstream fs(filename, std::ios::binary);
        const char magicS[4] = {'S', 'E', 'Q', 'U'};
        uint32_t t = terms;
        uint32_t n = N;
        fs.write(magicS, 4);
        fs.write((char*)&t, 4);
        fs.write((char*)&n, 4);
        fs.write((char*)S.data(), S.size() * sizeof(uint64_t));
        fs.close();
        LOG(LOG_INFO) << "[Validation] Exported sequence to " << filename;
    }

    static void export_polynomial_matrix(const std::string& filename, 
                                         const std::vector<uint64_t>& data, 
                                         int length, int dim_rows, int dim_cols) 
    {
        // Format: [Magic:4][Len:4][Rows:4][Cols:4][Data...]
        std::ofstream fs(filename, std::ios::binary);
        const char magicP[4] = {'P', 'O', 'L', 'Y'};
        uint32_t len = length;
        uint32_t r = dim_rows;
        uint32_t c = dim_cols;
        
        fs.write(magicP, 4);
        fs.write((char*)&len, 4);
        fs.write((char*)&r, 4);
        fs.write((char*)&c, 4);
        
        // Data size: length * rows * (ceil(cols/64)) * 8 bytes
        // The vector 'data' is assumed to be correctly sized.
        fs.write((char*)data.data(), data.size() * sizeof(uint64_t));
        fs.close();
        LOG(LOG_INFO) << "[Validation] Exported polynomial matrix to " << filename;
    }
};
