// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <cstdint>

namespace lingen {

/// @brief Kernel-passable view of packed BW solution bit-vectors on device.
///
/// Layout: num_solutions contiguous vectors in row-major order.
/// Solution j, relation i is accessed as:
///   (d_data[j * words_per_vec + i/64] >> (i % 64)) & 1
///
/// Total buffer size: num_solutions * words_per_vec * sizeof(uint64_t) bytes.
struct BWKernelSolutionView {
    const uint64_t* d_data;     ///< Device pointer to packed bit-matrix
    uint32_t num_solutions;     ///< Number of kernel vectors (K)
    uint32_t words_per_vec;     ///< Words per vector = ceil(num_rows / 64)
    uint32_t num_rows;          ///< Original (unpadded) matrix row count
};

} // namespace lingen
