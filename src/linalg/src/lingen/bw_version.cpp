// SPDX-License-Identifier: LGPL-3.0-or-later
// This file is part of the Block Wiedemann implementation.

#include "bw_version.h"

#include <cuda_runtime.h>

#include <sstream>
#include <string>

namespace lingen {

std::string version_banner() {
    std::ostringstream oss;
    oss << "block-wiedemann " << LINALGBW_VERSION_STRING
        << " (git " << LINALGBW_GIT_HASH;

    // CUDART_VERSION is a compile-time integer like 12060 → "12.6".
    const int cudart_major = CUDART_VERSION / 1000;
    const int cudart_minor = (CUDART_VERSION % 1000) / 10;
    oss << ", CUDA " << cudart_major << '.' << cudart_minor;

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        oss << ", SM " << prop.major << prop.minor;
    } else {
        oss << ", SM ?";
    }

    oss << ')';
    return oss.str();
}

}  // namespace lingen
