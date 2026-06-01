// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include <cstdint>
#include <string>

namespace mpqs {
namespace matrix {

/// Format a 64-bit unsigned integer with thousands separators.
/// Example: 1234567 -> "1,234,567".
inline std::string fmtNum(uint64_t n) {
    std::string s = std::to_string(n);
    int pos = static_cast<int>(s.size()) - 3;
    while (pos > 0) { s.insert(pos, ","); pos -= 3; }
    return s;
}

} // namespace matrix
} // namespace mpqs
