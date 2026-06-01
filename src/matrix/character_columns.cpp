// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "character_columns.h"
#include "merge_tree.h"
#include "montgomery.cuh"
#include "gpu_char_cols.cuh"

#include <cassert>
#include <cmath>
#include <unordered_set>
#include <utility>

namespace mpqs {
namespace matrix {

// ---------------------------------------------------------------------------
// Private helpers (anonymous namespace)
// ---------------------------------------------------------------------------

namespace {

/// Simple primality test via trial division.  Sufficient for candidates < ~1000.
bool is_prime_small(uint32_t candidate) {
    if (candidate < 2) return false;
    if (candidate == 2) return true;
    if ((candidate & 1u) == 0) return false;
    uint32_t limit = static_cast<uint32_t>(std::sqrt(static_cast<double>(candidate))) + 1;
    for (uint32_t d = 3; d <= limit; d += 2) {
        if (candidate % d == 0) return false;
    }
    return true;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// CharacterColumnComputer
// ---------------------------------------------------------------------------

void CharacterColumnComputer::selectAuxPrimes(const uint512& N,
                                               const std::vector<uint32_t>& fb,
                                               uint32_t k) {
    std::unordered_set<uint32_t> fb_set(fb.begin(), fb.end());

    aux_primes_.clear();
    n_mod_q_.clear();
    aux_primes_.reserve(k);
    n_mod_q_.reserve(k);

    // Start at 3 and walk odd candidates; stop once we have k primes.
    for (uint32_t candidate = 3; aux_primes_.size() < k; candidate += 2) {
        if (!is_prime_small(candidate)) continue;
        if (fb_set.count(candidate)) continue;  // skip factor base primes

        uint32_t n_mod = static_cast<uint32_t>(N % candidate);
        if (jacobi_symbol_dev(n_mod, candidate) != 1) continue;  // require (N|q) = +1

        aux_primes_.push_back(candidate);
        n_mod_q_.push_back(n_mod);
    }
}

CharacterColumns CharacterColumnComputer::compute(
        const structures::HostRelationBatch& batch) const {
    CharacterColumns result;
    result.aux_primes = aux_primes_;
    result.k = static_cast<uint32_t>(aux_primes_.size());
    result.columns.resize(result.k);

    const uint32_t n_rels = static_cast<uint32_t>(batch.num_relations);

    for (uint32_t j = 0; j < result.k; ++j) {
        result.columns[j].resize(n_rels, 0);
        const uint32_t q  = aux_primes_[j];
        const uint32_t nq = n_mod_q_[j];

        for (uint32_t i = 0; i < n_rels; ++i) {
            // Reduce sqrt_Q[i] mod q using the fast uint512 % uint32_t operator.
            uint32_t sq_mod_q = static_cast<uint32_t>(batch.sqrt_Q[i] % q);

            // Q_i mod q = (sq_mod_q^2 - N_mod_q) mod q.
            // Use 64-bit arithmetic and add q to prevent underflow before final mod.
            uint64_t sq2    = static_cast<uint64_t>(sq_mod_q) * sq_mod_q;
            uint32_t Q_mod_q = static_cast<uint32_t>((sq2 % q + q - nq) % q);

            // Jacobi symbol: QNR (-1) → column bit 1; QR or degenerate → 0.
            int ls = jacobi_symbol_dev(Q_mod_q, q);
            result.columns[j][i] = (ls == -1) ? 1 : 0;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// AppendCharacterColumns
// ---------------------------------------------------------------------------

void AppendCharacterColumns(HostMatrixCSR& csr, const CharacterColumns& chars,
                            uint32_t n_rels) {
    assert(csr.n_rows == n_rels);

    const uint32_t base_col = csr.n_cols;  // character columns start here

    // Count extra column entries per row.
    std::vector<uint32_t> extra(n_rels, 0);
    for (uint32_t j = 0; j < chars.k; ++j) {
        for (uint32_t i = 0; i < n_rels; ++i) {
            extra[i] += chars.columns[j][i];
        }
    }

    // Compute total extra entries.
    uint32_t total_extra = 0;
    for (uint32_t i = 0; i < n_rels; ++i) total_extra += extra[i];

    const uint32_t old_nnz = static_cast<uint32_t>(csr.col_indices.size());

    // Build new row_offsets and col_indices with character entries appended.
    std::vector<uint32_t> new_offsets(n_rels + 1);
    std::vector<uint32_t> new_indices;
    new_indices.reserve(old_nnz + total_extra);

    new_offsets[0] = 0;
    for (uint32_t i = 0; i < n_rels; ++i) {
        // Copy existing entries for row i.
        const uint32_t old_start = csr.row_offsets[i];
        const uint32_t old_end   = csr.row_offsets[i + 1];
        for (uint32_t idx = old_start; idx < old_end; ++idx) {
            new_indices.push_back(csr.col_indices[idx]);
        }

        // Append character column entries (indices >= base_col, preserving sort order
        // since character indices are all larger than any existing column index).
        for (uint32_t j = 0; j < chars.k; ++j) {
            if (chars.columns[j][i] == 1) {
                new_indices.push_back(base_col + j);
            }
        }

        new_offsets[i + 1] = static_cast<uint32_t>(new_indices.size());
    }

    csr.row_offsets = std::move(new_offsets);
    csr.col_indices = std::move(new_indices);
    csr.n_cols += chars.k;
}

// ---------------------------------------------------------------------------
// ConcatenateCharacterColumns
// ---------------------------------------------------------------------------

CharacterColumns ConcatenateCharacterColumns(const CharacterColumns& a,
                                              const CharacterColumns& b) {
    assert(a.k == b.k);

    CharacterColumns result;
    result.aux_primes = a.aux_primes;
    result.k = a.k;
    result.columns.resize(result.k);

    for (uint32_t j = 0; j < result.k; ++j) {
        result.columns[j].reserve(a.columns[j].size() + b.columns[j].size());
        result.columns[j].insert(result.columns[j].end(),
                                 a.columns[j].begin(), a.columns[j].end());
        result.columns[j].insert(result.columns[j].end(),
                                 b.columns[j].begin(), b.columns[j].end());
    }

    return result;
}

// ---------------------------------------------------------------------------
// computeProductCharacterColumns
// ---------------------------------------------------------------------------

CharacterColumns computeProductCharacterColumns(
    const std::vector<uint32_t>& row_map,
    const MergeTree& tree,
    const structures::HostRelationBatch& smooth_batch,
    const structures::HostRelationBatch& partial_batch,
    size_t ns,
    const uint512& N,
    const std::vector<uint32_t>& fb,
    uint32_t k) {

    // 1. Select auxiliary primes (reuse existing logic)
    CharacterColumnComputer cc;
    cc.selectAuxPrimes(N, fb, k);
    const auto& aux_primes = cc.auxPrimes();

    // Precompute N mod q for each aux prime
    std::vector<uint32_t> n_mod_q(k);
    for (uint32_t j = 0; j < k; ++j)
        n_mod_q[j] = static_cast<uint32_t>(N % aux_primes[j]);

    // 2. Set up result
    CharacterColumns result;
    result.aux_primes = aux_primes;
    result.k = k;
    result.columns.resize(k);
    const uint32_t num_rows = static_cast<uint32_t>(row_map.size());
    for (uint32_t j = 0; j < k; ++j)
        result.columns[j].resize(num_rows, 0);

    // 3. Montgomery context (shared across all rows)
    math::Montgomery mont(N);

    // 4. Temporary buffer for leaf expansion
    std::vector<uint32_t> leaves;

    // 5. For each reduced row, compute product sqrt_Q mod N, then Legendre symbols
    for (uint32_t r = 0; r < num_rows; ++r) {
        // Expand merge tree node to leaf indices
        tree.expand(row_map[r], leaves);

        // Leaf i < ns -> smooth_batch, leaf i >= ns -> partial_batch.
        // Returns const reference to avoid copying 512-bit values.
        auto get_sqrt_Q = [&](uint32_t leaf) -> const uint512& {
            return (leaf < ns) ? smooth_batch.sqrt_Q[leaf]
                               : partial_batch.sqrt_Q[leaf - ns];
        };

        uint512 S;  // product sqrt_Q mod N
        if (leaves.size() == 1) {
            // Single-leaf row: no Montgomery multiply needed.
            // Character value is identical to the char-M1 per-relation computation.
            S = get_sqrt_Q(leaves[0]);
        } else {
            // Multi-leaf: accumulate product in Montgomery form.
            // The mod-N reduction introduces non-linear information not encodable
            // by GF(2) columns, making these character values genuinely independent.
            uint512 S_mont = mont.transform(get_sqrt_Q(leaves[0]));
            for (size_t l = 1; l < leaves.size(); ++l) {
                uint512 t_mont = mont.transform(get_sqrt_Q(leaves[l]));
                S_mont = mont.mul(S_mont, t_mont);
            }
            S = mont.reduce(S_mont);
        }

        // Evaluate Legendre symbols for each aux prime
        for (uint32_t j = 0; j < k; ++j) {
            uint32_t q = aux_primes[j];
            uint32_t sq_mod_q = static_cast<uint32_t>(S % q);
            // Q mod q = (sq^2 - N) mod q.  Use 64-bit arithmetic and add q
            // before final mod to prevent underflow on subtraction.
            uint64_t sq2 = static_cast<uint64_t>(sq_mod_q) * sq_mod_q;
            uint32_t Q_mod_q = static_cast<uint32_t>((sq2 % q + q - n_mod_q[j]) % q);
            int ls = jacobi_symbol_dev(Q_mod_q, q);
            result.columns[j][r] = (ls == -1) ? 1 : 0;
        }
    }

    return result;
}

}  // namespace matrix
}  // namespace mpqs
