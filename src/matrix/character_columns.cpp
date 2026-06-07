// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "character_columns.h"
#include "merge_tree.h"
#include "montgomery.cuh"
#include "gpu_char_cols.cuh"
#include "prime_algorithms.h"  // Stage 1: Tonelli_Shanks_u64, jacobi_u64, is_prime_u64

#include <algorithm>  // std::min
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
                                               CharMode mode,
                                               uint64_t lp1_bound,
                                               uint32_t k) {
    std::unordered_set<uint32_t> fb_set(fb.begin(), fb.end());

    aux_primes_.clear();
    n_mod_q_.clear();
    t_s_.clear();
    aux_primes_.reserve(k);
    n_mod_q_.reserve(k);
    t_s_.reserve(k);

    if (mode == CharMode::NORM) {
        // NORM (default): byte-identical to the legacy selection. Aux primes stay
        // < 2^32; storing them in the widened uint64 vector is value-preserving.
        // Start at 3 and walk odd candidates; stop once we have k primes.
        for (uint32_t candidate = 3; aux_primes_.size() < k; candidate += 2) {
            if (!is_prime_small(candidate)) continue;
            if (fb_set.count(candidate)) continue;  // skip factor base primes

            uint32_t n_mod = static_cast<uint32_t>(N % candidate);
            if (jacobi_symbol_dev(n_mod, candidate) != 1) continue;  // require (N|q) = +1

            aux_primes_.push_back(candidate);
            n_mod_q_.push_back(n_mod);
        }
        return;
    }

    // BRANCH: select aux primes > lp1_bound (guaranteed outside FB / large-prime
    // range) via a 64-bit walk, locking a fixed Tonelli root t_s per prime.
    // Start at the first odd integer strictly greater than lp1_bound.
    uint64_t candidate = lp1_bound + 1;
    if ((candidate & 1ull) == 0) ++candidate;  // make odd
    for (; aux_primes_.size() < k; candidate += 2) {
        if (!mpqs::sieve::is_prime_u64(candidate)) continue;  // Stage 1, not is_prime_small
        assert(candidate > lp1_bound);
        // FB primes are all < lp1_bound, so q > lp1_bound is automatically outside FB.
        assert(candidate > 0xFFFFFFFFull ||
               fb_set.count(static_cast<uint32_t>(candidate)) == 0);

        uint64_t n_mod = N.mod_uint64(candidate);  // Stage 1
        if (mpqs::sieve::jacobi_u64(n_mod, candidate) != 1) continue;  // require (N|q) = +1

        uint64_t t = mpqs::sieve::Tonelli_Shanks_u64(n_mod, candidate);  // Stage 1
        t = std::min(t, candidate - t);  // canonicalize the branch for determinism

        aux_primes_.push_back(candidate);
        t_s_.push_back(t);
        // n_mod_q_ is the NORM path's representation; intentionally not pushed here.
    }
}

CharacterColumns CharacterColumnComputer::compute(
        const structures::HostRelationBatch& batch, CharMode mode) const {
    CharacterColumns result;
    result.aux_primes = aux_primes_;
    result.k = static_cast<uint32_t>(aux_primes_.size());
    result.columns.resize(result.k);

    const uint32_t n_rels = static_cast<uint32_t>(batch.num_relations);

    if (mode == CharMode::BRANCH) {
        // --- Stage 5 BRANCH path: thin adapter, no symbol re-evaluation ---
        // The genus-correct branch-fixed char vector was evaluated per relation at
        // birth (Stage 4) on the SIGNED (ax+b) and persisted in batch.char_bits.
        // Here we merely unpack bit j into columns[j][i]. char_bits absent (norm-
        // loaded data) → 0. (selectAuxPrimes must have been called with BRANCH so
        // result.k == r matches the persisted bit width.)
        for (uint32_t j = 0; j < result.k; ++j)
            result.columns[j].resize(n_rels, 0);
        for (uint32_t i = 0; i < n_rels; ++i) {
            const uint32_t cb = (i < batch.char_bits.size()) ? batch.char_bits[i] : 0u;
            for (uint32_t j = 0; j < result.k; ++j)
                result.columns[j][i] = static_cast<uint8_t>((cb >> j) & 1u);
        }
        return result;
    }

    for (uint32_t j = 0; j < result.k; ++j) {
        result.columns[j].resize(n_rels, 0);
        // NORM path only (compute() reads n_mod_q_): q < 2^32, cast is value-preserving.
        const uint32_t q  = static_cast<uint32_t>(aux_primes_[j]);
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
    CharMode mode,
    uint64_t lp1_bound,
    uint32_t k) {

    // 1. Select auxiliary primes (reuse existing logic; mode/lp1_bound threaded
    //    through). NORM walks q < 2^32 from 3; BRANCH walks q > lp1_bound.
    CharacterColumnComputer cc;
    cc.selectAuxPrimes(N, fb, mode, lp1_bound, k);
    const auto& aux_primes = cc.auxPrimes();  // std::vector<uint64_t>

    // 2. Set up result
    CharacterColumns result;
    result.aux_primes = aux_primes;
    result.k = k;
    result.columns.resize(k);
    const uint32_t num_rows = static_cast<uint32_t>(row_map.size());
    for (uint32_t j = 0; j < k; ++j)
        result.columns[j].resize(num_rows, 0);

    // 3. Temporary buffer for leaf expansion (reused across rows).
    std::vector<uint32_t> leaves;

    if (mode == CharMode::BRANCH) {
        // --- Stage 5 BRANCH path: homomorphic XOR-composition over the merge tree ---
        // The branch-fixed field-element character (gpu_char_cols.cuh branchCharBit)
        // is a multiplicative homomorphism, so the r-bit char vector of a merged
        // (= product) row equals the XOR of its constituent leaves' raw per-relation
        // vectors — composed via the SAME tree.expand(row_map[r]) leaf set the
        // Montgomery sqrt_Q product uses (with multiplicity). The raw vectors were
        // evaluated at relation birth (Stage 4) on the SIGNED (ax+b); we NEVER
        // re-evaluate the symbol on the mod-N product (provably non-homomorphic;
        // audit F4, proto-v2 §3). char_bits absent (norm-loaded data) → treated as 0.
        auto leaf_char_bits = [&](uint32_t leaf) -> uint32_t {
            if (leaf < ns)
                return (leaf < smooth_batch.char_bits.size())
                           ? smooth_batch.char_bits[leaf] : 0u;
            const uint32_t p = leaf - static_cast<uint32_t>(ns);
            return (p < partial_batch.char_bits.size())
                       ? partial_batch.char_bits[p] : 0u;
        };

        for (uint32_t r = 0; r < num_rows; ++r) {
            tree.expand(row_map[r], leaves);  // leaves with multiplicity
            uint32_t v = 0u;
            for (uint32_t leaf : leaves) v ^= leaf_char_bits(leaf);
            // Unpack the r (= k) bits into column-major form.
            for (uint32_t j = 0; j < k; ++j)
                result.columns[j][r] = static_cast<uint8_t>((v >> j) & 1u);
        }

        return result;
    }

    // --- NORM path (default): byte-identical to before (mod-N product symbol) ---
    // Precompute N mod q for each aux prime (NORM: q < 2^32, value-preserving)
    std::vector<uint32_t> n_mod_q(k);
    for (uint32_t j = 0; j < k; ++j)
        n_mod_q[j] = static_cast<uint32_t>(N % aux_primes[j]);

    // Montgomery context (shared across all rows)
    math::Montgomery mont(N);

    // For each reduced row, compute product sqrt_Q mod N, then Legendre symbols
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
        // (NORM path: q < 2^32; the narrowing cast is value-preserving.)
        for (uint32_t j = 0; j < k; ++j) {
            uint32_t q = static_cast<uint32_t>(aux_primes[j]);
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
