// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include "matrix_constructor.h"  // HostMatrixCSR
#include "merge_tree.h"          // MergeTree
#include "mpqs_soa.h"            // HostRelationBatch
#include "uint512.cuh"           // uint512
#include <vector>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Character column data for k auxiliary primes over n relations.
struct CharacterColumns {
    std::vector<uint32_t> aux_primes;           ///< k auxiliary primes
    std::vector<std::vector<uint8_t>> columns;  ///< columns[j][i] in {0,1}, j=char, i=relation
    uint32_t k = 0;                             ///< number of character columns
};

/// Selects auxiliary primes and computes Legendre-symbol character columns.
class CharacterColumnComputer {
public:
    /// Select k auxiliary primes satisfying (N|q)=+1, q not in FB.
    /// @param N         The composite being factored.
    /// @param fb        Factor base primes.
    /// @param k         Number of character columns (default 32).
    void selectAuxPrimes(const uint512& N,
                         const std::vector<uint32_t>& fb,
                         uint32_t k = 32);

    /// Compute character column values for a batch of relations.
    /// Q_i = sqrt_Q_i^2 - N for each relation. Legendre symbol (Q_i | q_j) encoded as GF(2).
    /// N mod q is precomputed by selectAuxPrimes().
    /// @param batch     Relations (uses sqrt_Q values).
    /// @return CharacterColumns with k columns, each of length batch.num_relations.
    CharacterColumns compute(const structures::HostRelationBatch& batch) const;

    /// Access selected primes (for logging/debugging).
    const std::vector<uint32_t>& auxPrimes() const { return aux_primes_; }

    /// Access precomputed N mod q values (for GPU dispatch in M8c).
    const std::vector<uint32_t>& nModQ() const { return n_mod_q_; }

private:
    std::vector<uint32_t> aux_primes_;
    std::vector<uint32_t> n_mod_q_;  ///< precomputed N mod q for each aux prime
};

/// Append character columns to an existing CSR matrix.
/// Character column indices start at csr.n_cols (i.e., after all existing columns).
/// For row i, appends column (base + j) for each character j where columns[j][i] == 1.
/// @param csr       Matrix to modify (n_cols increased by chars.k).
/// @param chars     Character column data.
/// @param n_rels    Number of relations (must match csr.n_rows).
void AppendCharacterColumns(HostMatrixCSR& csr, const CharacterColumns& chars,
                            uint32_t n_rels);

/// Concatenate two CharacterColumns (smooth + partial) into one.
/// Both must have the same k and aux_primes.
/// @return Combined CharacterColumns with columns of length a.columns[j].size() + b.columns[j].size().
CharacterColumns ConcatenateCharacterColumns(const CharacterColumns& a,
                                              const CharacterColumns& b);

/// Compute character columns for a reduced (post-preprocessing) matrix.
/// For each reduced row, expands the merge tree to leaf indices, computes
/// the product of leaf sqrt_Q values mod N, and evaluates Legendre symbols.
/// The mod-N product introduces non-linear information not present in the
/// GF(2) matrix, making these columns genuinely independent (unlike pre-
/// preprocessing char cols which are redundant in the expanded matrix).
///
/// @param row_map        Reduced row -> merge tree node (from MergeResult).
/// @param tree           Merge tree from preprocessing.
/// @param smooth_batch   Raw smooth relations (leaves 0..ns-1).
/// @param partial_batch  Raw partial relations (leaves ns..ns+np-1).
/// @param ns             Number of smooth relations.
/// @param N              The composite being factored.
/// @param fb             Factor base primes (for aux prime selection).
/// @param k              Number of character columns (default 32).
/// @return CharacterColumns with k columns, each of length row_map.size().
CharacterColumns computeProductCharacterColumns(
    const std::vector<uint32_t>& row_map,
    const MergeTree& tree,
    const structures::HostRelationBatch& smooth_batch,
    const structures::HostRelationBatch& partial_batch,
    size_t ns,
    const uint512& N,
    const std::vector<uint32_t>& fb,
    uint32_t k = 32);

/// GPU-accelerated standard character column computation (M8c).
/// Drop-in replacement for CharacterColumnComputer::compute() in the legacy matrix path.
/// Requires aux_primes and n_mod_q from CharacterColumnComputer::auxPrimes()/nModQ().
/// Output is bit-identical to CharacterColumnComputer::compute() for all inputs.
CharacterColumns gpuComputeCharacterColumns(
    const structures::HostRelationBatch& batch,
    const std::vector<uint32_t>& aux_primes,
    const std::vector<uint32_t>& n_mod_q);

/// GPU character columns from device-resident sqrt_Q (M9d).
/// Eliminates D→H→D round-trip for sqrt_Q in legacy MatrixStage path.
/// @param d_sqrt_Q     Device pointer to sqrt_Q array (from RelationBatchView).
/// @param n_rels       Number of relations.
/// @param aux_primes   k auxiliary primes (host vector).
/// @param n_mod_q      N mod q for each aux prime (host vector).
/// @return CharacterColumns identical to the HostRelationBatch overload.
CharacterColumns gpuComputeCharacterColumns_device(
    const uint512* d_sqrt_Q,
    uint32_t n_rels,
    const std::vector<uint32_t>& aux_primes,
    const std::vector<uint32_t>& n_mod_q);

}  // namespace matrix
}  // namespace mpqs
