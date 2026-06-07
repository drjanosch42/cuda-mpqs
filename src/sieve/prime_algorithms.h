// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <vector>
#include <cstdint>
#include "uint512.cuh"
#include "common.h" // Defines factoringData

namespace mpqs {
namespace sieve {

// ============================================================================
// Modular Arithmetic (Wrappers around mpqs::math or specific implementations)
// ============================================================================

// Note: If these are generic, they could move to mpqs::math.
// If specific to setting up the factor base, they stay here.
uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t modulus);
int jacobi(uint32_t a, uint32_t n);
uint32_t Tonelli_Shanks(uint32_t n, uint32_t p);
uint32_t findRoot(const mpqs::uint512& d, const mpqs::uint512& N, uint32_t p);

// ----------------------------------------------------------------------------
// 64-bit, overflow-safe number-theory primitives (for branch-fixed character
// columns; aux primes are chosen > lp1_bound ~1e11, so they exceed uint32_t).
// All modular steps route through mpqs::math::{mul_mod,pow_mod} (u128-safe).
// ----------------------------------------------------------------------------
uint64_t Tonelli_Shanks_u64(uint64_t n, uint64_t p);
int jacobi_u64(uint64_t a, uint64_t n);
bool is_prime_u64(uint64_t n);

// Hensel Lifting is very specific to setting up roots for powers of primes
uint64_t liftRoot(uint64_t r, uint64_t reduced_N, uint32_t p);

uint32_t modInv(uint32_t a, uint32_t p);
uint64_t modInvSquare(uint64_t a, uint32_t p);

// ============================================================================
// Sieve setup functions
// ============================================================================

void generateFactorBase(factoringData* fData);
void determineParams(factoringData* fData);
void generateHypercubePath(int* direction, int8_t* sign, int dim);

// ============================================================================
// Polynomial Selection
// ============================================================================

void init_a_factors(factoringData* fData);
void advance_a_factors(factoringData* fData, int steps);
void advance(std::vector<uint32_t>& indices, int pivotPrime, int direction, int stepSize);
void recalc_a(factoringData* fData);
/**
 * @brief Generates the flattened array of factor indices for the next 'batch_size' steps.
 *
 * This function advances the Host state of fData exactly as if the steps were executed,
 * recording the configuration of 'a_factors' at each step.
 *
 * @param fData Pointer to the factoring state.
 * @param batch_size Number of steps to generate.
 * @return std::vector<uint32_t> Flattened vector of size [batch_size * shc_dim].
 */
std::vector<uint32_t> prepareNextBatchIndices(factoringData* fData, int batch_size);

} // namespace sieve
} // namespace mpqs
