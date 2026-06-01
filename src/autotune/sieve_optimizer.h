// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "autotune_types.h"
#include "runtime_estimator.h"

#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <tuple>
#include <vector>

namespace mpqs { struct MPQSConfig; }

namespace mpqs::autotune {

/// Single grid point for Phase B/C 3×3 grid exploration.
struct GridPoint {
    uint32_t F = 0;
    uint64_t L = 0;
    double T = 0.0;
};

/// 3×3 grid of (F, L, T) triples for gradient estimation.
using Grid3x3 = std::array<std::array<GridPoint, 3>, 3>;

/// Joint (F,L) optimizer over (fb_bound, sieve_bound, lp1_bound).
/// Minimizes total estimated pipeline time by probing parameter sets
/// via estimateRuntime(). Three phases: L-sweep (A), 3×3 grid (B),
/// gradient descent (C). Falls back to coordinate descent for legacy callers.
///
/// Algorithm: Phase A sweeps L at fixed F to find best L. Phase B evaluates
/// a 3×3 grid around (F, L). Phase C performs gradient descent with
/// adaptive learning rate. Converges when gradient norm < epsilon.
class SieveParameterOptimizer {
public:
    struct Config {
        Config() = default;
        uint32_t max_rounds = 3;               ///< LEGACY: unused by joint optimizer (backward compat)
        uint32_t max_probes_per_axis = 6;      ///< LEGACY: unused by joint optimizer (backward compat)
        uint32_t max_total_probes = 40;        ///< Global probe budget (raised for joint optimizer)
        double convergence_epsilon = 0.02;     ///< 2% relative improvement
        double wall_clock_timeout_sec = 600.0; ///< 10 min hard cap

        // Truncation params forwarded to estimateRuntime
        double truncation_frac = 0.12;
        double eta_convergence_threshold = 0.05;
        uint32_t min_eta_samples = 3;

        // === Joint (F,L) optimizer fields ===
        double alpha = 1.2;                    ///< Multiplicative step factor for grid/sweep
        double eta0 = 0.5;                     ///< Initial gradient descent learning rate
        double lambda = 0.7;                   ///< Per-iteration learning rate decay
        double alpha_decay = 0.85;             ///< Per-iteration grid resolution decay
        uint32_t max_gd_iterations = 8;        ///< Max Phase C (gradient descent) iterations
        uint32_t grad_avg_window = 3;          ///< Gradient norm averaging window (K_avg)
        double f_max_multiplier = 2.0;         ///< F ≤ f_max_multiplier × F_heuristic
        uint32_t F_heuristic = 0;              ///< Heuristic F from determineParams(); 0 = no cap
        double warm_start_confidence = 0.0;    ///< History confidence, forwarded from Stage 0
    };

    SieveParameterOptimizer();
    explicit SieveParameterOptimizer(Config cfg);

    /// Run coordinate descent optimization.
    /// @param base_config  Base MPQSConfig (N, device_id, etc.)
    /// @param warm_start   Initial (fb_bound, sieve_bound, lp1_bound) from Stage 0 or heuristics
    /// @param bounds       Search bounds from deriveSieveSearchBounds()
    SieveOptimizationResult optimize(
        const mpqs::MPQSConfig& base_config,
        std::tuple<uint32_t, uint32_t, uint64_t> warm_start,
        const SieveSearchBounds& bounds);

private:
    using ParamKey = std::tuple<uint32_t, uint32_t, uint64_t>;

    /// Probe with caching. Returns cached result if (F_rounded, M, L_rounded) seen before.
    RuntimeEstimate cachedEstimate(
        const mpqs::MPQSConfig& base_config,
        uint32_t fb_bound, uint32_t sieve_bound, uint64_t lp1_bound);

    /// Golden section search on a single axis.
    /// Returns (optimal_value_rounded, T_at_optimum).
    /// @param lo, hi      Axis bounds
    /// @param quantize    Rounding step (1000 for F, 1000000 for L)
    /// @param eval        Lambda: uint64_t -> double (param value -> total_est_sec)
    /// @param max_probes  Max evaluations on this axis
    std::pair<uint64_t, double> goldenSectionSearch(
        uint64_t lo, uint64_t hi, uint64_t quantize,
        std::function<double(uint64_t)> eval,
        uint32_t max_probes);

    // --- Joint (F,L) optimizer helpers ---

    /// Quantize F to nearest multiple of 1000.
    static uint32_t quantize_F(uint32_t f);

    /// Quantize L to nearest multiple of 1,000,000.
    static uint64_t quantize_L(uint64_t l);

    /// Minimum nonzero LP bound: max(q_L, 10 * F).
    /// Below this, LP matching is useless (too few witnesses).
    static uint64_t L_min(uint32_t F);

    /// Check if probe budget or wall-clock timeout is exhausted.
    /// Sets timed_out_ flag if wall-clock exceeded.
    bool checkBudget();

    /// Compute effective F upper bound: min(bounds.fb_hi, f_max_multiplier * F_heuristic).
    /// Returns bounds.fb_hi if F_heuristic == 0 (no cap).
    uint32_t computeFMaxEffective(const SieveSearchBounds& bounds) const;

    // --- Phase A: 1D L-sweep at fixed F ---

    /// Sweep L at fixed F to find optimal LP bound.
    /// Evaluates L₀, L=0 baseline, canonical F×50, then directional sweep.
    /// Returns (L_best, T_best). All probes go through cachedEstimate().
    /// @param base_config  Base pipeline config for probes
    /// @param F            Fixed factor base bound
    /// @param M            Fixed sieve interval
    /// @param L0           Initial LP bound from warm-start
    /// @param bounds       Search bounds (provides lp_hi)
    std::pair<uint64_t, double> sweepL(
        const mpqs::MPQSConfig& base_config,
        uint32_t F, uint32_t M, uint64_t L0,
        const SieveSearchBounds& bounds);

    // --- Phase B: 3×3 grid exploration ---

    /// Build and evaluate a 3×3 grid around (F_center, L_center).
    /// Returns (F_best, L_best, T_best). grid_out is populated for Phase C gradient estimation.
    /// @param base_config   Base pipeline config for probes
    /// @param F_center      Center F value
    /// @param L_center      Center L value
    /// @param M             Fixed sieve interval
    /// @param bounds        Search bounds
    /// @param delta_scale   Grid resolution scale factor (1.0 initially, decays by alpha_decay)
    /// @param grid_out      Output: 3×3 grid of (F, L, T) triples
    std::tuple<uint32_t, uint64_t, double> exploreGrid(
        const mpqs::MPQSConfig& base_config,
        uint32_t F_center, uint64_t L_center, uint32_t M,
        const SieveSearchBounds& bounds,
        double delta_scale,
        Grid3x3& grid_out);

    // --- Phase C: gradient descent ---

    /// Iterative gradient descent over (F,L) using grid-estimated gradients.
    /// Rebuilds the 3×3 grid each iteration with decaying resolution.
    /// @param base_config  Base pipeline config for probes
    /// @param F_start      Starting F (from Phase B best)
    /// @param L_start      Starting L (from Phase B best)
    /// @param M            Fixed sieve interval
    /// @param bounds       Search bounds
    /// @param grid         Initial grid from Phase B (modified in place for gradient re-estimation)
    /// @return (F_opt, L_opt) locally optimal point
    std::pair<uint32_t, uint64_t> gradientDescent(
        const mpqs::MPQSConfig& base_config,
        uint32_t F_start, uint64_t L_start, uint32_t M,
        const SieveSearchBounds& bounds,
        Grid3x3& grid);

    Config cfg_;
    std::map<ParamKey, RuntimeEstimate> cache_;
    uint32_t total_probes_ = 0;

    std::chrono::high_resolution_clock::time_point t_wall_start_;
    bool timed_out_ = false;

    // Phase-level probe counters (populated during optimize())
    uint32_t phase_a_probes_ = 0;
    uint32_t phase_b_probes_ = 0;
    uint32_t phase_c_probes_ = 0;
    uint32_t phase_c_iters_ = 0;
    bool phase_c_converged_ = false;  ///< True if Phase C exited via gradient convergence

    // Effective F upper bound (computed once in optimize())
    uint32_t F_max_effective_ = 0;
};

} // namespace mpqs::autotune
