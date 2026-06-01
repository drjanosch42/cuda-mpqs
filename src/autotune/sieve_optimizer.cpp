// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "sieve_optimizer.h"
#include "orchestrator.h"   // MPQSConfig
#include "hpc_logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>

namespace mpqs::autotune {

SieveParameterOptimizer::SieveParameterOptimizer()
    : cfg_() {}

SieveParameterOptimizer::SieveParameterOptimizer(Config cfg)
    : cfg_(cfg) {}

// ---------------------------------------------------------------------------
// Golden section search on a single axis
// ---------------------------------------------------------------------------

std::pair<uint64_t, double>
SieveParameterOptimizer::goldenSectionSearch(
    uint64_t lo, uint64_t hi, uint64_t quantize,
    std::function<double(uint64_t)> eval, uint32_t max_probes)
{
    constexpr double PHI = 0.6180339887498949;  // (sqrt(5) - 1) / 2

    // Quantize bounds to grid
    auto round_q = [quantize](uint64_t v) -> uint64_t {
        return (v / quantize) * quantize;
    };
    lo = round_q(lo);
    hi = round_q(hi);
    if (lo >= hi) { return {lo, eval(lo)}; }

    uint64_t x1 = round_q(hi - static_cast<uint64_t>(PHI * (hi - lo)));
    uint64_t x2 = round_q(lo + static_cast<uint64_t>(PHI * (hi - lo)));
    double f1 = eval(x1);
    double f2 = eval(x2);
    uint32_t probes_used = 2;

    while (probes_used < max_probes && (hi - lo) > quantize) {
        if (f1 < f2) {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = round_q(hi - static_cast<uint64_t>(PHI * (hi - lo)));
            f1 = eval(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = round_q(lo + static_cast<uint64_t>(PHI * (hi - lo)));
            f2 = eval(x2);
        }
        probes_used++;
    }

    // Return the better of the two interior points
    return (f1 < f2) ? std::pair{x1, f1} : std::pair{x2, f2};
}

// ---------------------------------------------------------------------------
// Probe with cache
// ---------------------------------------------------------------------------

RuntimeEstimate SieveParameterOptimizer::cachedEstimate(
    const mpqs::MPQSConfig& base_config,
    uint32_t fb_bound, uint32_t sieve_bound, uint64_t lp1_bound)
{
    ParamKey key{fb_bound, sieve_bound, lp1_bound};
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;

    auto est = estimateRuntime(base_config, fb_bound, sieve_bound, lp1_bound,
                               cfg_.truncation_frac, cfg_.eta_convergence_threshold,
                               cfg_.min_eta_samples);
    cache_[key] = est;
    total_probes_++;
    return est;
}

// ---------------------------------------------------------------------------
// Joint (F,L) optimizer helpers
// ---------------------------------------------------------------------------

uint32_t SieveParameterOptimizer::quantize_F(uint32_t f) {
    return (f / 1000) * 1000;
}

uint64_t SieveParameterOptimizer::quantize_L(uint64_t l) {
    return (l / 1'000'000) * 1'000'000;
}

uint64_t SieveParameterOptimizer::L_min(uint32_t F) {
    return std::max(uint64_t{1'000'000}, static_cast<uint64_t>(F) * 10);
}

bool SieveParameterOptimizer::checkBudget() {
    if (timed_out_) return false;
    if (total_probes_ >= cfg_.max_total_probes) return false;
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - t_wall_start_).count();
    if (elapsed >= cfg_.wall_clock_timeout_sec) {
        LOG(LOG_INFO) << "SieveParameterOptimizer: wall-clock budget exhausted ("
                      << std::fixed << std::setprecision(1) << elapsed << "s)";
        timed_out_ = true;
        return false;
    }
    return true;
}

uint32_t SieveParameterOptimizer::computeFMaxEffective(
    const SieveSearchBounds& bounds) const
{
    if (cfg_.F_heuristic == 0) return bounds.fb_hi;
    uint32_t f_cap = static_cast<uint32_t>(cfg_.f_max_multiplier * cfg_.F_heuristic);
    return std::min(bounds.fb_hi, f_cap);
}

// ---------------------------------------------------------------------------
// Phase A: 1D L-sweep at fixed F
// ---------------------------------------------------------------------------

std::pair<uint64_t, double> SieveParameterOptimizer::sweepL(
    const MPQSConfig& base_config,
    uint32_t F, uint32_t M, uint64_t L0,
    const SieveSearchBounds& bounds)
{
    uint32_t probes_before = total_probes_;

    // F² ceiling for LP constraint, further capped by bounds.lp_hi
    uint64_t f_sq = std::min(
        static_cast<uint64_t>(F) * static_cast<uint64_t>(F),
        bounds.lp_hi);

    // Step 1: Evaluate starting point
    auto est0 = cachedEstimate(base_config, F, M, L0);
    uint64_t L_best = L0;
    double T_best = est0.total_est_sec;

    // Step 2: Always evaluate L=0 (LP disabled) if L0 != 0
    if (L0 != 0 && checkBudget()) {
        double T_noLP = cachedEstimate(base_config, F, M, 0).total_est_sec;
        if (T_noLP < T_best) {
            L_best = 0;
            T_best = T_noLP;
        }
    }

    // Step 3: If starting from L=0, try canonical LP bound F×50
    if (L0 == 0 && checkBudget()) {
        uint64_t L_trial = quantize_L(static_cast<uint64_t>(F) * 50);
        L_trial = std::min(L_trial, f_sq);
        if (L_trial >= L_min(F)) {
            double T_trial = cachedEstimate(base_config, F, M, L_trial).total_est_sec;
            if (T_trial < T_best) {
                L0 = L_trial;       // pivot to LP-active starting point
                L_best = L_trial;
                T_best = T_trial;
            } else {
                // LP doesn't help at this F
                phase_a_probes_ = total_probes_ - probes_before;
                return {L_best, T_best};
            }
        }
    }

    // If L_best is still 0 and L0 was 0, LP exploration failed — return
    if (L_best == 0 && L0 == 0) {
        phase_a_probes_ = total_probes_ - probes_before;
        return {L_best, T_best};
    }

    // Step 4: Directional sweep from L_best
    // Try increasing L first
    int direction = 0;  // +1 = increasing, -1 = decreasing
    if (L_best > 0 && checkBudget()) {
        uint64_t L_up = quantize_L(static_cast<uint64_t>(L_best * cfg_.alpha));
        L_up = std::min(L_up, f_sq);
        if (L_up > L_best && L_up <= f_sq) {
            double T_up = cachedEstimate(base_config, F, M, L_up).total_est_sec;
            if (T_up < T_best) {
                direction = +1;
                L_best = L_up;
                T_best = T_up;
            } else {
                // Try decreasing
                uint64_t L_down = quantize_L(static_cast<uint64_t>(L_best / cfg_.alpha));
                L_down = std::max(L_down, L_min(F));
                if (L_down < L_best && checkBudget()) {
                    double T_down = cachedEstimate(base_config, F, M, L_down).total_est_sec;
                    if (T_down < T_best) {
                        direction = -1;
                        L_best = L_down;
                        T_best = T_down;
                    } else {
                        // Local minimum found at L_best
                        phase_a_probes_ = total_probes_ - probes_before;
                        return {L_best, T_best};
                    }
                } else {
                    phase_a_probes_ = total_probes_ - probes_before;
                    return {L_best, T_best};
                }
            }
        }
    }

    // Step 5: Continue in chosen direction with exponentially growing steps
    if (direction != 0) {
        uint32_t step_exp = 2;  // next step is alpha^2
        while (checkBudget()) {
            uint64_t L_next;
            if (direction == +1) {
                L_next = quantize_L(static_cast<uint64_t>(
                    L_best * std::pow(cfg_.alpha, step_exp)));
                L_next = std::min(L_next, f_sq);
                if (L_next <= L_best) break;  // hit constraint ceiling
            } else {
                double divisor = std::pow(cfg_.alpha, step_exp);
                L_next = quantize_L(static_cast<uint64_t>(L_best / divisor));
                L_next = std::max(L_next, L_min(F));
                if (L_next >= L_best) break;  // hit floor
            }

            double T_next = cachedEstimate(base_config, F, M, L_next).total_est_sec;
            if (T_next < T_best) {
                L_best = L_next;
                T_best = T_next;
                step_exp++;  // accelerate
            } else {
                // Convexity: passed the optimum, stop
                break;
            }
        }
    }

    phase_a_probes_ = total_probes_ - probes_before;
    return {L_best, T_best};
}

// ---------------------------------------------------------------------------
// Phase B: 3×3 grid exploration
// ---------------------------------------------------------------------------

std::tuple<uint32_t, uint64_t, double> SieveParameterOptimizer::exploreGrid(
    const MPQSConfig& base_config,
    uint32_t F_center, uint64_t L_center, uint32_t M,
    const SieveSearchBounds& bounds,
    double delta_scale,
    Grid3x3& grid_out)
{
    uint32_t probes_before = total_probes_;

    // --- Build F axis: [F/eff_alpha, F, F*eff_alpha] ---
    // effective_alpha = 1 + (alpha - 1) * delta_scale
    //   At delta_scale=1.0: eff_alpha = alpha (full resolution)
    //   At delta_scale<1.0: eff_alpha shrinks toward 1.0 (finer grid)
    double eff_alpha = 1.0 + (cfg_.alpha - 1.0) * delta_scale;
    std::array<uint32_t, 3> F_vals;
    F_vals[0] = quantize_F(static_cast<uint32_t>(F_center / eff_alpha));
    F_vals[1] = F_center;
    F_vals[2] = quantize_F(static_cast<uint32_t>(F_center * eff_alpha));

    // Clamp to search bounds
    for (auto& f : F_vals) {
        f = std::clamp(f, bounds.fb_lo, F_max_effective_);
        f = quantize_F(f);  // re-quantize after clamping
    }

    // --- Build L axis ---
    std::array<uint64_t, 3> L_vals;
    if (L_center == 0) {
        // LP disabled: grid explores L=0 row + one LP row at F×50
        uint64_t L_trial = quantize_L(static_cast<uint64_t>(F_center) * 50);
        L_vals[0] = 0;
        L_vals[1] = L_trial;
        L_vals[2] = quantize_L(static_cast<uint64_t>(L_trial * eff_alpha));
    } else {
        L_vals[0] = quantize_L(static_cast<uint64_t>(L_center / eff_alpha));
        L_vals[1] = L_center;
        L_vals[2] = quantize_L(static_cast<uint64_t>(L_center * eff_alpha));

        // Enforce L >= L_min for nonzero values
        for (auto& l : L_vals) {
            if (l > 0 && l < L_min(F_center)) l = L_min(F_center);
        }
    }

    // --- Evaluate 3×3 grid ---
    uint32_t F_best = F_center;
    uint64_t L_best = L_center;
    double T_best = std::numeric_limits<double>::infinity();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            uint32_t F_ij = F_vals[i];
            uint64_t L_ij = L_vals[j];

            // Enforce L ≤ F² constraint per-F (F varies across columns)
            uint64_t f_sq = static_cast<uint64_t>(F_ij) * static_cast<uint64_t>(F_ij);
            L_ij = std::min(L_ij, f_sq);
            L_ij = quantize_L(L_ij);

            // Snap sub-minimum nonzero L to 0
            if (L_ij > 0 && L_ij < L_min(F_ij)) L_ij = 0;

            // Evaluate (reuses cached Phase A probes at F=F_center for free)
            double T_ij = std::numeric_limits<double>::infinity();
            if (checkBudget()) {
                T_ij = cachedEstimate(base_config, F_ij, M, L_ij).total_est_sec;
            }

            grid_out[i][j] = {F_ij, L_ij, T_ij};

            if (T_ij < T_best) {
                F_best = F_ij;
                L_best = L_ij;
                T_best = T_ij;
            }
        }
    }

    // --- Optional: edge extension in L direction ---
    // If the best in a non-center F column is at the L-edge, probe one step beyond.
    for (int i : {0, 2}) {  // non-center F columns
        if (!checkBudget()) break;

        // Find best L in this column
        int j_best = 0;
        double T_col_best = grid_out[i][0].T;
        for (int j = 1; j < 3; ++j) {
            if (grid_out[i][j].T < T_col_best) {
                T_col_best = grid_out[i][j].T;
                j_best = j;
            }
        }

        uint32_t F_col = F_vals[i];
        uint64_t f_sq = static_cast<uint64_t>(F_col) * static_cast<uint64_t>(F_col);

        if (j_best == 0 && L_vals[0] > L_min(F_col)) {
            // Minimum at lower L edge — probe one step lower
            uint64_t L_ext = quantize_L(static_cast<uint64_t>(L_vals[0] / eff_alpha));
            L_ext = std::max(L_ext, L_min(F_col));
            L_ext = std::min(L_ext, f_sq);
            if (L_ext < L_vals[0] && checkBudget()) {
                double T_ext = cachedEstimate(base_config, F_col, M, L_ext).total_est_sec;
                if (T_ext < T_best) {
                    F_best = F_col; L_best = L_ext; T_best = T_ext;
                }
            }
        } else if (j_best == 2 && L_vals[2] < f_sq) {
            // Minimum at upper L edge — probe one step higher
            uint64_t L_ext = quantize_L(static_cast<uint64_t>(L_vals[2] * eff_alpha));
            L_ext = std::min(L_ext, f_sq);
            if (L_ext > L_vals[2] && checkBudget()) {
                double T_ext = cachedEstimate(base_config, F_col, M, L_ext).total_est_sec;
                if (T_ext < T_best) {
                    F_best = F_col; L_best = L_ext; T_best = T_ext;
                }
            }
        }
    }

    phase_b_probes_ += (total_probes_ - probes_before);
    return {F_best, L_best, T_best};
}

// ---------------------------------------------------------------------------
// Phase C: gradient descent over (F,L) with grid-estimated gradients
// ---------------------------------------------------------------------------

std::pair<uint32_t, uint64_t> SieveParameterOptimizer::gradientDescent(
    const MPQSConfig& base_config,
    uint32_t F_start, uint64_t L_start, uint32_t M,
    const SieveSearchBounds& bounds,
    Grid3x3& grid)
{
    uint32_t probes_before = total_probes_;

    uint32_t F_cur = F_start;
    uint64_t L_cur = L_start;
    double T_best = cachedEstimate(base_config, F_cur, M, L_cur).total_est_sec;

    double eta = cfg_.eta0;         // learning rate
    double delta_scale = 1.0;       // grid resolution scale factor

    // Gradient norm averaging window (ring buffer)
    std::deque<double> grad_norms;

    bool converged = false;
    uint32_t iters = 0;

    for (uint32_t iter = 0; iter < cfg_.max_gd_iterations; ++iter) {
        if (!checkBudget()) break;

        // --- Step 1: Estimate gradient from current grid ---
        // Central differences averaged over rows/columns
        double dT_dF = 0.0;
        int n_F = 0;
        for (int j = 0; j < 3; ++j) {
            double DF = static_cast<double>(grid[2][j].F) - static_cast<double>(grid[0][j].F);
            double DT = grid[2][j].T - grid[0][j].T;
            if (DF > 0) {
                dT_dF += DT / DF;
                ++n_F;
            }
        }
        if (n_F > 0) dT_dF /= n_F;

        double dT_dL = 0.0;
        int n_L = 0;
        for (int i = 0; i < 3; ++i) {
            double DL = static_cast<double>(grid[i][2].L) - static_cast<double>(grid[i][0].L);
            double DT = grid[i][2].T - grid[i][0].T;
            if (DL > 0) {
                dT_dL += DT / DL;
                ++n_L;
            }
        }
        if (n_L > 0) dT_dL /= n_L;

        // Convert to log-space (dimensionless) gradient
        double g_F = static_cast<double>(F_cur) * dT_dF;
        double g_L = (L_cur > 0) ? static_cast<double>(L_cur) * dT_dL : 0.0;
        double g_norm = std::sqrt(g_F * g_F + g_L * g_L);

        // --- Step 2: Update gradient averaging window ---
        grad_norms.push_back(g_norm);
        if (grad_norms.size() > cfg_.grad_avg_window)
            grad_norms.pop_front();

        // --- Step 3: Check convergence ---
        if (grad_norms.size() >= cfg_.grad_avg_window && T_best > 0) {
            double avg_grad = 0.0;
            for (double gn : grad_norms) avg_grad += gn;
            avg_grad /= grad_norms.size();

            if (avg_grad / T_best < cfg_.convergence_epsilon) {
                converged = true;
                iters = iter + 1;
                break;
            }
        }

        // --- Step 4: Compute step (negative gradient in log-space) ---
        if (g_norm < 1e-12) {
            converged = true;
            iters = iter + 1;
            break;  // zero gradient, at minimum
        }

        double F_new_d = F_cur * std::exp(-eta * g_F / g_norm);
        uint32_t F_new = quantize_F(static_cast<uint32_t>(std::round(F_new_d)));
        F_new = std::clamp(F_new, bounds.fb_lo, F_max_effective_);

        uint64_t L_new;
        if (L_cur > 0) {
            double L_new_d = L_cur * std::exp(-eta * g_L / g_norm);
            L_new = quantize_L(static_cast<uint64_t>(std::round(L_new_d)));
        } else {
            L_new = 0;  // stay at L=0 if currently there
        }

        // --- Step 5: Enforce constraints ---
        uint64_t f_sq_new = static_cast<uint64_t>(F_new) * static_cast<uint64_t>(F_new);
        L_new = std::min(L_new, f_sq_new);
        if (L_new > 0 && L_new < L_min(F_new)) L_new = 0;

        // --- Step 6: Evaluate candidate ---
        if (!checkBudget()) { iters = iter + 1; break; }
        double T_new = cachedEstimate(base_config, F_new, M, L_new).total_est_sec;

        if (T_new < T_best) {
            F_cur = F_new;
            L_cur = L_new;
            T_best = T_new;
        } else {
            // Step did not improve: halve learning rate, keep position
            eta *= 0.5;
        }

        // --- Step 7: Rebuild grid around new position ---
        delta_scale *= cfg_.alpha_decay;

        if (!checkBudget()) { iters = iter + 1; break; }
        exploreGrid(base_config, F_cur, L_cur, M, bounds, delta_scale, grid);

        // --- Step 8: Decay learning rate ---
        eta *= cfg_.lambda;

        iters = iter + 1;
    }

    phase_c_probes_ = total_probes_ - probes_before;
    phase_c_iters_ = iters;
    phase_c_converged_ = converged;
    return {F_cur, L_cur};
}

// ---------------------------------------------------------------------------
// Joint (F,L) optimizer
// ---------------------------------------------------------------------------

SieveOptimizationResult SieveParameterOptimizer::optimize(
    const MPQSConfig& base_config,
    std::tuple<uint32_t, uint32_t, uint64_t> warm_start,
    const SieveSearchBounds& bounds)
{
    auto [F, M, L] = warm_start;
    t_wall_start_ = std::chrono::high_resolution_clock::now();
    timed_out_ = false;
    total_probes_ = 0;
    cache_.clear();
    phase_a_probes_ = 0;
    phase_b_probes_ = 0;
    phase_c_probes_ = 0;
    phase_c_iters_ = 0;

    // =====================================================================
    // Initialization
    // =====================================================================

    // Compute effective F upper bound (F runaway prevention)
    F_max_effective_ = computeFMaxEffective(bounds);

    // Clamp warm-start F to bounds
    F = std::clamp(F, bounds.fb_lo, F_max_effective_);
    F = quantize_F(F);

    // Clamp warm-start L: enforce L ≤ F², clamp to bounds, quantize
    {
        uint64_t f_sq = static_cast<uint64_t>(F) * static_cast<uint64_t>(F);
        L = std::min(L, f_sq);
        L = std::min(L, bounds.lp_hi);
        L = quantize_L(L);
    }

    // =====================================================================
    // M Optimization (unchanged from current algorithm)
    // =====================================================================
    // Enumerate M candidates at the warm-start (F, L), fix M at best.
    // Round 1: all powers of 2 in [M_min_log2, M_max_log2].

    {
        auto best_est = cachedEstimate(base_config, F, M, L);
        double T_M_best = best_est.total_est_sec;

        for (uint32_t k = bounds.M_min_log2; k <= bounds.M_max_log2; ++k) {
            uint32_t Mc = 1u << k;
            if (Mc == M) continue;  // already evaluated
            if (!checkBudget()) break;
            double Tc = cachedEstimate(base_config, F, Mc, L).total_est_sec;
            if (Tc < T_M_best) {
                M = Mc;
                T_M_best = Tc;
            }
        }
    }

    // =====================================================================
    // Phase A: 1D L-sweep at fixed F
    // =====================================================================

    auto [L_star, T_after_A] = sweepL(base_config, F, M, L, bounds);
    L = L_star;

    LOG(LOG_DEBUG_1) << "[Autotune] Phase A: L*=" << L
                     << " T=" << std::fixed << std::setprecision(2) << T_after_A << "s"
                     << " (" << phase_a_probes_ << " probes)";

    // =====================================================================
    // Phase B: 3×3 grid exploration
    // =====================================================================

    Grid3x3 grid{};
    auto [F_B, L_B, T_B] = exploreGrid(
        base_config, F, L, M, bounds, 1.0 /*delta_scale*/, grid);
    F = F_B;
    L = L_B;
    double T_best = T_B;

    LOG(LOG_DEBUG_1) << "[Autotune] Phase B: F=" << F << " L=" << L
                     << " T=" << std::fixed << std::setprecision(2) << T_best << "s"
                     << " (" << phase_b_probes_ << " probes)";

    // =====================================================================
    // Warm-start adaptation: decide whether to skip Phase C
    // =====================================================================

    bool skip_phase_c = false;
    {
        // Estimate initial gradient from Phase B grid
        double dT_dF = 0.0;
        int n_F = 0;
        for (int j = 0; j < 3; ++j) {
            double DF = static_cast<double>(grid[2][j].F) - static_cast<double>(grid[0][j].F);
            double DT = grid[2][j].T - grid[0][j].T;
            if (DF > 0) { dT_dF += DT / DF; ++n_F; }
        }
        if (n_F > 0) dT_dF /= n_F;

        double dT_dL = 0.0;
        int n_L = 0;
        for (int i = 0; i < 3; ++i) {
            double DL = static_cast<double>(grid[i][2].L) - static_cast<double>(grid[i][0].L);
            double DT = grid[i][2].T - grid[i][0].T;
            if (DL > 0) { dT_dL += DT / DL; ++n_L; }
        }
        if (n_L > 0) dT_dL /= n_L;

        double g_F = static_cast<double>(F) * dT_dF;
        double g_L = (L > 0) ? static_cast<double>(L) * dT_dL : 0.0;
        double g_norm = std::sqrt(g_F * g_F + g_L * g_L);
        double relative_grad = (T_best > 0) ? g_norm / T_best : 0.0;

        double conf = cfg_.warm_start_confidence;

        if (conf >= 0.8 && relative_grad < 0.05) {
            // Very close to optimum: skip Phase C entirely
            skip_phase_c = true;
            LOG(LOG_DEBUG_1) << "[Autotune] Warm-start: skipping Phase C"
                             << " (conf=" << std::fixed << std::setprecision(2) << conf
                             << ", rel_grad=" << relative_grad << ")";
        } else if (conf >= 0.5 && relative_grad < 0.10) {
            // Near optimum: reduce iterations and step size
            cfg_.max_gd_iterations = 3;
            cfg_.eta0 = 0.2;
            cfg_.alpha = 1.1;
            LOG(LOG_DEBUG_1) << "[Autotune] Warm-start: reduced Phase C"
                             << " (conf=" << std::fixed << std::setprecision(2) << conf
                             << ", rel_grad=" << relative_grad << ")";
        }
        // else: full exploration with defaults
    }

    // =====================================================================
    // Phase C: gradient descent (may be skipped)
    // =====================================================================

    bool converged = skip_phase_c;  // if skipped, consider converged

    if (!skip_phase_c && checkBudget()) {
        auto [F_C, L_C] = gradientDescent(
            base_config, F, L, M, bounds, grid);
        double T_C = cachedEstimate(base_config, F_C, M, L_C).total_est_sec;
        if (T_C < T_best) {
            F = F_C;
            L = L_C;
            T_best = T_C;
        }
        // Check if Phase C converged (gradient-based exit, not budget/timeout)
        converged = phase_c_converged_;

        LOG(LOG_DEBUG_1) << "[Autotune] Phase C: F=" << F << " L=" << L
                         << " T=" << std::fixed << std::setprecision(2) << T_best << "s"
                         << " (" << phase_c_probes_ << " probes, "
                         << phase_c_iters_ << " iters, "
                         << (converged ? "converged" : "budget/timeout") << ")";
    }

    // =====================================================================
    // Final safety: enforce all constraints
    // =====================================================================

    F = std::clamp(F, bounds.fb_lo, F_max_effective_);
    F = quantize_F(F);
    {
        uint64_t f_sq = static_cast<uint64_t>(F) * static_cast<uint64_t>(F);
        L = std::min(L, f_sq);
    }
    L = quantize_L(L);

    // =====================================================================
    // Build result
    // =====================================================================

    auto now = std::chrono::high_resolution_clock::now();
    double wall_sec = std::chrono::duration<double>(now - t_wall_start_).count();

    SieveOptimizationResult result;
    result.optimal_fb_bound = F;
    result.optimal_sieve_bound = M;
    result.optimal_lp1_bound = L;
    result.estimate = cache_[{F, M, L}];
    result.rounds_completed = 1;  // always 1 "round" in joint optimizer
    result.total_probes = total_probes_;
    result.wall_clock_sec = wall_sec;
    result.converged = converged;

    // Phase-level metadata
    result.phase_a_probes = phase_a_probes_;
    result.phase_b_probes = phase_b_probes_;
    result.phase_c_probes = phase_c_probes_;
    result.phase_c_iterations = phase_c_iters_;
    result.phase_c_skipped = skip_phase_c;

    // Probe history
    for (auto& [key, est] : cache_)
        result.probe_history.push_back(est);

    // Confidence penalty if not converged
    if (!converged)
        result.estimate.confidence *= 0.5;

    return result;
}

} // namespace mpqs::autotune
