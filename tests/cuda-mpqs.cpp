// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/**
 * @file cuda-mpqs.cpp
 * @brief Driver for the CUDA-MPQS Factorization Pipeline.
 * 
 * This application configures and launches the full Multiple Polynomial Quadratic Sieve
 * pipeline to factor a large integer N. It validates the integration of:
 * 1. Parameter Tuning
 * 2. GPU Sieving
 * 3. Matrix Construction
 * 4. Block Wiedemann Linear Algebra
 * 5. Square Root Refinement
 * 
 * It verifies the result by multiplying the found factors.
 */

#include "orchestrator.h"
#include "hpc_logger.h"
#include "uint512.cuh"
#include "autotune_history.h"
#include "runtime_estimator.h"
#include "version.h"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <charconv>
#include <cstring>
#include <cstdlib>
#include <fstream>

using namespace mpqs;

// -----------------------------------------------------------------------------
// Argument Parsing
// -----------------------------------------------------------------------------

// Logging CLI flags — resolved into SinkConfig after parse_args()
struct LogFlags {
    int         log_level     = LOG_INFO;  // console severity threshold
    bool        mute          = false;
    std::string log_file;                  // empty = no file sink
    std::string error_log;                 // empty = no error file sink
    bool        log_date      = false;
    bool        log_no_time   = false;
    bool        log_no_stage  = false;
    bool        log_show_rank = false;
    int         log_wrap      = 120;       // 0 = disabled
    bool        log_csv       = false;
};

struct ParsedArgs {
    MPQSConfig config;

    bool estimate_only = false;
    LogFlags log_flags;

    // Tracking
    std::map<std::string, bool> set_flags;

    void mark(const std::string& name) {
        set_flags[name] = true;
    }
};

// Helper to safely parse uint32_t
bool parse_uint32(const char* str, uint32_t& value) {
    auto res = std::from_chars(str, str + std::strlen(str), value);
    if (res.ec == std::errc::invalid_argument) {
        std::cerr << "Error: '" << str << "' is not a valid number.\n";
        return false;
   } else if (res.ec == std::errc::result_out_of_range) {
        std::cerr << "Error: '" << str << "' is out of range for a 32-bit unsigned integer.\n";
        return false;
    }
    return true;
}

// Helper to safely parse uint64_t
bool parse_uint64(const char* str, uint64_t& value) {
    auto res = std::from_chars(str, str + std::strlen(str), value);
    if (res.ec == std::errc::invalid_argument) {
        std::cerr << "Error: '" << str << "' is not a valid number.\n";
        return false;
   } else if (res.ec == std::errc::result_out_of_range) {
        std::cerr << "Error: '" << str << "' is out of range for a 64-bit unsigned integer.\n";
        return false;
    }
    return true;
}

// Helper to safely parse double
bool parse_double(const char* str, double& value) {
    char* end = nullptr;
    value = std::strtod(str, &end);
    if (end == str || *end != '\0') {
        std::cerr << "Error: '" << str << "' is not a valid floating-point number.\n";
        return false;
    }
    return true;
}

// Helper to safely parse int (signed)
bool parse_int(const char* str, int& value) {
    auto res = std::from_chars(str, str + std::strlen(str), value);
    if (res.ec == std::errc::invalid_argument) {
        std::cerr << "Error: '" << str << "' is not a valid integer.\n";
        return false;
    } else if (res.ec == std::errc::result_out_of_range) {
        std::cerr << "Error: '" << str << "' is out of range.\n";
        return false;
    }
    return true;
}

/// Parse a numeric string with optional K/M/B/T suffix (base-1000).
/// If snap_pow2 is true, result is rounded to nearest power of 2.
/// Returns true on success.
bool parse_suffixed_uint64(const char* str, uint64_t& out, bool snap_pow2 = false) {
    char* end = nullptr;
    double val = std::strtod(str, &end);
    if (end == str) return false;  // no numeric part

    // Parse suffix
    if (*end == 'K' || *end == 'k') { val *= 1000; end++; }
    else if (*end == 'M' || *end == 'm') { val *= 1000000; end++; }
    else if (*end == 'B' || *end == 'b') { val *= 1000000000; end++; }
    else if (*end == 'T' || *end == 't') { val *= 1000000000000.0; end++; }

    if (*end != '\0') return false;  // trailing garbage
    if (val < 0 || val > 1e18) return false;  // out of range

    uint64_t result = static_cast<uint64_t>(val + 0.5);  // round

    if (snap_pow2 && result > 0) {
        // Snap to nearest power of 2
        uint64_t lo = 1;
        while (lo * 2 <= result) lo *= 2;
        uint64_t hi = lo * 2;
        result = (result - lo <= hi - result) ? lo : hi;
    }

    out = result;
    return true;
}

/// Parse a size string with optional K/M suffix (base-1024).
/// Returns parsed value. Prints error and calls exit(1) on failure.
uint64_t parse_size(const char* str) {
    uint64_t value = 0;
    auto [ptr, ec] = std::from_chars(str, str + std::strlen(str), value);
    if (ec == std::errc::invalid_argument || ptr == str) {
        std::cerr << "Error: '" << str << "' is not a valid size.\n";
        exit(1);
    }
    if (ec == std::errc::result_out_of_range) {
        std::cerr << "Error: '" << str << "' is out of range.\n";
        exit(1);
    }
    // Check for suffix
    if (*ptr == 'K' || *ptr == 'k') {
        value *= 1024ULL;
        ptr++;
    } else if (*ptr == 'M' || *ptr == 'm') {
        value *= 1024ULL * 1024ULL;
        ptr++;
    }
    if (*ptr != '\0') {
        std::cerr << "Error: '" << str << "' has invalid suffix (use K or M).\n";
        exit(1);
    }
    return value;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "\n--- Problem Definition ---\n"
              << "  --N <DEC>        Integer to factor (decimal string) [Default: ~80-digits]\n"
              << "  --RSA100         Sets N to RSA-100 (330 bits)\n"
              << "  --RSA110         Sets N to RSA-110 (364 bits)\n"
              << "  --RSA120         Sets N to RSA-120 (397 bits)\n"
              << "  --RSA129         Sets N to RSA-129 (426 bits)\n"
              << "  --RSA130         Sets N to RSA-130 (430 bits)\n"
              << "  --RSA140         Sets N to RSA-140 (463 bits)\n"
              << "  --RSA150         Sets N to RSA-150 (496 bits)\n"
              << "  --RSA155         Sets N to RSA-155 (512 bits)\n"
              << "  --device <ID>    GPU Device ID [Default: 0]\n"
              << "  --dir <PATH>     Working directory [Default: ./mpqs_work]\n"
              << "  --disk_io        Enable dumping relations to disk\n"
              << "  --dump_matrix    Dump finalized GF(2) matrix (matrix.csr + matrix_columns.txt) to <dir>\n"
              << "  --dump_kernel_vectors  Dump BW solutions (bw_*.bin, reduced-row space) and final\n"
              << "                         kernel_vectors.bin + .txt (original-relation space) to <dir>\n"
              << "  --dump_combine_provenance  Capture LP-combined relation constituents (probe/witness\n"
              << "                         roots, signs, exps, LP) to combine_provenance.bin in <dir>\n"
              << "\n--- Tuning / Sieving ---\n"
              << "  --fb_bound <N>   Factor Base Bound F (K/M/B/T suffix) [Default: Auto/20000]\n"
              << "  --lp1_bound <N>  Large Prime Bound 1 (K/M/B/T suffix)\n"
              << "  --lp1_max_witnesses <SIZE> LP witness capacity (K/M suffix, snaps to pow2) [Default: 1M]\n"
              << "  --lp_interval <N>    LP processing frequency: 0=auto/adaptive, N>0=every N batches\n"
              << "  --target_rels <N> Target Relations [Default: Auto]\n"
              << "  --dedup_safety_factor <F> Dedup oversample margin [Default: 1.05; auto 1.35 for <80d]\n"
              << "  --sieve_bound <N> Sieve Interval bound M (K/M suffix, snaps to pow2) [Default: Auto/65536]\n"
              << "  --params <p1,...,p8> Sieve with specified parameter tuple\n"
              << "\n--- Buffer Sizing (all accept K/M suffix, e.g. 512K, 4M) ---\n"
              << "  --accum_buf_size <SIZE>      Accumulation buffer capacity [Default: 512K]\n"
              << "  --partial_buf_size <SIZE>    Partial (LP staging) buffer [Default: 8x accum]\n"
              << "  --persistent_buf_size <SIZE> Persistent relation store [Default: target+accum]\n"
              << "  --lp1_combined_buf <SIZE>    LP match output buffer [Default: 32K]\n"
              << "  --lp1_hash_bits <N>          LP hash table directory bits [Default: auto]\n"
              << "\n--- Execution Modes ---\n"
              << "  --param_test     Run parameter test/exploration\n"
              << "  --full           Full Pipeline (Default)\n"
              << "  --sieve_only     Run Sieving only\n"
              << "  --linalg_only    Run Linear Algebra only (requires matrix)\n"
              << "  --sqrt_only      Run Sqrt step only (BROKEN: use --linalg_only instead)\n"
              << "  --matrix_only    Load v2 relations, run matrix preprocessing + BW + sqrt\n"
              << "  --matrix_mode <MODE>    Matrix construction: legacy, preprocess [Default: auto]\n"
              << "  --char_mode <MODE>      Character-column aux-prime selection: norm, branch, none [Default: none]\n"
              << "                            none = append ZERO character columns (scientific null control).\n"
              << "  --lp_preprocess_threshold <F>  DEPRECATED/INERT: AUTO no longer auto-selects preprocess from LP fraction; use --matrix_mode preprocess to opt in [Default: 0.55]\n"
              << "  --lp_matrix_threshold <F>  DEPRECATED: alias for --lp_preprocess_threshold (kept for backwards compatibility)\n"
              << "  --partial_subsample <F>  Subsample partials/LP-combined for matrix_only experiments [0.0-1.0, default: 1.0]\n"
              << "  --smooth_subsample <F>   Subsample pure smooths (LP-combined always kept) for matrix_only experiments [0.0-1.0, default: 1.0]\n"
              << "  --truncation_factor <F>  Matrix truncation enable flag (>0 enabled, 0 disabled) [Default: 1.05]\n"
              << "                            Actual target is excess-based; see --matrix_truncation_excess.\n"
              << "  --matrix_truncation_excess <N>  Excess rows above (n_cols + n_extra_cols) [Default: 200]\n"
              << "  --truncation_min_rows <N>  Skip preprocess truncation when reduced rows <= N [Default: 5000000].\n"
              << "                            Above N, truncation row-selection is a known limitation (can confine the kernel to the trivial subspace).\n"
              << "  --preprocess_lp_materialize_max <F>  Facet-3 gate: above this combined-smooth LP fraction, the\n"
              << "                            preprocess path skips materializing raw 1-partial 2-cycle rows (they\n"
              << "                            capture the genus character -> 0% nontrivial). [Default: 0.45]. Keeps\n"
              << "                            below-cliff partials (94d); 1.0 = never skip (pre-fix), 0.0 = always skip.\n"
              << "  --merge_max_weight <K>  DIAGNOSTIC: max column weight for higher-weight merges (preprocess\n"
              << "                            CPU path). Default 10 (no change). K=2 disables weight>=3 multi-cycle\n"
              << "                            merges, leaving singleton+weight-2 only (legacy-like 2-cycles).\n"
              << "  --force_preprocess      DIAGNOSTIC: force the preprocess expand+merge path even with 0\n"
              << "                            partials (else orchestrator force-legacies). Runs preprocess's\n"
              << "                            reduction on a smooths-only relation set. Default off.\n"
              << "  --matrix_gf2_floor_factor <F>   M12-S2: stop compact-merge when GF(2) cols fall below\n"
              << "                                  factor x initial_gf2_cols [0.0-1.0, default: 0.5]\n"
              << "  --matrix_gf2_min_floor <N>      M12-S2: absolute minimum GF(2) col floor [Default: 8192]\n"
              << "  --compact_cycles <N>    Max compact-merge cycles (GPU backend; default 5; 0=single pass)\n"
              << "  --matrix_backend <B>    Preprocessing backend: cpu, gpu, auto (gpu if available + >10K rows) [Default: cpu]\n"
              << "  --sqrt_legacy    Use CPU sqrt path (debug; default: GPU batched)\n"
              << "  --sqrt_diagnostic  Log extra sqrt diagnostics: HalveExponents validity, solution diversity\n"
              << "  --estimate_only  Run truncated sieve probe and print runtime estimate, then exit\n"
              << "\n--- Cluster Mode ---\n"
              << "  --cluster_mode <MODE>  Cluster mode: solo (default), coordinator, worker\n"
              << "  --coordinator_host <HOST>  Coordinator hostname (worker mode)\n"
              << "  --coordinator_port <PORT>  Coordinator port (default 9100)\n"
              << "  --listen_port <PORT>       Listen port (coordinator mode, default 9100)\n"
              << "  --expected_workers <N>     Number of workers to accept (coordinator, default 0)\n"
              << "  --cluster_init_timeout <SEC> Init window: worker retries + coordinator accept (default 300)\n"
              << "  --cluster_node_weights <W>  Comma-separated per-node weights (overrides SM*clock)\n"
              << "  --cluster_headroom <PCT>    Per-node headroom percent [Default: 10]\n"
              << "  --cluster_pool_oversize <X> a-value overflow-pool over-provisioning multiplier\n"
              << "                              (coordinator only; >1 enlarges on-demand pool) [Default: 1.0]\n"
              << "\n--- Sieve stage ---\n"
              << "  --probe_timeout <SEC>  Hard timeout for sieve probes (default 120.0)\n"
              << "  --sieve_batch_size <N> Sieving batch size (Default 0 = legacy sieving)\n"
              << "  --cuda_graph_unroll <N> Capture N batches as CUDA graph (0=disabled, default=0)\n"
              << "                          Must be even. Recommended: 2 or 4. Max: 16.\n"
              << "  --sieve_max_relations <N> Stop sieve after N relations (K/M/B/T suffix) [0=disabled]\n"
              << "  --sieve_max_batches <N>   Stop sieve after N batch iterations [0=disabled]\n"
              << "  --sieve_truncate_continue Continue pipeline (matrix/BW/sqrt) after truncation\n"
              << "  --sieve_gms_blocks <N> Number of CUDA blocks for metaSieve [Default: Auto/64]\n"
              << "  --sieve_hc_dim <N>     Hypercube dimension for polynomial construction\n"
              << "\n--- Block Wiedemann ---\n"
              << "  --bw_m <N>       Block size m [Default: 256]\n"
              << "  --bw_n <N>       Block size n [Default: 256]\n"
            #ifdef SIEVING_DEBUG_FLAG
              << "\n--- Debug (SIEVING_DEBUG_MODE) ---\n"
              << "  --metaSnapshot <k> Snapshot of metaSieve buckets at step k\n"
              << "  --metaP <idx>    Filter metaSnapshot to prime index idx\n"
              << "  --metaO <o>      Filter metaSnapshot to offset o\n"
              << "  --sasSnapshot <k> Snapshot of candidate relations at step k\n"
            #endif
              << "\n--- Autotune ---\n"
              << "  --autotune               Enable autotune before sieving\n"
              << "  --autotune_only          Run autotune only, print results, exit\n"
              << "  --autotune_stage0        Enable autotune stage 0 (overrides defaults)\n"
              << "  --autotune_stage1        Enable autotune stage 1 (overrides defaults)\n"
              << "  --autotune_stage2        Enable autotune stage 2 (overrides defaults)\n"
              << "  --autotune_stage3        Enable autotune stage 3 (overrides defaults)\n"
              << "  --autotune_max_iter <N>  Max autotune iterations [Default: 2]\n"
              << "  --autotune_timeout <sec> Autotune wall-clock timeout [Default: 300]\n"
              << "  --autotune_history <P>   History file path [Default: <work_dir>/autotune_history.json]\n"
              << "  --autotune_benign_history <P>  Benign history file [Default: <work_dir>/benign_history.json]\n"
              << "  --autotune_no_history    Disable parameter history load/save\n"
              << "  --autotune_candidates <P>   Candidates file for bootstrap mode\n"
              << "  --autotune_bootstrap         Run bootstrap: factor all candidates, build history\n"
              << "\n--- Logging ---\n"
              << "  --log_file <path>    Write full log (DEBUG_1) to file\n"
              << "  --error_log <path>   Write warnings/errors only to file\n"
              << "  --mute               Suppress all output except the factorization result\n"
              << "  --log_level <N>      Console verbosity threshold (-4=result, -1=info, 0=stats, 1=debug)\n"
              << "  --log_date           Include date in log timestamps\n"
              << "  --log_no_time        Suppress timestamps entirely\n"
              << "  --log_no_stage       Suppress [Stage N: Name] prefix\n"
              << "  --log_show_rank      Show [Rank N] prefix (multi-process)\n"
              << "  --log_wrap <N>       Console line wrap width (default: 120; 0=disabled)\n"
              << "  --log_csv            CSV format for --log_file output\n"
              << "\n--- Misc ---\n"
              << "  --verbose        Show statistics and buffer telemetry (alias: --log_level 0)\n"
              << "  --debug          Show per-batch debug telemetry (alias: --log_level 1)\n"
              << "  --version        Print version and exit\n"
              << "  --help           Show this help message.\n";
}


ParsedArgs parse_args(int argc, char** argv) {
    // RSA Factoring Challenge numbers are in the public domain (RSA Security, 1991–2007).
    // See: https://en.wikipedia.org/wiki/RSA_Factoring_Challenge
    const char* RSA100 = "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139";
    const char* RSA110 = "35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667";
    const char* RSA120 = "227010481295437363334259960947493668895875336466084780038173258247009162675779735389791151574049166747880487470296548479";
    const char* RSA129 = "114381625757888867669235779976146612010218296721242362562561842935706935245733897830597123563958705058989075147599290026879543541";
    const char* RSA130 = "1807082088687404805951656164405905566278102516769401349170127021450056662540244048387341127590812303371781887966563182013214880557";
    const char* RSA140 = "21290246318258757547497882016271517497806703963277216278233383215381949984056495911366573853021918316783107387995317230889569230873441936471";
    const char* RSA150 = "155089812478348440509606754370011861770654545830995430655466945774312632703463465954363335027577729025391453996787414027003501631772186840890795964683";
    const char* RSA155 = "10941738641570527421809707322040357612003732945449205990913842131476349984288934784717997257891267332497625752899781833797076537244027146743531593354333897";
    ParsedArgs args;
    bool has_explicit_autotune_stages = false;
    bool explicit_stage0 = false, explicit_stage1 = false;
    bool explicit_stage2 = false, explicit_stage3 = false;

    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        
        // --- Problem / Path ---
        if (arg == "--N" && i+1 < argc) {
            args.config.N = mpqs::uint512(argv[++i]);
        }  else if(arg == "--RSA100") {
            if(args.config.N.is_zero())
	      args.config.N = mpqs::uint512(RSA100);
	    else
	      std::cout << "Warning: N is already set to: " << args.config.N.to_string() << std::endl;
        } else if(arg == "--RSA110") {
            if(args.config.N.is_zero())
	      args.config.N = mpqs::uint512(RSA110);
	    else
	      std::cout << "Warning: N is already set to: " << args.config.N.to_string() << std::endl;
        } else if(arg == "--RSA120") {
            if(args.config.N.is_zero())
	      args.config.N = mpqs::uint512(RSA120);
	    else
	      std::cout << "Warning: N is already set to: " << args.config.N.to_string() << std::endl;
        } else if(arg == "--RSA129") {
            if(args.config.N.is_zero())
	      args.config.N = mpqs::uint512(RSA129);
	    else
	      std::cout << "Warning: N is already set to: " << args.config.N.to_string() << std::endl;
        } else if(arg == "--RSA130") {
            if(args.config.N.is_zero())
	      args.config.N = mpqs::uint512(RSA130);
	    else
	      std::cout << "Warning: N is already set to: " << args.config.N.to_string() << std::endl;
        } else if(arg == "--RSA140") {
            if(args.config.N.is_zero())
	      args.config.N = mpqs::uint512(RSA140);
	    else
	      std::cout << "Warning: N is already set to: " << args.config.N.to_string() << std::endl;
        } else if(arg == "--RSA150") {
            if(args.config.N.is_zero())
	      args.config.N = mpqs::uint512(RSA150);
	    else
	      std::cout << "Warning: N is already set to: " << args.config.N.to_string() << std::endl;
        } else if(arg == "--RSA155") {
            if(args.config.N.is_zero())
	      args.config.N = mpqs::uint512(RSA155);
	    else
	      std::cout << "Warning: N is already set to: " << args.config.N.to_string() << std::endl;
        } else if (arg == "--device" && i+1 < argc) {
	    if (!parse_uint32(argv[++i], args.config.device_id)) exit(1);
	    args.mark("device_id");
        } else if (arg == "--dir" && i+1 < argc) {
            args.config.work_dir = argv[++i];
        } else if (arg == "--disk_io") {
            args.config.disk_io = true;
        } else if (arg == "--dump_matrix") {
            args.config.dump_matrix = true;
        } else if (arg == "--dump_kernel_vectors") {
            args.config.dump_kernel_vectors = true;
        } else if (arg == "--dump_combine_provenance") {
            args.config.dump_combine_provenance = true;
        }
        else if (arg == "--force_preprocess") {
            args.config.force_preprocess = true;
            args.mark("force_preprocess");
        }

        // --- Modes ---
        else if (arg == "--full") args.config.mode = ExecutionMode::FULL_PIPELINE;
        else if (arg == "--sieve_only") args.config.mode = ExecutionMode::SIEVE_ONLY;
        else if (arg == "--linalg_only") args.config.mode = ExecutionMode::LINALG_ONLY;
        else if (arg == "--sqrt_only") args.config.mode = ExecutionMode::SQRT_ONLY;
        else if (arg == "--matrix_only") args.config.mode = ExecutionMode::MATRIX_ONLY;
        else if (arg == "--sqrt_legacy") args.config.sqrt_legacy = true;
        else if (arg == "--sqrt_diagnostic") args.config.sqrt_diagnostic = true;
        else if (arg == "--estimate_only") args.estimate_only = true;
        else if (arg == "--param_test") {
            args.config.mode = ExecutionMode::PARAM_TEST;
        }

        // --- Cluster mode ---
        else if (arg == "--cluster_mode" && i+1 < argc) {
            std::string mode = argv[++i];
            if (mode == "coordinator") {
                args.config.cluster_mode = ClusterMode::COORDINATOR;
            } else if (mode == "worker") {
                args.config.cluster_mode = ClusterMode::WORKER;
            } else if (mode == "solo") {
                args.config.cluster_mode = ClusterMode::SOLO;
            } else {
                std::cerr << "Error: unknown cluster mode '" << mode
                          << "'. Valid: solo, coordinator, worker\n";
                exit(1);
            }
        }
        else if (arg == "--coordinator_host" && i+1 < argc) {
            args.config.coordinator_host = argv[++i];
        }
        else if (arg == "--coordinator_port" && i+1 < argc) {
            args.config.coordinator_port = static_cast<uint16_t>(std::stoul(argv[++i]));
        }
        else if (arg == "--listen_port" && i+1 < argc) {
            args.config.listen_port = static_cast<uint16_t>(std::stoul(argv[++i]));
        }
        else if (arg == "--expected_workers" && i+1 < argc) {
            args.config.expected_workers = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
        else if (arg == "--cluster_init_timeout" && i+1 < argc) {
            args.config.cluster_init_timeout = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
        else if (arg == "--cluster_node_weights" && i+1 < argc) {
            args.config.cluster_node_weights = argv[++i];
        }
        else if (arg == "--cluster_headroom" && i+1 < argc) {
            args.config.cluster_headroom = std::stod(argv[++i]);
        }
        else if (arg == "--cluster_pool_oversize" && i+1 < argc) {
            args.config.cluster_pool_oversize = std::stod(argv[++i]);
        }

        // --- Tuning ---
        else if (arg == "--fb_bound" && i+1 < argc) {
            uint64_t val;
            if (!parse_suffixed_uint64(argv[++i], val, false)) {
                std::cerr << "Error: invalid value for --fb_bound\n"; exit(1);
            }
            args.config.fb_bound = static_cast<uint32_t>(val);
            args.mark("fb_bound");
        }
        else if (arg == "--lp1_bound" && i+1 < argc) {
            uint64_t val;
            if (!parse_suffixed_uint64(argv[++i], val, false)) {
                std::cerr << "Error: invalid value for --lp1_bound\n"; exit(1);
            }
            args.config.lp1_bound = val;
            args.mark("lp1_bound");
        }
        else if (arg == "--lp1_max_witnesses" && i+1 < argc) {
            uint64_t val;
            if (!parse_suffixed_uint64(argv[++i], val, true)) {
                std::cerr << "Error: invalid value for --lp1_max_witnesses\n"; exit(1);
            }
            args.config.lp1_max_witness_capacity = static_cast<uint32_t>(val);
            args.mark("lp1_max_witness_capacity");
        }
        else if (arg == "--lp_interval" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.lp_interval)) exit(1);
            args.mark("lp_interval");
        }
        else if (arg == "--target_rels" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.target_relations)) exit(1);
            args.mark("target_relations");
        }
        else if (arg == "--dedup_safety_factor" && i+1 < argc) {
            double val;
            if (!parse_double(argv[++i], val)) exit(1);
            if (val < 1.0 || val > 2.0) {
                std::cerr << "Warning: --dedup_safety_factor " << val
                          << " outside recommended range [1.0, 2.0]\n";
            }
            args.config.dedup_safety_factor = val;
            args.mark("dedup_safety_factor");
        }
        else if (arg == "--lp_matrix_threshold" && i + 1 < argc) {
            args.config.lp_matrix_threshold = std::stod(argv[++i]);
            // Deprecated alias: propagate to lp_preprocess_threshold if not already set
            if (args.set_flags.count("lp_preprocess_threshold") == 0) {
                args.config.lp_preprocess_threshold = args.config.lp_matrix_threshold;
            }
        }
        else if (arg == "--matrix_mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "legacy") {
                args.config.matrix_mode = MatrixMode::LEGACY;
            } else if (mode == "preprocess") {
                args.config.matrix_mode = MatrixMode::PREPROCESS;
            } else {
                std::cerr << "Error: unknown --matrix_mode '" << mode
                          << "'. Valid values: legacy, preprocess\n";
                exit(1);
            }
            args.mark("matrix_mode");
        }
        else if (arg == "--char_mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "norm") {
                args.config.char_mode = matrix::CharMode::NORM;
            } else if (mode == "branch") {
                args.config.char_mode = matrix::CharMode::BRANCH;
            } else if (mode == "none") {
                args.config.char_mode = matrix::CharMode::NONE;
            } else {
                std::cerr << "Error: unknown --char_mode '" << mode
                          << "'. Valid values: norm, branch, none\n";
                exit(1);
            }
            args.mark("char_mode");
        }
        else if (arg == "--lp_preprocess_threshold" && i + 1 < argc) {
            double val;
            if (!parse_double(argv[++i], val)) exit(1);
            args.config.lp_preprocess_threshold = val;
            args.mark("lp_preprocess_threshold");
        }
        else if (arg == "--preprocess_lp_materialize_max" && i + 1 < argc) {
            double val;
            if (!parse_double(argv[++i], val)) exit(1);
            args.config.preprocess_lp_materialize_max = val;
            args.mark("preprocess_lp_materialize_max");
        }
        else if (arg == "--partial_subsample" && i + 1 < argc) {
            double val;
            if (!parse_double(argv[++i], val)) exit(1);
            if (val < 0.0 || val > 1.0) {
                std::cerr << "Error: --partial_subsample must be in [0.0, 1.0]\n";
                exit(1);
            }
            args.config.partial_subsample = val;
            args.mark("partial_subsample");
        }
        else if (arg == "--smooth_subsample" && i + 1 < argc) {
            double val;
            if (!parse_double(argv[++i], val)) exit(1);
            if (val < 0.0 || val > 1.0) {
                std::cerr << "Error: --smooth_subsample must be in [0.0, 1.0]\n";
                exit(1);
            }
            args.config.smooth_subsample = val;
            args.mark("smooth_subsample");
        }
        else if (arg == "--truncation_factor" && i + 1 < argc) {
            double val;
            if (!parse_double(argv[++i], val)) exit(1);
            if (val < 0.0) {
                std::cerr << "Error: --truncation_factor must be >= 0.0\n";
                exit(1);
            }
            args.config.truncation_factor = val;
            args.mark("truncation_factor");
        }
        else if (arg == "--compact_cycles" && i + 1 < argc) {
            uint32_t val;
            if (!parse_uint32(argv[++i], val)) exit(1);
            args.config.compact_cycles = val;
            args.mark("compact_cycles");
        }
        else if (arg == "--matrix_truncation_excess" && i + 1 < argc) {
            uint32_t val;
            if (!parse_uint32(argv[++i], val)) exit(1);
            args.config.matrix_truncation_excess = val;
            args.mark("matrix_truncation_excess");
        }
        else if (arg == "--truncation_min_rows" && i + 1 < argc) {
            uint32_t val;
            if (!parse_uint32(argv[++i], val)) exit(1);
            args.config.truncation_min_rows = val;
            args.mark("truncation_min_rows");
        }
        else if (arg == "--merge_max_weight" && i + 1 < argc) {
            uint32_t val;
            if (!parse_uint32(argv[++i], val)) exit(1);
            args.config.merge_max_weight = val;
            args.mark("merge_max_weight");
        }
        else if (arg == "--matrix_gf2_floor_factor" && i + 1 < argc) {
            double val;
            if (!parse_double(argv[++i], val)) exit(1);
            if (val < 0.0 || val > 1.0) {
                std::cerr << "Error: --matrix_gf2_floor_factor must be in [0.0, 1.0]\n";
                exit(1);
            }
            args.config.matrix_gf2_floor_factor = val;
            args.mark("matrix_gf2_floor_factor");
        }
        else if (arg == "--matrix_gf2_min_floor" && i + 1 < argc) {
            uint32_t val;
            if (!parse_uint32(argv[++i], val)) exit(1);
            args.config.matrix_gf2_min_floor = val;
            args.mark("matrix_gf2_min_floor");
        }
        else if (arg == "--matrix_backend" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "cpu")       args.config.matrix_backend = 0;
            else if (val == "gpu")  args.config.matrix_backend = 1;
            else if (val == "auto") args.config.matrix_backend = 2;
            else {
                std::cerr << "Error: unknown --matrix_backend '" << val
                          << "'. Use cpu, gpu, or auto.\n";
                exit(1);
            }
            args.mark("matrix_backend");
        }
        else if (arg == "--sieve_bound" && i+1 < argc) {
            uint64_t val;
            if (!parse_suffixed_uint64(argv[++i], val, true)) {
                std::cerr << "Error: invalid value for --sieve_bound\n"; exit(1);
            }
            args.config.sieve_bound = static_cast<uint32_t>(val);
            args.mark("sieve_bound");
        }
        else if (arg == "--sieve_hc_dim" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.sieve_hcube_dimension)) exit(1);
            args.mark("sieve_hcube_dimension");
        }
        else if (arg == "--params" && i+1 < argc) {
            std::string input = argv[++i];
            input.erase(std::remove(input.begin(), input.end(), '('), input.end());
            input.erase(std::remove(input.begin(), input.end(), ')'), input.end());
            std::stringstream ss(input);
            std::string segment;
            int found = 0;
            while (std::getline(ss, segment, ',')) {
                if (segment.empty()) {
                    std::cerr << "Error: empty value in --params list.\n";
                    exit(1);
                }
                if (found < 8) {
                    if (!parse_uint32(segment.data(), args.config.params[found])) exit(1);
                }
                found++;
            }
            if (found < 8) {
                std::cerr << "Error: --params requires 8 comma-separated values, got " << found << ".\n";
                exit(1);
            }
            args.config.useParams = true;
            args.mark("params");
        }

        // --- Sieve ---
        else if (arg == "--probe_timeout" && i + 1 < argc) {
            args.config.probe_timeout = std::stod(argv[++i]);
            args.mark("probe_timeout");
        } else if (arg == "--sieve_batch_size" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.sieve_batch_size)) exit(1);
            args.mark("sieve_batch_size");
        } else if (arg == "--sieve_gms_blocks" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.sieve_gms_num_blocks)) exit(1);
            args.mark("sieve_gms_num_blocks");
        } else if (arg == "--cuda_graph_unroll" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.cuda_graph_unroll)) exit(1);
            uint32_t val = args.config.cuda_graph_unroll;
            if (val > 1 && val % 2 != 0) {
                std::cerr << "Warning: --cuda_graph_unroll must be even (double-buffer constraint). "
                          << "Rounded " << val << " -> " << (val + 1) << "\n";
                val = val + 1;
            }
            if (val > 16) {
                std::cerr << "Warning: --cuda_graph_unroll " << val << " has diminishing returns. "
                          << "Capping at 16.\n";
                val = 16;
            }
            args.config.cuda_graph_unroll = val;
            args.mark("cuda_graph_unroll");
        }

        // --- Truncated Sieve ---
        else if (arg == "--sieve_max_relations" && i+1 < argc) {
            if (!parse_suffixed_uint64(argv[++i], args.config.sieve_max_relations)) {
                std::cerr << "Error: invalid value for --sieve_max_relations\n"; exit(1);
            }
            args.mark("sieve_max_relations");
        }
        else if (arg == "--sieve_max_batches" && i+1 < argc) {
            if (!parse_uint64(argv[++i], args.config.sieve_max_batches)) exit(1);
            args.mark("sieve_max_batches");
        }
        else if (arg == "--sieve_truncate_continue") {
            args.config.sieve_truncate_continue = true;
        }

        // --- Buffer Sizes ---
        else if (arg == "--accum_buf_size" && i+1 < argc) {
            args.config.accum_buffer_size = parse_size(argv[++i]);
            args.mark("accum_buffer_size");
        }
        else if (arg == "--partial_buf_size" && i+1 < argc) {
            args.config.partial_buffer_size = parse_size(argv[++i]);
            args.mark("partial_buffer_size");
        }
        else if (arg == "--persistent_buf_size" && i+1 < argc) {
            args.config.persistent_buffer_size = parse_size(argv[++i]);
            args.mark("persistent_buffer_size");
        }
        else if (arg == "--lp1_combined_buf" && i+1 < argc) {
            args.config.lp1_max_combined_output = parse_size(argv[++i]);
            args.mark("lp1_max_combined_output");
        }
        else if (arg == "--lp1_hash_bits" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.lp1_hash_bits)) exit(1);
            args.mark("lp1_hash_bits");
        }

        // --- BW ---
        else if (arg == "--bw_m" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.bw_m)) exit(1);
            args.mark("bw_m");
        }
        else if (arg == "--bw_n" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.bw_n)) exit(1);
            args.mark("bw_n");
        }

        #ifdef SIEVING_DEBUG_FLAG
        // --- Debug ---
        else if (arg == "--metaSnapshot" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.meta_snapshot_step)) exit(1);
            args.config.meta_snapshot_enabled = true;
        }
        else if (arg == "--metaP" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.meta_P)) exit(1);
            args.config.meta_P_enabled = true;
        }
        else if (arg == "--metaO" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.meta_O)) exit(1);
            args.config.meta_O_enabled = true;
        }
        else if (arg == "--sasSnapshot" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.sas_snapshot_step)) exit(1);
            args.config.sas_snapshot_enabled = true;
        }
        #endif

        // --- Autotune ---
        else if (arg == "--autotune") {
            args.config.autotune_enabled = true;
        }
        else if (arg == "--autotune_only") {
            args.config.mode = ExecutionMode::AUTOTUNE_ONLY;
            args.config.autotune_enabled = true;
        }
        else if (arg == "--autotune_stage0") {
            args.config.autotune_enabled = true;
            has_explicit_autotune_stages = true;
            explicit_stage0 = true;
        }
        else if (arg == "--autotune_stage1") {
            args.config.autotune_enabled = true;
            has_explicit_autotune_stages = true;
            explicit_stage1 = true;
        }
        else if (arg == "--autotune_stage2") {
            args.config.autotune_enabled = true;
            has_explicit_autotune_stages = true;
            explicit_stage2 = true;
        }
        else if (arg == "--autotune_stage3") {
            args.config.autotune_enabled = true;
            has_explicit_autotune_stages = true;
            explicit_stage3 = true;
        }
        else if (arg == "--autotune_max_iter" && i+1 < argc) {
            if (!parse_uint32(argv[++i], args.config.autotune_config.max_iterations)) exit(1);
        }
        else if (arg == "--autotune_timeout" && i+1 < argc) {
            if (!parse_double(argv[++i], args.config.autotune_config.timeout_sec)) exit(1);
        }
        else if (arg == "--autotune_history" && i+1 < argc) {
            args.config.autotune_config.history_file = argv[++i];
        }
        else if (arg == "--autotune_benign_history" && i+1 < argc) {
            args.config.autotune_config.benign_history_file = argv[++i];
        }
        else if (arg == "--autotune_no_history") {
            args.config.autotune_config.load_history = false;
            args.config.autotune_config.save_history = false;
        }
        else if (arg == "--autotune_candidates" && i+1 < argc) {
            args.config.autotune_config.candidates_file = argv[++i];
        }
        else if (arg == "--autotune_bootstrap") {
            args.config.autotune_config.bootstrap = true;
        }

        // --- Logging ---
        else if (arg == "--log_file" && i+1 < argc) {
            args.log_flags.log_file = argv[++i];
        }
        else if (arg == "--error_log" && i+1 < argc) {
            args.log_flags.error_log = argv[++i];
        }
        else if (arg == "--mute") {
            args.log_flags.mute = true;
        }
        else if (arg == "--log_level" && i+1 < argc) {
            if (!parse_int(argv[++i], args.log_flags.log_level)) exit(1);
            if (args.log_flags.log_level < LOG_RESULT || args.log_flags.log_level > LOG_DEBUG_3) {
                std::cerr << "--log_level must be in [" << LOG_RESULT << ".." << LOG_DEBUG_3 << "]\n";
                exit(1);
            }
        }
        else if (arg == "--log_date") {
            args.log_flags.log_date = true;
        }
        else if (arg == "--log_no_time") {
            args.log_flags.log_no_time = true;
        }
        else if (arg == "--log_no_stage") {
            args.log_flags.log_no_stage = true;
        }
        else if (arg == "--log_show_rank") {
            args.log_flags.log_show_rank = true;
        }
        else if (arg == "--log_wrap" && i+1 < argc) {
            if (!parse_int(argv[++i], args.log_flags.log_wrap)) exit(1);
        }
        else if (arg == "--log_csv") {
            args.log_flags.log_csv = true;
        }

        // --- Misc ---
        else if (arg == "--verbose") {
            args.log_flags.log_level = LOG_STATS;
        }
        else if (arg == "--debug") {
            args.log_flags.log_level = LOG_DEBUG_1;
        }
        else if (arg == "--version") {
            std::cout << "cuda-mpqs " << CUDAMPQS_VERSION
                      << " (lingen " << CUDAMPQS_LINALG_VERSION
                      << ", git " << CUDAMPQS_BUILD_GIT_SHA
                      << ", built " << CUDAMPQS_BUILD_DATE << ")\n";
            exit(0);
        }
        else if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            exit(1);
        }

    }

    // Autotune: if explicit stages were specified, override defaults
    if (has_explicit_autotune_stages) {
        args.config.autotune_config.enable_stage0 = explicit_stage0;
        args.config.autotune_config.enable_stage1 = explicit_stage1;
        args.config.autotune_config.enable_stage2 = explicit_stage2;
        args.config.autotune_config.enable_stage3 = explicit_stage3;
    }

    // Copy pin tracking flags into config for downstream consumers
    args.config.pinned_params = args.set_flags;

    // Default
    if(args.config.N.is_zero()) {
        const char* N_str = "6024065079889642469495026789749787328504528247460180000248150504804066095061017"; // Default ~80 digit composite
	args.config.N = mpqs::uint512(N_str);
    }
      

    return args;
}

// -----------------------------------------------------------------------------
// Main Verification Logic
// -----------------------------------------------------------------------------

bool verify_factors(const mpqs::uint512& N, const std::vector<mpqs::uint512>& factors) {
    if (factors.empty()) return false;

    mpqs::uint512 product = mpqs::uint512((uint32_t)1);

    // Truncated 512-bit multiplication (uint512::mult discards bits >= 2^512).
    // No overflow can occur here: every factor f satisfies 1 < f < N, the
    // accumulated product equals N exactly when verification succeeds, and
    // the orchestrator only accepts inputs N < 2^512. Therefore the
    // intermediate product never exceeds N < 2^512 and the truncation is
    // mathematically a no-op.
    
    LOG(LOG_INFO) << "[CUDA-MPQS] Verifying Product...";
    for (const auto& f : factors) {
        LOG(LOG_INFO) << "[CUDA-MPQS]   Factor: " << f.to_string();
        
        if (f.is_one() || f == N) {
            LOG(LOG_WARNING) << "[CUDA-MPQS]   Trivial factor detected!";
        }
        
        // Product accumulation
        product.mult(f);
    }

    LOG(LOG_INFO) << "[CUDA-MPQS]   Calculated Product: " << product.to_string();
    LOG(LOG_INFO) << "[CUDA-MPQS]   Target N:           " << N.to_string();

    return (product == N);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    // 1. Parse Args
    ParsedArgs args = parse_args(argc, argv);

    // Resolve default autotune history path
    if (args.config.autotune_config.history_file.empty()) {
        args.config.autotune_config.history_file =
            args.config.work_dir + "/autotune_history.json";
    }
    if (args.config.autotune_config.benign_history_file.empty()) {
        args.config.autotune_config.benign_history_file =
            args.config.work_dir + "/benign_history.json";
    }

    // 2. Build LogConfig from parsed flags
    LogConfig log_cfg;
    const auto& lf = args.log_flags;

    // Console sink (always present)
    {
        SinkConfig console;
        console.type = SinkConfig::CONSOLE;
        if (lf.mute) {
            console.min_severity   = LOG_RESULT;
            console.show_date      = false;
            console.show_time      = false;
            console.show_stage     = false;
            console.show_rank      = false;
            console.show_module    = false;
            console.show_submodule = false;
            console.show_level     = false;
            console.wrap_width     = 0;
        } else {
            console.min_severity   = lf.log_level;
            console.show_date      = lf.log_date;
            console.show_time      = !lf.log_no_time;
            console.show_stage     = !lf.log_no_stage;
            console.show_rank      = lf.log_show_rank;
            console.show_level     = false;
            console.wrap_width     = lf.log_wrap;
        }
        log_cfg.sinks.push_back(std::move(console));
    }

    // File sink (optional)
    if (!lf.log_file.empty()) {
        SinkConfig file_sink;
        file_sink.type           = SinkConfig::FILE;
        file_sink.path           = lf.log_file;
        file_sink.min_severity   = LOG_DEBUG_3;  // capture all levels to disk
        file_sink.show_date      = true;
        file_sink.show_time      = true;
        file_sink.show_rank      = true;
        file_sink.show_stage     = true;
        file_sink.show_module    = true;
        file_sink.show_submodule = true;
        file_sink.show_level     = true;
        file_sink.wrap_width     = 0;
        file_sink.csv_format     = lf.log_csv;
        log_cfg.sinks.push_back(std::move(file_sink));
    }

    // Error file sink (optional)
    if (!lf.error_log.empty()) {
        SinkConfig err_sink;
        err_sink.type           = SinkConfig::ERROR_FILE;
        err_sink.path           = lf.error_log;
        err_sink.min_severity   = LOG_WARNING;
        err_sink.max_severity   = LOG_ERROR_CRITICAL;
        err_sink.show_date      = true;
        err_sink.show_time      = true;
        err_sink.show_rank      = true;
        err_sink.show_stage     = true;
        err_sink.show_module    = true;
        err_sink.show_submodule = true;
        err_sink.show_level     = true;
        err_sink.wrap_width     = 0;
        err_sink.csv_format     = false;
        log_cfg.sinks.push_back(std::move(err_sink));
    }

    HPCLogger::Get().Init(log_cfg);

    // --- Estimate-only mode ---
    // Solo: run truncated sieve probe and print estimate (existing path).
    // Cluster: set config flag and let Run() handle time-limited sieve + estimate.
    if (args.estimate_only) {
        if (args.config.cluster_mode != ClusterMode::SOLO) {
            // Cluster estimate — handled inside Run()
            args.config.estimate_only = true;
        } else {
            // Solo estimate — existing self-contained path
            try {
                LOG(LOG_INFO) << "[CUDA-MPQS] === Runtime Estimation Mode ===";
                LOG(LOG_INFO) << "[CUDA-MPQS] Target: " << args.config.N.to_string();

                auto est = mpqs::autotune::estimateRuntime(
                    args.config,
                    args.config.fb_bound,
                    args.config.sieve_bound,
                    args.config.lp1_bound);

                LOG(LOG_INFO) << "[Estimate] --- Runtime Estimate ---";
                LOG(LOG_INFO) << "[Estimate] Parameters: F=" << est.fb_bound
                              << " M=" << est.sieve_bound
                              << " L=" << est.lp1_bound;
                LOG(LOG_INFO) << "[Estimate] FB size:        " << est.fb_size;
                LOG(LOG_INFO) << "[Estimate] Sieve time:     "
                              << std::fixed << std::setprecision(1)
                              << est.sieve_total_sec << " s";
                LOG(LOG_INFO) << "[Estimate] Matrix time:    "
                              << est.matrix_est_sec << " s";
                LOG(LOG_INFO) << "[Estimate] LinAlg time:    "
                              << est.linalg_est_sec << " s";
                LOG(LOG_INFO) << "[Estimate] Total estimate: "
                              << est.total_est_sec << " s";
                LOG(LOG_INFO) << "[Estimate] Confidence:     "
                              << std::setprecision(2) << est.confidence;
                LOG(LOG_INFO) << "[Estimate] ETA samples:    " << est.eta_samples;
                LOG(LOG_INFO) << "[Estimate] Throughput:     "
                              << std::setprecision(1) << est.relations_per_sec
                              << " rels/s";
                if (est.lp1_bound > 0) {
                    LOG(LOG_INFO) << "[Estimate] LP fill proj:   "
                                  << std::setprecision(1)
                                  << est.projected_witness_fill_pct << " %";
                    LOG(LOG_INFO) << "[Estimate] LP overflows:   "
                                  << est.overflow_events;
                }

                return 0;
            } catch (const std::exception& e) {
                LOG(LOG_ERROR_CRITICAL) << "[Estimate] Failed: " << e.what();
                return 1;
            }
        }
    }

    // --- Bootstrap mode (M4) ---
    if (args.config.autotune_config.bootstrap) {
        if (args.config.autotune_config.candidates_file.empty()) {
            LOG(LOG_ERROR_CRITICAL)
                << "[Bootstrap] --autotune_bootstrap requires --autotune_candidates <file>";
            return 1;
        }

        // Load candidates (one decimal per line, skip empty and # comments)
        std::vector<mpqs::uint512> candidates;
        {
            std::ifstream fin(args.config.autotune_config.candidates_file);
            if (!fin.is_open()) {
                LOG(LOG_ERROR_CRITICAL)
                    << "[Bootstrap] Cannot open candidates file: "
                    << args.config.autotune_config.candidates_file;
                return 1;
            }
            std::string line;
            while (std::getline(fin, line)) {
                // Trim whitespace
                auto start = line.find_first_not_of(" \t\r\n");
                if (start == std::string::npos) continue;
                line = line.substr(start);
                if (line.empty() || line[0] == '#') continue;
                candidates.emplace_back(line.c_str());
            }
        }

        if (candidates.empty()) {
            LOG(LOG_ERROR_CRITICAL) << "[Bootstrap] No candidates found in file.";
            return 1;
        }

        // Sort by bit_length ascending (small -> large)
        std::sort(candidates.begin(), candidates.end(),
            [](const mpqs::uint512& a, const mpqs::uint512& b) {
                return a.msb() < b.msb();
            });

        LOG(LOG_INFO) << "[Bootstrap] Processing " << candidates.size()
                      << " candidates (" << candidates.front().msb()
                      << " to " << candidates.back().msb() << " bits)";

        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            const auto& N = candidates[idx];
            LOG(LOG_INFO) << "[Bootstrap] [" << (idx+1) << "/"
                          << candidates.size() << "] Factoring "
                          << N.to_string().size() << "-digit composite ("
                          << N.msb() << " bits)...";

            // Copy base config, set N, force full pipeline
            MPQSConfig run_config = args.config;
            run_config.N = N;
            run_config.mode = ExecutionMode::FULL_PIPELINE;

            try {
                MPQSOrchestrator orch(run_config);
                orch.Run();

                auto factors = orch.GetFactors();
                if (!factors.empty() && verify_factors(N, factors)) {
                    LOG(LOG_INFO) << "[Bootstrap] SUCCESS: "
                                  << N.to_string().size() << "-digit composite factored.";
                } else {
                    LOG(LOG_WARNING) << "[Bootstrap] FAILED to factor "
                                     << N.to_string().size() << "-digit composite.";
                }
            } catch (const std::exception& e) {
                LOG(LOG_WARNING) << "[Bootstrap] Exception for "
                                 << N.to_string().size() << "-digit: " << e.what();
            }
            // History is saved by the post-sieve update in orchestrator
        }

        LOG(LOG_INFO) << "[Bootstrap] Complete. Processed "
                      << candidates.size() << " candidates.";
        return 0;
    }

    // 3. Setup Orchestrator
    try {
        LOG(LOG_INFO) << "[CUDA-MPQS] === CUDA-MPQS Start ===";
        LOG(LOG_INFO) << "[CUDA-MPQS] Target: " << args.config.N.to_string();

        MPQSOrchestrator orchestrator(args.config);
        
        // 4. Run Pipeline
        orchestrator.Run();

        // 5. Verify Results
        if (args.config.estimate_only) {
            LOG(LOG_INFO) << "[CUDA-MPQS] Cluster estimate complete.";
            return 0;
        }
        if (args.config.mode == ExecutionMode::FULL_PIPELINE || args.config.mode == ExecutionMode::SQRT_ONLY) {
            std::vector<mpqs::uint512> factors = orchestrator.GetFactors();
            
            if (factors.empty()) {
                LOG(LOG_ERROR_CRITICAL) << "[CUDA-MPQS] FAILURE: No factors returned by orchestrator.";
                return 1;
            }

            bool success = verify_factors(args.config.N, factors);
            
            if (success) {
                LOG(LOG_INFO) << "[CUDA-MPQS] SUCCESS: Factorization Verified.";
                return 0;
            } else {
                LOG(LOG_ERROR_CRITICAL) << "[CUDA-MPQS] FAILURE: Product of factors does not match N.";
                return 1;
            }
        } else {
            LOG(LOG_INFO) << "[CUDA-MPQS] Partial pipeline executed successfully. No factorization check performed.";
            return 0;
        }

    } catch (const std::exception& e) {
        LOG(LOG_ERROR_CRITICAL) << "[CUDA-MPQS] Unhandled Exception: " << e.what();
        return 1;
    }
}
