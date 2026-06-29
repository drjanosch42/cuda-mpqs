// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// =============================================================================
// test_checkpoint_io — contractual tests for the mid-sieve checkpoint (S1)
// =============================================================================
//
// (A) m-trailer-test: make the serialize_v2 / deserialize_v2 trailing-byte tolerance
//     CONTRACTUAL. deserialize_v2 reads section-by-section and returns f.good() without
//     checking EOF (relation_io.cpp), so arbitrary bytes appended after the payload MUST
//     be ignored. The checkpoint trailer+footer rely on this exactly.
//
// (B) Checkpoint round-trip: writeCheckpointAtomic → readCheckpoint reproduces the trailer
//     fields and the relation payload byte-for-byte.
//
// (C) Atomicity surface: a second write retains exactly one prior generation as
//     sieve.ckpt.prev; a torn live file (footer chopped) fails readCheckpoint but
//     loadLatestCheckpoint falls back to the intact .prev.
//
// CPU-only host test (compiled as .cu so the host_device math headers compile). No kernels.
// =============================================================================

#include "relation_io.h"
#include "sieve_checkpoint.h"
#include "mpqs_soa.h"
#include "uint512.cuh"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <unistd.h>

using mpqs::uint512;
using mpqs::structures::HostRelationBatch;

namespace {

int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); ++g_failures; } } while (0)

// Build a small, CSR-valid batch with `n` relations, each carrying 2-3 factors.
HostRelationBatch makeBatch(size_t n, uint32_t seed) {
    HostRelationBatch b;
    b.factor_offsets.push_back(0);
    for (size_t i = 0; i < n; ++i) {
        uint512 q{};
        q.limbs[0] = static_cast<uint32_t>(0x1000 + i * 7 + seed);
        q.limbs[1] = static_cast<uint32_t>(seed ^ (i * 13));
        b.sqrt_Q.push_back(q);
        b.signs.push_back((i & 1) ? 0xFFu : 1u);          // alternate sign encoding
        b.val_2_exps.push_back(static_cast<int32_t>(i % 5));
        b.large_primes.push_back((unsigned __int128)(1000003ull + i * 100 + seed));
        b.char_bits.push_back(0u);                        // non-branch: serialize_v2 won't emit these

        uint32_t nf = 2 + static_cast<uint32_t>(i % 2);   // 2 or 3 factors
        for (uint32_t f = 0; f < nf; ++f) {
            b.factor_indices.push_back((seed + i + f) % 97 + 1);
            b.factor_counts.push_back(static_cast<uint8_t>(1 + (f % 3)));
        }
        b.num_factors += nf;
        b.factor_offsets.push_back(b.num_factors);
        b.num_relations++;
    }
    return b;
}

bool batchEqual(const HostRelationBatch& a, const HostRelationBatch& b) {
    if (a.num_relations != b.num_relations) return false;
    if (a.num_factors  != b.num_factors)  return false;
    if (a.signs != b.signs) return false;
    if (a.val_2_exps != b.val_2_exps) return false;
    if (a.large_primes != b.large_primes) return false;
    if (a.factor_offsets != b.factor_offsets) return false;
    if (a.factor_indices != b.factor_indices) return false;
    if (a.factor_counts != b.factor_counts) return false;
    if (a.sqrt_Q.size() != b.sqrt_Q.size()) return false;
    for (size_t i = 0; i < a.sqrt_Q.size(); ++i)
        if (std::memcmp(a.sqrt_Q[i].limbs, b.sqrt_Q[i].limbs, 64) != 0) return false;
    return true;
}

mpqs::io::V2Metadata makeMeta() {
    mpqs::io::V2Metadata m;
    m.N.limbs[0] = 0xDEADBEEFu;
    m.N.limbs[1] = 0x12345678u;
    m.factor_base = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    m.lp_bound = 1000000000ull;
    m.sieve_bound = 131072u;
    return m;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    fs::path tdir = fs::temp_directory_path() /
        ("ckpt_io_test_" + std::to_string(static_cast<long>(::getpid())));
    fs::remove_all(tdir);
    fs::create_directories(tdir);

    HostRelationBatch smooths  = makeBatch(7, 11);
    HostRelationBatch partials = makeBatch(4, 23);
    mpqs::io::V2Metadata meta = makeMeta();

    // ---------------------------------------------------------------------
    // (A) m-trailer-test: deserialize_v2 ignores appended trailing bytes.
    // ---------------------------------------------------------------------
    {
        std::string v2 = (tdir / "relations.v2").string();
        CHECK(mpqs::io::serialize_v2(v2, smooths, partials, meta), "serialize_v2");

        HostRelationBatch s1, p1; mpqs::io::V2Metadata m1;
        CHECK(mpqs::io::deserialize_v2(v2, s1, p1, m1), "deserialize_v2 (clean)");

        // Append arbitrary garbage (incl. the checkpoint trailer magic) after the payload.
        {
            std::ofstream f(v2, std::ios::binary | std::ios::app);
            const char junk[] = "MPQS_CKPT\x01\x02\x03\x04 trailing bytes that must be ignored";
            f.write(junk, sizeof(junk));
            std::vector<char> blob(512, '\xAB');
            f.write(blob.data(), blob.size());
        }

        HostRelationBatch s2, p2; mpqs::io::V2Metadata m2;
        CHECK(mpqs::io::deserialize_v2(v2, s2, p2, m2), "deserialize_v2 (with trailing bytes)");

        CHECK(batchEqual(s1, s2), "smooths identical despite trailing bytes");
        CHECK(batchEqual(p1, p2), "partials identical despite trailing bytes");
        CHECK(batchEqual(s1, smooths), "round-trip smooths == original");
        CHECK(batchEqual(p1, partials), "round-trip partials == original");
    }

    // ---------------------------------------------------------------------
    // (B) Checkpoint round-trip.
    // ---------------------------------------------------------------------
    {
        std::string cdir = (tdir / "checkpoint").string();
        mpqs::ckpt::CheckpointTrailer tr;
        tr.global_a_index       = 123456789ull;
        tr.target_relations     = 250000ull;
        tr.loaded_smooths_raw   = 9ull;                       // pretend pre-dedup raw count
        tr.loaded_smooths_dedup = smooths.num_relations;      // must equal payload smooths
        tr.loaded_partials      = partials.num_relations;
        tr.lp1_bound            = meta.lp_bound;
        tr.sieve_bound          = meta.sieve_bound;
        tr.N                    = meta.N;
        tr.cluster_section_present = 0;
        tr.elapsed_sieve_sec    = 4242ull;

        CHECK(mpqs::ckpt::writeCheckpointAtomic(cdir, smooths, partials, meta, tr),
              "writeCheckpointAtomic");
        CHECK(fs::exists(cdir + "/sieve.ckpt"), "sieve.ckpt created");
        CHECK(!fs::exists(cdir + "/sieve.ckpt.tmp"), "no stray .tmp after commit");

        mpqs::ckpt::CheckpointLoadResult res;
        CHECK(mpqs::ckpt::readCheckpoint(cdir + "/sieve.ckpt", res), "readCheckpoint");
        CHECK(res.ok, "readCheckpoint ok");
        CHECK(res.trailer.global_a_index == tr.global_a_index, "trailer a_index");
        CHECK(res.trailer.target_relations == tr.target_relations, "trailer target");
        CHECK(res.trailer.loaded_smooths_raw == tr.loaded_smooths_raw, "trailer raw");
        CHECK(res.trailer.loaded_smooths_dedup == tr.loaded_smooths_dedup, "trailer dedup");
        CHECK(res.trailer.loaded_partials == tr.loaded_partials, "trailer partials");
        CHECK(res.trailer.lp1_bound == tr.lp1_bound, "trailer lp1_bound");
        CHECK(res.trailer.sieve_bound == tr.sieve_bound, "trailer sieve_bound");
        CHECK(res.trailer.elapsed_sieve_sec == tr.elapsed_sieve_sec, "trailer elapsed");
        CHECK(std::memcmp(res.trailer.N.limbs, tr.N.limbs, 64) == 0, "trailer N");
        CHECK(batchEqual(res.smooths, smooths), "ckpt payload smooths == original");
        CHECK(batchEqual(res.partials, partials), "ckpt payload partials == original");
        // Solo file (cluster_section_present == 0) ⇒ no cluster block decoded.
        CHECK(res.cluster.initial_high_water.empty(), "solo ckpt has no cluster block");
        CHECK(res.trailer.cluster_section_present == 0, "solo trailer flag == 0");

        // (C) Second write retains exactly one prior generation as .prev.
        tr.global_a_index = 999999999ull;
        CHECK(mpqs::ckpt::writeCheckpointAtomic(cdir, smooths, partials, meta, tr),
              "writeCheckpointAtomic (2nd)");
        CHECK(fs::exists(cdir + "/sieve.ckpt.prev"), ".prev retained after 2nd write");

        mpqs::ckpt::CheckpointLoadResult res2;
        CHECK(mpqs::ckpt::readCheckpoint(cdir + "/sieve.ckpt", res2), "readCheckpoint (2nd)");
        CHECK(res2.trailer.global_a_index == 999999999ull, "live = newest generation");

        // (C) Torn live file: chop the footer → readCheckpoint(live) fails, but
        //     loadLatestCheckpoint falls back to the intact .prev.
        {
            uint64_t sz = fs::file_size(cdir + "/sieve.ckpt");
            fs::resize_file(cdir + "/sieve.ckpt", sz - (mpqs::ckpt::CKPT_FOOTER_SIZE + 4));
        }
        mpqs::ckpt::CheckpointLoadResult torn;
        CHECK(!mpqs::ckpt::readCheckpoint(cdir + "/sieve.ckpt", torn),
              "torn live file rejected (no footer magic)");

        mpqs::ckpt::CheckpointLoadResult fb;
        CHECK(mpqs::ckpt::loadLatestCheckpoint(cdir, fb), "loadLatestCheckpoint falls back to .prev");
        CHECK(fb.ok && fb.trailer.global_a_index == 123456789ull,
              "fallback picks the prior committed generation");
    }

    // ---------------------------------------------------------------------
    // (D) S3 cluster-block round-trip: write a checkpoint WITH a variable-size
    //     cluster block (completedPrefixCursor + per-node initial-range high-water
    //     array) through the variable footer; read it back; assert the fields survive
    //     and the sanity invariants hold (prefix ≤ next_; per-node hw ≤ its range size).
    // ---------------------------------------------------------------------
    {
        std::string cdir = (tdir / "checkpoint_cluster").string();
        mpqs::ckpt::CheckpointTrailer tr;
        tr.global_a_index          = 0;                       // cluster: sentinel (progress in block)
        tr.target_relations        = 250000ull;
        tr.loaded_smooths_raw      = smooths.num_relations;
        tr.loaded_smooths_dedup    = smooths.num_relations;
        tr.loaded_partials         = partials.num_relations;
        tr.lp1_bound               = meta.lp_bound;
        tr.sieve_bound             = meta.sieve_bound;
        tr.N                       = meta.N;
        tr.cluster_section_present = 1;                        // an S3 cluster block follows
        tr.elapsed_sieve_sec       = 7777ull;

        // Synthetic cluster state: 4 nodes (coord + 3 workers). Per-node initial range
        // sizes (orig_count) and high-water marks. overflow_start = sum of ranges.
        const std::vector<uint64_t> orig_count = {100000, 80000, 80000, 60000};
        uint64_t overflow_start = 0;
        for (uint64_t c : orig_count) overflow_start += c;    // = 320000
        const uint64_t next_cursor = overflow_start + 50000;  // pool advanced 50k into overflow

        mpqs::ckpt::CheckpointClusterBlock cb;
        cb.completed_prefix_cursor = overflow_start + 40000;  // ≤ next_cursor (some in-flight tail)
        cb.initial_high_water      = {100000, 73216, 0, 60000};  // node 2 not yet done (hw=0)

        // Invariants the writer's producer must satisfy (assert on the synthetic data).
        CHECK(cb.completed_prefix_cursor <= next_cursor, "prefix cursor ≤ next_ (B2)");
        for (size_t i = 0; i < cb.initial_high_water.size(); ++i)
            CHECK(cb.initial_high_water[i] <= orig_count[i], "per-node hw ≤ initial range size (M1)");

        CHECK(mpqs::ckpt::writeCheckpointAtomic(cdir, smooths, partials, meta, tr, &cb),
              "writeCheckpointAtomic (cluster)");

        mpqs::ckpt::CheckpointLoadResult res;
        CHECK(mpqs::ckpt::readCheckpoint(cdir + "/sieve.ckpt", res), "readCheckpoint (cluster)");
        CHECK(res.ok, "readCheckpoint cluster ok");
        CHECK(res.trailer.cluster_section_present == 1, "cluster trailer flag == 1");
        CHECK(res.cluster.completed_prefix_cursor == cb.completed_prefix_cursor,
              "cluster prefix cursor round-trips");
        CHECK(res.cluster.initial_high_water == cb.initial_high_water,
              "per-node high-water array round-trips (size + values)");
        CHECK(res.cluster.initial_high_water.size() == orig_count.size(),
              "node count round-trips");
        CHECK(batchEqual(res.smooths, smooths), "cluster ckpt payload smooths == original");
        CHECK(batchEqual(res.partials, partials), "cluster ckpt payload partials == original");

        // The cluster block must NOT disturb the v2 payload for ordinary consumers:
        // deserialize_v2 still ignores trailer + cluster block + footer.
        HostRelationBatch s2, p2; mpqs::io::V2Metadata m2;
        CHECK(mpqs::io::deserialize_v2(cdir + "/sieve.ckpt", s2, p2, m2),
              "deserialize_v2 of a cluster checkpoint (ignores trailer+block+footer)");
        CHECK(batchEqual(s2, smooths), "deserialize_v2 smooths == original (cluster file)");

        // An empty-node-array cluster block (0 nodes) must also round-trip.
        mpqs::ckpt::CheckpointClusterBlock cb0;
        cb0.completed_prefix_cursor = 12345ull;
        cb0.initial_high_water.clear();
        CHECK(mpqs::ckpt::writeCheckpointAtomic((tdir / "ck0").string(),
              smooths, partials, meta, tr, &cb0), "writeCheckpointAtomic (0-node cluster)");
        mpqs::ckpt::CheckpointLoadResult res0;
        CHECK(mpqs::ckpt::readCheckpoint((tdir / "ck0").string() + "/sieve.ckpt", res0),
              "readCheckpoint (0-node cluster)");
        CHECK(res0.cluster.completed_prefix_cursor == 12345ull, "0-node prefix round-trips");
        CHECK(res0.cluster.initial_high_water.empty(), "0-node high-water array empty");
    }

    fs::remove_all(tdir);

    if (g_failures == 0) {
        std::printf("checkpoint_io: ALL CHECKS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "checkpoint_io: %d CHECK(s) FAILED\n", g_failures);
    return 1;
}
