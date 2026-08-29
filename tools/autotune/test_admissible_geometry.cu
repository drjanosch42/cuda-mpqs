// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Committed regression test for the v1.0.6 SM-aligned launch geometry.
//
// It exercises the REAL production validator (mpqs::autotune::KernelLaunchValidator) through
// its explicit-DeviceLimits constructor, so the test is deterministic on any machine and needs
// no CUDA device. Two rule sets are compared on the SAME tuples:
//
//   * allow_nonpow2_geometry = false  — the legacy / wide regime, i.e. the v1.0.5 rules
//     verbatim: all eight parameters must be powers of two;
//   * allow_nonpow2_geometry = true   — the narrow BATCH regime, where
//     {subCubeSize, metaGridDim, sasGridDim} (and the derived num_polyBlocksPerThreadBlock)
//     may be SM-aligned instead, subject to the exact-partition and GATHER-decomposition
//     invariants that the power-of-two world used to guarantee for free.
//
// Normative admissible set, for the narrow batch path:
//   pbs, ni, bpc, mB, sB power-of-two;  np % (mG*pbs) == 0 with npbptb = np/(mG*pbs) >= 1;
//   mG*pbs <= np;  mG <= np;  sG | np;  np/sG a power of two;  sG <= np.
//
// Asserts:
//   (A) BIT-IDENTITY of the old rule set: every tuple's verdict under
//       allow_nonpow2_geometry=false matches the v1.0.5 "all 8 pow2 + old arithmetic"
//       predicate, including the new G1/G2/G3 checks being tautologies there.
//   (B) The SM-aligned tuples are ACCEPTED in batch mode: A100 864/108/864,
//       A100 fallback 432/108/432, RTX 5070 Ti 560/70/560, and rejected in legacy/wide mode.
//   (C) The deliberately-broken tuples are REJECTED in BOTH modes, each for its own reason:
//       an (I1)/exact-partition violation, a non-divisor sasGridDim, a non-power-of-two
//       GATHER chunk, and sasGridDim > subCubeSize (G3).
//   (D) admissibleSasGridDim() reproduces "round up to the next power of two, clamped to np"
//       on every power-of-two np, and lands on the SM-aligned grid for np = 864 / 560.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "kernel_launch_validator.h"

#include <cstdint>
#include <cstdio>

using mpqs::autotune::DeviceLimits;
using mpqs::autotune::KernelLaunchValidator;
using mpqs::autotune::Params8;
using mpqs::autotune::SieveConstants;
using mpqs::autotune::admissibleSasGridDim;

static int g_failures = 0;

static void check(bool cond, const char* what) {
    if (!cond) { std::printf("  FAIL: %s\n", what); ++g_failures; }
    else       { std::printf("  ok:   %s\n", what); }
}

// Synthetic A100-class device: 108 SMs, 40 GiB, 164 KiB opt-in smem, 1024 threads/block.
static DeviceLimits makeDevice(uint32_t sm_count) {
    DeviceLimits d{};
    d.totalGlobalMem              = 40ull * 1024 * 1024 * 1024;
    d.maxSharedMemPerBlock        = 163 * 1024;
    d.sharedMemPerMultiprocessor  = 164 * 1024;
    d.maxThreadsPerBlock          = 1024;
    d.maxThreadsPerMP             = 2048;
    d.multiProcessorCount         = sm_count;
    d.maxGridDimX                 = 2147483647;
    d.maxBlockDimX                = 1024;
    return d;
}

// Narrow (uint8) sieve constants at an RSA-100-like operating point, with shc_dim large
// enough that np = 864 clears LEQ(np <= 2^(shc_dim-1)) = 1024.
static SieveConstants makeConstants(bool relax) {
    SieveConstants sc{};
    sc.shc_dim            = 11;         // 2^(11-1) = 1024 >= 864
    sc.M                  = 262144;
    sc.sievingBlockSize   = 65536;
    sc.globalBucketSize   = 32768;
    sc.bigPrimeStartIndex = 2048;
    sc.accumulatorBytes   = 1;
    sc.allow_nonpow2_geometry = relax;
    return sc;
}

// The v1.0.5 predicate, transcribed literally, as the bit-identity oracle for (A).
static bool legacyOracle(const Params8& p, const SieveConstants& sc) {
    for (uint32_t x : p) if (x == 0 || (x & (x - 1)) != 0) return false;
    const uint32_t np = p[0], ni = p[1], pbs = p[2], bpc = p[3], mG = p[4], sG = p[6];
    if (np > (1u << (sc.shc_dim - 1))) return false;
    if (bpc > ni) return false;
    if (mG > np) return false;
    if (mG * pbs > np) return false;
    if (sG > np) return false;
    const uint32_t npbptb = np / (mG * pbs);
    if (npbptb == 0 || (npbptb & (npbptb - 1)) != 0) return false;
    if (npbptb * pbs * mG != np) return false;
    if ((ni / bpc) * bpc != ni) return false;
    return true;
}

int main() {
    const DeviceLimits devA100 = makeDevice(108);
    const KernelLaunchValidator vStrict(devA100, makeConstants(false));  // legacy / wide
    const KernelLaunchValidator vBatch (devA100, makeConstants(true));   // narrow batch

    // ---------------------------------------------------------------------
    // (A) Bit-identity of the strict rule set on a broad pow2 sweep.
    // ---------------------------------------------------------------------
    std::printf("(A) strict regime == power-of-two-only predicate on the pow2 sweep\n");
    const SieveConstants scStrict = makeConstants(false);
    uint32_t compared = 0, mismatches = 0;
    for (uint32_t np : {128u, 256u, 512u, 1024u})
      for (uint32_t ni : {1u, 2u, 4u, 8u, 16u})
        for (uint32_t pbs : {1u, 2u, 4u, 8u, 16u})
          for (uint32_t bpc : {1u, 2u, 4u, 8u, 16u})
            for (uint32_t mG : {32u, 64u, 128u, 256u})
              for (uint32_t sG : {32u, 128u, 256u, 512u}) {
                  const Params8 p = {np, ni, pbs, bpc, mG, 1024u, sG, 1024u};
                  ++compared;
                  if (vStrict.isValid(p) != legacyOracle(p, scStrict)) ++mismatches;
              }
    std::printf("  compared %u tuples\n", compared);
    check(mismatches == 0, "strict-regime verdicts identical to the power-of-two-only predicate");

    // The record pin must stay valid in BOTH regimes (it is a pow2 tuple).
    const Params8 recordPin = {512u, 8u, 4u, 8u, 128u, 1024u, 256u, 1024u};
    check(vStrict.isValid(recordPin), "RTX record pin 512,8,4,8,128,1024,256,1024 valid (strict)");
    check(vBatch.isValid(recordPin),  "RTX record pin valid (batch, relaxation is a superset)");
    const Params8 a100Pin = {512u, 8u, 8u, 8u, 64u, 1024u, 512u, 1024u};
    check(vStrict.isValid(a100Pin), "A100 pin of record 512,8,8,8,64,1024,512,1024 valid (strict)");
    check(vBatch.isValid(a100Pin),  "A100 pin of record valid (batch)");

    // ---------------------------------------------------------------------
    // (B) SM-aligned tuples: accepted in batch mode, rejected in legacy/wide mode.
    // ---------------------------------------------------------------------
    std::printf("(B) SM-aligned tuples\n");
    const Params8 a100Exact  = {864u, 8u, 8u, 8u, 108u, 1024u, 864u, 1024u};  // 108 SMs, 1.000 waves
    const Params8 a100Fallbk = {432u, 8u, 4u, 8u, 108u, 1024u, 432u, 1024u};  // shc_dim=10 fallback
    const Params8 rtxExact   = {560u, 8u, 8u, 8u,  70u, 1024u, 560u, 1024u};  // 70 SMs (5070 Ti)
    check(vBatch.isValid(a100Exact),   "864,8,8,8,108,1024,864,1024 accepted (narrow batch)");
    check(vBatch.isValid(a100Fallbk),  "432,8,4,8,108,1024,432,1024 accepted (narrow batch)");
    check(vBatch.isValid(rtxExact),    "560,8,8,8,70,1024,560,1024 accepted (narrow batch)");
    check(!vStrict.isValid(a100Exact), "864-tuple REJECTED in legacy/wide mode");
    check(!vStrict.isValid(a100Fallbk),"432-tuple REJECTED in legacy/wide mode");
    check(!vStrict.isValid(rtxExact),  "560-tuple REJECTED in legacy/wide mode");

    // The exact decompositions the tuples rely on (documented so a future edit cannot
    // silently change what "exact wave" means here):
    //   864 = 108 * 1 * 8   => npbptb 1, SCATTER waves 864/(108*8*1) ... grid 108 = 1.000/SM
    //   864 / 864 = chunk 1 (pow2)  => GATHER grid 864 = 8.000 waves on 108 SMs
    check(864u % (108u * 8u) == 0u, "864 = 108*8*npbptb exactly (I1)");
    check(560u % ( 70u * 8u) == 0u, "560 = 70*8*npbptb exactly (I1)");

    // ---------------------------------------------------------------------
    // (C) Deliberately-broken tuples must be rejected in BOTH regimes.
    // ---------------------------------------------------------------------
    std::printf("(C) invalid tuples rejected in both regimes\n");
    // (I1) violation: 216*8 = 1728 > 864 — metaGridDim*polyBlockSize exceeds np.
    const Params8 badPartition = {864u, 8u, 8u, 8u, 216u, 1024u, 864u, 1024u};
    // Exact-partition violation with mG*pbs <= np but np % (mG*pbs) != 0: 864 % (100*8) != 0.
    const Params8 badPartition2 = {864u, 8u, 8u, 8u, 100u, 1024u, 864u, 1024u};
    // Non-divisor GATHER grid: 500 does not divide 864 (G1).
    const Params8 badSasDivisor = {864u, 8u, 8u, 8u, 108u, 1024u, 500u, 1024u};
    // Divisor but non-pow2 chunk: 864/288 = 3 (G2).
    const Params8 badSasChunk   = {864u, 8u, 8u, 8u, 108u, 1024u, 288u, 1024u};
    // Zero-trip GATHER: sasGridDim > np (G3).
    const Params8 badSasTooBig  = {512u, 8u, 4u, 8u, 128u, 1024u, 1024u, 1024u};
    check(!vBatch.isValid(badPartition)  && !vStrict.isValid(badPartition),
          "216-grid (I1) violation rejected in both regimes");
    check(!vBatch.isValid(badPartition2) && !vStrict.isValid(badPartition2),
          "np % (mG*pbs) != 0 rejected in both regimes");
    check(!vBatch.isValid(badSasDivisor) && !vStrict.isValid(badSasDivisor),
          "non-divisor sasGridDim (G1) rejected in both regimes");
    check(!vBatch.isValid(badSasChunk)   && !vStrict.isValid(badSasChunk),
          "non-pow2 GATHER chunk (G2) rejected in both regimes");
    check(!vBatch.isValid(badSasTooBig)  && !vStrict.isValid(badSasTooBig),
          "sasGridDim > subCubeSize (G3) rejected in both regimes");

    // diagnose() must name the actual violated invariant, not a generic message.
    const std::string dDiv = vBatch.diagnose(badSasDivisor);
    const std::string dChk = vBatch.diagnose(badSasChunk);
    const std::string dPar = vBatch.diagnose(badPartition2);
    std::printf("  diagnose(non-divisor sas) = %s\n", dDiv.c_str());
    std::printf("  diagnose(non-pow2 chunk)  = %s\n", dChk.c_str());
    std::printf("  diagnose(bad partition)   = %s\n", dPar.c_str());
    check(dDiv.find("does not divide") != std::string::npos,
          "diagnose() names the G1 divisibility failure");
    check(dChk.find("chunk") != std::string::npos,
          "diagnose() names the G2 chunk failure");
    check(dPar.find("SCATTER exact-partition") != std::string::npos,
          "diagnose() names the SCATTER exact-partition failure");

    // ---------------------------------------------------------------------
    // (D) admissibleSasGridDim: pow2 back-compatibility + SM-aligned behaviour.
    // ---------------------------------------------------------------------
    std::printf("(D) admissibleSasGridDim\n");
    auto nextPow2ClampedTo = [](uint32_t np, uint32_t m) -> uint32_t {
        if (m == 0) return 0;
        uint32_t v = m - 1;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        ++v;
        return (v > np) ? np : v;
    };
    uint32_t pow2Mismatches = 0;
    for (uint32_t np : {64u, 128u, 256u, 512u, 1024u, 2048u})
        for (uint32_t m = 1; m <= 2 * np; ++m)
            if (admissibleSasGridDim(np, m) != nextPow2ClampedTo(np, m)) ++pow2Mismatches;
    check(pow2Mismatches == 0,
          "pow2 np: equals the previous next-pow2-then-clamp result for every min_sas");
    // np = 864 = 2^5 * 27 => admissible {27,54,108,216,432,864}. LP min_sas = ceil(864*8/64) = 108.
    check(admissibleSasGridDim(864u, 108u) == 108u, "np=864, min_sas=108 -> 108 (SM-aligned)");
    check(admissibleSasGridDim(864u, 109u) == 216u, "np=864, min_sas=109 -> 216 (next admissible)");
    check(admissibleSasGridDim(864u, 1u)   ==  27u, "np=864, min_sas=1   -> 27 (odd part)");
    check(admissibleSasGridDim(864u, 9999u)== 864u, "np=864, min_sas>np  -> np (clamped)");
    // np = 560 = 2^4 * 35 => admissible {35,70,140,280,560}. LP min_sas = ceil(560*8/64) = 70.
    check(admissibleSasGridDim(560u, 70u)  ==  70u, "np=560, min_sas=70  -> 70 (SM-aligned)");
    check(admissibleSasGridDim(0u, 8u)     ==   0u, "np=0 -> 0 (degenerate input)");

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
