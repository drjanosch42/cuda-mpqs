# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski

# Main Driver
add_executable(cuda-mpqs cuda-mpqs.cpp)

# Critical: Resolve device symbols because we have extensive RDC (Relocatable Device Code)
# across multiple static libraries (sieve, postproc, linalg, common).
set_target_properties(cuda-mpqs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cuda-mpqs PRIVATE
    mpqs_orchestrator
    cudampqs_cluster
    mpqs_autotune
    mpqs_sieve
    mpqs_postproc
    mpqs_matrix
    mpqs_sqrt
    lingen_ops
    cudampqs_build_flags
)

# Standalone relation validator — exhaustive CPU-side correctness + LP primality
# checker for saved relations (.v2 / .soa). CPU-only (no kernel launches); compiled
# as CUDA only so the __host__ __device__ math headers compile. Links mpqs_common
# (relation_io + uint512/math_utils headers) and inherits OpenMP from build flags.
add_executable(relation_validator
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure/relation_validator.cu)
set_target_properties(relation_validator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# Pass -fopenmp to the nvcc host compiler so the OpenMP pragma is honoured
# (OpenMP::OpenMP_CXX only supplies the runtime + the CXX-language flag).
target_compile_options(relation_validator PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_libraries(relation_validator PRIVATE mpqs_common cudampqs_build_flags)

# 64-bit number-theory primitives regression test (Stage 1: branch-fixed character
# columns). CPU-only host test (no kernel launches); compiled as CUDA so the
# __host__ __device__ math headers compile. Each primitive (Tonelli_Shanks_u64,
# jacobi_u64, is_prime_u64, uint512::mod_uint64) is checked against an independent
# 128-bit oracle and cross-checked against the existing uint32 implementations.
# Links mpqs_sieve (provides the *_u64 free functions) + mpqs_common (uint512).
add_executable(test_nt_primitives_u64
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure/test_nt_primitives_u64.cu)
set_target_properties(test_nt_primitives_u64 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(test_nt_primitives_u64 PRIVATE mpqs_sieve mpqs_common cudampqs_build_flags)
# Registered with CTest (root-level enable_testing() is in the top CMakeLists).
add_test(NAME nt_primitives_u64 COMMAND test_nt_primitives_u64)

# Aux-prime selection regression test (Stage 2: branch-fixed character columns).
# Verifies CharacterColumnComputer::selectAuxPrimes() under CharMode {NORM, BRANCH}:
# BRANCH selects q > lp1_bound via a 64-bit walk with a fixed Tonelli root t_s
# (t_s^2 == N mod q), deterministically; NORM reproduces the legacy start-at-3
# uint32 selection bit-for-bit (checked against an inline independent oracle).
# CPU-only host test (no kernel launches); compiled as CUDA so the
# __host__ __device__ math headers compile. Links mpqs_matrix (selectAuxPrimes +
# jacobi_symbol_dev), mpqs_sieve (*_u64 primitives), and mpqs_common (uint512).
add_executable(test_aux_prime_selection
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure/test_aux_prime_selection.cu)
set_target_properties(test_aux_prime_selection PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(test_aux_prime_selection PRIVATE mpqs_matrix mpqs_sieve mpqs_common cudampqs_build_flags)
add_test(NAME aux_prime_selection COMMAND test_aux_prime_selection)

# Branch-fixed character-bit evaluator genus-correctness test (Stage 3).
# Certifies mpqs::matrix::branchCharBit (src/matrix/gpu_char_cols.cuh) against the
# genus-correct Python reference: a generator turns branch_bit_raw/tonelli_root/
# legendre_pm from the genus-correct prototype into a C++ fixture of direct +
# homomorphism triples. The test checks (1) reference parity, (2) the F2 homomorphism
# branchCharBit(product) == bit_i XOR bit_j, and (3) host==device parity over
# >=1e4 random 64-bit triples via a device kernel.
# The fixture (branch_char_fixture.h) is pre-generated and committed under
# tools/sqrt_failure/. The generator (gen_branch_char_fixture.py) imports a
# genus-correct Python reference that lives in the development tree only, so the
# build consumes the checked-in header directly rather than regenerating it.
# To refresh the fixture, re-run gen_branch_char_fixture.py in the dev tree and
# commit the result.
add_executable(test_branch_char_bit
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure/test_branch_char_bit.cu)
target_include_directories(test_branch_char_bit PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure)
set_target_properties(test_branch_char_bit PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(test_branch_char_bit PRIVATE mpqs_matrix mpqs_sieve mpqs_common cudampqs_build_flags)
add_test(NAME branch_char_bit COMMAND test_branch_char_bit)

# Stage 4 (branch-fixed character columns): birth-capture + persistence test.
# Verifies (a) PARITY — the relation-birth char-vector formula (calculate_sqrt_of_QX
# 5-arg + uint512::mod_uint64 + sub_mod + branchCharBit) matches an independent
# signed-residue Legendre oracle; (b) I/O ROUND-TRIP — relation_io::serialize_v2/
# deserialize_v2 preserves per-relation char_bits and the {r, aux_primes, t_s}
# metadata, while a norm-mode .v2 (no char flag) still loads with empty char_bits and
# has_char_bits==false (back-compat); (c) CLUSTER WIRE — cluster::serializeRelationBatch
# round-trips each relation's char vector. CPU-only host test (no kernel launches);
# compiled as CUDA so the __host__ __device__ math headers compile. Links mpqs_matrix
# (branchCharBit), mpqs_sieve (*_u64 primitives), cudampqs_cluster (wire), mpqs_common.
add_executable(test_char_bits_persist
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure/test_char_bits_persist.cu)
set_target_properties(test_char_bits_persist PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# serialization.h pulls in the sieve common.h (factoringData); matrix headers for
# branchCharBit. Add both include dirs explicitly (PRIVATE link does not propagate them).
target_include_directories(test_char_bits_persist PRIVATE
    ${CMAKE_SOURCE_DIR}/src/sieve
    ${CMAKE_SOURCE_DIR}/src/matrix
    ${CMAKE_SOURCE_DIR}/src/cluster)
target_link_libraries(test_char_bits_persist PRIVATE
    mpqs_matrix mpqs_sieve cudampqs_cluster mpqs_common cudampqs_build_flags)
add_test(NAME char_bits_persist COMMAND test_char_bits_persist)

# Stage 5 (branch-fixed character columns): CPU XOR-propagation homomorphism test.
# Certifies the four CPU branch paths wired in Stage 5:
#   (A) LP-combination XOR (largeprime.cu / cluster cpu_lp.cu): combined char vector
#       == XOR of constituents == branchCharBit of the TRUE product field element;
#       norm-mode (0^0) stays 0.
#   (B) merge-tree reduction XOR (computeProductCharacterColumns BRANCH): each reduced
#       row's composed columns == XOR-with-multiplicity of its leaves' persisted
#       char_bits along tree.expand(row_map[r]) == branchCharBit of the product.
#   (C) per-relation adapter (compute(batch, BRANCH)) unpacks char_bits verbatim.
#   (D) append-after-reduction structural check: AppendCharacterColumns adds exactly
#       32 columns at indices >= the pre-append n_cols (never inside any reduction step).
# Pure host test (no kernel launches); compiled as CUDA for the __host__ __device__
# math headers. Links mpqs_matrix (propagation + branchCharBit), mpqs_sieve (*_u64
# primitives), mpqs_common (uint512 / HostRelationBatch).
add_executable(test_char_xor_propagation
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure/test_char_xor_propagation.cu)
set_target_properties(test_char_xor_propagation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(test_char_xor_propagation PRIVATE
    ${CMAKE_SOURCE_DIR}/src/sieve
    ${CMAKE_SOURCE_DIR}/src/matrix)
target_link_libraries(test_char_xor_propagation PRIVATE
    mpqs_matrix mpqs_sieve mpqs_common cudampqs_build_flags)
add_test(NAME char_xor_propagation COMMAND test_char_xor_propagation)

# Stage 6 (branch-fixed character columns): M9v2 PACKED propagation parity test.
# Certifies the host-side mirror of the packed two-array char-vector reduction
# (seed d_char_bits -> XOR-compose d_ws_char_bits at the merge site via the
# ROW_WS_BIT selector -> compaction relocation -> final ptr_val/ws_idx gather at
# preprocess.cpp) is BIT-FOR-BIT identical to the authoritative CPU merge-tree
# leaf-XOR oracle computeProductCharacterColumns(BRANCH) and to branchCharBit of
# the true product field element (Stage-3 genus reference), plus the NORM no-op
# guard (all-0 seeds -> all-0 composed/gathered vectors). Pure host test (no kernel
# launches); compiled as CUDA for the __host__ __device__ math headers. Links
# mpqs_matrix (oracle + branchCharBit + ROW_WS_BIT), mpqs_sieve (*_u64), mpqs_common.
add_executable(test_packed_char_gather
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure/test_packed_char_gather.cu)
set_target_properties(test_packed_char_gather PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(test_packed_char_gather PRIVATE
    ${CMAKE_SOURCE_DIR}/src/sieve
    ${CMAKE_SOURCE_DIR}/src/matrix)
target_link_libraries(test_packed_char_gather PRIVATE
    mpqs_matrix mpqs_sieve mpqs_common cudampqs_build_flags)
add_test(NAME packed_char_gather COMMAND test_packed_char_gather)

# Stage 6 (branch-fixed character columns): DEVICE-LEVEL packed parity test.
# Drives the REAL M9v2 packed GPU pipeline (gpuPreprocessMatrix_packed, BRANCH) on a
# controlled relation set with engineered unique tag columns + forced weight-2 merges,
# then asserts the device-composed/gathered/appended 32 branch char columns equal the
# CPU oracle (XOR of each reduced row's constituent original-row seeds) BIT-FOR-BIT,
# plus a norm-mode all-0 no-op guard. Launches CUDA kernels — skips cleanly (exit 0)
# when no CUDA device is present. Links mpqs_matrix (packed pipeline), mpqs_sieve
# (*_u64 primitives), mpqs_common (RelationBatch / uint512).
add_executable(test_packed_char_device_parity
    ${CMAKE_SOURCE_DIR}/tools/sqrt_failure/test_packed_char_device_parity.cu)
set_target_properties(test_packed_char_device_parity PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(test_packed_char_device_parity PRIVATE
    ${CMAKE_SOURCE_DIR}/src/sieve
    ${CMAKE_SOURCE_DIR}/src/matrix)
target_link_libraries(test_packed_char_device_parity PRIVATE
    mpqs_matrix mpqs_sieve mpqs_common cudampqs_build_flags)
add_test(NAME packed_char_device_parity COMMAND test_packed_char_device_parity)

# Optional Sqrt specific benchmark (EXCLUDE_FROM_ALL: uses AoS Relation API, needs M2 update for SoA)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sqrt_benchmark.cpp")
    add_executable(bench_sqrt EXCLUDE_FROM_ALL sqrt_benchmark.cpp
        ${CMAKE_SOURCE_DIR}/src/sqrt/fake_relation_generator.cpp)
    set_target_properties(bench_sqrt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(bench_sqrt PRIVATE mpqs_sqrt mpqs_common cudampqs_build_flags)
endif()
