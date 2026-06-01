# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2024-2026 Fabian Januszewski
# This file is part of the Block Wiedemann implementation.
# See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

import sys
import os
import struct
import numpy as np

# Import trusted reference
try:
    import block_wiedemann_lingen_v5 as bw_ref
except ImportError:
    print("Error: Ensure block_wiedemann_lingen_v5.py is in the same directory.")
    sys.exit(1)

# --- FNV1A hash function ---

def fnv1a_64_hash(data_list):
    """
    Implements 64-bit FNV-1a hash matching the C++ fnv1a_hash.
    Expects a list or array of 64-bit integers.
    """
    FNV_OFFSET_BASIS = 0xcbf29ce484222325
    FNV_PRIME = 0x100000001b3
    MASK_64 = 0xFFFFFFFFFFFFFFFF

    hash_val = FNV_OFFSET_BASIS
    for val in data_list:
        # Pack the 64-bit integer into 8 bytes (little-endian) to match C++ loop
        bytes_8 = struct.pack('<Q', int(val))
        for byte in bytes_8:
            hash_val ^= byte
            hash_val = (hash_val * FNV_PRIME) & MASK_64
            
    return hash_val

# --- Binary Loaders ---

def load_bin_matrix(fname):
    """Parses [Magic:4][Rows:4][Cols:4][Data:Rows*Cols uint8]"""
    if not os.path.exists(fname): raise FileNotFoundError(f"Missing {fname}")
    with open(fname, 'rb') as f:
        magic = f.read(4)
        if magic != b'MATA': raise ValueError(f"Bad magic {magic} in {fname}")
        rows, = struct.unpack('<I', f.read(4))
        cols, = struct.unpack('<I', f.read(4))
        data = np.frombuffer(f.read(), dtype=np.uint8)

    raw_hashing = data.copy()

    return data.reshape(rows, cols), raw_hashing

def load_bin_vector(fname):
    """Parses [Magic:4][Rows:4][Blk:4][Data:Packed u64]"""
    if not os.path.exists(fname): raise FileNotFoundError(f"Missing {fname}")
    with open(fname, 'rb') as f:
        magic = f.read(4)
        if magic != b'VECT': raise ValueError(f"Bad magic {magic} in {fname}")
        rows, = struct.unpack('<I', f.read(4))
        blk, = struct.unpack('<I', f.read(4))
        raw = np.frombuffer(f.read(), dtype=np.uint64)

    raw_hashing = raw.copy()
        
    words_per_row = (blk + 63) // 64
    mat = np.zeros((rows, blk), dtype=int)
    raw = raw.reshape(rows, words_per_row)
    for r in range(rows):
        for w in range(words_per_row):
            val = int(raw[r, w])
            for b in range(64):
                if w*64 + b < blk:
                    if (val >> b) & 1: mat[r, w*64 + b] = 1
    return mat, raw_hashing


def load_bin_sequence(fname):
    """Parses [Magic:4][Terms:4][Rows:4][Cols:4] (SEQ2) or [Magic:4][Terms:4][Blk:4] (SEQU)"""
    if not os.path.exists(fname): raise FileNotFoundError(f"Missing {fname}")
    with open(fname, 'rb') as f:
        magic = f.read(4)
        m = 0
        n = 0
        terms = 0
        
        if magic == b'SEQ2':
            terms, = struct.unpack('<I', f.read(4))
            m, = struct.unpack('<I', f.read(4))
            n, = struct.unpack('<I', f.read(4))
        elif magic == b'SEQU':
            terms, = struct.unpack('<I', f.read(4))
            blk, = struct.unpack('<I', f.read(4))
            m = blk
            n = blk
        else:
            raise ValueError(f"Bad magic {magic} in {fname}")

        raw = np.frombuffer(f.read(), dtype=np.uint64)

    raw_hashing = raw.copy() 
       
    words_per_row = (n + 63) // 64
    words_total = m * words_per_row
    
    if len(raw) != terms * words_total:
        raise ValueError(f"Size mismatch in {fname}: expected {terms*words_total} words, got {len(raw)}")

    raw = raw.reshape(terms, words_total)
    seq = []
    for t in range(terms):
        mat = np.zeros((m, n), dtype=int)
        for r in range(m):
            for w in range(words_per_row):
                val = int(raw[t, r*words_per_row + w])
                for b in range(64):
                    if w*64 + b < n:
                        if (val >> b) & 1: mat[r, w*64+b] = 1
        seq.append(mat)
    return seq, raw_hashing

def load_bin_poly(fname):
    """Parses [Magic:4][Len:4][Rows:4][Cols:4][Data:Packed u64]"""
    if not os.path.exists(fname): return None
    with open(fname, 'rb') as f:
        magic = f.read(4)
        if magic != b'POLY': raise ValueError(f"Bad magic {magic} in {fname}")
        length, = struct.unpack('<I', f.read(4))
        rows, = struct.unpack('<I', f.read(4))
        cols, = struct.unpack('<I', f.read(4))
        raw = np.frombuffer(f.read(), dtype=np.uint64)

    raw_hashing = raw.copy()
        
    words_row = (cols+63)//64
    raw = raw.reshape(length, rows * words_row)
    poly = []
    for k in range(length):
        mat = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for w in range(words_row):
                val = int(raw[k, r*words_row + w])
                for b in range(64):
                    if w*64+b < cols:
                        if (val >> b) & 1: mat[r, w*64+b] = 1
        poly.append(mat)
    return poly, raw_hashing

def load_bin_solution(fname):
    """Parses [Magic:4][Rows:4][Cols=1:4][Data:Packed u64]"""
    if not os.path.exists(fname): return None
    with open(fname, 'rb') as f:
        magic = f.read(4)
        if magic != b'SOLN': raise ValueError(f"Bad magic {magic} in {fname}")
        rows, = struct.unpack('<I', f.read(4))
        cols, = struct.unpack('<I', f.read(4))
        raw = np.frombuffer(f.read(), dtype=np.uint64)

    raw_hashing = raw.copy()
        
    w_vec = np.zeros((rows, 1), dtype=int)
    for r in range(rows):
        word = r // 64
        bit = r % 64
        if word < len(raw):
            if (raw[word] >> bit) & 1:
                w_vec[r, 0] = 1
    return w_vec, raw_hashing

# --- Helpers ---

def compare_coefficients(Pi_cpp, f_ref, n):
    limit = min(len(Pi_cpp), len(f_ref))
    mismatches = 0
    max_to_print = 5
    for k in range(limit):
        # f_ref is [dim x n], Pi_cpp is [dim x dim] but we only care about first n cols
        # Wait, f_ref is shape [dim x n] in Python ref (generator).
        # Pi_cpp is stored as [dim x dim] but logically we used first n cols.
        # Actually, Python reference defines generator as matrix with coefficients (m+n) x n.
        # Mission 5 implemented build_f_init as dim x dim.
        # Mission 7 implemented elimination on dim x dim.
        # So Pi_cpp is dim x dim.
        # We compare Pi_cpp[k][:, :n] with f_ref[k].
        
        mat_cpp = Pi_cpp[k][:, :n]
        mat_ref = f_ref[k] 
        diff = (mat_cpp != mat_ref)
        if np.any(diff):
            rows, cols = np.where(diff)
            for r, c in zip(rows, cols):
                mismatches += 1
                if mismatches <= max_to_print:
                    print(f"      Mismatch at Deg={k}, Row={r}, Col={c}: C++={mat_cpp[r,c]}, Ref={mat_ref[r,c]}")
    return mismatches

def check_row_annihilation(S, Pi_poly, m, n):
    deg_pi = len(Pi_poly)
    len_S = len(S)
    dim = Pi_poly[0].shape[0]
    valid_rows = []
    
    for r in range(dim):
        failed = False
        for t in range(deg_pi, len_S):
            acc = np.zeros(m, dtype=int)
            for k in range(deg_pi):
                s_idx = t - k
                if s_idx < 0: continue
                pi_row = Pi_poly[k][r, :n] 
                s_T = S[s_idx].T          
                acc ^= (pi_row @ s_T) % 2
            if np.any(acc):
                failed = True
                break
        if not failed: valid_rows.append(r)
    return valid_rows

def gf2_rank(vectors, N):
    """
    Computes the rank of a set of binary vectors of length N over GF(2).
    Vectors can be a list of (N, 1) ndarrays or flat arrays.
    """
    if not vectors:
        return 0
    
    # Convert to list of flat arrays for processing
    # Gaussian Elimination with pivot tracking
    basis = []
    
    for v in vectors:
        row = v.flatten().copy()
        
        # Reduce against existing basis
        for b_row, pivot_idx in basis:
            if row[pivot_idx]:
                row ^= b_row
        
        # If not zero, it's independent
        if np.any(row):
            # Find first non-zero bit
            pivot = np.argmax(row) 
            basis.append((row, pivot))
            
    return len(basis)

# --- Main ---

def run_verify(prefix, transpose_mode=False):
    print(f"=== Verification Pipeline: {prefix} ===")
    
    # --- Load Inputs ---
    try:
        A_raw, A_h = load_bin_matrix(f"{prefix}_A.bin")
        X, X_h = load_bin_vector(f"{prefix}_X.bin")
        Y, Y_h = load_bin_vector(f"{prefix}_Y.bin")
        B = A_raw.T if transpose_mode else A_raw
        m, n = X.shape[1], Y.shape[1]
    except Exception as e:
        print(f"[FAIL] Loading artifacts: {e}"); return

    A_hash = fnv1a_64_hash(A_h)
    print(f"  [HASH] A hash: {hex(A_hash)}")        
    X_hash = fnv1a_64_hash(X_h)
    print(f"  [HASH] X hash: {hex(X_hash)}")        
    Y_hash = fnv1a_64_hash(Y_h)
    print(f"  [HASH] Y hash: {hex(Y_hash)}")        
        
    ref_solver = bw_ref.BlockWiedemannGF2(B, m=m, n=n)

    # --- Stage 1 (Sequence) ---
    print("\n--- Stage 1: Sequence S ---")
    try:
        S_cpp, S_h = load_bin_sequence(f"{prefix}_S.bin")
        S_cpp_hash = fnv1a_64_hash(S_h)
        print(f"  [HASH] S_cpp hash: {hex(S_cpp_hash)}")

        cpp_m, cpp_n = S_cpp[0].shape
        if cpp_m != m or cpp_n != n:
            print(f"  [FAIL] Sequence dimension mismatch. C++: {cpp_m}x{cpp_n}, Expected: {m}x{n}")
            return
        
        S_ref = ref_solver.generate_krylov_sequence(X, Y, len(S_cpp))
        coincidence = [np.array_equal(a, b) for a, b in zip(S_cpp, S_ref)]
        if np.all(coincidence):
            print("  [PASS] Bit-exact match.")
        else:
            first_diff = np.argmin(coincidence)
            print(f"  [FAIL] Mismatch in Sequence S at index {first_diff}.")
            return
    except FileNotFoundError: print("  [SKIP] No S.bin"); return

    # --- Stage 2 (Generator) ---
    print("\n--- Stage 2: Generator Pi ---")
    Pi_cpp, Pi_h = load_bin_poly(f"{prefix}_Pi.bin")
    
    if Pi_cpp:
        Pi_cpp_hash = fnv1a_64_hash(Pi_h)
        print(f"  [HASH] Pi_cpp hash: {hex(Pi_cpp_hash)}")        
        
        # Note: We FORCE early_stop=False to match C++ full-length behavior
        f_ref, degs_ref = ref_solver.block_berlekamp_massey(S_cpp, early_stop=False)
        mismatches = compare_coefficients(Pi_cpp, f_ref, n)
        if mismatches == 0:
            print("  [PASS] Generator Coefficients: Exact Match")
        else:
            print(f"  [FAIL] Generator Coefficients: {mismatches} mismatches.")
            return
    else:
        print("  [SKIP] No Pi.bin"); return

    # --- Stage 3 (Solutions) ---
    print("\n--- Stage 3: Solution Completeness Check ---")
    
    # 1. Exhaustive Search for Reference Solutions
    print("  [Ref] Exhaustively evaluating ALL generator rows to find full Kernel Space...")
    
    # Precompute B powers for speed
    if ref_solver._last_Z is None: ref_solver._last_Z = Y.copy()
    B_pow_Z = []
    curr_Z = ref_solver._last_Z.copy()
    L = len(S_cpp)
    for _ in range(L + 2):
        B_pow_Z.append(curr_Z)
        curr_Z = (ref_solver.B @ curr_Z) % 2
        
    dim = ref_solver.m + ref_solver.n
    ref_solutions = []
    
    # Iterate every single row of the verified generator
    for r in range(dim):
        w_ref, deg = ref_solver._eval_row_on_Z(B_pow_Z, f_ref, r)
        if deg < 0: continue
        
        # Strip valuation to find kernel vector
        w_strip, v, ok = ref_solver._strip_B_valuation(w_ref, max_steps=L)
        
        # If valid kernel vector (Bw=0) and non-zero
        if ok and np.any(w_strip):
            ref_solutions.append(w_strip)

    rank_ref = gf2_rank(ref_solutions, ref_solver.N)
    print(f"  [Ref] Found {len(ref_solutions)} candidate vectors spanning Rank {rank_ref}.")

    # 2. Analyze C++ Solutions
    import glob
    sol_files = sorted(glob.glob(f"{prefix}_sol_*.bin"))
    
    if not sol_files:
        print("  [WARN] No C++ solutions found.")
    else:
        cpp_solutions = []
        for fname in sol_files:
            w, _ = load_bin_solution(fname)
            # Verify it is actually a kernel vector
            if np.sum((B @ w) % 2) == 0:
                cpp_solutions.append(w)
            else:
                print(f"  [FAIL] C++ solution {fname} is invalid (Bw != 0).")

        rank_cpp = gf2_rank(cpp_solutions, ref_solver.N)
        print(f"  [Cpp] Loaded {len(cpp_solutions)} valid solutions spanning Rank {rank_cpp}.")
        
        # 3. Completeness Verdict
        if rank_cpp == rank_ref:
            print(f"  [PASS] C++ found the FULL solution space (Rank {rank_cpp}). No solutions overlooked.")
        elif rank_cpp < rank_ref:
            print(f"  [FAIL] C++ overlooked solutions! Ref Rank {rank_ref} > Cpp Rank {rank_cpp}.")
        else:
            print(f"  [???] C++ Rank ({rank_cpp}) > Ref Rank ({rank_ref}). This implies C++ found valid vectors NOT in the generator span. Impossible if Pi matches.")

def run_verify_old(prefix, transpose_mode=False):
    print(f"=== Verification Pipeline: {prefix} ===")
    
    try:
        A_raw = load_bin_matrix(f"{prefix}_A.bin")
        X = load_bin_vector(f"{prefix}_X.bin")
        Y = load_bin_vector(f"{prefix}_Y.bin")
        B = A_raw.T if transpose_mode else A_raw
        m, n = X.shape[1], Y.shape[1]
    except Exception as e:
        print(f"[FAIL] Loading artifacts: {e}"); return
        
    ref_solver = bw_ref.BlockWiedemannGF2(B, m=m, n=n)

    # Stage 1
    print("\n--- Stage 1: Sequence S ---")
    try:
        S_cpp = load_bin_sequence(f"{prefix}_S.bin")
        S_ref = ref_solver.generate_krylov_sequence(X, Y, len(S_cpp))
        if np.all([np.array_equal(a, b) for a, b in zip(S_cpp, S_ref)]):
            print("  [PASS] Bit-exact match.")
        else:
            print("  [FAIL] Mismatch in Sequence S.")
            return
    except FileNotFoundError: print("  [SKIP] No S.bin"); return

    # Stage 2
    print("\n--- Stage 2: Generator Pi ---")
    Pi_cpp = load_bin_poly(f"{prefix}_Pi.bin")
    
    # 1. Check Start State (Basis Pairs)
    print("  [Init Check] Computing Reference Basis Pairs...")
    try:
        s_val, basis_pairs = ref_solver._thome_find_s_and_basis_pairs(S_cpp)
        print(f"    Ref: t0={s_val}, Basis Pairs={basis_pairs}...")
        # Compare manually with C++ output
    except Exception as e:
        print(f"    [FAIL] Reference Basis search: {e}")

    if Pi_cpp:
        print("  [Gen Check] Running Reference BBM...")
        f_ref, degs_ref = ref_solver.block_berlekamp_massey(S_cpp, early_stop=False)
        
        mismatches = compare_coefficients(Pi_cpp, f_ref, n)
        if mismatches == 0:
            print("  [PASS] Generator Coefficients: Exact Match")
        else:
            print(f"  [WARN] Generator Coefficients: {mismatches} mismatches.")
            print("    -> Generators likely structurally different (different basis).")
            
            valid_rows = check_row_annihilation(S_cpp, Pi_cpp, m, n)
            if valid_rows:
                print(f"  [PASS] Mathematical Annihilation holds for {len(valid_rows)} rows.")
            else:
                print("  [FAIL] Mathematical Annihilation FAILED.")
    else: print("  [SKIP] No Pi.bin")

    # Stage 3
    print("\n--- Stage 3: Solution w ---")
    import glob
    sol_files = sorted(glob.glob(f"{prefix}_sol_*.bin"))
    
    if not sol_files:
        print("  [SKIP] No C++ solution files.")
        return

    print(f"  [Cpp Sol] Found {len(sol_files)} C++ solutions.")
    
    # 1. Precompute B powers for fast evaluation (Reference)
    print("  [Ref Sol] Precomputing B^k Z for exhaustive verification...")
    # Use internal helper to get the cache
    if ref_solver._last_Z is None: ref_solver._last_Z = Y.copy() # Ensure Z is set
    B_pow_Z = []
    curr_Z = ref_solver._last_Z.copy()
    L = len(S_cpp)
    for _ in range(L + 2):
        B_pow_Z.append(curr_Z)
        curr_Z = (ref_solver.B @ curr_Z) % 2
    
    # 2. Check each C++ solution
    all_pass = True
    
    # We will search through ALL rows of the generator to see if we can reproduce the C++ vector
    # This proves the C++ vector is valid and derived from the same generator.
    dim = ref_solver.m + ref_solver.n
    
    for i, fname in enumerate(sol_files):
        sol_cpp = load_bin_solution(fname)
        
        # Quick Validity Check: B*w = 0
        resid = np.sum((B @ sol_cpp) % 2)
        weight = np.sum(sol_cpp)
        
        if resid != 0:
            print(f"  [FAIL] Sol {i}: Invalid! Residual weight {resid}.")
            all_pass = False
            continue
            
        # Generator Derivation Check
        # We try to find which row 'r' of Pi produced this solution
        match_found = False
        match_row = -1
        
        # Optimization: Most likely C++ picked rows 0..M or sorted by Gamma. 
        # We scan all rows.
        for r in range(dim):
            # Evaluate without valuation stripping first
            w_ref, deg = ref_solver._eval_row_on_Z(B_pow_Z, f_ref, r)
            if deg < 0: continue
            
            # Direct match?
            if np.array_equal(w_ref, sol_cpp):
                match_found = True; match_row = r; break
            
            # Valuation match? (w_ref might need B*w_ref to match sol_cpp)
            w_strip, v, ok = ref_solver._strip_B_valuation(w_ref, max_steps=L)
            if ok and np.array_equal(w_strip, sol_cpp):
                match_found = True; match_row = r; break
                
        if match_found:
            print(f"  [PASS] Sol {i} (Wt: {weight}): Matches Ref Generator Row {match_row}.")
        else:
            print(f"  [WARN] Sol {i} (Wt: {weight}): Valid kernel vector, but not found in Ref Generator scan (or valuation depth exceeded).")
            # If it's a valid kernel vector, it's acceptable, even if we can't trivially map it back to a specific single row 
            # (e.g., if C++ combined rows).
            
    if all_pass:
        print("  [PASS] All C++ solutions are valid kernel vectors.")
    else:
        print("  [FAIL] Nota all C++ solutions are valid kernel vectors or not all found.")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python verify_bw_pipeline.py <prefix>")
    else: run_verify(sys.argv[1])
