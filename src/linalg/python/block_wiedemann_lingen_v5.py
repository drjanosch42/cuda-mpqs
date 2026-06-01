"""
SPDX-License-Identifier: LGPL-3.0-only
Copyright (c) 2024-2026 Fabian Januszewski
This file is part of the Block Wiedemann implementation.
See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

Block Wiedemann over GF(2) (NumPy reference implementation).

This module implements a block Wiedemann pipeline for finding nonzero vectors w
in the right kernel of a singular matrix B over GF(2), i.e. B w = 0.

The code follows the standard 3-phase structure described by Coppersmith (block
Wiedemann + block Berlekamp–Massey) and Thomé (MSLGDC divide-and-conquer
acceleration for the generator stage).

Important indexing convention used by this implementation:
- Phase 1 stores the Krylov sequence as S_k = X^T * B^(k+1) * Z (mod 2),
  because Y is chosen as Y = B Z and the recurrence is started from Y.
  This is equivalent to the usual presentation S_k = X^T * B^k * Y with Y = BZ,
  but it shifts the exponent by +1 relative to Z.

Representation conventions (transpose convention):
- Sequence coefficients S_k are m×n matrices over GF(2).
- A generator candidate is a polynomial vector u(X) with n components, stored as
  one row of a polynomial matrix f(X) whose coefficients are (m+n)×n matrices.
- Nominal degrees γ are integer bounds used for pivot/shift decisions; they are
  bounds and may exceed the true degree due to cancellations.

Performance note:
- Thomé’s MSLGDC recursion structure is implemented, but polynomial/matrix
  multiplications are currently naive (no FFT/NTT), so the implementation is
  mathematically faithful but not asymptotically subquadratic as in the paper.

Key References
-------------
[Thomé2003] E. Thomé, "Subquadratic computation of vector generating polynomials 
            and improvement of the block Wiedemann algorithm", JSC 2003.
[Coppersmith1994] D. Coppersmith, "Solving linear equations over GF(2) via block 
                  Wiedemann algorithm", Math. Comp. 1994.
"""

import numpy as np

class BlockWiedemannGF2:
    """
    Block Wiedemann solver for sparse linear systems over GF(2).

    Given a singular N×N matrix B over GF(2), the solver attempts to construct
    a nonzero kernel vector w with B w = 0 using random block projections.

    Phases:
    - Phase 1: Generate a block Krylov sequence S_k = X^T B^(k+1) Z for k=0..L-1.
    - Phase 2: Compute a vector generating polynomial u(X) (represented inside f(X))
      for the matrix sequence A(X)=∑ S_k X^k, using either:
      * block Berlekamp–Massey (Coppersmith-style iterative update), or
      * Thomé’s MSLGDC (recursive divide-and-conquer product of degree-1 updates).
    - Phase 3: Reconstruct a kernel vector from a selected row of the generator.

    Notes on correctness assumptions:
    - The theoretical guarantees in Coppersmith/Thomé assume non-degenerate random
      projections (informally: X and Z behave “generically” for the given B).
    - The implementation enforces the key invariant that exactly m rows are “shifted”
      (their nominal degree increases) at each Phase-2 update step.

    Attributes
    ----------
    B : ndarray
        The N×N system matrix over GF(2)
    N : int
        Matrix dimension
    m, n : int
        Block dimensions (m ≥ n recommended)
    verbose : bool
        Enable logging output
    """

    def __init__(self, matrix, m=4, n=4, verbose=False):
        """
        Initialize Block Wiedemann solver.

        Parameters
        ----------
        matrix : ndarray
            N×N sparse matrix over integers (reduced mod 2)
        m : int, optional
            Number of left projection vectors (rows in Krylov sequence)
        n : int, optional
            Number of right projection vectors (columns in Krylov sequence)
        verbose : bool, optional
            Enable detailed logging

        Notes
        -----
        For stability, m ≥ n is recommended. The complexity of Phase 2 scales as
        O(mn²k²) for Coppersmith or O(mn³k log²k) for Thomé, where k ≈ N/n.
        """
        self.B = matrix % 2
        self.N = matrix.shape[0]
        self.m = int(m)
        self.n = int(n)
        self.verbose = bool(verbose)

        # Storage for reconstruction phase
        self._last_Z = None

        if self.m < self.n:
            print("[Warning] m < n increases probability of failure (Coppersmith/Thomé condition)")

    def _log(self, msg):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[BW] {msg}")

    # =========================================================================
    # Phase 1: Sequence Generation
    # =========================================================================

    def generate_krylov_sequence(self, X, Z, length):
        """
        Generate the block Krylov sequence used by block Wiedemann.

        Parameters
        ----------
        X : ndarray, shape (N, m), dtype int
            Left projection block over GF(2).
        Z : ndarray, shape (N, n), dtype int
            Right starting block over GF(2).
        length : int
            Number of sequence terms to compute.

        Returns
        -------
        sequence : list[ndarray]
            List of length `length` with matrices S_k of shape (m, n) over GF(2).

        Definition (this implementation)
        -------------------------------
        This routine stores the sequence as
            S_k = X^T * B^(k+1) * Z  (mod 2),   for k = 0..length-1.
        Equivalently, defining Y := B Z, it matches the standard presentation
            S_k = X^T * B^k * Y.

        Side effects
        -----------
        Stores a copy of Z in `self._last_Z` for use during reconstruction.
        """
        self._log(f"Phase 1: Generating {length} Krylov terms...")
        
        self._last_Z = Z.copy()
        
        # We define Y_0 = B * Z. 
        # The sequence is S_k = X^T * B^k * Y_0 = X^T * B^{k+1} * Z
        current_Y = (self.B @ Z) % 2
        
        sequence = []
        for _ in range(length):
            # S_k is m x n
            S = (X.T @ current_Y) % 2
            sequence.append(S)
            current_Y = (self.B @ current_Y) % 2
            
        return sequence

    # =========================================================================
    # Core Arithmetic: Matrix Polynomials over GF(2)
    # =========================================================================

    def _mat_poly_mul(self, P, Q):
        """
        Multiply two polynomial matrices via convolution.

        Parameters
        ----------
        A_poly : list of ndarray
            Polynomial A(z) = Σ A_i z^i, each A_i is p×q matrix
        B_poly : list of ndarray
            Polynomial B(z) = Σ B_i z^i, each B_i is q×r matrix

        Returns
        -------
        result : list of ndarray
            Product C(z) = A(z)B(z), each C_k = Σ_{i+j=k} A_i B_j (mod 2)

        Complexity
        ----------
        O(deg(A)·deg(B)·pqr) scalar multiplications (naive)
        Can be improved to O(pqr·deg log deg) using FFT
        """
        deg_p = len(P) - 1
        deg_q = len(Q) - 1
        deg_r = deg_p + deg_q
        
        rows = P[0].shape[0]
        cols = Q[0].shape[1]
        
        # Initialize result with zeros
        R = [np.zeros((rows, cols), dtype=int) for _ in range(deg_r + 1)]
        
        # Convolve
        for i, p_mat in enumerate(P):
            for j, q_mat in enumerate(Q):
                if np.any(p_mat) and np.any(q_mat):
                    R[i + j] ^= (p_mat @ q_mat) % 2
        return R

    def _apply_poly_to_sequence(self, pi_poly, seq_D):
        """
        Apply a polynomial matrix Π(X) to a truncated power series D(X) modulo X^L.

        Given:
        - Π(X) = ∑_{i=0..d} Π_i X^i  with Π_i in K^{p×q},
        - D(X) = ∑_{k=0..L-1} D_k X^k with D_k in K^{q×r},

        this computes the truncated product
            E(X) = Π(X) D(X) mod X^L,
        i.e. E_k = ∑_{i=0..min(d,k)} Π_i D_{k-i}.

        Usage in MSLGDC:
        After computing Π_left for the low half, the recursion forms
            D' = (Π_left * D) div X^half
        by computing E = Π_left D mod X^b and then slicing coefficients E[half:].
        """
        L = len(seq_D)
        deg_pi = len(pi_poly)
        rows_pi = pi_poly[0].shape[0]
        cols_d = seq_D[0].shape[1]
        
        result = [np.zeros((rows_pi, cols_d), dtype=int) for _ in range(L)]
        
        for i in range(deg_pi):
            pi_i = pi_poly[i]
            # We only affect terms up to L-1
            limit = L - i
            for j in range(limit):
                result[i + j] ^= (pi_i @ seq_D[j]) % 2
                
        return result

    # =========================================================================
    # Phase 2 Initialization (Coppersmith & Thomé)
    # =========================================================================

    def _thome_find_s_and_basis_pairs(self, sequence):
        """
        Thomé §3.3 initialization helper.

        Given coefficient matrices a_i = sequence[i] (shape m×n),
        find the smallest s such that the columns of a_0,...,a_{s-1}
        span GF(2)^m, and return m pairs (i_k, j_k) identifying
        canonical r_k = e_{j_k} so that a_{i_k} r_k = (column j_k of a_{i_k})
        forms a basis of GF(2)^m.
        
        Returns
        -------
        s : int
        Minimal number of coefficients needed (>=1).
        basis_pairs : list[tuple[int,int]]
        Length-m list of (i_k, j_k) pairs.
        """
        m = self.m
        n = self.n

        if len(sequence) == 0:
            raise ValueError("Empty sequence: cannot determine s/basis pairs")

        # Maintain an incremental column-space basis of GF(2)^m.
        #
        # basis_by_pivot[r] holds a vector (length m) whose leading 1 is at row r.
        # This is standard incremental elimination for independence testing.
        basis_by_pivot = {}  # pivot_row -> vector (np.ndarray shape (m,))
        basis_pairs = []     # record (i, j) each time we increase rank

        def add_vector_return_pivot(v):
            """Try to add v (1D length-m) to basis; return pivot row if independent, else None."""
            v = v.copy() % 2
            # eliminate known pivots
            for piv in sorted(basis_by_pivot.keys()):
                if v[piv]:
                    v ^= basis_by_pivot[piv]
            if not np.any(v):
                return None
            # choose first 1 as pivot (smallest row index)
            piv = int(np.flatnonzero(v)[0])
            basis_by_pivot[piv] = v
            return piv

        # Scan coefficients in order i=0,1,... and columns j=0,...,n-1
        # Stop at first i for which rank reaches m; then s = i+1.
        for i, a_i in enumerate(sequence):
            if a_i.shape != (m, n):
                raise ValueError(f"sequence[{i}] has shape {a_i.shape}, expected {(m, n)}")

            # process each canonical column vector a_i * e_j = a_i[:, j]
            for j in range(n):
                pivot = add_vector_return_pivot(a_i[:, j])
                if pivot is not None:
                    basis_pairs.append((i, j))
                    if len(basis_pairs) == m:
                        s = i + 1
                        return s, basis_pairs

        # If we get here, rank never reached m
        raise ValueError(
            f"Could not find s: span of all available a_i columns has rank {len(basis_pairs)} < m={m}. "
            "Need longer sequence or different random projections."
        )

    def _build_f_init_thome(self, sequence):
        """
        Construct Thomé/Coppersmith initialization f^(t0)(X) and nominal degrees γ.
        Thomé §3.3 initializer (up to transpose convention).

        Given the first coefficients of the matrix sequence A(X)=∑ a_i X^i (with a_i=S_i),
        find the smallest s such that the columns of a_0,...,a_{s-1} span K^m, and set t0=s.

        The initializer builds a sparse (transpose-convention) polynomial matrix f_init(X)
        with coefficients of shape (m+n)×n such that:
        - The first n rows represent the n canonical candidates (identity at degree 0).
        - The next m rows represent monomial candidates X^(t0-i_k) e_{j_k} chosen so that
          rank(X^0 e^(t0)) = m holds (Thomé §3.3, condition C2).

        Nominal degrees:
        Returns γ as a length-(m+n) list where all entries are initialized to t0.
        This matches the paper’s §3.3 initialization (γ is a bound, not the true degree).
        """
        s, basis_pairs = self._thome_find_s_and_basis_pairs(sequence)
        t0 = s
        dim = self.m + self.n

        self._log(f"Thomé init: s={s}, t0={t0}")
        
        # f_init degree 0..t0
        f_init = [np.zeros((dim, self.n), dtype=int) for _ in range(t0 + 1)]
        
        # First n candidates = I_n at degree 0
        for j in range(self.n):
            f_init[0][j, j] = 1
            
        # Remaining m candidates: X^{s - i_k} * e_{j_k}
        # Stored at row (n+k), column j_k, degree (s - i_k)
        for k, (i_k, j_k) in enumerate(basis_pairs):
            deg = t0 - i_k
            if deg < 0 or deg > t0:
                raise ValueError(f"Internal error: deg={deg} out of range for pair {(i_k, j_k)} with t0={t0}")
            f_init[deg][self.n + k, j_k] = 1

        # Thomé sets all nominal degrees gamma_j initially to t0 (= s)
        gamma = [t0] * dim

        if self.verbose and False:
            self._log(f"[init] f_init: {f_init}")
            self._log(f"[init] gamma: {gamma}")
            self._log(f"[init] t0: {t0}")
        
        return f_init, gamma, t0

    def _build_f_init_coppersmith(self, sequence):
        """
        Coppersmith-phase2 initializer.
        
        Recommended: use the same §3.3 initializer as in Thomé, since Thomé's paper
        uses the same starting framework/invariants and swaps only the generator stage.
        """
        return self._build_f_init_thome(sequence)

    # =========================================================================
    # Phase 2a: Coppersmith's Block Berlekamp-Massey (Iterative)
    # =========================================================================

    def block_berlekamp_massey(self,
                               sequence,
                               early_stop=True,
                               stop_check_interval=8,
                               stop_max_check=None):
        """
        Block Berlekamp–Massey / Coppersmith Phase-2 (iterative generator update).

        Given the sequence S_k (the coefficients of A(X)), iteratively updates a polynomial
        matrix f(X) so that its rows provide candidate vector generating polynomials.

        At each iteration step t, the method:
        - Forms a discrepancy matrix Δ from the coefficient of X^t in A(X) f(X).
        - Performs degree-guided elimination to compute a unimodular row transform τ.
        - Shifts (multiplies by X) exactly m rows to enforce the invariant that the total
          nominal-degree increase per step is m.

        Optional early termination:
        If enabled, periodically checks a small number of best candidates (smallest γ)
        using an explicit annihilation test against the computed finite sequence.
        This is a safe engineering termination criterion relative to available data, but it
        is not the exact “unexpected C1 event” trigger described in Coppersmith’s text.

        Returns
        -------
        f_poly : list[ndarray]
            Coefficients of f(X) as (m+n)×n matrices over GF(2).
        nominal_degrees : list[int]
            Nominal degree bounds γ for each of the (m+n) candidate rows.
        """
        self._log("Phase 2a: Running Block Berlekamp-Massey (Iterative)")
        f_poly, nominal_degrees, t0 = self._build_f_init_coppersmith(sequence)

        num_terms = len(sequence)
        dim = self.m + self.n

        # Optional default: check a handful of candidates only
        if stop_max_check is None:
            stop_max_check = min(dim, 8 * self.n)

        for t in range(t0, num_terms):
            # 1. Discrepancy
            delta = np.zeros((dim, self.m), dtype=int)
            for k, f_k in enumerate(f_poly):
                idx = t - k
                if 0 <= idx < num_terms:
                    delta ^= (f_k @ sequence[idx].T) % 2

            # 2. Elimination (Tau)
            tau = np.eye(dim, dtype=int)
            curr_delta = delta.copy()

            perm = sorted(range(dim), key=lambda x: nominal_degrees[x])

            for col in range(self.m):
                pivot_row = -1
                for p_idx in perm:
                    if curr_delta[p_idx, col] == 1:
                        pivot_row = p_idx
                        break
                    
                if pivot_row != -1:
                    perm.remove(pivot_row)
                    row_mask = (curr_delta[:, col] == 1)
                    row_mask[pivot_row] = False
                    if np.any(row_mask):
                        curr_delta[row_mask] ^= curr_delta[pivot_row]
                        tau[row_mask] ^= tau[pivot_row]

            # 3. Shift Strategy: EXACTLY m shifts
            is_zero = np.all(curr_delta == 0, axis=1)
            non_zero_indices = np.where(~is_zero)[0].tolist()
            zero_indices = np.where(is_zero)[0].tolist()

            zero_indices.sort(key=lambda x: nominal_degrees[x])
            shift_indices = set(non_zero_indices)
            while len(shift_indices) < self.m and zero_indices:
                shift_indices.add(zero_indices.pop(0))

            # 4. Update generator
            next_len = len(f_poly) + 1
            f_next = [np.zeros((dim, self.n), dtype=int) for _ in range(next_len)]

            for k, f_k in enumerate(f_poly):
                tf_k = (tau @ f_k) % 2
                for r in range(dim):
                    if r in shift_indices:
                        f_next[k + 1][r] = tf_k[r]
                    else:
                        f_next[k][r] = tf_k[r]

            while len(f_next) > 1 and np.all(f_next[-1] == 0):
                f_next.pop()

            f_poly = f_next

            for r in shift_indices:
                nominal_degrees[r] += 1

            # ------------------------------------------------------------
            #  termination check (degree-driven, not full scan)
            # ------------------------------------------------------------
            if early_stop and ((t - t0 + 1) % stop_check_interval == 0):
                candidates = self._select_generator_rows(
                    sequence=sequence,
                    f_poly=f_poly,
                    nominal_degrees=nominal_degrees,
                    max_check=stop_max_check,
                    max_return=1,
                    require_annihilation=True
                )
                if candidates:
                    self._log(f"[BBM early-stop] Found annihilating row r={candidates[0]} at t={t}.")
                    break

        return f_poly, nominal_degrees

    # =========================================================================
    # Phase 2b: Thomé's MSLGDC (Recursive)
    # =========================================================================

    def thome_lingen(self, sequence):
        """
        Thomé MSLGDC (Program 4.1): recursive generator computation for matrix sequences.

        This routine computes a polynomial matrix f(X) whose rows contain candidate
        vector generating polynomials for the matrix power series A(X)=∑ S_k X^k.

        High-level structure:
        1) Build the §3.3 initialization (f^(t0)(X), γ^(t0)) with t0 = s.
        2) Construct the residual context e(X) mod X^b needed by the recursion.
        3) Run MSLGDC recursion to compute Π(X)=P^(t0)⋯P^(t0+b-1).
        4) Output f(X)=Π(X) f^(t0)(X) and the updated nominal degrees γ.

        Nominal degrees:
        This implementation initializes γ with all entries equal to t0 (a degree bound), as
        in Thomé §3.3, and then updates γ via the base-step shifts.

        Complexity note:
        The recursion matches Thomé’s divide-and-conquer structure, but polynomial/matrix
        products are implemented naively, so the asymptotic FFT-based bounds from the paper
        do not apply to this code as written.

        Returns
        -------
        f_final : list[ndarray]
            Polynomial coefficients of f(X), each of shape (m+n)×n over GF(2).
        final_degrees : list[int]
            Final nominal degree bounds γ for each of the (m+n) candidate rows.
        """
        self._log("Phase 2b: Running Thomé MSLGDC (Recursive)")
        
        f_init, nominal_degrees, t0 = self._build_f_init_thome(sequence)
        dim = self.m + self.n
        L = len(sequence)
        
        # 1. Construct initial Residual Sequence D
        # D_t = coeff of z^{t0+t} in A(z) * F_init(z)
        # Note: We construct the residual sequence D explicitly for the recursion
        D_seq = []
        
        # Precompute transposed sequences for speed
        S_T = [s.T for s in sequence]
        
        # F_init has non-zeros at:
        #   (r, c) at degree d  =>  term is z^d * E_{r,c}
        #   Contribution to A(z)F(z) at degree K comes from A_{K-d} * E_{r,c}
        #   which is col c of S_{K-d}.
        #   We store the TRANSPOSE of the residual, so we store (S_{K-d})^T row c at row r.
        
        # Map of non-zeros in F_init: (row, col) -> degree
        # Since F_init is sparse (monomials), this is efficient.
        f_map = {}
        for d, mat in enumerate(f_init):
            rows, cols = np.nonzero(mat)
            for r, c in zip(rows, cols):
                f_map[r] = (d, c)
                
        # Generate D_0 ... D_{L-t0-1}
        # We need enough terms for the recursion depth.
        len_D = L - t0
        
        for t in range(len_D):
            # Target degree in product is t0 + t
            d_mat = np.zeros((dim, self.m), dtype=int)
            
            for r, (d_f, c_f) in f_map.items():
                # We need S_{K - d_f} where K = t0 + t
                s_idx = (t0 + t) - d_f
                if 0 <= s_idx < L:
                    # Copy row c_f of S_T[s_idx] to row r of d_mat
                    d_mat[r, :] = S_T[s_idx][c_f, :]
            
            D_seq.append(d_mat)
            
        # 2. Run Recursion
        # Returns Pi(z) such that Final Residual = Pi(z) * Initial Residual
        pi_poly, final_deg = self._recursive_mslgdc(D_seq, nominal_degrees)
        
        # 3. Compute Final Generator
        # F_final(z) = Pi(z) * F_init(z)
        f_final = self._mat_poly_mul(pi_poly, f_init)
        
        return f_final, final_deg

    def _recursive_mslgdc(self, D, degrees):
        """
        Recursive core of MSLGDC (Program 4.1).

        Parameters
        ----------
        D : list of ndarray
            Context residual e(X) mod X^b as list [e_0,...,e_{b-1}]
            Each e_i is (m+n)×m matrix (transpose convention)
        nominal_degrees : list of int
            Degree bounds γ = (γ_0,...,γ_{m+n-1})

        Returns
        -------
        pi_poly : list of ndarray
            Polynomial Pi(z) = P^{(t)}...P^{(t+b-1)} as matrix list
        new_degrees : list of int
            Updated degree bounds after b iterations

        Algorithm
        ---------
        Base case (b=1):
            Compute P^{(t)}(z) = pi_0 + pi_1·z from e_0 and γ
            using pivot elimination (ALGO 1)

        Recursive case (b>1):
            1. Split: k = b//2
            2. Left: Pi_1 = MSLGDC(D[0:k], γ)
            3. Update: D' = (Pi_1·D) div X^k  [take coefficients k,...,b-1]
            4. Right: Pi_2 = MSLGDC(D', γ')  [γ' from step 2]
            5. Combine: Pi = Pi_2·Pi_1

        Invariant
        ---------
        If context (D,γ) at time t satisfies conditions C1-C2, then
        output Pi satisfies: f^{(t+b)} = Pi·f^{(t)} with updated γ^{(t+b)}.
        """
        k = len(D)
        
        # Base Case
        if k == 1:
            return self._thome_base_step(D[0], degrees)
            
        # Split
        half = k // 2
        
        # 1. Left Recursive Call
        pi_L, deg_L = self._recursive_mslgdc(D[:half], degrees)
        
        # 2. Update Residual (Middle Product)
        # New D = (Pi_L * D) div z^half
        # We need terms from index `half` to `k` of the product
        D_new = self._apply_poly_to_sequence(pi_L, D) 
        D_right_input = D_new[half:]
        
        # 3. Right Recursive Call
        pi_R, deg_R = self._recursive_mslgdc(D_right_input, deg_L)
        
        # 4. Combine: Pi = Pi_R * Pi_L
        pi_comb = self._mat_poly_mul(pi_R, pi_L)
        
        return pi_comb, deg_R

    def _thome_base_step(self, delta, degrees):
        """
        Base case iterator (ALGO 1) - single iteration of Berlekamp-Massey-style update.

        Parameters
        ----------
        delta : ndarray, shape (m+n, m)
            Discrepancy matrix δ = X^0 e^{(t)} (coefficient 0 of residual)
        nominal_degrees : list of int
            Current degree bounds γ

        Returns
        -------
        [pi_0, pi_1] : list of ndarray
            Polynomial P^{(t)}(z) = pi_0 + pi_1·z
            Each pi_i is (m+n)×(m+n) transformation matrix
        new_degrees : list of int
            Updated degree bounds γ^{(t+1)}

        Algorithm (Theorem 3.3)
        -----------------------
        1. Perform Gaussian elimination on δ using pivot strategy guided by γ
           (eliminate in order of increasing γ_j to maintain stability)
        2. Transform matrix τ tracks row operations
        3. Force exactly m "non-zero" rows (those with degree increase)
        4. Construct P^{(t)}(z):
           - Rows in non-zero set: contribute to pi_1 (degree increases by 1)
           - Rows in zero set: contribute to pi_0 (degree unchanged)

        Notes
        -----
        - This is kept unchanged from original implementation as requested
        - Identical logic to Coppersmith's iterative update
        - Pivot order critical for numerical stability and termination proof
        """

        dim = self.m + self.n
        tau = np.eye(dim, dtype=int)
        curr_delta = delta.copy()

        # Sort for stability (do not subtract higher-degree rows from lower-degree rows)
        perm = sorted(range(dim), key=lambda x: degrees[x])

        # Elimination to row-reduce Delta (on its m columns)
        for col in range(self.m):
            pivot_row = -1
            for p in perm:
                if curr_delta[p, col] == 1:
                    pivot_row = p
                    break

            if pivot_row != -1:
                perm.remove(pivot_row)
                mask = (curr_delta[:, col] == 1)
                mask[pivot_row] = False
                if np.any(mask):
                    curr_delta[mask] ^= curr_delta[pivot_row]
                    tau[mask] ^= tau[pivot_row]

        # ------------------------------------------------------------
        # SHIFT STRATEGY (patched):
        # Choose exactly m rows to shift (multiply by z), prioritizing
        # rows with nonzero reduced discrepancy, and filling with zero
        # rows if necessary (tie-break by smallest nominal degree).
        # ------------------------------------------------------------
        is_zero = np.all(curr_delta == 0, axis=1)
        non_zero_indices = np.where(~is_zero)[0].tolist()
        zero_indices = np.where(is_zero)[0].tolist()

        # If we have > m nonzero rows (should not happen with m columns, but be safe),
        # pick the lowest-degree ones to shift.
        non_zero_indices.sort(key=lambda r: degrees[r])

        shift_set = set(non_zero_indices[:self.m])

        if len(shift_set) < self.m:
            zero_indices.sort(key=lambda r: degrees[r])
            for r in zero_indices:
                shift_set.add(r)
                if len(shift_set) == self.m:
                    break

        # Construct P(z) = pi_0 + pi_1 z
        pi_0 = np.zeros((dim, dim), dtype=int)
        pi_1 = np.zeros((dim, dim), dtype=int)
        new_degrees = list(degrees)

        for r in range(dim):
            if r in shift_set:
                pi_1[r] = tau[r]
                new_degrees[r] += 1
            else:
                pi_0[r] = tau[r]

        return [pi_0, pi_1], new_degrees

    # =========================================================================
    # Phase 3: Reconstruction
    # =========================================================================

    ### HELPERS ###
    def _row_effective_degree(self, f_poly, row):
        """Highest k such that f_poly[k][row] is nonzero; -1 if identically zero."""
        for k in range(len(f_poly) - 1, -1, -1):
            if np.any(f_poly[k][row]):
                return k
        return -1

    def _rank_generator_rows(self, f_poly, nominal_degrees):
        """
        Return nonzero rows, sorted by increasing nominal degree (gamma),
        tie-breaking by effective degree then row index.
        """
        dim = self.m + self.n
        rows = []
        for r in range(dim):
            deg_eff = self._row_effective_degree(f_poly, r)
            if deg_eff >= 0:
                rows.append((nominal_degrees[r], deg_eff, r))
        rows.sort()
        return [r for (_, _, r) in rows]

    def _select_generator_rows(self,
                               sequence,
                               f_poly,
                               nominal_degrees,
                               max_check=None,
                               max_return=None,
                               require_annihilation=True):
        """
        Degree-driven generator row selection.
        
        Parameters
        ----------
        require_annihilation:
        - True: return only rows that actually annihilate `sequence`.
        - False: return the best rows by nominal degree without checking.
        
        max_check:
        how many candidate rows to test with _check_annihilation (caps cost).
        
        max_return:
        how many rows to return.

        Returns
        -------
        list[int] rows
        """
        dim = self.m + self.n
        ordered = self._rank_generator_rows(f_poly, nominal_degrees)

        if max_check is None:
            max_check = len(ordered)
        else:
            max_check = min(max_check, len(ordered))

        if max_return is None:
            max_return = self.n  # a sensible default

        if not require_annihilation:
            return ordered[:max_return]

        out = []
        for r in ordered[:max_check]:
            if self._check_annihilation(sequence, f_poly, r):
                out.append(r)
                if len(out) >= max_return:
                    break
        return out

    def _eval_row_on_Z(self, B_pow_Z, f_poly, row):
        """
        Evaluate one candidate generator row on Z using precomputed B^k Z (reverse order).

        Given a candidate row u(X)=∑_{k=0..deg} u_k X^k (u_k in K^n) stored in `f_poly`,
        this constructs the block-Wiedemann reconstruction vector (valuation 0 case)
            w = ∑_{k=0..deg} B^(deg-k) Z * u_k  (mod 2).

        Why the exponent is reversed:
        Because Phase 1 stores S_k = X^T B^(k+1) Z, the standard annihilation relation
        translates into a kernel vector obtained from a reversed evaluation in B.

        Parameters
        ----------
        B_pow_Z : list[ndarray]
            Precomputed list with B_pow_Z[i] = B^i Z, each N×n over GF(2).
        f_poly : list[ndarray]
            Polynomial coefficients of f(X), each (m+n)×n.
        row : int
            Candidate row index in 0..m+n-1.

        Returns
        -------
        w : ndarray, shape (N, 1)
            The reconstructed vector candidate.
        deg : int
            Effective degree of the row (highest k with nonzero u_k), or -1 if the row is zero.
        """
        deg = -1
        for k in range(len(f_poly) - 1, -1, -1):
            if np.any(f_poly[k][row]):
                deg = k
                break
        if deg < 0:
            return np.zeros((self.N, 1), dtype=int), -1

        # Need B_pow_Z[0..deg] available
        if deg >= len(B_pow_Z):
            return np.zeros((self.N, 1), dtype=int), -1

        w = np.zeros((self.N, 1), dtype=int)
        for k in range(deg + 1):
            ck = f_poly[k][row].reshape(self.n, 1)
            if np.any(ck):
                power_idx = deg - k
                w ^= (B_pow_Z[power_idx] @ ck) % 2

        return w, deg
    
    def _strip_B_valuation(self, w, max_steps=None):
        """
        Remove a potential B-valuation by repeated multiplication w <- B w.

        In block Wiedemann reconstruction, the vector obtained from the generator may be
        nonzero but satisfy B^v w in ker(B) for some v>=0 (a nonzero valuation in B).

        This helper applies B repeatedly until either:
        - Bw = 0 (success: w is a kernel vector), or
        - w becomes 0 / the step budget is exhausted.

        This is an O(v) add-on in terms of sparse matrix-vector products and is intended
        as a replacement for scanning many “shift” attempts in reconstruction.
        """
        if max_steps is None:
            max_steps = self.N  # conservative; in practice small

        if not np.any(w):
            return w, 0, False

        steps = 0
        while steps < max_steps:
            Bw = (self.B @ w) % 2
            if not np.any(Bw):
                return w, steps, True
            w = Bw
            steps += 1

            if not np.any(w):
                return w, steps, False

        return w, steps, False
    
    def reconstruct_solution(self, f_poly, nominal_degrees, sequence):
        """
        Reconstruct a nonzero kernel vector w with B w = 0 from a computed generator.

        Algorithm outline:
        1) Precompute B^k Z for k=0..L once (dominant sparse cost, O(L)).
        2) Select a small number of promising candidate rows (small nominal degree γ and,
           when enabled, passing an annihilation check on the available finite sequence).
        3) For each candidate row, compute w by reverse evaluation using the cached B^k Z.
        4) If needed, strip a small B-valuation by repeated multiplication w <- B w.

        Complexity:
        The dominant sparse-matrix cost is the single precomputation of B^k Z plus a small
        number of additional multiplies for valuation stripping, avoiding O(L^2) shift scans.

        Returns
        -------
        w : ndarray, shape (N, 1)
            A kernel vector (possibly the zero vector on failure).
        """
        self._log("Phase 3: Reconstruction (O(L) eval + valuation stripping)")

        if self._last_Z is None:
            self._log("[-] No stored Z from Phase 1.")
            return np.zeros((self.N, 1), dtype=int)

        L = len(sequence)
        dim = self.m + self.n

        # 1) Precompute B^k Z for k=0..L
        self._log(f"Pre-computing {L} powers of B applied to Z...")
        B_pow_Z = []
        curr_Z = self._last_Z.copy()
        for _ in range(L + 1):
            B_pow_Z.append(curr_Z)
            curr_Z = (self.B @ curr_Z) % 2

        # 2) Choose candidate generator rows (degree-driven; limited checking)
        candidates = self._select_generator_rows(
            sequence=sequence,
            f_poly=f_poly,
            nominal_degrees=nominal_degrees,
            max_check=min(dim, 8 * self.n),
            max_return=max(1, self.n),
            require_annihilation=True
        )
        if not candidates:
            candidates = self._select_generator_rows(
                sequence=sequence,
                f_poly=f_poly,
                nominal_degrees=nominal_degrees,
                max_check=min(dim, 8 * self.n),
                max_return=max(1, self.n),
                require_annihilation=False
            )

        # 3) Try candidates
        for r in candidates:
            w, deg_row = self._eval_row_on_Z(B_pow_Z, f_poly, r)
            if deg_row < 0:
                continue

            if not np.any(w):
                # w==0 can happen; try next candidate
                continue

            # 4) Strip valuation if needed
            w2, v, ok = self._strip_B_valuation(w, max_steps=L + 2)
            if ok and np.any(w2):
                self._log(f"[SUCCESS] Found solution from row {r} (deg={deg_row}, valuation={v})")
                return w2

        self._log("[-] Failed to reconstruct a nonzero kernel vector. Try longer L or different projections.")
        return np.zeros((self.N, 1), dtype=int)

    def _check_annihilation(self, sequence, f_poly, row):
        """
        Test whether a candidate row u(X) annihilates the available truncated sequence.

        For a fixed row u(X)=∑ u_k X^k (stored in `f_poly`), this checks the finite
        convolution identities implied by A(X)u(X)=0 up to the available length, i.e.
        it verifies that the computed coefficients vanish for all t in the checked range.

        This is a finite-data consistency test (sufficient for safe candidate filtering),
        not a proof that u(X) is a true generator for the infinite series.
        """
        deg_f = len(f_poly)
        deg_seq = len(sequence)
        
        # Check convolution sum for t from deg_f to deg_seq
        # sum_{k} f_k[row] * S_{t-k}^T
        
        for t in range(deg_f, deg_seq):
            acc = np.zeros(self.m, dtype=int)
            for k in range(deg_f):
                idx = t - k
                f_vec = f_poly[k][row] # shape (n,)
                s_mat = sequence[idx]  # shape (m, n)
                
                # f * S^T = (S * f^T)^T
                term = s_mat @ f_vec
                acc ^= term
                
            if np.any(acc % 2):
                return False
        return True

def generate_exact_rank_matrix(N, rank):
    """
    Generates a random NxN matrix over GF(2) with exactly the specified rank.
    Uses B = P * L * D * U * Q decomposition.
    """
    if rank > N:
        raise ValueError(f"Rank {rank} cannot be greater than dimension {N}")

    # 1. Create D (diagonal with 'rank' ones)
    D = np.zeros((N, N), dtype=int)
    for i in range(rank):
        D[i, i] = 1

    # 2. Create random Lower Unit Triangular (L) and Upper Unit Triangular (U)
    # These are guaranteed to be invertible (det = 1)
    L = np.tril(np.random.randint(0, 2, (N, N)), k=-1)
    np.fill_diagonal(L, 1)
    
    U = np.triu(np.random.randint(0, 2, (N, N)), k=1)
    np.fill_diagonal(U, 1)

    # 3. Compute B = L * D * U
    # This matrix has rank 'rank' but the nullspace is trivial (last N-rank cols)
    B = (L @ D @ U) % 2

    # 4. Scramble rows and columns (Permutations)
    # This distributes the rank/nullspace complexity throughout the matrix
    row_perm = np.random.permutation(N)
    col_perm = np.random.permutation(N)
    
    B = B[row_perm, :]    # Permute rows
    B = B[:, col_perm]    # Permute cols

    return B
    
if __name__ == "__main__":
    # Toy Test
    print("--- Running Block Wiedemann Test ---")
    np.random.seed(42)
    N = 1000
    target_rank = N - 5
    m, n = 16, 16
    print(f"N = {N}, rank = {target_rank}, m = {m}, n = {n}")
    
    # Random singular matrix
    # B = np.random.randint(0, 2, (N, N))
    # B[0] = 0 # Ensure singularity
    B = generate_exact_rank_matrix(N, target_rank)
    
    bw = BlockWiedemannGF2(B, m=m, n=n, verbose=True)
    
    X = np.random.randint(0, 2, (N, m))
    Z = np.random.randint(0, 2, (N, n))
    
    # L = N/n + N/m approx
    L =  2*(N // n + m // n) + 10
    print(f"L 2(N/n+m/n)+10 = {L}")
    print("-" * 50)
    seq = bw.generate_krylov_sequence(X, Z, L)
    
    # Test BBM
    print("\n[Testing Iterative BBM]")
    f_bbm, degs_bbm = bw.block_berlekamp_massey(seq)
    sol_bbm = bw.reconstruct_solution(f_bbm, degs_bbm, seq)
    
    if sol_bbm is not None:
        print(f"BBM Solution found, norm: {np.sum(sol_bbm)}")
    else:
        print("BBM failed to find solution")

    # Test Thomé
    print("\n[Testing Recursive MSLGDC]")
    f_th, degs_th = bw.thome_lingen(seq)
    sol_th = bw.reconstruct_solution(f_th, degs_th, seq)
    
    if sol_th is not None:
        print(f"Thomé Solution found, norm: {np.sum(sol_th)}")
    else:
        print("Thomé failed to find solution")
