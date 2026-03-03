import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import root_scalar
from tqdm import tqdm
from .utils import get_sigma, spd_logm


class PCA_MP:
    """Stable PCA via Mirror-Prox algorithm.
    It solves the min-max formulation: max_M min_w sum_l w_l trace <Sigma_l, M>
    where M is a PSD matrix with trace k and eigenvalues in [0,1] (simplex),
    and w lies on the simplex.
    
    Supports different variants: 'stable', 'fair', and 'squared'.
    """
    def __init__(self, n_components=None, method=None):
        """
        Args:
            n_components: Number of components (rank k)
            method: Method variant. Options:
                - 'stable': StablePCA (default)
                - 'fair': Fair PCA variant
                - 'squared': Squared PCA variant
        """
        self.n_components = n_components
        self.M = None
        self.M_proj = None
        self.w = None
        self.method = 'stable' if method is None else method
        
    def fit(self, X_list, Sigma_list=None, M_init=None, eta_init=10,
            max_iter=1000, tol=1e-6, verbose=True, check_dual=False, 
            decay_every=500, eta_decay=0.5, use_fast_logm=True,
            init_iter=50, check_freq=100):
        """
        Args:
            X_list: list of L data matrices from different sources.
            Sigma_list: list of L covariance matrices. Defaults to None.
                If None, then X_list is used to compute empirical covariances.
            M_init: initial value for M. Defaults to None.
            eta_init (int, optional): initial multiplier factor for learning rate. Defaults to 1.
            max_iter (int, optional): maximum number of iterations. Defaults to 1000.
            tol (float, optional): tolerance for convergence. Defaults to 1e-6.
            verbose: 
                - False or 0: no printing.
                - True: print every 100 iterations.
                - int > 0: print every `verbose` iterations.
            check_dual (bool, optional): whether to check duality gap. Defaults to False.
            decay_every (int, optional): number of iterations to wait before decaying learning rate. Defaults to 500.
            eta_decay (float, optional): factor by which to decay learning rate. Defaults to 0.5.
            use_fast_logm (bool, optional): if True, use partial eigendecomposition for log(M) 
                when dimension is large. This is much faster for low-rank M. Defaults to True.
            init_iter (int, optional): number of initial iterations before using fast_logm.
                During first init_iter iterations, full eigendecomposition is used to ensure 
                accuracy. After init_iter, fast_logm is enabled. Defaults to 50.
            check_freq (int, optional): frequency of checking primal/dual values for convergence.
                Only checks every `check_freq` iterations to save computation time. 
                Defaults to 100. Set to 1 to check every iteration.
        """
        # Interpret verbose
        if isinstance(verbose, bool):
            print_freq = 100 if verbose else 0
        elif isinstance(verbose, int):
            print_freq = max(verbose, 0)
        else:
            raise ValueError("verbose must be bool or int.")
        
        # --------- Prepare Data --------- #
        L = len(X_list)
        d = X_list[0].shape[1]
        if Sigma_list is None:
            Sigma_list = [get_sigma(X, demean=False) for X in X_list]
        
        # Store flags for use in _update_M_w
        self._use_fast_logm = use_fast_logm
        
        # Step 1: Apply method-specific transformation to Sigma_list (regularization)
        # This computes Sigma_reg_list from original Sigma_list
        Sigma_reg_list = []
        for l in range(L):
            Sigma_l = Sigma_list[l]
            eval, _ = np.linalg.eigh(Sigma_l)
            if self.method == 'fair':
                idx = np.argsort(eval)[::-1]
                topk_eigvals = eval[idx][:self.n_components]
                topk_mean = np.mean(topk_eigvals)
                Sigma_l_reg = Sigma_l - topk_mean * np.eye(Sigma_l.shape[0])
            elif self.method == 'squared':
                sum_eigvals = np.sum(eval)
                Sigma_l_reg = Sigma_l - (sum_eigvals / self.n_components) * np.eye(Sigma_l.shape[0])
            elif self.method == 'stable':
                Sigma_l_reg = Sigma_l.copy()
            else:
                raise ValueError(f"Unknown method: {self.method}. Must be 'stable', 'fair', or 'squared'.")
            Sigma_reg_list.append(Sigma_l_reg)
        
        # Step 2: Normalize based on Sigma_reg_list scale (not original Sigma_list)
        # The normalization scale should depend on the scale of regularized Sigma
        L_op = np.max([np.linalg.norm(Sigma_reg, ord=2) for Sigma_reg in Sigma_reg_list])
        Sigma_reg_list_normalized = [Sigma_reg / L_op for Sigma_reg in Sigma_reg_list]
        self.L_op = L_op
        
        Sigma_stack = np.stack(Sigma_reg_list_normalized)

        # --------- Initialize Parameters --------- #        
        w_curr = np.ones(L) / L
        Sigma_w = np.einsum('i,ijk->jk', w_curr, Sigma_stack) 
        M_curr = np.eye(d) * self.n_components / d if M_init is None else M_init

        # parameters for learning rate (now L_op_normalized = 1)
        a = np.log(L)
        b = self.n_components * np.log(d / self.n_components)
        # Since Sigma are normalized, effective L_op = 1
        eta = eta_init * np.sqrt(np.log(L) / (self.n_components * np.log(d / self.n_components))) / 4
        eta_M = eta / a
        eta_w = eta / b

        if print_freq > 0:
            print(f"Initial step size eta_M = {eta_M:.3e} and eta_w = {eta_w:.3e}")

        # Use incremental averaging to avoid O(N × p²) cost at each iteration
        # Initialize running averages
        M_avg = M_curr.copy()
        w_avg = w_curr.copy()
        n_avg = 0  # Count of iterates averaged so far
        
        # Track best model (smallest duality gap) if check_dual
        best_duality_gap = float('inf')
        best_M_avg = None
        best_w_avg = None
        best_iter = 0
        prev_duality_gap = float('inf')
        
        primal = self._compute_primal(Sigma_stack, M_curr)
        prev_primal = primal # for monitoring local change
        # optimization loop
        for iter in range(max_iter):
            # Determine if we should use fast logm (only after init_iter iterations)
            # This ensures full accuracy in early iterations, then switches to fast approximation
            use_fast_this_iter = (hasattr(self, '_use_fast_logm') and 
                                  self._use_fast_logm and 
                                  iter >= init_iter and 
                                  M_curr.shape[0] > 30)
            
            # Adaptive n_eig selection: gradually reduce from 2k to 1k as we converge
            # This exploits the fact that M becomes more low-rank as iterations progress
            if use_fast_this_iter and hasattr(self, 'n_components') and self.n_components is not None:
                k = self.n_components
                # Linearly interpolate: start with 2k at iter=init_iter, reduce to k at iter=max_iter
                iter_progress = (iter - init_iter) / max(1, max_iter - init_iter)  # 0 to 1
                n_eig_factor = 2.0 - iter_progress  # 2.0 -> 1.0
                adaptive_n_eig = max(k, int(n_eig_factor * k))
            else:
                adaptive_n_eig = None  # Will use default in _update_M_w
            # ----------- First Step ---------- #
            ## starting from (M_curr, w_curr) to get (M_mid, w_mid)
            M_mid, w_mid = self._update_M_w(Sigma_stack, M_start=M_curr, w_start=w_curr, 
                M_eval=M_curr, w_eval=w_curr, 
                eta_M=eta_M, eta_w=eta_w, 
                k=self.n_components, use_fast_logm=use_fast_this_iter, 
                n_eig_override=adaptive_n_eig if use_fast_this_iter else None)

            # ----------- Second Step ---------- #
            ## starting from (M_curr, w_curr) to update (M_curr, w_curr), but using (M_mid, w_mid) for gradients evaluation
            M_curr, w_curr = self._update_M_w(Sigma_stack, M_start=M_curr, w_start=w_curr, 
                M_eval=M_mid, w_eval=w_mid, 
                eta_M=eta_M, eta_w=eta_w, 
                k=self.n_components, use_fast_logm=use_fast_this_iter, 
                n_eig_override=adaptive_n_eig if use_fast_this_iter else None)
            
            # ----------- Incremental Average Iterates ---------- #
            # Update running average incrementally: O(p²) instead of O(N × p²)
            # M_avg_new = (M_avg_old * n + M_mid) / (n + 1)
            n_avg += 1
            M_avg = (M_avg * (n_avg - 1) + M_mid) / n_avg
            w_avg = (w_avg * (n_avg - 1) + w_mid) / n_avg
            
            # ----------- Check primal/dual and convergence (only every check_freq iterations) ---------- #
            # Skip expensive primal/dual computation most of the time to save computation
            # Only compute when needed for convergence checking or printing
            should_check = ((iter + 1) % check_freq == 0) or (iter + 1 >= max_iter) or (iter + 1 < check_freq)
            
            if should_check:
                primal_curr = self._compute_primal(Sigma_stack, M_avg)
                primal_diff = np.abs(prev_primal - primal_curr)
                
                # compute dual and duality gap if needed
                if check_dual:
                    dual_curr = self._compute_dual(Sigma_stack, w_avg)
                    dual_gap = np.abs(primal_curr - dual_curr) * L_op  # Scale by L_op for consistency
                    
                    # Track best model (smallest duality gap)
                    if dual_gap < best_duality_gap:
                        best_duality_gap = dual_gap
                        best_M_avg = M_avg.copy()
                        best_w_avg = w_avg.copy()
                        best_iter = iter + 1
                else:
                    dual_curr = None
                    dual_gap = None
            else:
                # Skip computation, use cached values or None
                primal_curr = None
                primal_diff = None
                dual_curr = None
                dual_gap = None

            if print_freq > 0 and (iter + 1) % print_freq == 0:
                # Need to compute primal if not already computed for printing
                if primal_curr is None:
                    primal_curr = self._compute_primal(Sigma_stack, M_avg)
                    primal_diff = np.abs(prev_primal - primal_curr) if prev_primal is not None else None
                if check_dual and dual_curr is None:
                    dual_curr = self._compute_dual(Sigma_stack, w_avg)
                    dual_gap = np.abs(primal_curr - dual_curr) * L_op if primal_curr is not None else None
                
                log_info = f"Iter {iter + 1}"
                if primal_diff is not None:
                    log_info += f" | Diff primal: {primal_diff:.6e}"
                if check_dual and dual_gap is not None:
                    log_info += f" | duality gap: {dual_gap:.6e}"
                log_info += f" | eta: {eta:.3e}"
                print(log_info)

            # ----------- Check convergence (only if we computed primal_diff) ---------- #
            # Note: primal_diff and dual_gap are computed on normalized Sigma,
            # so their values are scaled by 1/L_op. However, relative errors are preserved,
            # so we can use the original tolerance thresholds.
            # Ensure minimum of 500 iterations for stability
            min_iterations = 500
            converged = False
            early_stop_duality_gap = False
            
            if should_check and primal_diff is not None:
                if iter + 1 >= min_iterations:
                    if not check_dual:
                        # original stopping rule: only primal difference
                        if primal_diff < tol:
                            converged = True
                            if print_freq > 0:
                                print(f"Converged at iteration {iter + 1}")
                    else:
                        # Check if duality gap is increasing (early stop condition)
                        if dual_gap is not None:
                            if dual_gap > prev_duality_gap and prev_duality_gap < float('inf'):
                                # Duality gap is increasing, trigger early stop
                                early_stop_duality_gap = True
                                if print_freq > 0:
                                    print(
                                        f"\nEarly stopping at iteration {iter + 1}: "
                                        f"duality gap increased ({prev_duality_gap:.6e} -> {dual_gap:.6e})"
                                    )
                                    print(
                                        f"Best model found at iteration {best_iter} "
                                        f"with duality gap {best_duality_gap:.6e}"
                                    )
                            else:
                                # Update prev_duality_gap for next check
                                prev_duality_gap = dual_gap
                        
                        # Regular convergence check
                        if (primal_diff < tol) and (dual_gap is not None) and (dual_gap < 1e-2):
                            converged = True
                            if print_freq > 0:
                                print(
                                    f"Converged at iteration {iter + 1} "
                                    f"(primal diff={primal_diff:.3e}, duality gap={dual_gap:.3e})"
                                )
            
            # Stop if converged or duality gap is increasing
            if converged or early_stop_duality_gap:
                # If early stop due to increasing duality gap, use best model
                if early_stop_duality_gap and best_M_avg is not None:
                    M_avg = best_M_avg
                    w_avg = best_w_avg
                    if print_freq > 0:
                        print(f"Using best model from iteration {best_iter}")
                break
            
            # Update prev_primal only when we computed primal_curr
            if should_check and primal_curr is not None:
                prev_primal = primal_curr
            
            # ----------- Adaptive step-size schedule ---------- #
            # simple decay: every `decay_every` iterations, shrink eta
            if (decay_every is not None) and ((iter + 1) % decay_every == 0):
                eta_old_M = eta_M
                eta_old_w = eta_w
                eta_M = max(1e-4, eta_M * eta_decay)
                eta_w = max(1e-4, eta_w * eta_decay)
                if print_freq > 0:
                    print(
                        f"[Step-size decay] Iter {iter + 1}: "
                        f"eta_M: {eta_old_M:.3e} -> {eta_M:.3e} | eta_w: {eta_old_w:.3e} -> {eta_w:.3e}"
                    )

        # --------- Finalize Projection Matrix --------- #
        self.M = M_avg
        self.w = w_avg
        
        # Considering that M is not sparse, we only output the top k eigenvectors
        eval, evec = np.linalg.eigh(self.M)
        # Sort eigenvectors in descending order of eigenvalues
        sorted_indices = np.argsort(eval)[::-1]
        evec_sorted = evec[:, sorted_indices]
        # Select top k eigenvalues and eigenvectors
        top_k_eigenvectors = evec_sorted[:, :self.n_components]
        self.components_ = top_k_eigenvectors.T
        self.M_proj = top_k_eigenvectors @ top_k_eigenvectors.T
        
        X_all = np.vstack(X_list)
        self.mean_ = X_all.mean(axis=0)
        X_centered = X_all - self.mean_
        X_proj = X_centered @ self.components_.T
        self.explained_variance_ = np.var(X_proj, axis=0)
        
        # --------- Compute and print final duality gap (even if check_dual=False) --------- #
        # Note: Sigma_stack is normalized, so we need to restore the scale by L_op
        primal_final = self._compute_primal(Sigma_stack, self.M) * L_op
        dual_final = self._compute_dual(Sigma_stack, self.w) * L_op
        duality_gap_final = np.abs(primal_final - dual_final)
        
        # If check_dual and we found a better model earlier, report it
        if check_dual and best_duality_gap < float('inf') and best_duality_gap < duality_gap_final:
            if print_freq > 0:
                print(f"\nNote: Best duality gap was {best_duality_gap:.6e} at iteration {best_iter}")
                print(f"      Final duality gap is {duality_gap_final:.6e}")
        
        if print_freq > 0:
            print(f"\nFinal results:")
            print(f"  Primal value: {primal_final:.6e}")
            print(f"  Dual value: {dual_final:.6e}")
            print(f"  Duality gap: {duality_gap_final:.6e}")
            if check_dual and best_iter > 0:
                print(f"  Best duality gap: {best_duality_gap:.6e} (at iteration {best_iter})")
        
        # Restore primal values by multiplying by L_op (since we optimized on normalized Sigma)
        self.primal_M = self._compute_primal(Sigma_stack, self.M) * L_op
        self.primal_M_proj = self._compute_primal(Sigma_stack, self.M_proj) * L_op
        
        # Store best duality gap info for reference
        if check_dual:
            self.best_duality_gap = best_duality_gap if best_duality_gap < float('inf') else duality_gap_final
            self.best_iter = best_iter if best_iter > 0 else iter + 1
        else:
            self.best_duality_gap = duality_gap_final
            self.best_iter = iter + 1
        
        
    # ----------------------------------------------- #
    # Helper functions
    # ----------------------------------------------- #
    def _compute_primal(self, Sigma_stack, M):
        """
        Primal problem: given M, compute max_w -<M, Sigma(w)>
        """
        vals = np.array([-np.trace(Sigma.T @ M) for Sigma in Sigma_stack], dtype=float)
        return float(vals.max())
    
    def _compute_dual(self, Sigma_stack, w):
        """
        Dual problem:Given w, compute min_M -<M, Sigma(w)>
        """
        Sigma_w = np.einsum('i,ijk->jk', w, Sigma_stack)

        # eigen-decomposition
        eval, evec = np.linalg.eigh(Sigma_w)
        idx = np.argsort(eval)[::-1][:self.n_components]
        U = evec[:, idx]
        M = U @ U.T
        return float(-np.trace(Sigma_w.T @ M))

    def _update_M_w(self, Sigma_stack, M_start, w_start, M_eval, w_eval, eta_M, eta_w, k, use_fast_logm=False, n_eig_override=None):
        """
        Returns:
            M_new, w_new
        """
        """
        Update M and w using the mirror-prox algorithm.
        Args:
            Sigma_stack: stack of L covariance matrices.
            M_start: starting iterate for M.
            w_start: starting iterate for w.
            M_eval: iterate M for evaluating gradients.
            w_eval: iterate w for evaluating traces.
            eta_M: learning rate for M.
            eta_w: learning rate for w.
            k: number of components.
            use_fast_logm: if True, use partial eigendecomposition for log(M).
        Returns:
            M_new: new iterate for M.
            w_new: new iterate for w.
        """
        # for omega
        traces = np.array([np.trace(Sigma.T @ M_eval) for Sigma in Sigma_stack])
        w_new = w_start * np.exp(- eta_w * traces)
        w_new = np.maximum(w_new, 0.0)   # clip tiny negative noise
        w_new = w_new / w_new.sum()

        # Mirror-prox "gradient" step
        # Optimization: exploit low-rank structure of log_M
        #
        # Key insight: log_M = U_k @ diag(log(λ_k) - log(tol)) @ U_k.T + log(tol) * I
        # - In U_k subspace: log_M has non-constant eigenvalues
        # - In orthogonal complement: log_M ≈ log(tol) * I (constant)
        # - So A's eigenvalues in orthogonal complement ≈ eta_M * (Sigma_w eigenvalues) + log(tol)
        # - If eta_M is small, these are dominated by log(tol) ≈ -9.21
        #
        # Strategy: Only compute A's eigenvalues in U_k subspace and a few from Sigma_w
        
        p = M_start.shape[0]
        Sigma_w = np.einsum('i,ijk->jk', w_eval, Sigma_stack)  # (p, p)
        
        # Use optimized low-rank computation if enabled
        if use_fast_logm and p > 30 and k < p // 4:
            # Step 1: Get log_M components (U_k and log eigenvalues)
            # Use n_eig_override if provided (for adaptive selection), otherwise default to 2k
            if n_eig_override is not None:
                n_eig_fast = min(n_eig_override, p - 1)
            else:
                n_eig_fast = min(2 * k + 10, p - 1)
            log_M, U_k, log_eval_k = spd_logm(M_start, n_eig=n_eig_fast, return_components=True)
            # U_k: (p, n_eig_fast), log_eval_k: (n_eig_fast,) where log_eval_k = log(λ) - log(tol)
            
            try:
                # Step 2: Compute A in U_k subspace
                # A_kk = U_k.T @ A @ U_k = U_k.T @ (eta_M * Sigma_w + log_M) @ U_k
                # Since log_M = U_k @ diag(log_eval_k) @ U_k.T + log(tol) * I,
                # we have: A_kk = eta_M * (U_k.T @ Sigma_w @ U_k) + diag(log_eval_k) + log(tol) * I
                log_tol = np.log(1e-8)  # Same tol as in spd_logm
                Sigma_w_proj = U_k.T @ Sigma_w @ U_k  # (n_eig_fast, n_eig_fast)
                A_kk = eta_M * Sigma_w_proj + np.diag(log_eval_k) + log_tol * np.eye(n_eig_fast)
                
                # Step 3: Compute eigenvalues in U_k subspace
                eval_k, evec_k = np.linalg.eigh(A_kk)  # O(n_eig_fast³), fast!
                # Note: eigh returns ascending order, so reverse
                eval_k = eval_k[::-1]
                evec_k = evec_k[:, ::-1]
                
                # Step 4: Project eigenvectors back to full space
                evec_k_full = U_k @ evec_k  # (p, n_eig_fast)
                
                # Step 5: Handle eigenvalues in orthogonal complement
                # Key insight: Since eta_M is typically 0.01-1 and log(tol) ≈ -18.42 is a large negative,
                # in the orthogonal complement where log_M ≈ log(tol) * I,
                # A's eigenvalues ≈ eta_M * (Sigma_w eigenvalues) + log(tol).
                # Since eta_M * (Sigma_w eigenvalues) is typically << |log(tol)|,
                # log(tol) dominates, so we can directly set these eigenvalues to log(tol).
                #
                # Only the U_k subspace has non-constant log_M eigenvalues, so those are the
                # only ones we need to compute accurately.
                
                eval = eval_k
                evec = evec_k_full
                
                # Pad with log(tol) eigenvalues for _solve_for_nu (need enough for accuracy)
                # These represent eigenvalues in the orthogonal complement, all dominated by log(tol)
                n_eig_needed = min(2 * k, p)
                n_pad = max(0, n_eig_needed - len(eval))
                if n_pad > 0:
                    eval_pad = np.full(n_pad, log_tol)
                    # For eigenvectors in orthogonal complement, use QR decomposition
                    # to get orthonormal basis (exact eigenvectors don't matter since eigenvalues are constant)
                    if n_eig_fast < p:
                        Q, _ = np.linalg.qr(U_k, mode='complete')
                        evec_pad = Q[:, n_eig_fast:min(n_eig_fast + n_pad, p)]
                        if evec_pad.shape[1] < n_pad:
                            # Shouldn't happen, but pad with zeros if needed
                            zeros_pad = np.zeros((p, n_pad - evec_pad.shape[1]))
                            evec_pad = np.hstack([evec_pad, zeros_pad])
                    else:
                        evec_pad = np.zeros((p, n_pad))
                    
                    eval = np.concatenate([eval, eval_pad])
                    evec = np.hstack([evec, evec_pad])
                    # Sort by eigenvalue (descending)
                    idx_sort = np.argsort(eval)[::-1]
                    eval = eval[idx_sort]
                    evec = evec[:, idx_sort]
                    
            except Exception as e:
                # Fallback to full computation if optimization fails
                if use_fast_logm:
                    log_M = spd_logm(M_start, n_eig=n_eig_fast)
                else:
                    log_M = spd_logm(M_start)
                A = eta_M * Sigma_w + log_M
                eval, evec = np.linalg.eigh(A)
        else:
            # Full eigendecomposition for small matrices or when k is large
            log_M = spd_logm(M_start)
            A = eta_M * Sigma_w + log_M
            eval, evec = np.linalg.eigh(A)

        nu = self._solve_for_nu(eval, k)
        evals = np.minimum(np.exp(eval + nu), 1.0)

        M_new = evec @ np.diag(evals) @ evec.T
        return M_new, w_new

            
    def _solve_for_nu(self, lambdas, k, nu_bounds=(-20, 20)):
        """
        Solve for nu in the equation:
        sum_{j} min(exp(lambdas[j] + nu), 1) = k.

        Parameters:
            lambdas (np.ndarray): Array of lambda_j values.
            k (float): Target sum.
            nu_guess (float): Initial guess for nu.
            nu_bounds (tuple): Bounds for nu (low, high).

        Returns:
            float: The solution for nu.
        """
        def f(nu):
            return np.sum(np.minimum(np.exp(lambdas + nu), 1.0)) - k

        # Brent's method requires f(a) and f(b) to have opposite signs
        sol = root_scalar(
            f,
            bracket=nu_bounds,
            method='brentq',
            xtol=1e-6
        )
        return sol.root
    
    def _best_rank_k_approximation(self, A, k):
        """
        Compute the best rank-k approximation of A using its eigenvalues and eigenvectors.
        """
        epsilon = 0.05
        eval, evec = np.linalg.eigh(A)
        idx = np.argsort(eval)[::-1][:k]
        U_k = evec[:, idx]
        alpha = 1 - epsilon
        beta = (k * epsilon) / A.shape[0]
        P_k = U_k @ U_k.T
        M = alpha * P_k + beta * np.eye(A.shape[0]) # 
        return M


class PCA_Dual:
    """PCA Dual class.
    It implements multi-source PCA via dual formulation, which first solves for the optimal weights on the simplex
    and then computes the top-k eigenvectors of the weighted covariance matrix.
    It supports different variants including 'fair', 'squared', and 'stable' PCA.
    """
    def __init__(self, n_components=None, method=None):
        self.n_components = n_components
        self.P = None
        self.w = None
        self.method = 'stable' if method is None else method

    def fit(self, X_list, Sigma_list=None, lr=None, max_iter=4000, tol=1e-5, check_dual=True, verbose=True, progress=True):
        # --------- Prepare Data ---------
        L = len(X_list)
        if Sigma_list is None: 
            Sigma_list = [get_sigma(X, demean=False) for X in X_list] 
        
        # Normalize for numerical stability: divide all Sigma by L_op
        L_op = np.max([np.linalg.norm(Sigma, ord=2) for Sigma in Sigma_list])
        Sigma_list_normalized = [Sigma / L_op for Sigma in Sigma_list]
        self.L_op = L_op
        
        Sigma_reg_list = [] 
        # get top-k eigenvalues of each source 
        for l in range(L): 
            Sigma_l = Sigma_list_normalized[l] 
            eval, _ = np.linalg.eigh(Sigma_l) 
            if self.method == 'fair': 
                idx = np.argsort(eval)[::-1] 
                topk_eigvals = eval[idx][:self.n_components] 
                topk_mean = np.mean(topk_eigvals) 
                Sigma_l_reg = Sigma_l - topk_mean * np.eye(Sigma_l.shape[0]) 
            elif self.method == 'squared': 
                sum_eigvals = np.sum(eval) 
                Sigma_l_reg = Sigma_l - (sum_eigvals / self.n_components) * np.eye(Sigma_l.shape[0]) 
            elif self.method == 'stable': 
                Sigma_l_reg = Sigma_l.copy() 
            Sigma_reg_list.append(Sigma_l_reg)

        # --------- Optimize weights w via mirror descent ---------
        w_opt, phi_val, info = self._mirror_descent_phi(Sigma_reg_list, self.n_components, 
                                                        lr=lr, max_iter=max_iter, tol=tol, 
                                                        check_dual=check_dual, verbose=verbose, progress=progress)
        
        # --------- Finalize Projection Matrix ---------
        Sigma_reg_stack = np.stack(Sigma_reg_list)
        Sigma_opt = np.einsum('i,ijk->jk', w_opt, Sigma_reg_stack)
        eval, evec = np.linalg.eigh(Sigma_opt)
        idx = np.argsort(eval)[::-1]
        idx_k = idx[:self.n_components]
        U_k = evec[:, idx_k]
        self.P = U_k @ U_k.T
        self.w = w_opt
        self.phi_val = phi_val
        self.info = info
        self.components_ = U_k.T
        X_all = np.vstack(X_list)
        self.mean_ = X_all.mean(axis=0)
        X_centered = X_all - self.mean_
        X_proj = X_centered @ self.components_.T
        self.explained_variance_ = np.var(X_proj, axis=0)
    
    def _mirror_descent_phi(self, Sigma_list, k, lr=None, max_iter=4000,
                        tol=1e-5, check_dual=False, verbose=False, progress=True):
        """
        Mirror descent on the simplex to minimize:
            phi(w) = sum_{i=1}^k lambda_i( sum_{l=1}^L w_l Sigma^{(l)} ).
        Args:
            Sigma_list: List of L symmetric d x d matrices Sigma^{(l)}.
            k: Number of components.
            lr: Learning rate for mirror descent. If None, choose automatically.
            max_iter: Maximum number of iterations.
            tol: Convergence threshold for ||w_new - w||_1.
            check_dual: Whether to compute duality gap for monitoring.
            verbose:
                - False or 0: no printing.
                - True: print every 100 iterations.
                - int > 0: print every `verbose` iterations.
            progress: Whether to display tqdm progress bar.

        Returns:
            w: Final weight vector on the simplex.
            phi_val: phi(w_last) at the last iterate.
            info: dict with:
                - 'phi_hist'
                - 'gap_hist'
                - 'iters'
                - 'w_hist'
        """
        Sigma_list = [np.asarray(Sigma) for Sigma in Sigma_list]
        Sigma_stack = np.stack(Sigma_list)  # (L, d, d)
        L, d, _ = Sigma_stack.shape

        # Interpret verbose
        if isinstance(verbose, bool):
            print_freq = 100 if verbose else 0
        elif isinstance(verbose, int):
            print_freq = max(verbose, 0)
        else:
            raise ValueError("verbose must be bool or int.")

        if lr is None:
            lr = self._choose_learning_rate(Sigma_list, k, max_iter)

        use_bar = print_freq > 0
        if use_bar:
            print(f"[Mirror Descent] Start training for {max_iter} iterations.")
            print(f"[Mirror Descent] Learning rate: {lr:.4e}")

        # Initialize
        w = np.ones(L) / L
        w_hist = [w.copy()]
        phi_hist = []
        gap_hist = []

        it = 0

        # tqdm progress bar
        iterator = tqdm(
            range(max_iter),
            disable=not progress,
            desc="Mirror Descent",
            leave=True
        )

        for t in iterator:
            it = t + 1

            # Weighted covariance Σ(w_t)
            Sigma_w = np.tensordot(w, Sigma_stack, axes=(0, 0))  # (d, d)

            # Eigendecomposition for top-k eigenvalues/vectors
            if (d > 30 and k < d // 2):
                eval, evec = eigsh(Sigma_w, k=k, which='LA')
                idx = np.argsort(eval)[::-1]
                top_evals = eval[idx]
                U_k = evec[:, idx]     # (d, k)
            else:
                eval, evec = np.linalg.eigh(Sigma_w)
                idx = np.argsort(eval)[::-1]
                top_evals = eval[idx[:k]]
                U_k = evec[:, idx[:k]]

            # phi(w_t) = sum of top-k eigenvalues
            phi_val = float(np.sum(top_evals))
            phi_hist.append(phi_val)

            # Gradient in w: grad_l = <Σ_l, M_t>, M_t = U_k U_k^T
            T = np.einsum('lij,jk->lik', Sigma_stack, U_k)  # (L, d, k)
            grad = np.einsum('jk,ljk->l', U_k, T)           # (L,)

            # Entropic MD update
            unnorm = w * np.exp(-lr * grad)
            w_new = unnorm / np.sum(unnorm)
            w_hist.append(w_new.copy())

            # L1 step size
            delta = np.linalg.norm(w_new - w, ord=1)

            # Duality gap on averaged w
            duality_gap = None
            if check_dual:
                w_avg = np.mean(w_hist, axis=0)
                duality_gap = self._check_duality_gap(Sigma_list, w_avg)
                gap_hist.append(duality_gap)

            # Logging
            if use_bar and (it == 1 or it % print_freq == 0):
                if len(phi_hist) > 1:
                    diff_primal = abs(phi_hist[-1] - phi_hist[-2])
                else:
                    diff_primal = float('nan')

                if duality_gap is not None:
                    msg = (f"Iter {it:4d} | Diff primal: {diff_primal:.6e} | "
                        f"duality gap: {duality_gap:.6e} | L1 step: {delta:.6e}")
                else:
                    msg = (f"Iter {it:4d} | Diff primal: {diff_primal:.6e} | "
                        f"L1 step: {delta:.6e}")

                # This prints *with* the tqdm bar, like:
                # 10%|█         | 101/1000 [..] 
                # Iter 101 | Diff primal: ...
                iterator.write(msg)

            # Update tqdm postfix (optional, nice UX)
            if use_bar:
                postfix = {"phi": phi_val}
                if duality_gap is not None:
                    postfix["gap"] = duality_gap
                iterator.set_postfix(postfix)

            # Stopping rule
            if delta < tol:
                w = w_new
                break

            w = w_new

        # Final averaged w
        w = np.mean(w_hist, axis=0)

        info = {
            'phi_hist': phi_hist,
            'gap_hist': gap_hist,
            'iters': it,
            'w_hist': w_hist,
        }
        return w, phi_val, info

    def _choose_learning_rate(self, Sigma_list, k, T):
        # L = number of sources
        L = len(Sigma_list)

        # Estimate operator norms
        op_norms = []
        for S in Sigma_list:
            # largest eigenvalue in absolute value (since symmetric)
            eval = np.linalg.eigvalsh(S)
            op_norms.append(np.max(np.abs(eval)))
        S_max = max(op_norms)
        G_est = k * S_max
        lr = 5 * np.sqrt(2 * np.log(L)) / (G_est * np.sqrt(T))
        return lr

    def _check_duality_gap(self, Sigma_list, w):
        """
        1. compute M* = \argmin_M -<M, Sigma(w)>
        2. then the dual value is min_M -<M, Sigma(w)> = -<M*, Sigma(w)>
        3. the primal value is max_w -<M*, Sigma(w)>
        4. the duality gap is the difference between the primal and dual values
        """

        d = Sigma_list[0].shape[0]
        Sigma_w = np.tensordot(w, Sigma_list, axes=(0, 0)) # (d, d)

        # --- partial eigendecomp (fast for k << d) ---
        k = self.n_components
        if (d > 30 and k < d // 2):
            eval, evec = eigsh(Sigma_w, k=k, which='LA')  # largest algebraic
            idx = np.argsort(eval)[::-1]
            top_evals = eval[idx]
            U_k = evec[:, idx]
        else:
            eval, evec = np.linalg.eigh(Sigma_w)
            idx = np.argsort(eval)[::-1]
            top_evals = eval[idx[:k]]
            U_k = evec[:, idx[:k]]
        dual_val = - float(np.sum(top_evals))

        M_w = U_k @ U_k.T
        vals = np.array([-np.trace(Sigma.T @ M_w) for Sigma in Sigma_list], dtype=float)
        primal_val = float(vals.max())

        duality_gap = np.abs(primal_val - dual_val)
        
        return duality_gap
