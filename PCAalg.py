import numpy as np
from scipy.linalg import logm, expm, eigh
from scipy.optimize import root_scalar


class StablePCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.M = None
        self.w = None
        
    def fit(self, X_list, M_init=None, w_init=None, eta_init=None,
            max_iter=1000, tol=1e-6, verbose=True, adaptive=True,
            check_dual=False):
        # --------- Prepare Data --------- #
        L = len(X_list)
        d = X_list[0].shape[1]
        Sigma_list = [self._getSigma(X, demean=False) for X in X_list]
        Sigma_stack = np.stack(Sigma_list)

        # --------- Initialize Parameters --------- #        
        self.w = np.ones(L) / L if w_init is None else w_init
        self.M = np.einsum('i,ijk->jk', self.w, Sigma_stack) if M_init is None else M_init
        self.M = self._best_rank_k_approximation(self.M, self.n_components)
        M_bar = self.M.copy()
        w_bar = self.w.copy()
        # parameters for adaptative learning rate
        a = self.n_components * np.log(d)
        b = np.log(L)
        if adaptive:
            eta = eta_init if eta_init is not None else 0.5
            Z_cumsum = 0.
        else:
            eta = 1 / (8 * np.max([np.linalg.norm(Sigma, ord=2) for Sigma in Sigma_list]) * np.sqrt(a * b))
        
        primal = self._compute_primal(Sigma_list, self.M)
        # optimization loop
        for iter in range(max_iter):
            # ----------- Middle Step ---------- #
            M_bar, w_bar = self._update_M_w(Sigma_stack, self.M, self.w, M_bar, w_bar, eta, a, b, self.n_components)
            # ----------- Proximal Step ---------- #
            M_curr, w_curr = self._update_M_w(Sigma_stack, self.M, self.w, M_bar, w_bar, eta, a, b, self.n_components)
            # ----------- Adaptive learning rate ---------- #
            if adaptive:
                Z = a * (self._schatten_1_norm(M_bar - self.M) ** 2 
                        + self._schatten_1_norm(M_curr - M_bar) ** 2) \
                            + b * (np.linalg.norm(w_bar - self.w, ord=1) ** 2
                                    + np.linalg.norm(w_curr - w_bar, ord=1) ** 2)
                Z_cumsum += Z / (5 * eta ** 2)
                if eta > 1e-2:
                    eta = eta / np.sqrt(1 + Z_cumsum)
                else:
                    eta = 1 / (16 * np.max([np.linalg.norm(Sigma, ord=2) for Sigma in Sigma_list]) * np.sqrt(a * b))
            
            # ----------- Update Parameters ---------- #
            self.M = M_curr
            self.w = w_curr
            primal_curr = self._compute_primal(Sigma_list, self.M)
            if iter % 50 == 0:
                log_info = f"Iter {iter +1} | Diff primal: {np.abs(primal - primal_curr):.6f}"
                if check_dual:
                    dual_curr = self._compute_dual(Sigma_list, self.w)
                    log_info += f" | duality gap: {primal_curr-dual_curr:.6f}"
                if verbose:
                    print(log_info)
            
            # Check convergence
            if np.abs(primal - primal_curr) < tol:
                dual_curr = self._compute_dual(Sigma_list, self.w)
                if check_dual and np.abs(primal_curr - dual_curr) < 1e-2:
                    if verbose:
                        print(f"Converged at iteration {iter + 1}")
                    break
                else:
                    if verbose:
                        print(f"Converged at iteration {iter + 1} but duality gap may not be small.")
                    break
            primal = primal_curr

        # --------- Finalize Projection Matrix --------- #
        # Considering that M is not sparse, we only output the top k eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.M)
        # Sort eigenvectors in descending order of eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        # Select top k eigenvalues and eigenvectors
        top_k_eigenvalues = eigenvalues[sorted_indices][:self.n_components]
        top_k_eigenvectors = eigenvectors_sorted[:, :self.n_components]
        self.components_ = top_k_eigenvectors
        
        # Sort eigenvectors in descending order of explained variance
        X_centered = np.vstack(X_list) - np.vstack(X_list).mean(axis=0)
        X_proj = X_centered @ top_k_eigenvectors
        explained_variance = np.var(X_proj, axis=0)
        sorted_indices = np.argsort(explained_variance)[::-1]
        self.mean_ = np.vstack(X_list).mean(axis=0)[sorted_indices]
        self.components_ = top_k_eigenvectors[:, sorted_indices]
        self.explained_variance_ = explained_variance[sorted_indices]
        self.M_truncated = top_k_eigenvectors @ np.diag(top_k_eigenvalues) @ top_k_eigenvectors.T
        self.M_proj = top_k_eigenvectors @ top_k_eigenvectors.T
        
        self.primal_M = self._compute_primal(Sigma_list, self.M)
        self.primal_M_truncated = self._compute_primal(Sigma_list, self.M_truncated)
        self.primal_M_proj = self._compute_primal(Sigma_list, self.M_proj)
        
    # ----------------------------------------------- #
    # Helper functions
    # ----------------------------------------------- #
    
    def _compute_primal(self, Sigma_list, M):
        """
        Compute the primal objective function value.
        """
        return np.max([-np.trace(Sigma.T @ M) for Sigma in Sigma_list])
    
    def _compute_dual(self, Sigma_list, w):
        Sigma_stack = np.stack(Sigma_list)
        A = np.einsum('i,ijk->jk', w, Sigma_stack)
        # conduct eigenvalue decomposition to A
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argsort(eigvals)[::-1][:self.n_components]
        U = eigvecs[:, idx]
        M = U @ U.T
        # compute the dual objective function value
        dual = np.sum([-wl * np.trace(Sigma.T @ M) for Sigma, wl in zip(Sigma_list, w)])
        return dual 
    
    def _update_M_w(self, Sigma_list, M, w, M_bar, w_bar, eta, a, b, k):
        Sigma_stack = np.stack(Sigma_list)
        # for omega
        w_new = w * np.exp([- eta/b * np.trace(Sigma.T @ M_bar) for Sigma in Sigma_list])
        w_new = w_new / w_new.sum()
        # for M
        A = eta / a * np.einsum('i,ijk->jk', w_bar, Sigma_stack) + logm(M)
        lambs, U = eigh(A)
        nu = self._solve_for_nu(lambs, k)
        M_new = U @ np.diag(np.minimum(np.exp(lambs + nu), 1)) @ U.T
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
            xtol=1e-8
        )
        return sol.root
    
    def _getSigma(self, X, demean=True):
        if demean:
            X = X - X.mean(axis=0)
        return X.T @ X / X.shape[0]
    
    def _best_rank_k_approximation(self, A, k):
        """
        Compute the best rank-k approximation of A using its eigenvalues and eigenvectors.
        """
        epsilon = 0.05
        lambdas, U = eigh(A)
        idx = np.argsort(lambdas)[::-1][:k]
        Q_k = U[:, idx]
        alpha = 1 - epsilon
        beta = (k * epsilon) / A.shape[0]
        P_k = Q_k @ Q_k.T
        M = alpha * P_k + beta * np.eye(A.shape[0])
        return M
    
    def _schatten_1_norm(self, A):
        eigenvalues = np.linalg.eigvalsh(A)
        schatten_1_norm = np.sum(np.abs(eigenvalues))
        return schatten_1_norm