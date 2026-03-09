import numpy as np


def get_sigma(X, demean=True):
    """
    Compute the covariance matrix of X.
    
    Args:
        X: Input data matrix of shape (n, d)
        demean: Whether to subtract the mean (default: True)
    
    Returns:
        Covariance matrix of shape (d, d)
    """
    if demean:
        X = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    return (X.T @ X) / n


def real_sym(M):
    """Force matrix M to be real and symmetric.
    
    Args:
        M: Input matrix
    
    Returns:
        Real symmetric matrix
    """
    M = np.real_if_close(M, tol=1000)
    return (M + M.T) / 2


def real_vec(v):
    """Force v to be real float vector.
    
    Args:
        v: Input vector
    
    Returns:
        Real float vector
    """
    v = np.real_if_close(v, tol=1000)
    return np.asarray(v.real, dtype=float)


def randomized_eigh(A, k, n_oversample=10, n_iter=2, random_state=None):
    """
    Randomized eigendecomposition for symmetric matrices.
    Computes top k eigenvalues and eigenvectors using randomized SVD approach.
    
    Based on Halko et al. (2011) "Finding structure with randomness: 
    Probabilistic algorithms for constructing approximate matrix decompositions"
    
    Args:
        A: Symmetric matrix of shape (p, p)
        k: Number of top eigenvalues/vectors to compute
        n_oversample: Extra samples for better accuracy (default: 10)
        n_iter: Number of power iterations for better accuracy (default: 2)
        random_state: Random seed for reproducibility
    
    Returns:
        eval: Top k eigenvalues in descending order (k,)
        evec: Top k eigenvectors (p, k), columns are eigenvectors
    """
    # Use a local RNG to keep reproducibility without mutating NumPy's global RNG state.
    rng = np.random.default_rng(random_state) if random_state is not None else np.random
    
    p = A.shape[0]
    if k >= p:
        # If k >= p, just do full eigendecomposition
        eval, evec = np.linalg.eigh(A)
        idx = np.argsort(eval)[::-1]
        return eval[idx], evec[:, idx]
    
    # Step 1: Generate random test matrix
    l = k + n_oversample  # Number of random vectors
    Omega = rng.standard_normal((p, l))
    
    # Step 2: Power iteration to find approximate range of A
    # Y = (A^T @ A)^n_iter @ Omega ≈ A^(2*n_iter) @ Omega
    # For symmetric A, A^T = A, so Y = A^(2*n_iter) @ Omega
    Y = Omega.copy()
    for i in range(n_iter):
        Y = A @ Y  # (p, l)
        # Orthonormalize to prevent rank deficiency
        Q, R = np.linalg.qr(Y, mode='reduced')
        Y = Q
    
    # Final projection: Y = A @ Y (one more time for symmetric A)
    Y = A @ Y  # (p, l)
    
    # Step 3: Orthonormalize to get Q
    Q, _ = np.linalg.qr(Y, mode='reduced')  # (p, l)
    
    # Step 4: Project A to lower-dimensional space
    B = Q.T @ A @ Q  # (l, l), much smaller!
    
    # Step 5: Eigendecomposition in small space
    eval_B, evec_B = np.linalg.eigh(B)  # O(l³), very fast!
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eval_B)[::-1]
    eval_B_sorted = eval_B[idx]
    evec_B_sorted = evec_B[:, idx]
    
    # Step 6: Project eigenvectors back to original space
    evec_full = Q @ evec_B_sorted  # (p, l)
    
    # Step 7: Extract top k
    eval = eval_B_sorted[:k]
    evec = evec_full[:, :k]
    
    # Optional: Refinement step for better accuracy
    # For symmetric matrices, we can do a few refinement iterations
    if n_iter >= 1:
        # One refinement iteration
        Y_refined = A @ evec  # (p, k)
        Q_refined, _ = np.linalg.qr(Y_refined, mode='reduced')
        B_refined = Q_refined.T @ A @ Q_refined
        eval_refined, evec_B_refined = np.linalg.eigh(B_refined)
        idx_refined = np.argsort(eval_refined)[::-1]
        eval = eval_refined[idx_refined][:k]
        evec = Q_refined @ evec_B_refined[:, idx_refined][:, :k]
    
    return eval, evec


def spd_logm(M, n_eig=None, tol=1e-8, return_components=False, use_randomized=False, random_state=None):
    """Matrix log for (approximately) SPD matrix M via eigen-decomposition.
    Optimized for low-rank matrices by computing only top eigenvalues.
    Returns a real symmetric matrix.
    
    Args:
        M: Input matrix (approximately symmetric positive definite)
        n_eig: Number of top eigenvalues to compute. If None, computes all.
               For low-rank M, set to ~2*k where k is the target rank.
               If specified and n_eig < d, uses partial eigendecomposition.
        tol: Tolerance for clipping eigenvalues away from zero.
        return_components: If True, also return U_k and log_eval_k for optimization.
                          U_k is the top n_eig eigenvectors, log_eval_k are the 
                          corresponding log eigenvalues (without the log(tol) shift).
        use_randomized: If True, use randomized SVD instead of eigsh (default: False)
        random_state: Random seed for randomized SVD
    
    Returns:
        If return_components=False:
            Real symmetric matrix (log of M)
        If return_components=True:
            (log_M, U_k, log_eval_k) where:
            - log_M: Real symmetric matrix (log of M)
            - U_k: Top n_eig eigenvectors (p × n_eig)
            - log_eval_k: Top n_eig log eigenvalues (shifted, i.e., log(λ) - log(tol))
    """
    M_sym = real_sym(M)
    d = M_sym.shape[0]
    
    # For large dimensions, use partial eigendecomposition if n_eig is specified
    if n_eig is not None and n_eig < d:
        log_tol = np.log(tol)
        
        # For low-rank matrices (n_eig << d), eigsh is often faster than randomized SVD
        # because it's highly optimized LAPACK and doesn't need power iterations
        # Use randomized only if n_eig is relatively large (e.g., > 20)
        use_eigsh_direct = (not use_randomized) or (n_eig <= 20)
        
        if use_eigsh_direct:
            # Direct eigsh for low-rank matrices - fastest option
            from scipy.sparse.linalg import eigsh
            # Compute top n_eig eigenvalues (use a bit more for numerical stability)
            k_compute = min(n_eig + 5, d - 1)  # Reduced from 10 to 5 for low-rank
            eval, evec = eigsh(M_sym, k=k_compute, which='LA')
            # Note: eigsh returns eigenvalues in ascending order, so reverse
            eval = eval[::-1]
            evec = evec[:, ::-1]
        else:
            # Use randomized SVD for larger n_eig or when explicitly requested
            # For speed, use n_iter=1 instead of 2 (slightly less accurate but faster)
            # In practice, for low-rank matrices, n_iter=1 is often sufficient
            eval, evec = randomized_eigh(M_sym, k=n_eig, n_oversample=10, n_iter=1, random_state=random_state)
            # randomized_eigh already returns descending order
        
        # Clip eigenvalues and compute log
        eval_clipped = np.clip(eval[:n_eig], tol, None)
        log_eval_computed = np.log(eval_clipped)
        
        # Approximate: log(M) ≈ U @ diag(log(λ)) @ U.T + log(ε) * (I - U @ U.T)
        # where λ are computed eigenvalues and ε is small value for uncomputed ones
        # This equals: U @ diag(log(λ) - log(ε)) @ U.T + log(ε) * I
        
        log_eval_shifted = log_eval_computed - log_tol
        U = evec[:, :n_eig]
        
        # Construct the log matrix efficiently
        logM_approx = U @ np.diag(log_eval_shifted) @ U.T + log_tol * np.eye(d)
        
        # For low-rank matrices, logM_approx should already be symmetric
        # Skip real_sym for performance (we already used symmetric M_sym)
        log_M_result = logM_approx
        
        if return_components:
            return log_M_result, U, log_eval_shifted
        else:
            return log_M_result
        
    else:
        # Full eigendecomposition (original implementation)
        eval, evec = np.linalg.eigh(M_sym)
        eval_clipped = np.clip(eval, tol, None)
        log_eig = np.log(eval_clipped)
        A = evec @ np.diag(log_eig) @ evec.T
        # A should already be symmetric (evec from eigh is orthonormal)
        # But for numerical stability, we keep it as is (symmetric by construction)
        log_M_result = A
        
        if return_components and n_eig is not None:
            # Extract top n_eig components
            idx_top = np.argsort(eval_clipped)[::-1][:n_eig]
            U_k = evec[:, idx_top]
            log_eval_k = log_eig[idx_top] - np.log(tol)  # Shift by log(tol) for consistency
            return log_M_result, U_k, log_eval_k
        elif return_components:
            # Return all components
            log_tol = np.log(tol)
            return log_M_result, evec, log_eig - log_tol
        else:
            return log_M_result
