import numpy as np


def subspace_principal_angles(A, V):
    """
    A, V: (p, k) with full column rank
    returns the k principal angles in radians
    """
    # Orthonormal bases
    QA, _ = np.linalg.qr(A)
    QV, _ = np.linalg.qr(V)

    # Singular values of QA^T QV = cos(theta_i)
    M = QA.T @ QV
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, -1.0, 1.0)
    thetas = np.arccos(s)
    return thetas

def subspace_error(A, V):
    thetas = subspace_principal_angles(A, V)
    # could use max angle, mean angle, or sin of max angle
    return np.sin(thetas).max(), thetas

def proj_error(M1, M2):
    """Compute frobenius norm between projection matrices of two subspaces.
    """
    return np.linalg.norm(M1 - M2, ord='fro')


def explained_variance_ratio_subspace(V, X):
    """
    V: (p, k) basis (not necessarily orthonormal)
    X: (n, p) data matrix
    returns trace(V^T Σ V) / trace(Σ)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Sigma = np.cov(Xc, rowvar=False)  # (p, p)

    # Orthonormalize V to avoid scaling issues
    Q, _ = np.linalg.qr(V)
    num = np.trace(Q.T @ Sigma @ Q)
    den = 1
    # den = np.trace(Sigma)
    return num / den

def explained_variance(M, X):
    Sigma = X.T @ X / X.shape[0]
    return np.trace(Sigma @ M)


def get_sigma(X, demean=True):
    """Compute empirical covariance matrix.
    
    Args:
        X: (n, p) data matrix
        demean: Whether to subtract the mean. Defaults to True.
    
    Returns:
        (p, p) covariance matrix
    """
    if demean:
        X = X - X.mean(axis=0)
    return X.T @ X / X.shape[0]


def real_sym(A):
    """Force A to be real symmetric float.
    
    Args:
        A: Input matrix
    
    Returns:
        Real symmetric matrix
    """
    A_sym = 0.5 * (A + A.T)
    A_sym = np.real_if_close(A_sym, tol=1000)
    return np.asarray(A_sym.real, dtype=float)


def real_vec(v):
    """Force v to be real float vector.
    
    Args:
        v: Input vector
    
    Returns:
        Real float vector
    """
    v = np.real_if_close(v, tol=1000)
    return np.asarray(v.real, dtype=float)


def spd_logm(M):
    """Matrix log for (approximately) SPD matrix M via eigen-decomposition.
    Returns a real symmetric matrix.
    
    Args:
        M: Input matrix (approximately symmetric positive definite)
    
    Returns:
        Real symmetric matrix (log of M)
    """
    M_sym = real_sym(M)
    eval, evec = np.linalg.eigh(M_sym)
    # clip eigenvalues away from zero to avoid log of nonpositive
    eval_clipped = np.clip(eval, 1e-10, None)
    log_eig = np.log(eval_clipped)
    A = evec @ np.diag(log_eig) @ evec.T
    return real_sym(A)
