import numpy as np
import cvxpy as cp


def truncated_svd(X, rank):
    """
    Rank-`rank` approximation of X using SVD.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, len(s))
    U_r = U[:, :r]
    s_r = s[:r]
    Vt_r = Vt[:r, :]
    return U_r @ np.diag(s_r) @ Vt_r


def fair_pca_multisource(groups, d, sdp_solver="SCS", lp_solver="ECOS"):
    """
    Multi-source Fair PCA, the implementation is from the paper:
    "The Price of Fair PCA: One extra dimension".


    Parameters
    ----------
    groups : list of array-like
        List [X^(1), ..., X^(L)] with shapes (m_l, n).
        All groups must have the same n (feature dimension).
    d : int
        Target dimension (must satisfy d < n).
    sdp_solver : str
        CVXPY solver name for the SDP step.
    lp_solver : str
        CVXPY solver name for the LP step.

    Returns
    -------
    P_star : ndarray, shape (n, n)
        The final projection matrix P^* with exact rank d.
    """
    # --------- basic checks ----------
    L = len(groups)
    if L < 2:
        raise ValueError("Need at least two groups for Fair PCA.")
    groups = [np.asarray(G, dtype=float) for G in groups]
    n = groups[0].shape[1]
    for G in groups:
        if G.shape[1] != n:
            raise ValueError("All group matrices must have the same number of columns.")
    if d >= n:
        raise ValueError("Need d < n.")

    # --------- Step 1: rank-d approximations ----------
    A_hats = [truncated_svd(G, d) for G in groups]

    m_list = [G.shape[0] for G in groups]
    norms2 = [np.linalg.norm(Ah, "fro") ** 2 for Ah in A_hats]
    Gram_list = [G.T @ G for G in groups]  # G_l^T G_l, each (n, n)

    # --------- Step 2: SDP in P, z ----------
    P = cp.Variable((n, n), PSD=True)
    z = cp.Variable()

    constraints = []
    for l in range(L):
        constraints.append(
            z >= (1.0 / m_list[l]) * (norms2[l] - cp.trace(Gram_list[l] @ P))
        )
    constraints += [
        cp.trace(P) <= d,
        P << cp.Constant(np.eye(n)),  # P ⪯ I
    ]

    prob_sdp = cp.Problem(cp.Minimize(z), constraints)
    
    # Note: CVXPY >= 1.6.4 fixes the negative axis index bug for large SDP problems
    # Determine which solvers support PSD constraints (SCS and CVXOPT)
    psd_capable_solvers = []
    if "SCS" in cp.installed_solvers():
        psd_capable_solvers.append("SCS")
    if "CVXOPT" in cp.installed_solvers():
        psd_capable_solvers.append("CVXOPT")
    
    # Build solver list (prioritize specified solver if it's PSD-capable)
    solver_list = []
    if sdp_solver in psd_capable_solvers:
        solver_list.append(sdp_solver)
    solver_list.extend([s for s in psd_capable_solvers if s != sdp_solver])
    
    if not solver_list:
        raise RuntimeError(
            "No PSD-capable solvers available. Install SCS (recommended) or CVXOPT. "
            f"Installed solvers: {cp.installed_solvers()}"
        )
    
    # Try solving with available PSD-capable solvers
    solved = False
    last_error = None
    
    for solver in solver_list:
        try:
            prob_sdp.solve(solver=solver, verbose=False)
            if P.value is not None:
                solved = True
                break
        except Exception as err:
            last_error = err
            continue
    
    if not solved:
        error_msg = f"SDP solver failed. Tried solvers: {solver_list}\n"
        if last_error:
            error_msg += f"Last error: {str(last_error)}\n"
        error_msg += f"Ensure cvxpy >= 1.6.4 is installed for large dimension support."
        raise RuntimeError(error_msg)

    P_hat = 0.5 * (P.value + P.value.T)  # symmetrize numerically

    # --------- Step 3: eigendecomposition of P_hat ----------
    lam_hat, U_eig = np.linalg.eigh(P_hat)   # ascending
    idx = np.argsort(-lam_hat)               # descending
    lam_hat = lam_hat[idx]
    U_eig = U_eig[:, idx]

    # --------- Step 4: LP over λ and z in eigenbasis ----------
    # a_lj = u_j^T G_l u_j
    a_lj = np.stack(
        [np.array([u.T @ Gram_list[l] @ u for u in U_eig.T]) for l in range(L)],
        axis=0
    )  # shape (L, n)

    lam = cp.Variable(n)
    z_lp = cp.Variable()

    constraints_lp = []
    for l in range(L):
        # <G_l, Σ_j λ_j u_j u_j^T> = Σ_j λ_j a_lj[l, j]
        constraints_lp.append(
            z_lp >= (1.0 / m_list[l]) * (norms2[l] - a_lj[l] @ lam)
        )
    constraints_lp += [
        cp.sum(lam) <= d,
        lam >= 0,
        lam <= 1,
    ]

    prob_lp = cp.Problem(cp.Minimize(z_lp), constraints_lp)
    # Try the specified solver, fall back to CLARABEL if not available
    try:
        prob_lp.solve(solver=lp_solver, verbose=False)
        if lam.value is None:
            # If solver didn't work, try CLARABEL
            prob_lp.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        # If solver is not available, use CLARABEL
        prob_lp.solve(solver=cp.CLARABEL, verbose=False)
    if lam.value is None:
        raise RuntimeError("LP did not converge. Try a different solver.")

    lam_bar = np.clip(lam.value, 0.0, 1.0)

    # --------- Step 5: λ*_j = 1 - sqrt(1 - λ̄_j), P* = Σ_j λ*_j u_j u_j^T ----------
    lam_star = 1.0 - np.sqrt(1.0 - lam_bar)
    lam_star = np.clip(lam_star, 0.0, 1.0)

    # --------- Step 6: Ensure exact rank d by keeping only top d eigenvalues ----------
    # Sort eigenvalues in descending order
    idx_sorted = np.argsort(-lam_star)  # descending order
    lam_star_sorted = lam_star[idx_sorted]
    U_eig_sorted = U_eig[:, idx_sorted]
    
    # Keep only top d eigenvalues, set rest to zero
    lam_star_rank_d = np.zeros_like(lam_star_sorted)
    lam_star_rank_d[:d] = lam_star_sorted[:d]
    
    # Construct projection matrix with exact rank d
    P_star = U_eig_sorted @ np.diag(lam_star_rank_d) @ U_eig_sorted.T

    return P_star


if __name__ == "__main__":
    # Simple test
    np.random.seed(0)
    X1 = np.random.randn(100, 5)
    X2 = np.random.randn(150, 5) + 1.0
    groups = [X1, X2]
    d = 2

    P_star = fair_pca_multisource(groups, d)
    eigvals, eigvecs = np.linalg.eigh(P_star)
    eigvals = np.sort(eigvals)[::-1]
    print("Projection matrix P*:")
    print(P_star)
    print("Top-{} eigenvalues of P*: {}".format(d, eigvals[:5]))