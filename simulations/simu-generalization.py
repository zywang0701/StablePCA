"""
Out-of-distribution generalization simulation.
Runs k ∈ {3, 5, 10}, L ∈ {2, 4, 6, 8, 10}, n_runs=50 per (k, L).
Saves results to pkl files: results-generalization_k{k}_L{L}.pkl
"""

import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
import sys
from pathlib import Path
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

from src.PCAalg import PCA_MP
from src.utils import proj_error, explained_variance, subspace_capture
from src.data import generate_stablepca_data

# ----------------------------
# Simulation parameters
# ----------------------------
n = 2000
k_list = [3, 5, 10]
m = 5
p = 40
L_list = [2, 4, 6, 8, 10]
n_runs = 100
METHODS = ["pooled", "stable", "squared", "fair"]
results_dir = str(_project_root / "saved_results")
os.makedirs(results_dir, exist_ok=True)


# ----------------------------
# Helper: one run for given k, L and seed
# ----------------------------
def run_one_simulation(k, L, seed, verbose=False, pooled_only=False):
    np.random.seed(seed)

    # Generate data: true subspace is always rank 3 (hardcoded)
    k_true = 3
    X_list, X_ood_list, A, _, _ = generate_stablepca_data(
        n=n, p=p, k=k_true, m=m, L=L, random_state=seed
    )

    # True subspace: V_true is (p x k_true) orthonormal basis
    eigvals, eigvecs = np.linalg.eigh(A @ A.T / A.shape[0])
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    V_true = eigvecs[:, :k_true]
    P_true = V_true @ V_true.T

    # err: k=3 use proj_error (Frobenius); k>3 use subspace_capture (1-capture, fair when k_est>k_true)
    def _err(P_est):
        if k == 3:
            return proj_error(P_est, P_true)
        else:
            return 1.0 - subspace_capture(V_true, P_est)

    # ------------------------------------------------------------------
    # 1) Pooled PCA
    # ------------------------------------------------------------------
    X_pooled = np.vstack(X_list)
    pca_pooled = PCA(n_components=k).fit(X_pooled)
    V_pooled = pca_pooled.components_.T
    P_pooled = V_pooled @ V_pooled.T

    err_pooled = _err(P_pooled)
    expl_var_pooled_src = min(explained_variance(P_pooled, X) for X in X_list)
    expl_var_pooled_ood = min(explained_variance(P_pooled, X) for X in X_ood_list)

    if pooled_only:
        return {
            "L": L,
            "seed": seed,
            "err": [err_pooled, np.nan, np.nan, np.nan],
            "expl_var_src": [expl_var_pooled_src, np.nan, np.nan, np.nan],
            "expl_var_ood": [expl_var_pooled_ood, np.nan, np.nan, np.nan],
        }

    # ------------------------------------------------------------------
    # 2) Stable PCA
    # ------------------------------------------------------------------
    pca_stable = PCA_MP(n_components=k, method='stable')
    pca_stable.fit(
        X_list, max_iter=500, eta_init=100,
        verbose=500 if verbose else False, decay_every=1000,
        check_dual=False
    )
    P_stable = pca_stable.P

    # ------------------------------------------------------------------
    # 3) Squared PCA
    # ------------------------------------------------------------------
    pca_squared = PCA_MP(n_components=k, method='squared')
    pca_squared.fit(
        X_list, max_iter=500, eta_init=100,
        verbose=500 if verbose else False, decay_every=1000,
        check_dual=False
    )
    P_squared = pca_squared.P

    # ------------------------------------------------------------------
    # 4) Fair PCA
    # ------------------------------------------------------------------
    pca_fair = PCA_MP(n_components=k, method='fair')
    pca_fair.fit(
        X_list, max_iter=500, eta_init=100,
        verbose=500 if verbose else False, decay_every=1000,
        check_dual=False
    )
    P_fair = pca_fair.P

    err_stable = _err(P_stable)
    err_squared = _err(P_squared)
    err_fair = _err(P_fair)
    expl_var_stable_src = min(explained_variance(P_stable, X) for X in X_list)
    expl_var_squared_src = min(explained_variance(P_squared, X) for X in X_list)
    expl_var_fair_src = min(explained_variance(P_fair, X) for X in X_list)
    expl_var_stable_ood = min(explained_variance(P_stable, X) for X in X_ood_list)
    expl_var_squared_ood = min(explained_variance(P_squared, X) for X in X_ood_list)
    expl_var_fair_ood = min(explained_variance(P_fair, X) for X in X_ood_list)

    return {
        "L": L,
        "seed": seed,
        "err": [err_pooled, err_stable, err_squared, err_fair],
        "expl_var_src": [expl_var_pooled_src, expl_var_stable_src, expl_var_squared_src, expl_var_fair_src],
        "expl_var_ood": [expl_var_pooled_ood, expl_var_stable_ood, expl_var_squared_ood, expl_var_fair_ood],
    }


# ----------------------------
# Large simulation: run for each (k, L), save one pkl per (k, L)
# ----------------------------
def run_large_simulation(n_runs_override=None, pooled_only=False, k_use=None):
    """Run n_runs simulations per (k, L). Save results to pkl for each (k, L)."""
    nr = n_runs_override if n_runs_override is not None else n_runs
    if k_use is None:
        k_use = [3] if pooled_only else k_list
    mode = "pooled-only" if pooled_only else "all methods"
    print(f"Running ({mode})")

    for k in k_use:
        for L in L_list:
            print(f"\n{'='*60}\nRunning k={k}, L={L} (n_runs={nr})\n{'='*60}")

            seed_arr = np.zeros(nr, dtype=int)
            err_mat = np.zeros((nr, 4))
            expl_var_src_mat = np.zeros((nr, 4))
            expl_var_ood_mat = np.zeros((nr, 4))

            for r in tqdm(range(nr), desc=f"k={k}, L={L}", unit="run"):
                try:
                    result = run_one_simulation(k=k, L=L, seed=r, verbose=False, pooled_only=pooled_only)
                    seed_arr[r] = result["seed"]
                    err_mat[r, :] = result["err"]
                    expl_var_src_mat[r, :] = result["expl_var_src"]
                    expl_var_ood_mat[r, :] = result["expl_var_ood"]
                except Exception as e:
                    print(f"\n  ERROR at run {r}: {e}")
                    err_mat[r, :] = np.nan
                    expl_var_src_mat[r, :] = np.nan
                    expl_var_ood_mat[r, :] = np.nan
                    seed_arr[r] = r

            results = {
                "k": k,
                "L": L,
                "n_runs": nr,
                "METHODS": METHODS,
                "seed": seed_arr,
                "err": err_mat,
                "expl_var_src": expl_var_src_mat,
                "expl_var_ood": expl_var_ood_mat,
            }
            fname = os.path.join(results_dir, f"results-generalization_k{k}_L{L}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved to {fname}")
            se_scale = np.sqrt(nr) if nr > 1 else 1.0
            if pooled_only:
                se_err = np.nanstd(err_mat[:, 0], ddof=1) / se_scale
                se_src = np.nanstd(expl_var_src_mat[:, 0], ddof=1) / se_scale
                se_ood = np.nanstd(expl_var_ood_mat[:, 0], ddof=1) / se_scale
                print(f"  err: {np.nanmean(err_mat[:, 0]):.4f} ± {se_err:.4f} (SE)")
                print(f"  expl_var_src: {np.nanmean(expl_var_src_mat[:, 0]):.4f} ± {se_src:.4f} (SE)")
                print(f"  expl_var_ood: {np.nanmean(expl_var_ood_mat[:, 0]):.4f} ± {se_ood:.4f} (SE)")
            else:
                for j, method in enumerate(METHODS):
                    mval = np.nanmean(err_mat[:, j])
                    se_val = np.nanstd(err_mat[:, j], ddof=1) / se_scale
                    print(f"  {method}: {mval:.4f} ± {se_val:.4f} (SE)")


# ----------------------------
# Single test run (quick sanity check)
# ----------------------------
def run_single_test():
    """Quick single-run test."""
    print("=" * 60)
    print("Running Single Simulation Test (k=3, L=6)")
    print("=" * 60)

    result = run_one_simulation(k=3, L=6, seed=1, verbose=True)

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print("\n Projection Errors:")
    print(f"  {'Method':<12} {'Error':<12}")
    print(f"  {'-'*11} {'-'*11}")
    for method, error in zip(METHODS, result["err"]):
        print(f"  {method.capitalize():<12} {error:.4f}")
    print("\n Worst-case Explained Variance (Source):")
    for method, var in zip(METHODS, result["expl_var_src"]):
        print(f"  {method.capitalize():<12} {var:.4f}")
    print("\n Worst-case Explained Variance (OOD):")
    for method, var in zip(METHODS, result["expl_var_ood"]):
        print(f"  {method.capitalize():<12} {var:.4f}")
    print("\n" + "=" * 60)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generalization simulation")
    parser.add_argument("--test", action="store_true", help="Run single test only")
    parser.add_argument("--n_runs", type=int, default=100, help="Override n_runs (for quick testing)")
    parser.add_argument("--pooled-only", action="store_true",
        help="Only run Pooled PCA (for verification vs summary_OOD.png)")
    parser.add_argument("--k", type=int, default=3,
        help="Which k to run (default: 3). Use with all methods.")
    args = parser.parse_args()

    n_runs_override = args.n_runs
    if args.pooled_only:
        n_runs_override = 100
        print("Pooled-only mode: k=3, n_runs=100 (verify vs summary_OOD.png)")
    elif n_runs_override is not None:
        print(f"Using n_runs={n_runs_override} (override)")

    if args.test:
        run_single_test()
    else:
        k_use = [3] if args.pooled_only else [args.k]
        print(f"Running large simulation: k={k_use[0]}, {'pooled-only' if args.pooled_only else 'all methods'} (--test, --pooled-only, --n_runs N, --k K)")
        run_large_simulation(n_runs_override=n_runs_override, pooled_only=args.pooled_only, k_use=k_use)
