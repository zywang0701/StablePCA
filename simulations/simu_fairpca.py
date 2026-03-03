"""
Simulation script to compare PCA_MP (method='fair') and fair_pca_multisource.
Runs n_runs=100 per dimension d. Saves results to pkl files.
"""

import numpy as np
import time
import pickle
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import generate_stablepca_data_fix
from src.PCAalg import PCA_MP
from src.prev_methods import fair_pca_multisource

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
d_list = [30, 50, 70, 100, 200, 300]
n_runs = 100
n = 10000
k = 3
m = 5
L = 3
results_dir = "saved_results"
os.makedirs(results_dir, exist_ok=True)

# PCA_MP unified settings
PCA_MP_KWARGS = dict(
    eta_init=100,
    max_iter=500,
    tol=1e-6,
    check_dual=False,
    verbose=False,
    use_fast_logm=True,
    init_iter=100,
    check_freq=500,
    decay_every=200,
    eta_decay=0.2,
)


# ------------------------------------------------------------
# Helper function to compute regularized variance
# ------------------------------------------------------------
def compute_regularized_variance(X_list, projection_matrix, k, L_op=None):
    """Compute worst-case regularized variance for fair PCA."""
    Sigma_list_emp = [X.T @ X / X.shape[0] for X in X_list]
    if L_op is None:
        L_op = np.max([np.linalg.norm(Sigma, ord=2) for Sigma in Sigma_list_emp])
    Sigma_list_normalized = [Sigma / L_op for Sigma in Sigma_list_emp]

    Sigma_reg_list = []
    for l in range(len(X_list)):
        Sigma_l = Sigma_list_normalized[l]
        eval, _ = np.linalg.eigh(Sigma_l)
        idx = np.argsort(eval)[::-1]
        topk_eigvals = eval[idx][:k]
        topk_mean = np.mean(topk_eigvals)
        Sigma_l_reg = Sigma_l - topk_mean * np.eye(Sigma_l.shape[0])
        Sigma_reg_list.append(Sigma_l_reg)

    reg_var = min(np.trace(Sigma_reg @ projection_matrix) * L_op
                  for Sigma_reg in Sigma_reg_list)
    return reg_var


# ------------------------------------------------------------
# Single simulation run
# ------------------------------------------------------------
def run_one_simulation(d, run_id, random_state1=0, verbose=False):
    """Run a single simulation with given dimension d."""
    X_list, Sigma_list, _, _ = generate_stablepca_data_fix(
        n=n, p=d, k=k, m=m, L=L,
        random_state1=random_state1,
        random_state2=run_id
    )

    # PCA_MP (method='fair')
    if verbose:
        print("Running PCA_MP (method='fair')...")
    pca_mp = PCA_MP(n_components=k, method='fair')
    start_time = time.time()
    pca_mp.fit(X_list, **PCA_MP_KWARGS)
    time_mp = time.time() - start_time
    reg_var_mp = compute_regularized_variance(X_list, pca_mp.P, k)

    # fair_pca_multisource
    if verbose:
        print("Running fair_pca_multisource...")
    start_time = time.time()
    P_fair = fair_pca_multisource(X_list, d=k, sdp_solver="SCS", lp_solver="CLARABEL")
    time_fair = time.time() - start_time
    reg_var_fair = compute_regularized_variance(X_list, P_fair, k)

    reg_var_diff = np.abs(reg_var_mp - reg_var_fair)
    return {
        'reg_var_diff': reg_var_diff,
        'reg_var_mp': reg_var_mp,
        'reg_var_fair': reg_var_fair,
        'time_mp': time_mp,
        'time_fair': time_fair,
        'run_id': run_id,
        'd': d
    }


# ------------------------------------------------------------
# Large simulation
# ------------------------------------------------------------
def run_large_simulation(n_runs_override=None):
    """Run all simulations across all dimensions."""
    nr = n_runs_override if n_runs_override is not None else n_runs
    all_results = []

    for d in d_list:
        print(f"\n{'='*60}")
        print(f"Running simulations for d = {d} (n_runs = {nr})")
        print(f"{'='*60}")

        results_d = []
        for run_id in tqdm(range(nr), desc=f"d={d}", unit="run"):
            try:
                result = run_one_simulation(d, run_id=run_id, verbose=False)
                results_d.append(result)
            except Exception as e:
                print(f"\n  ERROR in run {run_id+1}: {e}")
                import traceback
                traceback.print_exc()

        fname = os.path.join(results_dir, f"results_fairpca_d{d}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(results_d, f)
        print(f"Saved {len(results_d)} results to {fname}")

        if results_d:
            reg_var_diffs = [r['reg_var_diff'] for r in results_d]
            times_mp = [r['time_mp'] for r in results_d]
            times_fair = [r['time_fair'] for r in results_d]
            print(f"Summary: |reg_var_diff| = {np.mean(reg_var_diffs):.6e} ± {np.std(reg_var_diffs):.6e}")
            print(f"  Time PCA_MP: {np.mean(times_mp):.2f}s, fair: {np.mean(times_fair):.2f}s")

        all_results.extend(results_d)

    fname_all = os.path.join(results_dir, "results_fairpca_all.pkl")
    with open(fname_all, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to {fname_all}")
    return all_results


# ------------------------------------------------------------
# Single test run
# ------------------------------------------------------------
def run_single_test():
    """Quick single-run test."""
    print("=" * 60)
    print("Running Single Simulation Test (d=50)")
    print("=" * 60)

    result = run_one_simulation(d=50, run_id=0, verbose=True)

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"  reg_var_diff: {result['reg_var_diff']:.6e}")
    print(f"  reg_var_mp:   {result['reg_var_mp']:.6e}")
    print(f"  reg_var_fair: {result['reg_var_fair']:.6e}")
    print(f"  time_mp:     {result['time_mp']:.2f}s")
    print(f"  time_fair:   {result['time_fair']:.2f}s")
    print("=" * 60)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fair PCA comparison simulation")
    parser.add_argument("--test", action="store_true", help="Run single test only")
    parser.add_argument("--n_runs", type=int, default=None, help="Override n_runs")
    args = parser.parse_args()

    n_runs_override = args.n_runs
    if n_runs_override is not None:
        print(f"Using n_runs={n_runs_override} (override)")

    if args.test:
        run_single_test()
    else:
        print("Running large simulation (use --test for single run)")
        run_large_simulation(n_runs_override=n_runs_override)
