"""
Finite-sample performance simulation.
Runs n_runs=100 per (n, p) combination. Saves results to pkl files.
"""

import numpy as np
import pickle
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import generate_stablepca_data_fix
from src.PCAalg import PCA_MP

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
n_list = [100, 300, 600, 1200, 2500, 5000, 10000, 20000, 40000]
p_list = [10, 20, 30]
n_runs = 100
k, m, L = 3, 5, 4
results_dir = "saved_results"
os.makedirs(results_dir, exist_ok=True)

# PCA_MP unified settings
PCA_MP_KWARGS = dict(
    eta_init=100,
    max_iter=500,
    check_dual=False,
    decay_every=200,
    eta_decay=0.2,
    verbose=False,
    use_fast_logm=True,
    check_freq=500,
    init_iter=100,
)

# ------------------------------------------------------------
# Helper to compute metrics
# ------------------------------------------------------------
def compute_metrics(X_list, Sigma_list, pca_stable, pca_stable_pop):
    """Compute projection loss, certificate gap, and optimality gap."""
    proj_loss = np.linalg.norm(pca_stable.M - pca_stable.P, ord="fro")
    Sigma_hat_list = [X_list[i].T @ X_list[i] / X_list[i].shape[0] for i in range(L)]
    certificate = (
        np.min([np.trace(Sigma_hat_list[i] @ pca_stable.M) for i in range(L)])
        - np.min([np.trace(Sigma_hat_list[i] @ pca_stable.P) for i in range(L)])
    )
    val_gap = (
        np.min([np.trace(Sigma_list[i] @ pca_stable_pop.M) for i in range(L)])
        - np.min([np.trace(Sigma_list[i] @ pca_stable.M) for i in range(L)])
    )
    est_gap = np.linalg.norm(pca_stable.M - pca_stable_pop.M, ord="fro")
    return {
        "proj_loss": proj_loss,
        "certificate": certificate,
        "val_gap": val_gap,
        "est_gap": est_gap,
    }


# ------------------------------------------------------------
# Single simulation run
# ------------------------------------------------------------
def run_one_simulation(n, p, seed=0, verbose=False):
    """Run a single simulation with given parameters."""
    X_list, Sigma_list, _, _ = generate_stablepca_data_fix(
        n=n, p=p, k=k, m=m, L=L,
        random_state1=0,
        random_state2=seed
    )

    kwargs = {**PCA_MP_KWARGS, "verbose": 100 if verbose else False, "check_dual": verbose}

    # --- StablePCA (empirical) ---
    if verbose:
        print("Running Stable PCA (empirical)...")
    pca_stable = PCA_MP(n_components=k)
    pca_stable.fit(X_list, **kwargs)

    # --- StablePCA (population covariance) ---
    if verbose:
        print("Running Stable PCA (population covariance)...")
    pca_stable_pop = PCA_MP(n_components=k)
    pca_stable_pop.fit(X_list, Sigma_list, **kwargs)

    metrics = compute_metrics(X_list, Sigma_list, pca_stable, pca_stable_pop)
    metrics.update({"n": n, "p": p, "seed": seed})
    return metrics


# ------------------------------------------------------------
# Large simulation
# ------------------------------------------------------------
def run_large_simulation(n_runs_override=None):
    """Run n_runs simulations per (n, p). Save results to pkl for each (n, p)."""
    nr = n_runs_override if n_runs_override is not None else n_runs

    for p in p_list:
        for n in n_list:
            print(f"\n=== Running n = {n}, p = {p} (n_runs = {nr}) ===")
            all_results = []

            for run in tqdm(range(1, nr + 1), desc=f"n={n},p={p}", unit="run"):
                try:
                    metrics = run_one_simulation(n=n, p=p, seed=run, verbose=False)
                    metrics["run"] = run
                    all_results.append(metrics)
                except Exception as e:
                    print(f"\n  ERROR at run {run}: {e}")
                    import traceback
                    traceback.print_exc()

            fname = os.path.join(results_dir, f"results-Finite_n{n}_p{p}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(all_results, f)
            print(f"Saved {len(all_results)} results to {fname}")


# ------------------------------------------------------------
# Single test run
# ------------------------------------------------------------
def run_single_test():
    """Quick single-run test."""
    print("=" * 60)
    print("Running Single Simulation Test (n=1000, p=10)")
    print("=" * 60)

    result = run_one_simulation(n=1000, p=10, seed=0, verbose=True)

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print(f"  {'Metric':<30} {'Value':<15}")
    print(f"  {'-'*29} {'-'*14}")
    print(f"  {'Projection Loss':<30} {result['proj_loss']:.4f}")
    print(f"  {'Certificate Gap':<30} {result['certificate']:.4f}")
    print(f"  {'Optimality Gap (val_gap)':<30} {result['val_gap']:.4f}")
    print(f"  {'Estimation Gap (est_gap)':<30} {result['est_gap']:.4f}")
    print("=" * 60)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finite-sample simulation")
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
