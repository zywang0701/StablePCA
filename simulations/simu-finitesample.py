import numpy as np
import pickle
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import generate_stablepca_data_fix
from src.PCAalg import PCA_MP

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
n_list = [400, 800, 1200, 1600, 2000]
p_list = [15, 30]
num_runs = 100
k, m, L = 3, 5, 4
results_dir = "saved_results"
os.makedirs(results_dir, exist_ok=True)

# ------------------------------------------------------------
# Helper to compute metrics
# ------------------------------------------------------------
def compute_metrics(X_list, Sigma_list, pca_stable, pca_stable_pop):
    """Compute projection loss, certificate gap, and optimality gap."""
    # projection difference between relaxed and projected solution
    proj_loss = np.linalg.norm(pca_stable.M - pca_stable.M_proj, ord="fro")

    # certificate gap on empirical covariance
    # empirical covariance matrices
    Sigma_hat_list = [X_list[i].T @ X_list[i] / X_list[i].shape[0] for i in range(L)]
    certificate = (
        np.min([np.trace(Sigma_hat_list[i] @ pca_stable.M) for i in range(L)])
        - np.min([np.trace(Sigma_hat_list[i] @ pca_stable.M_proj) for i in range(L)])
    )

    # population-level global optimality gap
    val_gap = (
        np.min([np.trace(Sigma_list[i] @ pca_stable_pop.M) for i in range(L)])
        - np.min([np.trace(Sigma_list[i] @ pca_stable.M) for i in range(L)])
    )

    # estimation error between empirical and population solution
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
def run_one_simulation(n, p, seed=0):
    """Run a single simulation with given parameters."""
    X_list, Sigma_list, _, _ = generate_stablepca_data_fix(
        n=n, p=p, k=k, m=m, L=L,
        random_state1=0,
        random_state2=seed
    )
   
    # --- StablePCA (empirical) ---
    print("Running Stable PCA (empirical)...")
    pca_stable = PCA_MP(n_components=k)
    pca_stable.fit(
        X_list, eta_init=10, max_iter=4000,
        check_dual=True, decay_every=500,
        verbose=500
    )
    
    # --- StablePCA (population covariance) ---
    print("Running Stable PCA (population covariance)...")
    pca_stable_pop = PCA_MP(n_components=k)
    pca_stable_pop.fit(
        X_list, Sigma_list,
        eta_init=10, max_iter=4000,
        check_dual=True, decay_every=500,
        verbose=500
    )

    # compute metrics
    metrics = compute_metrics(X_list, Sigma_list, pca_stable, pca_stable_pop)
    metrics.update({"n": n, "p": p, "seed": seed})
    
    return metrics

# ------------------------------------------------------------
# Simulation Loop
# Note: Please uncomment this section when running large-scale simulations
# ------------------------------------------------------------
# for p in p_list:
#     for n in n_list:
#         print(f"\n=== Running n = {n}, p = {p} ===")
#         all_results = []
# 
#         for run in range(1, num_runs + 1):
#             X_list, Sigma_list, _, _ = generate_stablepca_data_fix(
#                 n=n, p=p, k=k, m=m, L=L,
#                 random_state1=0,
#                 random_state2=run
#             )
# 
#             # --- StablePCA (empirical) ---
#             pca_stable = PCA_MP(n_components=k)
#             pca_stable.fit(
#                 X_list, eta_init=10, max_iter=2000,
#                 check_dual=False, decay_every=500,
#                 eta_decay=0.5, verbose=False
#             )
# 
#             # --- StablePCA (population covariance) ---
#             pca_stable_pop = PCA_MP(n_components=k)
#             pca_stable_pop.fit(
#                 X_list, Sigma_list,
#                 eta_init=10, max_iter=2000,
#                 check_dual=False, decay_every=500,
#                 eta_decay=0.5, verbose=False
#             )
# 
#             # compute metrics
#             metrics = compute_metrics(X_list, Sigma_list, pca_stable, pca_stable_pop)
#             metrics.update({"run": run, "n": n, "p": p})
#             all_results.append(metrics)
# 
#             if run % 10 == 0:
#                 print(f"  Completed {run}/{num_runs} runs")
# 
#         # save results for this n
#         fname = os.path.join(results_dir, f"results-Finite_n{n}_p{p}.pkl")
#         with open(fname, "wb") as f:
#             pickle.dump(all_results, f)
# 
#         print(f"Saved {len(all_results)} results to {fname}")


# ------------------------------------------------------------
# Single test run
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running Single Simulation Test (n = 800, p = 15)")
    print("=" * 60)
    
    result = run_one_simulation(n=800, p=15, seed=0)
    
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    
    # Display metrics
    print(f"  {'Metric':<30} {'Value':<15}")
    print(f"  {'-'*29} {'-'*14}")
    print(f"  {'Projection Loss':<30} {result['proj_loss']:.4f}")
    print(f"  {'Certificate Gap':<30} {result['certificate']:.4f}")
    print(f"  {'Optimality Gap (val_gap)':<30} {result['val_gap']:.4f}")
    print(f"  {'Estimation Gap (est_gap)':<30} {result['est_gap']:.4f}")

