import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.PCAalg import PCA_MP
from src.prev_methods import fair_pca_multisource
from src.utils import proj_error, explained_variance
from src.data import generate_stablepca_data

# ----------------------------
# Simulation parameters
# ----------------------------
n = 2000
k = 3
m = 5
p = 40
L_list = [2, 4, 6, 8, 10]
n_runs = 100

# ----------------------------
# Helper: one run for given L and seed
# ----------------------------
def run_one_simulation(L, seed):
    np.random.seed(seed)

    # Generate data
    X_list, X_ood_list, A, _, _ = generate_stablepca_data(
        n=n, p=p, k=k, m=m, L=L, random_state=seed
    )

    # True subspace
    eigvals, eigvecs = np.linalg.eigh(A @ A.T / A.shape[0])
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    V_true = eigvecs[:, :k]
    P_true = V_true @ V_true.T

    # ------------------------------------------------------------------
    # 1) Pooled PCA
    # ------------------------------------------------------------------
    X_pooled = np.vstack(X_list)
    pca_pooled = PCA(n_components=k).fit(X_pooled)
    V_pooled = pca_pooled.components_.T
    P_pooled = V_pooled @ V_pooled.T

    # ------------------------------------------------------------------
    # 2) Stable PCA
    # ------------------------------------------------------------------
    print("Running Stable PCA...")
    pca_stable = PCA_MP(n_components=k, method='stable')
    pca_stable.fit(X_list, max_iter=2000, eta_init=10, verbose=500, decay_every=500, check_dual=True)

    P_stable = pca_stable.M_proj

    # ------------------------------------------------------------------
    # 3) Squared PCA
    # ------------------------------------------------------------------
    print("Running Squared PCA...")
    pca_squared = PCA_MP(n_components=k, method='squared')
    pca_squared.fit(X_list, max_iter=2000, eta_init=10, verbose=500, decay_every=500, check_dual=True)
    P_squared = pca_squared.M_proj

    # ------------------------------------------------------------------
    # 4) Fair PCA
    # ------------------------------------------------------------------
    print("Running Fair PCA...")
    # Note: we use the fair PCA implementation from the previous methods for comparison.
    # You can also use the fair PCA implementation from by PCA_MP(n_components=k, method='fair').
    P_fair = fair_pca_multisource(X_list, d=k)

    # ------------------------------------------------------------------
    # Projection errors
    # ------------------------------------------------------------------
    err_pooled = proj_error(P_pooled, P_true)
    err_stable = proj_error(P_stable, P_true)
    err_squared = proj_error(P_squared, P_true)
    err_fair   = proj_error(P_fair,   P_true)

    # ------------------------------------------------------------------
    # Worst-case explained variance (source)
    # ------------------------------------------------------------------
    expl_var_pooled_src = min(explained_variance(P_pooled, X) for X in X_list)
    expl_var_stable_src = min(explained_variance(P_stable, X) for X in X_list)
    expl_var_squared_src = min(explained_variance(P_squared, X) for X in X_list)
    expl_var_fair_src   = min(explained_variance(P_fair, X)   for X in X_list)

    # ------------------------------------------------------------------
    # Worst-case explained variance (OOD)
    # ------------------------------------------------------------------
    expl_var_pooled_ood = min(explained_variance(P_pooled, X) for X in X_ood_list)
    expl_var_stable_ood = min(explained_variance(P_stable, X) for X in X_ood_list)
    expl_var_squared_ood = min(explained_variance(P_squared, X) for X in X_ood_list)
    expl_var_fair_ood   = min(explained_variance(P_fair, X)   for X in X_ood_list)

    return {
        "L": L,
        "seed": seed,
        "err": {
            "pooled": err_pooled,
            "stable": err_stable,
            "squared": err_squared,
            "fair": err_fair,
        },
        "expl_var_src": {
            "pooled": expl_var_pooled_src,
            "stable": expl_var_stable_src,
            "squared": expl_var_squared_src,
            "fair": expl_var_fair_src,
        },
        "expl_var_ood": {
            "pooled": expl_var_pooled_ood,
            "stable": expl_var_stable_ood,
            "squared": expl_var_squared_ood,
            "fair": expl_var_fair_ood,
        },
    }

# ----------------------------
# Main simulation loop. Please uncomment this section to run the simulation.
# Note: please set the check_dual=False and verbose=False in the above codes to save time.
# ----------------------------
# for L in L_list:
#     results = []
#     print(f"\n========== Running simulations for L = {L} ==========")
#     for seed in range(n_runs):
#         result = run_one_simulation(L, seed)
#         results.append(result)
#         print(f"Run {seed+1}/{n_runs} completed. ")

#     # Save results to pickle
#     filename = f"saved_results/results-OOD_L{L}.pkl"
#     os.makedirs("saved_results", exist_ok=True)
#     with open(filename, "wb") as f:
#         pickle.dump(results, f)
#     print(f"Saved results for L={L} to {filename}")


# ----------------------------
# Single test run
# ----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running Single Simulation Test (L = 6)")
    print("=" * 60)
    
    result = run_one_simulation(L=6, seed=21)
    
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    
    # Projection Errors
    print("\n Projection Errors:")
    print(f"  {'Method':<12} {'Error':<12}")
    print(f"  {'-'*11} {'-'*11}")
    for method, error in result["err"].items():
        print(f"  {method.capitalize():<12} {error:.4f}")
    
    # Explained Variance (Source)
    print("\n Worst-case Explained Variance (Source Data):")
    print(f"  {'Method':<12} {'Explained Variance':<20}")
    print(f"  {'-'*11} {'-'*19}")
    for method, var in result["expl_var_src"].items():
        print(f"  {method.capitalize():<12} {var:.4f}")
    
    # Explained Variance (OOD)
    print("\n Worst-case Explained Variance (OOD Data):")
    print(f"  {'Method':<12} {'Explained Variance':<20}")
    print(f"  {'-'*11} {'-'*19}")
    for method, var in result["expl_var_ood"].items():
        print(f"  {method.capitalize():<12} {var:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Simulation completed (L={result['L']}, seed={result['seed']})")
    print("=" * 60)