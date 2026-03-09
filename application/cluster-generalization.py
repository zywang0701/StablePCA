import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import scanpy as sc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.PCAalg import PCA_Dual
from tqdm import tqdm

def explained_variance_samples(X, M):
    """
    X: (n, d) data matrix for one batch
    M: (d, d) projection matrix (e.g. U U^T)

    Returns: scalar
        (1/n) sum_i (||x_i||^2 - ||x_i - x_i M||^2)
    """
    # Project
    X_proj = X @ M          # (n, d)
    resid = X - X_proj      # (n, d)

    sq_norm = np.sum(X**2, axis=1)        # (n,)
    sq_resid = np.sum(resid**2, axis=1)   # (n,)
    return np.mean(sq_norm - sq_resid)


def batch_scale(X):
    """
    Scale used in the normalized metric denominator:
        (1/n) * sum_i ||x_i||^2
    """
    sq_norm = np.sum(X**2, axis=1)  # (n,)
    return float(np.mean(sq_norm))


def rotation_test(
    X_list,
    n_components=50,
    train_size=8,
    max_groups=6,
    job_id=0,
    num_jobs=20,
    shuffle_seed=0,
    max_iter=1000,
    tol=1e-6,
    verbose=False,
):
    """
    Rotation test for PCA_Dual (stable and pooled PCA), sharded across jobs.

    X_list: list of 12 (n_l, d) arrays, one per batch.
    n_components: rank k.
    train_size: number of batches used for training (8 here).
    max_groups: max # of rotations *per job* (can be None to use all assigned combos).
    job_id: integer in [0, num_jobs-1] identifying this job.
    num_jobs: total number of parallel jobs.
    shuffle_seed: seed used to shuffle the global list of combos.
    max_iter, tol, verbose: passed to PCA_Dual.fit.
    """
    L = len(X_list)
    all_combos = list(combinations(range(L), train_size))  # global list of train combos
    n_total = len(all_combos)

    # 1. Shuffle globally in a deterministic way
    rng = np.random.default_rng(shuffle_seed)
    idx_perm = rng.permutation(n_total)
    all_combos = [all_combos[i] for i in idx_perm]

    # 2. Partition the shuffled list into num_jobs shards
    #    We'll give each job a contiguous block.
    if num_jobs < 1:
        raise ValueError("num_jobs must be >= 1")
    if not (0 <= job_id < num_jobs):
        raise ValueError(f"job_id must be in [0, {num_jobs-1}], got {job_id}")

    # Compute equal-ish shard boundaries
    # per_job ≈ ceil(n_total / num_jobs)
    per_job = int(np.ceil(n_total / num_jobs))
    start = job_id * per_job
    end = min(n_total, start + per_job)

    # This job sees only this slice of combos
    shard_combos = all_combos[start:end]

    # 3. Optionally limit to max_groups within this shard
    if max_groups is not None and max_groups < len(shard_combos):
        shard_combos = shard_combos[:max_groups]

    print(f"[Job {job_id}/{num_jobs}] Total combos: {n_total}, "
          f"assigned: {len(shard_combos)} (from indices {start} to {end-1})")

    stable_in_worst, stable_out_worst = [], []
    fair_in_worst, fair_out_worst = [], []
    pooled_in_worst, pooled_out_worst = [], []
    squared_in_worst, squared_out_worst = [], []

    # Track the largest scale we've ever seen across all batches
    # scale(X) = (1/n) sum_i ||x_i||^2 for a given batch matrix X
    max_scale_global = -np.inf

    train_idx_list, test_idx_list = [], []

    print(f"[Job {job_id}] Starting rotation test over {len(shard_combos)} groups...")
    for combo in tqdm(shard_combos, desc=f"Rotations (job {job_id})", unit="group"):
        train_idx = list(combo)
        test_idx = [i for i in range(L) if i not in train_idx]
        train_idx_list.append(train_idx)
        test_idx_list.append(test_idx)

        # 1. prepare training data
        X_train_list = [X_list[i] for i in train_idx]

        # --- Stable PCA ---
        print(" ------> Training StablePCA...")
        pca_stable = PCA_Dual(n_components=n_components, method='stable')
        pca_stable.fit(
            X_list=X_train_list,
            lr=1e-1,
            max_iter=max_iter,
            tol=tol,
            check_dual=False,
            verbose=verbose,
        )
        M_stable = pca_stable.P
        
        # --- Squared PCA ---
        print(" ------> Training SquaredPCA...")
        pca_squared = PCA_Dual(n_components=n_components, method='squared')
        pca_squared.fit(
            X_list=X_train_list,
            lr=0.02,
            max_iter=max_iter,
            tol=tol,
            check_dual=False,
            verbose=verbose,
        )
        M_squared = pca_squared.P

        # --- Fair PCA ---
        print(" ------> Training FairPCA...")
        pca_fair = PCA_Dual(n_components=n_components, method='fair')
        pca_fair.fit(
            X_list=X_train_list,
            lr=0.01,
            max_iter=max_iter,
            tol=tol,
            check_dual=False,
            verbose=verbose,
        )
        M_fair = pca_fair.P

        # --- Pooled PCA ---
        X_pooled = np.vstack([X_list[i] for i in train_idx])
        pca_pooled = PCA(n_components=n_components)
        pca_pooled.fit(X_pooled)
        M_pooled = pca_pooled.components_.T @ pca_pooled.components_

        # --------- Evaluate in-dist (train batches) ----------
        for i in train_idx:
            max_scale_global = max(max_scale_global, batch_scale(X_list[i]))
        stable_train_scores = [explained_variance_samples(X_list[i], M_stable) for i in train_idx]
        fair_train_scores = [explained_variance_samples(X_list[i], M_fair) for i in train_idx]
        pooled_train_scores = [explained_variance_samples(X_list[i], M_pooled) for i in train_idx]
        squared_train_scores = [explained_variance_samples(X_list[i], M_squared) for i in train_idx]

        
        stable_in_worst.append(min(stable_train_scores))
        fair_in_worst.append(min(fair_train_scores))
        pooled_in_worst.append(min(pooled_train_scores))
        squared_in_worst.append(min(squared_train_scores))

        # --------- Evaluate out-of-dist (test batches) ----------
        for i in test_idx:
            max_scale_global = max(max_scale_global, batch_scale(X_list[i]))
        stable_test_scores = [explained_variance_samples(X_list[i], M_stable) for i in test_idx]
        fair_test_scores = [explained_variance_samples(X_list[i], M_fair) for i in test_idx]
        pooled_test_scores = [explained_variance_samples(X_list[i], M_pooled) for i in test_idx]
        squared_test_scores = [explained_variance_samples(X_list[i], M_squared) for i in test_idx]


        stable_out_worst.append(min(stable_test_scores))
        fair_out_worst.append(min(fair_test_scores))
        pooled_out_worst.append(min(pooled_test_scores))
        squared_out_worst.append(min(squared_test_scores))

    print(f"[Job {job_id}] Rotation test finished.")
    results = {
        "train_idx_list": train_idx_list,
        "test_idx_list": test_idx_list,
        "stable_in": stable_in_worst,
        "stable_out": stable_out_worst,
        "fair_in": fair_in_worst,
        "fair_out": fair_out_worst,
        "pooled_in": pooled_in_worst,
        "pooled_out": pooled_out_worst,
        "squared_in": squared_in_worst,
        "squared_out": squared_out_worst,
        "max_scale_global": float(max_scale_global),
    }
    return results
 

if __name__ == "__main__":
    import os
    import argparse
    import pickle
    
    # Try application/RNA.pkl first (run from project root), then Real-SingleCell/RNA.pkl
    for p in ["application/RNA.pkl", "Real-SingleCell/RNA.pkl"]:
        if Path(p).exists():
            with open(p, "rb") as f:
                X_list = pickle.load(f)
            break
    else:
        raise FileNotFoundError(
            "RNA.pkl not found. Run: python application/data-preprocess.py "
            "(requires full12batch.h5ad in application/). See README for data source."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, default=None,
                        help="Job index (0-based). If None, use SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--num-jobs", type=int, default=20,
                        help="Total number of parallel jobs.")
    parser.add_argument("--max-groups", type=int, default=10,
                        help="Max # of rotations per job.")
    parser.add_argument("--n-components", type=int, nargs='+', default=[50, 100, 150],
                        help="List of n_components values to test (default: [50, 100, 150]).")
    args = parser.parse_args()

    # Resolve job_id
    if args.job_id is not None:
        job_id = args.job_id
    else:
        job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    num_jobs = args.num_jobs
    n_components_list = args.n_components

    print(f"Running rotation test for n_components: {n_components_list}")
    print(f"Job ID: {job_id}, Total jobs: {num_jobs}")
    
    # Run rotation test for each n_components value
    all_results = {}
    
    for n_comp in n_components_list:
        print(f"\n{'='*60}")
        print(f"Running rotation test with n_components={n_comp}")
        print(f"{'='*60}")
        
        results = rotation_test(
            X_list,
            n_components=n_comp,
            train_size=8,
            max_groups=args.max_groups,
            job_id=job_id,
            num_jobs=num_jobs,
            shuffle_seed=0,
            max_iter=1000,
            tol=1e-6,
            verbose=100,
        )
        
        all_results[f"n_components_{n_comp}"] = results
        print(f"Completed n_components={n_comp}")

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/rotation_test_results_job{job_id}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n{'='*60}")
    print(f"Saved results to {out_path}")
    print(f"Results contain {len(all_results)} different n_components values: {list(all_results.keys())}")
    print(f"{'='*60}")
