"""
Preprocess full12batch.h5ad and save RNA.pkl for cluster-generalization experiments.

The data is publicly available from:
https://openproblems.bio/events/2021-09_neurips
(NeurIPS 2021: Multimodal Single-Cell Data Integration)

We select the 12 batches with RNA+GEX+ADT multimodalities.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler


def preprocess_h5ad(input_path: str, output_path: str, hvg_num: int = 1000) -> None:
    """
    Load h5ad, extract GEX (RNA) per batch, select HVGs, standardize, and save as pickle.

    Args:
        input_path: Path to full12batch.h5ad
        output_path: Path to save RNA.pkl
        hvg_num: Number of highly variable genes to select
    """
    adata = sc.read_h5ad(input_path)
    print(f"Loaded {adata.shape[0]} cells, {adata.shape[1]} features")

    metadata = adata.obs
    genes = adata.var

    # Split depending on omics
    omics = genes["feature_types"].unique()
    adata_omics = {
        view: adata[:, adata.var["feature_types"] == view].copy() for view in omics
    }

    # Split batches for each omics (RNA/GEX only here)
    batches = metadata["batch"].unique()
    adata_RNA = adata_omics["GEX"]
    RNA_batch = {
        batch: adata_RNA[adata_RNA.obs["batch"] == batch].copy() for batch in batches
    }

    # HVGs selection on RNA data
    sc.pp.highly_variable_genes(adata_RNA, n_top_genes=hvg_num, flavor="seurat_v3")
    hvg = adata_RNA.var[adata_RNA.var["highly_variable"]].index
    for batch, adata_batch in RNA_batch.items():
        adata_batch = adata_batch[:, hvg].copy()
        RNA_batch[batch] = adata_batch
        print(
            f"Batch {batch} has {adata_batch.shape[0]} cells and "
            f"{adata_batch.shape[1]} genes after HVG selection"
        )

    batch_cell_num = [adata_batch.shape[0] for adata_batch in RNA_batch.values()]
    RNA_batch_np = [adata_batch.X.toarray() for adata_batch in RNA_batch.values()]
    RNA = np.vstack(RNA_batch_np)

    # Standardize data
    scaler = StandardScaler()
    features = scaler.fit_transform(RNA)

    # Split back into batches
    X_list = []
    start = 0
    for num in batch_cell_num:
        X_list.append(features[start : start + num, :])
        start += num

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(X_list, f)
    print(f"Saved RNA.pkl to {output_path} ({len(X_list)} batches)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess full12batch.h5ad and save RNA.pkl"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="application/full12batch.h5ad",
        help="Path to full12batch.h5ad (default: application/full12batch.h5ad)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="application/RNA.pkl",
        help="Path to save RNA.pkl (default: application/RNA.pkl)",
    )
    parser.add_argument(
        "--hvg",
        type=int,
        default=1000,
        help="Number of highly variable genes (default: 1000)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root if needed
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not output_path.is_absolute():
        output_path = project_root / output_path

    preprocess_h5ad(str(input_path), str(output_path), hvg_num=args.hvg)
