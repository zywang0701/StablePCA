# StablePCA (2026-Mar Release)

This folder contains the code to reproduce the results in the paper:

**StablePCA: Distributionally Robust Learning of Shared Representations from Multi-Source Data**  
[https://arxiv.org/pdf/2505.00940](https://arxiv.org/pdf/2505.00940)

## Folder Structure

```
StablePCA/
├── src/                    # Core algorithms
├── illustrations/          # Illustration notebooks
├── simulations/            # Simulation scripts
├── application/            # Real-data application (single-cell RNA)
├── README.md
├── requirements.txt
└── LICENSE
```

## Contents

### `src/` — Core Algorithms

- **PCAalg.py**: Main implementations
  - `PCA_MP`: Mirror-Prox algorithm for StablePCA, FairPCA, SquaredPCA
  - `PCA_Dual`: Dual formulation for multi-source PCA
- **data.py**: Data generation for simulations
- **utils.py**: Utilities
- **prev_methods.py**: Previous SDP-based FairPCA method

### `illustrations/` — Illustration Notebooks

- **illus-1.ipynb**: Illustration of StablePCA vs. PooledPCA
- **illus-2.ipynb**: Illustration of StablePCA vs. alternative PCAs

### `simulations/` — Simulation Scripts

- **simu-generalization.py**: Generalization experiment (train/test split across sources)
- **simu-finitesample.py**: Finite-sample convergence
- **simu_fairpca.py**: Fair PCA comparison (PCA_MP vs SDP-based `fair_pca_multisource`)

### `application/` — Real Single-Cell RNA Application

- **cluster-generalization.py**: Cluster-based generalization experiment on RNA data
- **RNA.pkl**: Preprocessed RNA dataset (12 batches) — *not included*; prepare from h5ad using the commented code at the top of `cluster-generalization.py`, then save to `application/RNA.pkl`
- **plot-singlecell.ipynb**: Visualization of results (boxplots, worst-case explained variance)
- **downstream.ipynb**: Downstream analysis and visualization

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run from the project root (e.g., `StablePCA/`):
   ```bash
   cd StablePCA
   python simulations/simu-generalization.py
   python simulations/simu_fairpca.py
   python application/cluster-generalization.py
   ```

3. For notebooks, ensure the working directory is the project root so `from src.PCAalg import ...` works.

## Methods

- **StablePCA**: Minimizes worst-case explained variance across sources
- **FairPCA**: Fair variant with regularized covariance
- **SquaredPCA**: Squared-loss variant
- **Pooled PCA**: Standard PCA on concatenated data (baseline)
