# StablePCA (2026-Mar Release)

This repository contains the code to reproduce the results in the paper:

**StablePCA: Distributionally Robust Learning of Shared Representations from Multi-Source Data**  
[https://arxiv.org/pdf/2505.00940](https://arxiv.org/pdf/2505.00940)

## Quick Start

1. **Install dependencies:**
  ```bash
   pip install -r requirements.txt
  ```
2. **RNA data** (for application experiments): Download `full12batch.h5ad` and run `data-preprocess.py` to generate `RNA.pkl`. See [Data](#data).
3. **Run from project root** so `from src.PCAalg import ...` works in notebooks.

---

## Workflow Overview


| Section           | Step 1: Run script          | Step 2: Plot notebook       | Figures / Tables                   |
| ----------------- | --------------------------- | --------------------------- | ---------------------------------- |
| **Illustrations** | —                           | `illus-1.ipynb`             | Figure 1 (main)                    |
|                   | —                           | `illus-2.ipynb`             | Figure 6 (supplement)              |
| **Simulations**   | `simu-generalization.py`    | `plot-generalization.ipynb` | Fig. 2 (main), Figs. 8–9 (supp.)   |
|                   | `simu-finitesample.py`      | `plot-finitesample.ipynb`   | Fig. 7, Table 2 (supp.)            |
|                   | `simu-fairpca.py`           | `plot-fairpca.ipynb`        | Table 1 (main)                     |
| **Application**   | `cluster-generalization.py` | `plot-singlecell.ipynb`     | Fig. 3 (main), Figs. 10–11 (supp.) |
|                   | —                           | `downstream.ipynb`          | Fig. 4 (main), Figs. 12–13 (supp.) |


**Note:** For simulations, run the `simu_*.py` scripts first to save results; readers may use a single run for quick testing. Then open the corresponding `plot-*.ipynb` to reproduce figures and tables.

---

## Folder Structure

```
StablePCA/
├── src/                    # Core algorithms
├── illustrations/          # Illustration notebooks (Figures 1, 6)
├── simulations/            # Simulation scripts + plot notebooks
├── application/            # Single-cell RNA application
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Detailed Workflow

### `illustrations/` — Reproduce illustration figures


| Notebook          | Description                                 | Output                |
| ----------------- | ------------------------------------------- | --------------------- |
| **illus-1.ipynb** | StablePCA vs. PooledPCA                     | Figure 1 (main text)  |
| **illus-2.ipynb** | StablePCA vs. alternative multi-source PCAs | Figure 6 (supplement) |


Run the notebooks directly; no prior script execution needed.

---

### `simulations/` — Reproduce simulated experiments

**Workflow:** Run `simu_*.py` → save results → run `plot-*.ipynb` to generate figures/tables.


| Script                     | Plot notebook                 | Figures / Tables                          |
| -------------------------- | ----------------------------- | ----------------------------------------- |
| **simu-generalization.py** | **plot-generalization.ipynb** | Figure 2 (main), Figures 8–9 (supplement) |
| **simu-finitesample.py**   | **plot-finitesample.ipynb**   | Figure 7, Table 2 (supplement)            |
| **simu-fairpca.py**        | **plot-fairpca.ipynb**        | Table 1 (main text)                       |


**Example:**

```bash
# Run simulation (or a single run for quick test)
python simulations/simu-generalization.py
python simulations/simu-finitesample.py
python simulations/simu-fairpca.py

# Then open the corresponding plot notebook
jupyter notebook simulations/plot-generalization.ipynb
```

---

### `application/` — Reproduce single-cell RNA experiments

**Workflow:** Download data → run `data-preprocess.py` → run `cluster-generalization.py` → run plot notebooks.


| Script / Notebook             | Description                                   | Figures / Tables                            |
| ----------------------------- | --------------------------------------------- | ------------------------------------------- |
| **data-preprocess.py**        | Preprocess h5ad → RNA.pkl                     | —                                           |
| **cluster-generalization.py** | Generalization experiment on single-cell data | —                                           |
| **plot-singlecell.ipynb**     | Visualize generalization results              | Figure 3 (main), Figures 10–11 (supplement) |
| **downstream.ipynb**          | Downstream visualization                      | Figure 4 (main), Figures 12–13 (supplement) |


**Example:**

```bash
# 1. Download full12batch.h5ad (see Data section) and place in application/

# 2. Preprocess data to generate RNA.pkl
python application/data-preprocess.py

# 3. Run cluster-generalization experiment
python application/cluster-generalization.py

# 4. Open plot notebooks
jupyter notebook application/plot-singlecell.ipynb
jupyter notebook application/downstream.ipynb
```

---

## Data

### RNA.pkl (single-cell application)

The single-cell data is **publicly available** from:

**[NeurIPS 2021: Multimodal Single-Cell Data Integration](https://openproblems.bio/events/2021-09_neurips)**

We select the **12 batches** with **RNA+GEX+ADT** multimodalities. Download the source data, place `full12batch.h5ad` in `application/`, then run:

```bash
python application/data-preprocess.py
```

This generates `application/RNA.pkl` (12 batches of preprocessed GEX/RNA data with 1000 HVGs, standardized).

---

## Contents

### `src/` — Core algorithms

- **PCAalg.py**: Main implementations
  - `PCA_MP`: Mirror-Prox algorithm for StablePCA, FairPCA, SquaredPCA
  - `PCA_Dual`: Dual formulation for multi-source PCA
- **data.py**: Data generation for simulations
- **utils.py**: Utilities
- **prev_methods.py**: Previous SDP-based FairPCA method

### Methods

- **StablePCA**: Minimizes worst-case explained variance across sources
- **FairPCA**: Fair variant with regularized covariance
- **SquaredPCA**: Squared-loss variant
- **Pooled PCA**: Standard PCA on concatenated data (baseline)

