# StablePCA

Stable Principal Component Analysis (StablePCA) for Multi-Source Data

## Overview

StablePCA is a principal component analysis method for multi-source data that obtains stable and fair dimensionality reduction representations across different data sources by optimizing a min-max objective function. This project implements two main algorithms:
- **PCA_MP**: Implementation based on the Mirror-Prox algorithm
- **PCA_Dual**: Implementation based on the dual formulation

## Project Structure

```
StablePCA/
├── src/                    # Core source code
├── illustrations/          # Paper illustration examples
├── simulations/            # Simulation experiments and visualization
├── application/            # Real-world application examples
├── saved_results/          # Saved experimental results
└── Figures/                # Generated figures
```

For detailed project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Installation

### Install Dependencies with pip

```bash
pip install -r requirements.txt
```

### Main Dependencies

- **Core Computing**: numpy, scipy, scikit-learn
- **Optimization**: cvxpy (for baseline methods)
- **Visualization**: matplotlib, seaborn, umap-learn
- **Data Handling**: pandas, h5py, anndata, scanpy (for single-cell data analysis)
- **Jupyter**: jupyter, ipykernel, notebook
- **Utilities**: tqdm (progress bars)

See `requirements.txt` for the complete dependency list.

## Quick Start

### 1. Basic Usage

```python
from src.PCAalg import PCA_MP
import numpy as np

# Prepare multi-source data
X_list = [X1, X2, X3, X4]  # Each X_i is an (n_i, p) data matrix

# Create model and fit
pca = PCA_MP(n_components=10, method='stable')
pca.fit(X_list)

# Get projection matrix
pca.M_proj  # Project using components_
```

### 2. Run Simulation Experiments

#### Finite-Sample Performance Experiments

```bash
python simulations/simu-finitesample.py
```

Results will be saved to the `saved_results/` directory. Visualize results using `simulations/plot-finitesample.ipynb`.

#### Out-of-Distribution Generalization Experiments

```bash
python simulations/simu-generalization.py
```

Visualize results using `simulations/plot-generalization.ipynb`.

### 3. Visualize Experimental Results

- **Finite-sample results**: Run `simulations/plot-finitesample.ipynb`
- **Generalization results**: Run `simulations/plot-generalization.ipynb`
- **Paper illustrations**: Run `illustrations/illus-1.ipynb` and `illustrations/illus-2.ipynb`

### 4. Real-World Application Examples

#### Single-Cell Data Analysis

The project includes a complete analysis pipeline for single-cell data:

1. **Clustering Generalization Experiment**:
   ```bash
   python application/cluster-generalization.py
   ```

2. **Downstream Analysis**:
   - Open `application/downstream.ipynb`
   - Follow the instructions in the notebook to run the complete analysis pipeline

3. **Visualization**:
   - Run `application/plot-singlecell.ipynb` to view visualization results

## Algorithm Description

### PCA_MP (Mirror-Prox Algorithm)

`PCA_MP` solves a min-max optimization problem via the Mirror-Prox algorithm:
- Primal problem: Find the optimal projection matrix that minimizes the worst-case loss across all data sources
- Supports three variants: `'stable'` (default), `'fair'`, and `'squared'`

### PCA_Dual (Dual Algorithm)

`PCA_Dual` first optimizes weights on the simplex, then computes the top-k eigenvectors of the weighted covariance matrix:
- More computationally efficient, suitable for large-scale data
- Also supports `'stable'`, `'fair'`, and `'squared'` variants

## Results

All experimental results are saved in the `saved_results/` directory:
- `results-Finite_*.pkl`: Finite-sample experiment results
- `results-OOD_*.pkl`: Out-of-distribution generalization results
- `real_application/`: Real application results

Generated figures are saved in the `Figures/` directory.

## Citation

If you use this project, please cite the relevant paper (please add specific citation information).

## License

Please see the LICENSE file (if it exists) or add license information.

## Contact

For questions or suggestions, please contact via GitHub Issues.
