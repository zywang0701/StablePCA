# Project Structure

```
StablePCA-online/
├── README.md                      # Project description and quick start guide
├── requirements.txt               # Python dependencies
├── PROJECT_STRUCTURE.md           # Project structure documentation (this file)
├── .gitignore                     # Git ignore rules
│
├── src/                           # Core source code
│   ├── __init__.py
│   ├── PCAalg.py                 # Main algorithm implementations (PCA_MP, PCA_Dual)
│   ├── data.py                   # Data generation functions for simulations
│   ├── utils.py                  # Utility functions (subspace error, explained variance, etc.)
│   └── prev_methods.py           # Baseline method implementations (Fair PCA, etc.)
│
├── illustrations/                 # Paper figure examples
│   ├── illus-1.ipynb             # Illustration example 1
│   └── illus-2.ipynb             # Illustration example 2
│
├── simulations/                   # Simulation experiments and result visualization
│   ├── simu-finitesample.py      # Finite-sample performance experiments
│   ├── simu-generalization.py    # Out-of-distribution generalization experiments
│   ├── plot-finitesample.ipynb   # Finite-sample results visualization
│   └── plot-generalization.ipynb # Generalization results visualization
│
├── application/                   # Real-world applications
│   ├── cluster-generalization.py # Clustering generalization experiment script
│   ├── downstream.ipynb          # Downstream analysis pipeline
│   ├── plot-singlecell.ipynb     # Single-cell data visualization
│   ├── visualization.py          # Visualization utility functions
│   ├── full12batch.h5ad          # Single-cell data file
│   └── RNA.pkl                   # Preprocessed RNA data
│
├── saved_results/                 # Saved experimental results
│   ├── results-Finite_*.pkl      # Finite-sample experiment results
│   ├── results-OOD_*.pkl         # Out-of-distribution generalization results
│   └── real_application/         # Real application results
│       └── rotation_test_results_job*.pkl
│
└── Figures/                       # Generated figures
    ├── illus_pca_diff.png        # PCA difference illustration
    ├── illus_stablepca.png       # StablePCA illustration
    ├── summary_Finite.png        # Finite-sample results summary
    ├── summary_OOD.png           # Out-of-distribution generalization results summary
    ├── single_cell.png           # Single-cell data results
    ├── visualize_downstream.png  # Downstream analysis visualization
    └── visualize_downstream.pdf  # Downstream analysis visualization (PDF format)
```

## Directory Descriptions

### 1. `src/` - Core Source Code
Contains the main implementations of the StablePCA algorithms:
- **PCAalg.py**: Implements `PCA_MP` (Mirror-Prox algorithm) and `PCA_Dual` (dual formulation) methods
- **data.py**: Provides data generation functions for simulation experiments
- **utils.py**: Contains utility functions for subspace error computation, explained variance ratio, etc.
- **prev_methods.py**: Implements baseline methods (e.g., Fair PCA)

### 2. `illustrations/` - Paper Illustrations
Contains Jupyter notebooks for generating paper figures that demonstrate intuitive examples of the algorithms.

### 3. `simulations/` - Simulation Experiments
Contains simulation experiment scripts and result visualizations:
- **simu-finitesample.py**: Finite-sample performance evaluation experiments
- **simu-generalization.py**: Out-of-distribution (OOD) generalization performance evaluation experiments
- **plot-*.ipynb**: Notebooks for visualizing experimental results

### 4. `application/` - Real Applications
Contains examples of applications on real data:
- **cluster-generalization.py**: Clustering generalization experiments
- **downstream.ipynb**: Main pipeline for downstream analysis
- **plot-singlecell.ipynb**: Visualization of single-cell data
- **visualization.py**: Provides t-SNE and UMAP visualization functions

### 5. `saved_results/` - Experimental Results
Stores all experimental results (in pickle format) for subsequent analysis and reproduction.

### 6. `Figures/` - Generated Figures
Stores all generated figure files, including paper illustrations and experimental result visualizations.
