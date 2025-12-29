# StablePCA-V2

[Brief project description - e.g., "Stable Principal Component Analysis for Multi-Source Data"]

## Project Structure

```
StablePCA-V2/
├── src/                    # (1) Core source code
├── illustrations/          # (2) Paper figure notebooks
├── simulations/            # (3) Simulation experiments and plotting
├── application/            # (4) Real-world applications
├── saved_results/          # (5) Experimental results
└── figures/                # (6) Generated figures
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using conda (if environment.yml is provided)
conda env create -f environment.yml
conda activate stablepca
```

## Quick Start

### Run Simulations

```bash
# Finite-sample performance
python simulations/main-finite.py

# Out-of-distribution generalization
python simulations/main-OOD.py

# Single-cell application (first part)
python simulations/main-singlecell.py
```

### Reproduce Figures

1. **Illustrative examples** (main text figures):
   - `illustrations/exp-illus-Nov22.ipynb` - StablePCA vs PooledPCA
   - `illustrations/exp-illus-Nov23.ipynb` - StablePCA vs SquaredPCA vs FairPCA

2. **Result visualizations**:
   - `simulations/plot-Finite.ipynb` - Finite-sample results
   - `simulations/plot-OOD.ipynb` - OOD results
   - `simulations/plot-singlecell.ipynb` - Single-cell results

### Real-world Application

1. **Single-cell analysis (Part 1)**:
   ```bash
   python simulations/main-singlecell.py
   ```

2. **Downstream analysis (Part 2)**:
   - Open `application/downstream/pipeline.ipynb`
   - Follow instructions in the notebook

## Dependencies

See `requirements.txt` for the full list. Main dependencies include:
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas (if used)
- jupyter
- tqdm
- (add any other specific packages used)

## Results

- Simulation results: `saved_results/`
- Figures: `figures/`

## Citation

[Add citation information]

## License

[Add license information]

