# Low-Rank Matrix Approximation Algorithms

This project implements and benchmarks low-rank matrix approximation algorithms including:
- **rSVD-SRFT**: Randomized SVD with Subsampled Random Fourier Transform
- **RRQR**: Rank-Revealing QR with column pivoting
- **rSVD**: Randomized SVD with Gaussian random projection

This README was generated with the assisstance of Claude.

## Installation

Prerequisites: Python e3.12

Install dependencies:
```bash
uv sync
```

## Running the Scripts

### 1. Gene Expression Analysis

Analyzes real gene expression data from the GSE2553 dataset:

```bash
python -m src.genomics
```

This script:
- Downloads and loads GSE2553 gene expression matrix
- Runs rSVD-SRFT with different power iteration counts (q=0, 1, 4)
- Runs RRQR for comparison
- Outputs spectral error results for ranks k={40, 60, 80, 100}

### 2. Runtime Benchmarks

Benchmarks algorithm performance on random matrices:

```bash
python -m src.runtimes
```

This script:
- Tests on matrices of sizes 1024�1024, 2048�2048, 4096�4096
- Compares truncated SVD, rSVD (Gaussian), rSVD-SRFT, and RRQR
- Reports timing and Frobenius error for various ranks
- Prints detailed comparison tables

### 3. Image Low-Rank Approximation (CLI)

Computes low-rank approximations of images:

```bash
python -m src.image_lowrank_cli <image_path> --max-rank <k> --num-runs <n> [--grayscale true/false]
```

**Parameters:**
- `image_path`: Path to input image (e.g., `images/boston.jpg`)
- `--max-rank`: Maximum rank to test (e.g., `100`)
- `--num-runs`: Number of rank values to sample between 1 and max-rank (e.g., `10`)
- `--grayscale`: Convert to grayscale (default: `false`)

**Example:**
```bash
python -m src.image_lowrank_cli images/boston.jpg --max-rank 100 --num-runs 10 --grayscale false
```

**Output:** Saves to `results/image_{name}_{H}x{W}/`:
- Approximated images for each algorithm and rank
- Frobenius and spectral error plots
- Memory usage comparison plot
- CSV file with error metrics (`errors_vs_rank.csv`)

### 4. Plot Combined Error Comparison

Creates comparison plots across multiple images:

```bash
python -m src.plot_image_errors
```

This script:
- Reads error CSVs from `results/image_*/errors_vs_rank.csv`
- Generates side-by-side comparison of Frobenius and spectral errors
- Saves combined plot to `results/error_plots.png`

## Typical Workflow

To analyze and compare image approximations:

1. Run low-rank approximation on multiple images:
   ```bash
   python -m src.image_lowrank_cli images/trefethen.jpg --max-rank 100 --num-runs 10
   python -m src.image_lowrank_cli images/boston.jpg --max-rank 100 --num-runs 10
   ```

2. Generate comparison plots:
   ```bash
   python -m src.plot_image_errors
   ```

