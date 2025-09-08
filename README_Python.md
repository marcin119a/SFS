# SFS Python Package

## Introduction

This Python package is a port of the R package SFS, developed for a new sampling algorithm to find the set of feasible solutions (SFS) from an initial solution of non-negative matrix factorization (NMF). 

Non-negative matrix factorization takes a non-negative matrix **M(K × G)** and approximates it by two other non-negative matrices **P(K × N)** and **E(N × G)** such that:

**M ≈ PE**

Other solutions with the same approximation could be constructed with an invertible matrix **A(N × N)** such that:

**P̃ = PA ≥ 0** and **Ẽ = A⁻¹E ≥ 0**

are new solutions. There exist trivial ambiguities where **A** is either a diagonal matrix or a permutation matrix, but besides these trivial ambiguities others could exist as well. The scaling ambiguity is removed by assuming the columns of **P** sum to one. 

The goal of the main function `sampleSFS` in this package is to approximate the whole SFS that exists for **P** and **E** besides the ambiguities. The advantage of this algorithm is that it has a simple implementation and can be applied for an arbitrary dimension of **N**. A further description can be found in the corresponding paper *R. Laursen and A. Hobolth, A sampling algorithm to compute the set of feasible solutions for non-negative matrix factorization with an arbitrary rank*.

## Package Functions

The package includes the following functions:

- `sampleSFS`: Main function that can find the SFS from an initial NMF
- `NMFPois`: Can find an initial NMF solution from a data matrix
- `gkl_dev`: Internal function for `NMFPois`, that calculates the generalized Kullback-Leibler divergence
- `samplesToSVD`: Will transform SFS solutions from `sampleSFS` relative to SVD solution

## Installation

### Requirements

- Python 3.7 or higher
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- scikit-learn >= 0.23.0

### Install from source

```bash
# Clone the repository
git clone https://github.com/ragnhildlaursen/SFS.git
cd SFS

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Usage Example

To illustrate the functions, let's assume we have a given matrix of data **M (4 × 6)**:

```python
import numpy as np
from sfs import NMFPois, sampleSFS, samplesToSVD

# Create example data matrix
M = np.array([[20, 3, 24, 19, 2, 15],
              [9, 14, 25, 30, 15, 10],
              [30, 6, 12, 10, 11, 7],
              [9, 27, 5, 11, 19, 15]], dtype=float)

print("Data matrix M:")
print(M)
```

### Step 1: Create Initial NMF Solution

First, we need to create an initial NMF solution using the function `NMFPois`. The input for this function is a matrix **M** and a rank **N**, which we choose to be **3**:

```python
# Create initial NMF solution
rank = 3
initial_fit = NMFPois(M, rank)

P = initial_fit['P']
E = initial_fit['E']
gkl = initial_fit['gkl']

print(f"Initial P shape: {P.shape}")
print(f"Initial E shape: {E.shape}")
print(f"Initial GKLD: {gkl:.6f}")

# Check approximation quality
approximation = P @ E
print("Approximation of M:")
print(approximation)
```

### Step 2: Find the Set of Feasible Solutions

Now, as an initial solution has been constructed, one can find the **SFS** with the function `sampleSFS`:

```python
# Find the set of feasible solutions
sfs_result = sampleSFS(P, E, maxIter=10000, check=1000)

print(f"Total iterations: {sfs_result['totalIter']}")
print(f"Final average change: {sfs_result['avgChangeFinal']:.2e}")

# Access the results
P_min = sfs_result['Pminimum']
P_max = sfs_result['Pmaximum']
E_min = sfs_result['Eminimum']
E_max = sfs_result['Emaximum']

print("P matrix ranges (min, max):")
for i in range(rank):
    print(f"Column {i+1}:")
    for j in range(P.shape[0]):
        print(f"  Row {j+1}: [{P_min[i,j]:.4f}, {P_max[i,j]:.4f}]")
```

### Step 3: Convert to SVD Representation (Optional)

You can also convert the results to SVD representation:

```python
# Get results from the last check
P_last = sfs_result['P_lastCheckResults']
E_last = sfs_result['E_lastCheckResults']

# Convert to SVD representation
svd_result = samplesToSVD(P_last, E_last, rank)

P_points = svd_result['P.points']
E_points = svd_result['E.points']

print(f"SVD P points shape: {P_points.shape}")
print(f"SVD E points shape: {E_points.shape}")
```

## Function Documentation

### `NMFPois(M, N, seed=None, arrange=True, tol=1e-5)`

Non-negative matrix factorization algorithm for Poisson data.

**Parameters:**
- `M` (np.ndarray): Non-negative data matrix
- `N` (int): Rank of the factorization
- `seed` (list, optional): Random seeds for initialization
- `arrange` (bool): Whether to arrange columns by row sums
- `tol` (float): Convergence tolerance

**Returns:**
- Dictionary with keys: 'P', 'E', 'gkl'

### `sampleSFS(P, E, maxIter=100000, check=1000, beta=0.5, eps=1e-10)`

Find the set of feasible solutions from initial NMF matrices.

**Parameters:**
- `P` (np.ndarray): Initial P matrix
- `E` (np.ndarray): Initial E matrix
- `maxIter` (int): Maximum iterations
- `check` (int): Check interval for convergence
- `beta` (float): Beta distribution parameter
- `eps` (float): Convergence epsilon

**Returns:**
- Dictionary with SFS results including min/max values and convergence info

### `samplesToSVD(Presults, Eresults, N, Mfit=None)`

Convert SFS results to SVD representation.

**Parameters:**
- `Presults` (np.ndarray): P results matrix
- `Eresults` (np.ndarray): E results matrix
- `N` (int): Rank
- `Mfit` (np.ndarray, optional): Reference factorization

**Returns:**
- Dictionary with 'P.points' and 'E.points'

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sfs

# Run specific test file
pytest tests/test_nmf_pois.py
```

## Development

### Code Style

The package follows PEP 8 style guidelines. Format code with:

```bash
black sfs/ tests/
```

### Linting

Check code quality with:

```bash
flake8 sfs/ tests/
```

## Differences from R Version

1. **Optimization**: The Python version uses `scipy.optimize.minimize` instead of SQUAREM for the NMF optimization. For better performance, consider implementing SQUAREM in Python.

2. **Random Number Generation**: Uses NumPy's random number generator instead of R's.

3. **Data Types**: All arrays are NumPy arrays with explicit float types.

4. **Error Handling**: Python-style exceptions instead of R's error system.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package, please cite the original paper:

*R. Laursen and A. Hobolth, A sampling algorithm to compute the set of feasible solutions for non-negative matrix factorization with an arbitrary rank.*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
