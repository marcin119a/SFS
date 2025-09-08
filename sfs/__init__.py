"""
SFS Package - Sampling the set of feasible solutions from a non-negative matrix factorization

This package is developed for a new sampling algorithm to find the set of feasible solutions (SFS) 
from an initial solution of non-negative matrix factorization (NMF).

The package includes the following functions:
- sampleSFS: Main function that can find the SFS from an initial NMF
- NMFPois: Can find an initial NMF solution from a data matrix
- gkl_dev: Internal function for NMFPois that calculates the generalized Kullback-Leibler
- samplesToSVD: Will transform SFS solutions from sampleSFS relative to SVD solution
"""

from .nmf_pois import NMFPois, gkl_dev
from .sample_sfs import sampleSFS
from .samples_to_svd import samplesToSVD

__version__ = "1.1.0"
__author__ = "Ragnhild Laursen"
__email__ = "ragnhild@math.au.dk"

__all__ = ["NMFPois", "gkl_dev", "sampleSFS", "samplesToSVD"]
