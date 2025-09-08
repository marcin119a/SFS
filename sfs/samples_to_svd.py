"""
Finding SVD representation from solutions of matrices P and E.
"""
import numpy as np
from typing import Dict, Optional


def samplesToSVD(Presults: np.ndarray, Eresults: np.ndarray, N: int, 
                 Mfit: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Find SVD representation from solutions of matrices P and E.
    
    Parameters
    ----------
    Presults : np.ndarray
        Matrix of results of P transposed stacked on top of each other. 
        Dimension is (N*results x nrow(P)).
    Eresults : np.ndarray
        Matrix of results of E stacked on top of each other. 
        Dimension is (N*results x ncol(E))
    N : int
        The rank of the factorization
    Mfit : np.ndarray, optional
        The initial factorization of P and E to use as a reference for the eigenvectors.
        Default is the factorization of the first matrix in Presults and Eresults.
        
    Returns
    -------
    dict
        Dictionary containing:
        - P.points : np.ndarray - Matrix of P results as SVD solution (results x (N-1))
        - E.points : np.ndarray - Matrix of E results as SVD solution (results x (N-1))
        
    Raises
    ------
    ValueError
        If dimensions don't match expected values
    """
    # Validate dimensions
    if Presults.shape[0] % N != 0:
        raise ValueError("Dimension mismatch: Presults first dimension should be divisible by N")
    
    if Eresults.shape[0] % N != 0:
        raise ValueError("Dimension mismatch: Eresults first dimension should be divisible by N")
    
    if Presults.shape[0] != Eresults.shape[0]:
        raise ValueError("Dimension mismatch: Presults and Eresults should have same first dimension")
    
    n_results = Presults.shape[0] // N
    K = Presults.shape[1]
    G = Eresults.shape[1]
    
    # Create default Mfit if not provided
    if Mfit is None:
        P_first = Presults[:N, :].T
        E_first = Eresults[:N, :]
        Mfit = P_first @ E_first
    
    # Perform SVD on Mfit
    svd_Mfit = np.linalg.svd(Mfit.T, full_matrices=False)
    svdV = svd_Mfit[2].T  # v matrix (G x N)
    svdU = svd_Mfit[0]    # u matrix (K x N)
    
    # Initialize output matrices
    P_points = np.zeros((Presults.shape[0], N - 1))
    E_points = np.zeros((Eresults.shape[0], N - 1))
    
    # Process each result
    for i in range(n_results):
        # Get P and E for this result (following R indexing)
        p = Presults[i*N:(i+1)*N, :]  # N x K (not transposed!)
        e = Eresults[i*N:(i+1)*N, :]  # N x G
        
        # Find Tmat for P: p @ svdV should be N x N
        Tmat_p = p @ svdV
        Tmat_p = Tmat_p / Tmat_p[:, 0:1]  # normalize by first column
        P_points[i*N:(i+1)*N, :] = Tmat_p[:, 1:N]
        
        # Find Tmat for E: e @ svdU should be N x N
        Tmat_e = e @ svdU
        Tmat_e = Tmat_e / Tmat_e[:, 0:1]  # normalize by first column
        E_points[i*N:(i+1)*N, :] = Tmat_e[:, 1:N]
    
    return {
        'P.points': P_points,
        'E.points': E_points
    }
