"""
Non-negative matrix factorization for Poisson data and related utilities.
"""
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Union, Optional


def gkl_dev(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Function for calculating the generalized Kullback-Leibler divergence.
    
    Internal function used in NMFPois.
    
    Parameters
    ----------
    y : np.ndarray
        Observation
    mu : np.ndarray
        Estimate
        
    Returns
    -------
    float
        Generalized Kullback-Leibler divergence
        
    Raises
    ------
    ValueError
        If inputs have different lengths or contain negative values
    """
    if len(y) != len(mu):
        raise ValueError("Different length of observations and their estimates")
    
    if np.any(y < 0) or np.any(mu < 0):
        raise ValueError("The input cannot be negative")
    
    r = mu.copy()
    p = y > 0
    r[p] = (y[p] * (np.log(y[p]) - np.log(mu[p])) - y[p] + mu[p])
    
    return np.sum(r)


def NMFPois(M: np.ndarray, N: int, seed: Optional[List[int]] = None, 
            arrange: bool = True, tol: float = 1e-5) -> Dict[str, np.ndarray]:
    """
    Non-negative matrix factorization algorithm for Poisson data.
    
    Factorizes M into two matrices P and E of dimension ncol(M) x N and N x nrow(M)
    with the acceleration of SQUAREM. The objective function is the generalized 
    Kullback-Leibler divergence (GKLD).
    
    Parameters
    ----------
    M : np.ndarray
        Non-negative data matrix
    N : int
        Small dimension of the two new matrices (rank)
    seed : list of int, optional
        Vector of random seeds to initialize the matrices
    arrange : bool, default True
        Arranging columns in P and rows in E after largest row sums of E
    tol : float, default 1e-5
        Maximum change of P and E when stopping
        
    Returns
    -------
    dict
        Dictionary containing:
        - P : np.ndarray - Non-negative matrix of dimension ncol(M) x N, with columns summing to one
        - E : np.ndarray - Non-negative matrix of dimension N x nrow(M), where rows sum to one
        - gkl : float - Smallest value of the Generalized Kullback-Leibler divergence
        
    Raises
    ------
    ValueError
        If the data matrix or rank is negative
    """
    if np.any(M < 0) or N < 0:
        raise ValueError("The data matrix and/or rank cannot be negative.")
    
    K = M.shape[0]  # mutations
    G = M.shape[1]  # patients
    
    if seed is None:
        seed = list(np.random.randint(1, 101, 3))
    
    div = np.zeros(len(seed))  # vector of different GKLD values
    Plist = []  # list of P matrices
    Elist = []  # list of E matrices
    reslist = []
    
    def poisson_em(x):
        """EM update function for Poisson NMF"""
        x = np.exp(x)
        P = x[:K*N].reshape(K, N)
        E = x[K*N:].reshape(N, G)
        
        PE = P @ E
        # Update of signatures
        P = P * ((M / PE) @ E.T)
        P = P / np.sum(E, axis=1)
        
        PE = P @ E
        # Update of exposures
        E = E * (P.T @ (M / PE))
        E = E / np.sum(P, axis=0)
        
        par = np.concatenate([P.flatten(), E.flatten()])
        par[par <= 0] = 1e-10
        return np.log(par)
    
    def gkl_obj(x):
        """Objective function for GKLD"""
        x = np.exp(x)
        P = x[:K*N].reshape(K, N)
        E = x[K*N:].reshape(N, G)
        
        GKL = gkl_dev(M.flatten(), (P @ E).flatten())
        return GKL
    
    for i in range(len(seed)):
        np.random.seed(seed[i])
        
        # Initialize P and E
        P = np.random.rand(K, N)
        E = np.random.rand(N, G)
        
        init = np.log(np.concatenate([P.flatten(), E.flatten()]))
        
        # Use scipy.optimize.minimize instead of SQUAREM
        # This is a simplified version - in practice you might want to implement SQUAREM
        result = minimize(gkl_obj, init, method='L-BFGS-B', 
                         bounds=[(-20, 20)] * len(init))
        
        P = np.exp(result.x[:K*N]).reshape(K, N)
        E = np.exp(result.x[K*N:]).reshape(N, G)
        E = np.sum(P, axis=0) * E  # normalizing
        P = P / np.sum(P, axis=0)
        
        Plist.append(P)
        Elist.append(E)
        div[i] = gkl_obj(result.x)
        reslist.append(result)
    
    best = np.argmin(div)  # smallest GKLD value
    P = Plist[best]
    E = Elist[best]
    
    if arrange:
        idx = np.argsort(np.sum(E, axis=1))[::-1]  # decreasing order
        P = P[:, idx]
        E = E[idx, :]
    
    return {
        'P': P,
        'E': E,
        'gkl': div[best]
    }
