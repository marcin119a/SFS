"""
Sampling algorithm to find the set of feasible solutions (SFS) from an initial NMF.
"""
import numpy as np
from typing import Dict, Tuple


def _create_amat(lambda_val: float, s: int, smix: int, N: int) -> np.ndarray:
    """
    Create the transformation matrix given a lambda and two different signatures.
    
    Parameters
    ----------
    lambda_val : float
        Lambda parameter
    s : int
        First signature index
    smix : int
        Second signature index
    N : int
        Dimension
        
    Returns
    -------
    np.ndarray
        Transformation matrix A
    """
    A = np.eye(N)
    A[s, s] = 1 - lambda_val
    A[smix, s] = lambda_val
    return A


def _lambda_range(P: np.ndarray, E: np.ndarray) -> Tuple[float, float]:
    """
    Create the feasible interval of lambda given a matrix P with two columns 
    and E with two rows.
    
    Parameters
    ----------
    P : np.ndarray
        Matrix P with two columns
    E : np.ndarray
        Matrix E with two rows
        
    Returns
    -------
    tuple
        (lmin, lmax) - minimum and maximum lambda values
    """
    p1 = P[:, 0]
    p2 = P[:, 1]
    pdiff = p1 - p2
    neg = pdiff < 0
    
    if np.any(neg):
        frac1 = p1[neg] / pdiff[neg]
        lmin = np.max(frac1)
    else:
        lmin = 0.0
    
    e1 = E[0, :]
    e2 = E[1, :]
    zero = (e1 > 0) | (e2 > 0)
    
    if np.any(zero):
        frac2 = e2[zero] / (e1[zero] + e2[zero])
        lmax = np.min(frac2)
    else:
        lmax = 1.0
    
    return lmin, lmax


def sampleSFS(P: np.ndarray, E: np.ndarray, maxIter: int = 100000, 
              check: int = 1000, beta: float = 0.5, eps: float = 1e-10) -> Dict:
    """
    Find the SFS from a given solution of P and E.
    
    Parameters
    ----------
    P : np.ndarray
        One of the factorized two matrices, which will control the stopping criteria
    E : np.ndarray
        The other of the factorized matrices
    maxIter : int, default 100000
        The maximum number of iterations
    check : int, default 1000
        Number of iterations between checking to stop
    beta : float, default 0.5
        The two shape parameters in the beta distribution to sample lambda
    eps : float, default 1e-10
        Epsilon for the stopping criteria
        
    Returns
    -------
    dict
        Dictionary containing:
        - avgChangeFinal : float - final average change for each entrance in P
        - avgChangeEachCheck : np.ndarray - average change for each entrance in P at each check
        - totalIter : int - total number of iterations needed for convergence
        - Pminimum : np.ndarray - minimum for each entry in P
        - Pmaximum : np.ndarray - maximum for each entry in P
        - Eminimum : np.ndarray - minimum for each entry in E
        - Emaximum : np.ndarray - maximum for each entry in E
        - P_lastCheckResults : np.ndarray - results of P for the last 'check' iterations
        - E_lastCheckResults : np.ndarray - results of E for the last 'check' iterations
        
    Raises
    ------
    ValueError
        If inputs contain negative values or have dimension mismatch
    """
    if np.any(P < 0) or np.any(E < 0):
        raise ValueError("Input matrices cannot contain negative values")
    
    N = P.shape[1]
    K = P.shape[0]
    G = E.shape[1]
    
    if E.shape[0] != N:
        raise ValueError("Dimension mismatch: E should have N rows")
    
    # Initialize storage matrices
    probs = np.zeros(((check + 2) * N, K))
    expos = np.zeros(((check + 2) * N, G))
    
    # Store initial values
    probs[:N, :] = P.T
    probs[N:2*N, :] = P.T
    expos[:N, :] = E
    expos[N:2*N, :] = E
    
    # Clean small values
    P = np.maximum(P, 1e-10)
    E = np.maximum(E, 1e-10)
    
    # Initialize tracking variables
    sig = np.tile(np.arange(N), check + 2)
    avgDev = np.zeros(maxIter // check)
    
    # Initialize min/max matrices
    probmin = np.zeros((N, K))
    probmax = np.zeros((N, K))
    exposmin = np.zeros((N, G))
    exposmax = np.zeros((N, G))
    
    diffnew = 1.0
    diffold = 0.0
    
    for i in range(maxIter):
        # Main sampling loop
        for s in range(N):
            # Choose a different signature to mix with
            smix = s
            while smix == s:
                smix = np.random.randint(0, N)
            
            # Get the two signatures and exposures
            both = [s, smix]
            Pmix = P[:, both]
            Emix = E[both, :]
            
            # Calculate feasible lambda range
            lmin, lmax = _lambda_range(Pmix, Emix)
            
            # Sample lambda from beta distribution
            gvar = np.random.gamma(beta, 1.0, 2)
            x = gvar[0] / np.sum(gvar)
            lambda_val = lmin * x + lmax * (1 - x)
            
            if abs(lambda_val) > 1e-10:
                # Create transformation matrices
                A = _create_amat(lambda_val, s, smix, N)
                Ainv = _create_amat(-lambda_val / (1 - lambda_val), s, smix, N)
                
                # Apply transformation
                P = P @ A
                E = Ainv @ E
                
                # Clean small values
                P = np.maximum(P, 1e-10)
                E = np.maximum(E, 1e-10)
        
        # Store results periodically
        iter_idx = i - (i // check) * check
        probs[N*(iter_idx + 2):N*(iter_idx + 3), :] = P.T
        expos[N*(iter_idx + 2):N*(iter_idx + 3), :] = E
        
        # Check for convergence
        if i > 0 and iter_idx == 0:
            # Calculate minimum and maximum of exposures
            for g in range(G):
                evec = expos[:, g]
                for n in range(N):
                    echoice = evec[sig == n]
                    exposmin[n, g] = np.min(echoice)
                    exposmax[n, g] = np.max(echoice)
            
            expos[:N, :] = exposmin
            expos[N:2*N, :] = exposmax
            
            # Calculate minimum and maximum of signature probabilities
            probdiff = np.zeros((N, K))
            for k in range(K):
                pvec = probs[:, k]
                for n in range(N):
                    pchoice = pvec[sig == n]
                    probdiff[n, k] = np.ptp(pchoice)  # peak-to-peak (max - min)
                    probmin[n, k] = np.min(pchoice)
                    probmax[n, k] = np.max(pchoice)
            
            probs[:N, :] = probmin
            probs[N:2*N, :] = probmax
            diffnew = np.mean(probdiff)
            
            if diffnew - diffold < eps:
                # Converged
                avgDev = avgDev[:i//check - 1]
                break
            else:
                diffold = diffnew
                if i//check - 1 < len(avgDev):
                    avgDev[i//check - 1] = diffnew
    
    return {
        'avgChangeFinal': diffnew,
        'avgChangeEachCheck': avgDev,
        'totalIter': i,
        'Pminimum': probmin,
        'Pmaximum': probmax,
        'Eminimum': exposmin,
        'Emaximum': exposmax,
        'P_lastCheckResults': probs[2*N:, :],
        'E_lastCheckResults': expos[2*N:, :]
    }
