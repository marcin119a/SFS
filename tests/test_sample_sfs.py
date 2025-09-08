"""
Tests for the sampleSFS function
"""
import numpy as np
import pytest
from sfs.sample_sfs import sampleSFS


class TestSampleSFS:
    """Test cases for the sampleSFS function"""
    
    def test_sample_sfs_basic(self):
        """Test basic functionality of sampleSFS"""
        # Create test matrices
        P = np.array([[0.3, 0.4, 0.3],
                      [0.2, 0.5, 0.3],
                      [0.4, 0.3, 0.3],
                      [0.1, 0.4, 0.5]], dtype=float)
        E = np.array([[0.2, 0.3, 0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]], dtype=float)
        
        result = sampleSFS(P, E, maxIter=1000, check=100)
        
        # Check that all expected keys are present
        expected_keys = ['avgChangeFinal', 'avgChangeEachCheck', 'totalIter',
                        'Pminimum', 'Pmaximum', 'Eminimum', 'Emaximum',
                        'P_lastCheckResults', 'E_lastCheckResults']
        for key in expected_keys:
            assert key in result
        
        # Check dimensions
        assert result['Pminimum'].shape == P.shape
        assert result['Pmaximum'].shape == P.shape
        assert result['Eminimum'].shape == E.shape
        assert result['Emaximum'].shape == E.shape
        
        # Check that minimums are <= maximums
        assert np.all(result['Pminimum'] <= result['Pmaximum'])
        assert np.all(result['Eminimum'] <= result['Emaximum'])
        
        # Check that results are non-negative
        assert np.all(result['Pminimum'] >= 0)
        assert np.all(result['Pmaximum'] >= 0)
        assert np.all(result['Eminimum'] >= 0)
        assert np.all(result['Emaximum'] >= 0)
    
    def test_sample_sfs_negative_inputs(self):
        """Test sampleSFS with negative inputs should raise error"""
        P = np.array([[0.3, 0.4, 0.3],
                      [0.2, 0.5, 0.3],
                      [0.4, 0.3, 0.3],
                      [0.1, 0.4, 0.5]], dtype=float)
        E = np.array([[0.2, 0.3, 0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]], dtype=float)
        
        P_neg = P.copy()
        P_neg[0, 0] = -0.1
        
        with pytest.raises(ValueError, match="cannot be negative"):
            sampleSFS(P_neg, E)
        
        E_neg = E.copy()
        E_neg[0, 0] = -0.1
        
        with pytest.raises(ValueError, match="cannot be negative"):
            sampleSFS(P, E_neg)
    
    def test_sample_sfs_dimension_mismatch(self):
        """Test sampleSFS with dimension mismatch should raise error"""
        P = np.array([[0.3, 0.4, 0.3],
                      [0.2, 0.5, 0.3],
                      [0.4, 0.3, 0.3],
                      [0.1, 0.4, 0.5]], dtype=float)
        E = np.array([[0.2, 0.3, 0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]], dtype=float)  # Wrong number of rows
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            sampleSFS(P, E)
    
    def test_sample_sfs_small_iterations(self):
        """Test sampleSFS with small number of iterations"""
        P = np.array([[0.3, 0.4, 0.3],
                      [0.2, 0.5, 0.3],
                      [0.4, 0.3, 0.3],
                      [0.1, 0.4, 0.5]], dtype=float)
        E = np.array([[0.2, 0.3, 0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]], dtype=float)
        
        result = sampleSFS(P, E, maxIter=100, check=50)
        
        assert result['totalIter'] <= 100
        assert isinstance(result['avgChangeFinal'], float)
    
    def test_sample_sfs_custom_parameters(self):
        """Test sampleSFS with custom parameters"""
        P = np.array([[0.3, 0.4, 0.3],
                      [0.2, 0.5, 0.3],
                      [0.4, 0.3, 0.3],
                      [0.1, 0.4, 0.5]], dtype=float)
        E = np.array([[0.2, 0.3, 0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]], dtype=float)
        
        result = sampleSFS(P, E, maxIter=500, check=100, beta=0.3, eps=1e-8)
        
        assert result['totalIter'] <= 500
        assert isinstance(result['avgChangeFinal'], float)
    
    def test_sample_sfs_convergence(self):
        """Test that sampleSFS converges (avgChangeFinal should be small)"""
        P = np.array([[0.3, 0.4, 0.3],
                      [0.2, 0.5, 0.3],
                      [0.4, 0.3, 0.3],
                      [0.1, 0.4, 0.5]], dtype=float)
        E = np.array([[0.2, 0.3, 0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]], dtype=float)
        
        result = sampleSFS(P, E, maxIter=10000, check=1000, eps=1e-6)
        
        # The final average change should be small if converged
        assert result['avgChangeFinal'] < 1e-5
