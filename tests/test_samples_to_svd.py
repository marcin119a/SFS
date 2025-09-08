"""
Tests for the samplesToSVD function
"""
import numpy as np
import pytest
from sfs.samples_to_svd import samplesToSVD


class TestSamplesToSVD:
    """Test cases for the samplesToSVD function"""
    
    def test_samples_to_svd_basic(self):
        """Test basic functionality of samplesToSVD"""
        # Create test data
        N = 3
        K = 4
        G = 6
        n_results = 2
        
        # Create Presults: (N*results x K)
        Presults = np.random.rand(N * n_results, K)
        # Create Eresults: (N*results x G)
        Eresults = np.random.rand(N * n_results, G)
        
        result = samplesToSVD(Presults, Eresults, N)
        
        # Check that all expected keys are present
        assert 'P.points' in result
        assert 'E.points' in result
        
        P_points = result['P.points']
        E_points = result['E.points']
        
        # Check dimensions
        assert P_points.shape == (N * n_results, N - 1)
        assert E_points.shape == (N * n_results, N - 1)
        
        # Check that results are finite
        assert np.all(np.isfinite(P_points))
        assert np.all(np.isfinite(E_points))
    
    def test_samples_to_svd_custom_mfit(self):
        """Test samplesToSVD with custom Mfit"""
        N = 3
        K = 4
        G = 6
        n_results = 2
        
        Presults = np.random.rand(N * n_results, K)
        Eresults = np.random.rand(N * n_results, G)
        
        # Create custom Mfit
        Mfit = np.random.rand(K, G)
        
        result = samplesToSVD(Presults, Eresults, N, Mfit=Mfit)
        
        P_points = result['P.points']
        E_points = result['E.points']
        
        # Check dimensions
        assert P_points.shape == (N * n_results, N - 1)
        assert E_points.shape == (N * n_results, N - 1)
    
    def test_samples_to_svd_dimension_mismatch(self):
        """Test samplesToSVD with dimension mismatch should raise error"""
        N = 3
        K = 4
        G = 6
        n_results = 2
        
        Presults = np.random.rand(N * n_results, K)
        Eresults = np.random.rand(N * n_results, G)
        
        # Wrong N
        with pytest.raises(ValueError, match="Dimension mismatch"):
            samplesToSVD(Presults, Eresults, N=2)
        
        # Wrong Presults dimensions
        Presults_wrong = np.random.rand(N * n_results + 1, K)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            samplesToSVD(Presults_wrong, Eresults, N)
        
        # Wrong Eresults dimensions
        Eresults_wrong = np.random.rand(N * n_results, G + 1)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            samplesToSVD(Presults, Eresults_wrong, N)
    
    def test_samples_to_svd_single_result(self):
        """Test samplesToSVD with single result"""
        N = 2
        K = 3
        G = 4
        n_results = 1
        
        Presults = np.random.rand(N * n_results, K)
        Eresults = np.random.rand(N * n_results, G)
        
        result = samplesToSVD(Presults, Eresults, N)
        
        P_points = result['P.points']
        E_points = result['E.points']
        
        # Check dimensions
        assert P_points.shape == (N * n_results, N - 1)
        assert E_points.shape == (N * n_results, N - 1)
    
    def test_samples_to_svd_large_data(self):
        """Test samplesToSVD with larger data"""
        N = 5
        K = 10
        G = 8
        n_results = 3
        
        Presults = np.random.rand(N * n_results, K)
        Eresults = np.random.rand(N * n_results, G)
        
        result = samplesToSVD(Presults, Eresults, N)
        
        P_points = result['P.points']
        E_points = result['E.points']
        
        # Check dimensions
        assert P_points.shape == (N * n_results, N - 1)
        assert E_points.shape == (N * n_results, N - 1)
    
    def test_samples_to_svd_rank_one(self):
        """Test samplesToSVD with rank 1 (edge case)"""
        N = 1
        K = 3
        G = 4
        n_results = 2
        
        Presults = np.random.rand(N * n_results, K)
        Eresults = np.random.rand(N * n_results, G)
        
        result = samplesToSVD(Presults, Eresults, N)
        
        P_points = result['P.points']
        E_points = result['E.points']
        
        # For rank 1, N-1 = 0, so points should be empty
        assert P_points.shape == (N * n_results, 0)
        assert E_points.shape == (N * n_results, 0)
