"""
Tests for the NMFPois function
"""
import numpy as np
import pytest
from sfs.nmf_pois import NMFPois


class TestNMFPois:
    """Test cases for the non-negative matrix factorization function"""
    
    def test_nmf_pois_basic(self):
        """Test basic functionality of NMFPois"""
        M = np.array([[20, 3, 24, 19, 2, 15],
                      [9, 14, 25, 30, 15, 10],
                      [30, 6, 12, 10, 11, 7],
                      [9, 27, 5, 11, 19, 15]], dtype=float)
        N = 3
        
        result = NMFPois(M, N)
        
        assert 'P' in result
        assert 'E' in result
        assert 'gkl' in result
        
        P = result['P']
        E = result['E']
        gkl = result['gkl']
        
        # Check dimensions
        assert P.shape == (4, 3)  # K x N
        assert E.shape == (3, 6)  # N x G
        assert isinstance(gkl, float)
        
        # Check non-negativity
        assert np.all(P >= 0)
        assert np.all(E >= 0)
        
        # Check normalization (columns of P should sum to 1)
        assert np.allclose(np.sum(P, axis=0), 1.0, atol=1e-10)
    
    def test_nmf_pois_negative_matrix(self):
        """Test NMFPois with negative matrix should raise error"""
        M = np.array([[20, -3, 24, 19, 2, 15],
                      [9, 14, 25, 30, 15, 10],
                      [30, 6, 12, 10, 11, 7],
                      [9, 27, 5, 11, 19, 15]], dtype=float)
        N = 3
        
        with pytest.raises(ValueError, match="cannot be negative"):
            NMFPois(M, N)
    
    def test_nmf_pois_negative_rank(self):
        """Test NMFPois with negative rank should raise error"""
        M = np.array([[20, 3, 24, 19, 2, 15],
                      [9, 14, 25, 30, 15, 10],
                      [30, 6, 12, 10, 11, 7],
                      [9, 27, 5, 11, 19, 15]], dtype=float)
        N = -1
        
        with pytest.raises(ValueError, match="cannot be negative"):
            NMFPois(M, N)
    
    def test_nmf_pois_rank_larger_than_min_dimension(self):
        """Test NMFPois with rank larger than min dimension"""
        M = np.array([[20, 3, 24, 19, 2, 15],
                      [9, 14, 25, 30, 15, 10],
                      [30, 6, 12, 10, 11, 7],
                      [9, 27, 5, 11, 19, 15]], dtype=float)
        N = 5  # Larger than min(4, 6) = 4
        
        result = NMFPois(M, N)
        P = result['P']
        E = result['E']
        
        assert P.shape == (4, 5)
        assert E.shape == (5, 6)
    
    def test_nmf_pois_custom_seed(self):
        """Test NMFPois with custom seed"""
        M = np.array([[20, 3, 24, 19, 2, 15],
                      [9, 14, 25, 30, 15, 10],
                      [30, 6, 12, 10, 11, 7],
                      [9, 27, 5, 11, 19, 15]], dtype=float)
        N = 3
        seed = [42, 43, 44]
        
        result1 = NMFPois(M, N, seed=seed)
        result2 = NMFPois(M, N, seed=seed)
        
        # Results should be identical with same seed
        assert np.allclose(result1['P'], result2['P'])
        assert np.allclose(result1['E'], result2['E'])
        assert result1['gkl'] == result2['gkl']
    
    def test_nmf_pois_arrange_false(self):
        """Test NMFPois with arrange=False"""
        M = np.array([[20, 3, 24, 19, 2, 15],
                      [9, 14, 25, 30, 15, 10],
                      [30, 6, 12, 10, 11, 7],
                      [9, 27, 5, 11, 19, 15]], dtype=float)
        N = 3
        
        result = NMFPois(M, N, arrange=False)
        P = result['P']
        E = result['E']
        
        # Check that columns of P still sum to 1
        assert np.allclose(np.sum(P, axis=0), 1.0, atol=1e-10)
    
    def test_nmf_pois_approximation_quality(self):
        """Test that the approximation M ≈ PE is reasonable"""
        M = np.array([[20, 3, 24, 19, 2, 15],
                      [9, 14, 25, 30, 15, 10],
                      [30, 6, 12, 10, 11, 7],
                      [9, 27, 5, 11, 19, 15]], dtype=float)
        N = 3
        
        result = NMFPois(M, N)
        P = result['P']
        E = result['E']
        approximation = P @ E
        
        # The approximation should be reasonably close to the original
        relative_error = np.mean(np.abs(M - approximation) / (M + 1e-10))
        assert relative_error < 1.0  # Allow for some error
