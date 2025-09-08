"""
Tests for the gkl_dev function
"""
import numpy as np
import pytest
from sfs.nmf_pois import gkl_dev


class TestGklDev:
    """Test cases for the generalized Kullback-Leibler divergence function"""
    
    def test_gkl_dev_basic(self):
        """Test basic functionality of gkl_dev"""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.1])
        result = gkl_dev(y, mu)
        assert isinstance(result, float)
        assert result >= 0
    
    def test_gkl_dev_identical_inputs(self):
        """Test gkl_dev with identical inputs should return 0"""
        y = np.array([1.0, 2.0, 3.0])
        mu = y.copy()
        result = gkl_dev(y, mu)
        assert result == 0.0
    
    def test_gkl_dev_different_lengths(self):
        """Test gkl_dev with different length inputs should raise error"""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Different length"):
            gkl_dev(y, mu)
    
    def test_gkl_dev_negative_inputs(self):
        """Test gkl_dev with negative inputs should raise error"""
        y = np.array([1.0, -2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="cannot be negative"):
            gkl_dev(y, mu)
        
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, -2.0, 3.0])
        with pytest.raises(ValueError, match="cannot be negative"):
            gkl_dev(y, mu)
    
    def test_gkl_dev_zero_observations(self):
        """Test gkl_dev with zero observations"""
        y = np.array([0.0, 2.0, 0.0])
        mu = np.array([1.0, 2.0, 3.0])
        result = gkl_dev(y, mu)
        assert isinstance(result, float)
        assert result >= 0
    
    def test_gkl_dev_zero_estimates(self):
        """Test gkl_dev with zero estimates should raise error"""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([0.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="cannot be negative"):
            gkl_dev(y, mu)
    
    def test_gkl_dev_large_values(self):
        """Test gkl_dev with large values"""
        y = np.array([100.0, 200.0, 300.0])
        mu = np.array([110.0, 190.0, 310.0])
        result = gkl_dev(y, mu)
        assert isinstance(result, float)
        assert result >= 0
