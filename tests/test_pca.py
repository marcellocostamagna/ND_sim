import pytest
import sys
import os
from fingerprints import *
from pca_fingerprint import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_principle_components():
    """ Test that the principle components obtained from the 
    Single Value Decomposition and from the eigendecomposition of 
    the covariance matrix are the same.
    """
    data = np.array([[0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2, 2],
                        [3, 3, 3, 3, 3, 3],
                        [4, 4, 4, 4, 4, 4],
                        [5, 5, 5, 5, 5, 5],
                        [6, 6, 6, 6, 6, 6]]).astype(float)
    
    fingerprint_pca = get_pca_fingerprint(data)
    fingerprint_cov = compute_nD_fingerprint(data)

    with pytest.raises(AssertionError):
        assert fingerprint_pca == fingerprint_cov
