import numpy as np
import pytest
from nd_sim.pca_transform import *

def generate_multivariate_not_rotated_data(n_dim, variances):
    mean = np.zeros(n_dim)
    cov = np.diag(variances)
    data = np.random.multivariate_normal(mean, cov, 1000000)
    return data

@pytest.mark.parametrize("n_dim", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("chirality", [False, True])
def test_pca(n_dim, chirality):
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    result = compute_pca_using_covariance(original_data, chirality=chirality)
    
    # Test variance along each principal component
    std_axes = np.eye(n_dim)
    if chirality:
        transformed_data, num_significant_indices = result
        assert len(result) == 2  # Check if two objects are returned
    else:
        transformed_data = result
        
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[:, i])
        variance = np.var(projected_data)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.01
    
    # Additional tests for chirality=True
    if chirality:
        # Check if the number of significant indices is correct
        assert num_significant_indices <= n_dim