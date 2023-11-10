import numpy as np
import pytest
from similarity.source.pca_tranform import *

def generate_multivariate_not_rotated_data(n_dim, variances):
    mean = np.zeros(n_dim)
    cov = np.diag(variances)
    data = np.random.multivariate_normal(mean, cov, 1000000)
    return data

@pytest.mark.parametrize("n_dim", [2, 3, 4, 5, 6])
def test_pca(n_dim):
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    transformed_data = compute_pca_using_covariance(original_data)
    
    # Test variance along each principal component
    std_axes = np.eye(n_dim)
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[:, i])
        variance = np.var(projected_data)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.01
    
