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
    transformed_data, eigenvectors = compute_pca_using_covariance(original_data)
    # Test eigenvectors are orthogonal
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            dot_product = np.dot(eigenvectors[:, i], eigenvectors[:, j])
            assert np.isclose(dot_product, 0, atol=1e-4)
    
    # Test variance along each principal component
    std_axes = np.eye(n_dim)
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[:, i])
        variance = np.var(projected_data)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.01
    
    # Test eigenvectors
    for i in range(n_dim):
        matches = [np.isclose(np.abs(np.dot(eigenvectors[:, i], basis_vector)), 1 , atol=0.1) for basis_vector in np.eye(n_dim)]
        assert any(matches)
