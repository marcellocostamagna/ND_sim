import numpy as np
import pytest
from source import pca_tranform

def generate_multivariate_not_rotated_data(n_dim, variances):
    mean = np.zeros(n_dim)
    cov = np.diag(variances)
    data = np.random.multivariate_normal(mean, cov, 10000)
    return data

##### 2D #####
def test_perform_PCA_on_known_data_2D():
    n_dim = 2
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(original_data)
    
    # Test eigenvectors are orthogonal
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            dot_product = np.dot(axes[i], axes[j])
            assert np.isclose(dot_product, 0, atol=1e-1)

    # Test eigenvalues
     # Test variance along each principal component
    std_axes = np.eye(n_dim)
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[i])
        variance = np.var(projected_data)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.1
    # Test eigenvectors
    for axis in axes:
        matches = [np.isclose(np.abs(np.dot(axis, basis_vector)), 1 , atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)

#### 3D ####

def test_perform_PCA_on_known_data_3D():
    n_dim = 3
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(original_data)

     # Test eigenvectors are orthogonal
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            dot_product = np.dot(axes[i], axes[j])
            assert np.isclose(dot_product, 0, atol=1e-1)

    # Test eigenvalues
    # Test eigenvalues
     # Test variance along each principal component
    std_axes = np.eye(n_dim)
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[i])
        variance = np.var(projected_data)
        # assert np.isclose(variance, variances[i], atol=1)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.1

    # Test eigenvectors
    for axis in axes:
        matches = [np.isclose(np.abs(np.dot(axis, basis_vector)), 1 , atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)

##### 4D ####

def test_perform_PCA_on_known_data_4D():
    n_dim = 4
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(original_data)

     # Test eigenvectors are orthogonal
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            dot_product = np.dot(axes[i], axes[j])
            assert np.isclose(dot_product, 0, atol=1e-1)

    # Test eigenvalues
     # Test variance along each principal component
    std_axes = np.eye(n_dim)
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[i])
        variance = np.var(projected_data)
        # assert np.isclose(variance, variances[i], atol=1)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.1

    # Test eigenvectors
    for axis in axes:
        matches = [np.isclose(np.abs(np.dot(axis, basis_vector)), 1 , atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)


##### 5D ####

def test_perform_PCA_on_known_data_5D():
    n_dim = 5
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(original_data)

     # Test eigenvectors are orthogonal
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            dot_product = np.dot(axes[i], axes[j])
            assert np.isclose(dot_product, 0, atol=1e-1)

    # Test eigenvalues
     # Test variance along each principal component
    std_axes = np.eye(n_dim)
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[i])
        variance = np.var(projected_data)
        # assert np.isclose(variance, variances[i], atol=1)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.1

    # Test eigenvectors
    for axis in axes:
        matches = [np.isclose(np.abs(np.dot(axis, basis_vector)), 1 , atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)


##### 6D ####

def test_perform_PCA_on_known_data_6D():
    n_dim = 6
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(original_data)

     # Test eigenvectors are orthogonal
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            dot_product = np.dot(axes[i], axes[j])
            assert np.isclose(dot_product, 0, atol=1e-1)

    # Test eigenvalues
     # Test variance along each principal component
    std_axes = np.eye(n_dim)
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[i])
        variance = np.var(projected_data)
        # assert np.isclose(variance, variances[i], atol=1)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.1

    # Test eigenvectors
    for axis in axes:
        matches = [np.isclose(np.abs(np.dot(axis, basis_vector)), 1 , atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)

def test_no_sign_ambiguity_1D():
    """
    Test that PCA correctly handles data with no sign ambiguity.
    The dataset is unidimensional with all positive points, 
    and PCA should generate a first eigenvector with all non-negative coordinates.
    """
    n_dim = 1
    n_points = 1000
    original_data = np.random.uniform(0, 1, size=(n_points, n_dim))  # all data points are positive

    # Add a second dimension with all zeros to facilitate PCA
    original_data = np.hstack((original_data, np.zeros((n_points, 1))))

    _, _, eigenvectors, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(original_data)

    # Validate that all coordinates of the first eigenvector are non-negative
    assert np.all(eigenvectors[:, 0] >= 0)


def test_mixed_signs_1D():
    """
    Test that PCA correctly handles data with mixed signs.
    The dataset is unidimensional with both negative and positive points.
    Additional unambiguous points are added to guide the PCA solution,
    and PCA should generate a first eigenvector with all non-negative coordinates.
    """
    n_dim = 1
    n_points = 1000
    original_data = np.random.uniform(-1, 1, size=(n_points, n_dim))  # data points are both negative and positive

    # Add a second dimension with all zeros to facilitate PCA
    original_data = np.hstack((original_data, np.zeros((n_points, 1))))

    # Find the maximum absolute value along the first dimension
    max_abs_val = np.max(np.abs(original_data[:, 0]))

    # Add an unambiguous positive point and two ambiguous points
    additional_points = np.array([[max_abs_val + 0.5, 0], [max_abs_val + 1, 0], [-(max_abs_val + 1), 0]])
    original_data = np.vstack((original_data, additional_points))

    _, _, eigenvectors, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(original_data)

    # Validate that all coordinates of the first eigenvector are non-negative
    assert np.all(eigenvectors[:, 0] >= 0)


def test_symmetric_signs_1D():
    """
    Test that PCA handles sign ambiguity correctly when flipping the sign of the first eigenvector.
    The dataset is unidimensional with an equal number of positive and negative points,
    creating a symmetric distribution around zero.
    After PCA, the sign of the first eigenvector is flipped and the data is transformed back to the original space.
    The absolute values of the original and the transformed data should match.
    """
    n_dim = 1
    n_points = 1000
    positive_points = np.random.uniform(0, 1, size=(n_points, n_dim))  # generate positive data points
    negative_points = -positive_points  # create symmetric negative points
    original_data = np.vstack((positive_points, negative_points))

    # Augment the original data to 2D by adding a second dimension filled with zeros
    original_data = np.hstack((original_data, np.zeros((2*n_points, 1))))

    original_data, transformed_data, eigenvectors, _ = pca_tranform.perform_PCA_and_get_transformed_data_cov(original_data)

    # Validate that all coordinates of the first eigenvector are non-negative
    assert np.all(eigenvectors[:, 0] >= 0)

    # Flip the sign of the first eigenvector
    flipped_eigenvectors = eigenvectors.copy()
    flipped_eigenvectors[:, 0] *= -1

    # Project the transformed data back onto the original space
    flipped_data = np.dot(transformed_data, flipped_eigenvectors.T)

    # Validate that the absolute values of the original and flipped data match
    assert np.allclose(abs(original_data[:, 0]), abs(flipped_data[:, 0]), atol=1e-4, rtol=1e-4)
