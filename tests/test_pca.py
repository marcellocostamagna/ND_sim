import numpy as np
import pytest
from ..source import pca
from scipy.spatial.transform import Rotation as R


def generate_multivariate_rotated_data(n_dim, variances, rotation):
    mean = np.zeros(n_dim)
    cov = np.diag(variances)
    data = np.random.multivariate_normal(mean, cov, 10000)
    data = data.dot(rotation)
    return data

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
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)
    
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

def test_perform_PCA_on_rotated_data_2D():
    n_dim = 2
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    rotation_angle = np.random.uniform(0, np.pi)
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation = np.array(((c, -s), (s, c)))

    original_data = generate_multivariate_rotated_data(n_dim, variances, rotation)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

     # Test eigenvectors are orthogonal
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            dot_product = np.dot(axes[i], axes[j])
            assert np.isclose(dot_product, 0, atol=1e-1)

    # Test eigenvalues
    # round the eigenvalues to be integers
     # Test variance along each principal component
    std_axes = np.eye(n_dim)
    for i in range(n_dim):
        projected_data = np.dot(transformed_data, std_axes[i])
        variance = np.var(projected_data)
        # assert np.isclose(variance, variances[i], atol=2)
        relative_error = np.abs(variance - variances[i]) / variances[i]
        assert relative_error < 0.1

    # Apply inverse rotation to eigenvectors and compare to original basis vectors
    # NOTE: we compare the module of the dot product to 1 because the eigenvectors
    # can be flipped
    inverse_rotation = np.linalg.inv(rotation)
    rotated_axes = np.dot(axes, inverse_rotation)
    for i in range(n_dim):
        matches = [np.isclose(np.abs(np.dot(rotated_axes[i], basis_vector)), 1 , atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)

#### 3D ####

def test_perform_PCA_on_known_data_3D():
    n_dim = 3
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

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

def test_perform_PCA_on_rotated_data_3D():
    n_dim = 3
    variances = np.sort(np.random.uniform(1, 50, n_dim))[::-1]
    rotation_angles = np.random.uniform(0, np.pi, 3)
    rotation = R.from_euler('xyz', rotation_angles).as_matrix()

    original_data = generate_multivariate_rotated_data(n_dim, variances, rotation)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

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

    # Apply inverse rotation to eigenvectors and compare to original basis vectors
    # NOTE: we compare the module of the dot product to 1 because the eigenvectors
    # can be flipped
    inverse_rotation = np.linalg.inv(rotation)
    rotated_axes = np.dot(axes, inverse_rotation)
    for i in range(n_dim):
        matches = [np.isclose(np.abs(np.dot(rotated_axes[i], basis_vector)), 1, atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)



##### 4D ####

def test_perform_PCA_on_known_data_4D():
    n_dim = 4
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

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

def test_perform_PCA_on_rotated_data_4D():
    n_dim = 4
    variances = np.sort(np.random.uniform(1, 50, n_dim))[::-1]
    rotation_angles = np.random.uniform(0, np.pi, 3)
    rotation_3D = R.from_euler('xyz', rotation_angles).as_matrix()

    rotation = np.eye(n_dim)
    rotation[:3, :3] = rotation_3D

    original_data = generate_multivariate_rotated_data(n_dim, variances, rotation)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

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

    # Apply inverse rotation to eigenvectors and compare to original basis vectors
    # NOTE: we compare the module of the dot product to 1 because the eigenvectors
    # can be flipped
    inverse_rotation = np.linalg.inv(rotation)
    rotated_axes = np.dot(axes, inverse_rotation)
    for i in range(n_dim):
        matches = [np.isclose(np.abs(np.dot(rotated_axes[i], basis_vector)), 1, atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)

##### 5D ####

def test_perform_PCA_on_known_data_5D():
    n_dim = 5
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

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

def test_perform_PCA_on_rotated_data_5D():
    n_dim = 5
    variances = np.sort(np.random.uniform(1, 50, n_dim))[::-1]
    rotation_angles = np.random.uniform(0, np.pi, 3)
    rotation_3D = R.from_euler('xyz', rotation_angles).as_matrix()

    rotation = np.eye(n_dim)
    rotation[:3, :3] = rotation_3D

    original_data = generate_multivariate_rotated_data(n_dim, variances, rotation)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

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

    # Apply inverse rotation to eigenvectors and compare to original basis vectors
    # NOTE: we compare the module of the dot product to 1 because the eigenvectors
    # can be flipped
    inverse_rotation = np.linalg.inv(rotation)
    rotated_axes = np.dot(axes, inverse_rotation)
    for i in range(n_dim):
        matches = [np.isclose(np.abs(np.dot(rotated_axes[i], basis_vector)), 1, atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)

##### 6D ####

def test_perform_PCA_on_known_data_6D():
    n_dim = 6
    variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
    original_data = generate_multivariate_not_rotated_data(n_dim, variances)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

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

def test_perform_PCA_on_rotated_data_6D():
    n_dim = 6
    variances = np.sort(np.random.uniform(1, 50, n_dim))[::-1]
    rotation_angles = np.random.uniform(0, np.pi, 3)
    rotation_3D = R.from_euler('xyz', rotation_angles).as_matrix()

    rotation = np.eye(n_dim)
    rotation[:3, :3] = rotation_3D

    original_data = generate_multivariate_rotated_data(n_dim, variances, rotation)
    _, transformed_data, axes, _ = pca.perform_PCA_and_get_transformed_data(original_data)

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

    # Apply inverse rotation to eigenvectors and compare to original basis vectors
    # NOTE: we compare the module of the dot product to 1 because the eigenvectors
    # can be flipped
    inverse_rotation = np.linalg.inv(rotation)
    rotated_axes = np.dot(axes, inverse_rotation)
    for i in range(n_dim):
        matches = [np.isclose(np.abs(np.dot(rotated_axes[i], basis_vector)), 1, atol=0.3) for basis_vector in np.eye(n_dim)]
        assert any(matches)

