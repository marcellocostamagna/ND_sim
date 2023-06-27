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











# def test_perform_PCA_on_known_data_4D():
#     n_dim = 4
#     variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
#     original_data = generate_multivariate_not_rotated_data(n_dim, variances)
#     _, _, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

#     # Test eigenvalues
#     # round the eigenvalues to be integers
#     eigenvalues = np.round(eigenvalues)
#     assert axes.shape[0] == eigenvalues.shape[0] == n_dim
#     np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(variances), decimal=1)
    
#     # Test eigenvectors
#     assert np.isclose(np.abs(np.dot(axes[0], [1, 0, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[1], [0, 1, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[2], [0, 0, 1, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[3], [0, 0, 0, 1])), 1, atol=1e-1)

# def test_perform_PCA_on_rotated_data_4D():
#     n_dim = 4
#     variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
#     rotation_angle = np.pi / 4
#     c, s = np.cos(rotation_angle), np.sin(rotation_angle)
#     rotation = np.array(((c, -s, 0, 0), (s, c, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

#     original_data = generate_multivariate_rotated_data(n_dim, variances, rotation)
#     _, _, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

#     # Test eigenvalues
#     # round the eigenvalues to be integers
#     eigenvalues = np.round(eigenvalues)
#     np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(variances), decimal=1)
    
#     # Test eigenvectors
#     assert np.isclose(np.abs(np.dot(axes[0], [1, 0, 0, 0])), np.abs(np.cos(rotation_angle)), atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[1], [0, 1, 0, 0])), np.abs(np.cos(rotation_angle)), atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[2], [0, 0, 1, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[3], [0, 0, 0, 1])), 1, atol=1e-1)

# #### 5D ####

# def test_perform_PCA_on_known_data_5D():
#     n_dim = 5
#     variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
#     original_data = generate_multivariate_not_rotated_data(n_dim, variances)
#     _, _, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

#     # Test eigenvalues
#     # round the eigenvalues to be integers
#     eigenvalues = np.round(eigenvalues)
#     assert axes.shape[0] == eigenvalues.shape[0] == n_dim
#     np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(variances), decimal=1)
    
#     # Test eigenvectors
#     assert np.isclose(np.abs(np.dot(axes[0], [1, 0, 0, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[1], [0, 1, 0, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[2], [0, 0, 1, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[3], [0, 0, 0, 1, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[4], [0, 0, 0, 0, 1])), 1, atol=1e-1)

# def test_perform_PCA_on_rotated_data_5D():
#     n_dim = 5
#     variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
#     rotation_angle = np.pi / 4
#     c, s = np.cos(rotation_angle), np.sin(rotation_angle)
#     rotation = np.array(((c, -s, 0, 0, 0), (s, c, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)))

#     original_data = generate_multivariate_rotated_data(n_dim, variances, rotation)
#     _, _, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

#     # Test eigenvalues
#     # round the eigenvalues to be integers
#     eigenvalues = np.round(eigenvalues)
#     np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(variances), decimal=1)
    
#     # Test eigenvectors
#     assert np.isclose(np.abs(np.dot(axes[0], [1, 0, 0, 0, 0])), np.abs(np.cos(rotation_angle)), atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[1], [0, 1, 0, 0, 0])), np.abs(np.cos(rotation_angle)), atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[2], [0, 0, 1, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[3], [0, 0, 0, 1, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[4], [0, 0, 0, 0, 1])), 1, atol=1e-1)

# #### 6D ####

# def test_perform_PCA_on_known_data_6D():
#     n_dim = 6
#     variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
#     original_data = generate_multivariate_not_rotated_data(n_dim, variances)
#     _, _, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

#     # Test eigenvalues
#     # round the eigenvalues to be integers
#     eigenvalues = np.round(eigenvalues)
#     assert axes.shape[0] == eigenvalues.shape[0] == n_dim
#     np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(variances), decimal=1)
    
#     # Test eigenvectors
#     assert np.isclose(np.abs(np.dot(axes[0], [1, 0, 0, 0, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[1], [0, 1, 0, 0, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[2], [0, 0, 1, 0, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[3], [0, 0, 0, 1, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[4], [0, 0, 0, 0, 1, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[5], [0, 0, 0, 0, 0, 1])), 1, atol=1e-1)

# def test_perform_PCA_on_rotated_data_6D():
#     n_dim = 6
#     variances = np.sort(np.random.randint(1, 50, n_dim))[::-1]
#     rotation_angle = np.pi / 4
#     c, s = np.cos(rotation_angle), np.sin(rotation_angle)
#     rotation = np.array(((c, -s, 0, 0, 0, 0), (s, c, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)))

#     original_data = generate_multivariate_rotated_data(n_dim, variances, rotation)
#     _, _, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

#     # Test eigenvalues
#     # round the eigenvalues to be integers
#     eigenvalues = np.round(eigenvalues)
#     np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(variances), decimal=1)

#     # Test eigenvectors
#     assert np.isclose(np.abs(np.dot(axes[0], [1, 0, 0, 0, 0, 0])), np.abs(np.cos(rotation_angle)), atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[1], [0, 1, 0, 0, 0, 0])), np.abs(np.cos(rotation_angle)), atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[2], [0, 0, 1, 0, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[3], [0, 0, 0, 1, 0, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[4], [0, 0, 0, 0, 1, 0])), 1, atol=1e-1)
#     assert np.isclose(np.abs(np.dot(axes[5], [0, 0, 0, 0, 0, 1])), 1, atol=1e-1)




# # def test_perform_PCA_and_get_transformed_data():
# #     # Define a simple 2D data to perform PCA
# #     original_data = np.array([[1, 2], [3, 4], [5, 6]])

# #     # Call your function
# #     original, transformed_data, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

   
# #     np.testing.assert_array_equal(original, original_data)

# #     # Check if axes and eigenvalues are of correct shape
# #     assert axes.shape[0] == eigenvalues.shape[0] == original_data.shape[1]

# #     # Add more assertions as needed to verify the correctness of transformed_data, axes, and eigenvalues


# #def generate_data(n_dim, variances, rotation):
# #     """
# #     Generate multivariate data with given variances and rotation
# #     """
# #     # Generate data along each axis
# #     data = np.diag(np.sqrt(variances)).dot(np.random.randn(n_dim, 1000)).T

# #     # Apply rotation
# #     data = data.dot(rotation)

# #     return data