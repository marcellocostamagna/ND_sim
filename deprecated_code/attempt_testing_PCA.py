import numpy as np
# import pytest
# from ..source import pca
from scipy.spatial.transform import Rotation as R

def generate_data(n_dim, variances, rotation):
    """
    Generate multivariate data with given variances and rotation
    """
    # Generate data along each axis
    data = np.diag(np.sqrt(variances)).dot(np.random.randn(n_dim, 100)).T

    # Apply rotation
    data = data.dot(rotation)

    return data

def test_perform_PCA_and_get_transformed_data_2D():
    n_dim = 2
    variances = np.array([4, 1])
    rotation_angle = np.pi / 4
    rotation = R.from_euler('z', rotation_angle).as_matrix()

    original_data = generate_data(n_dim, variances, rotation)

    original, transformed_data, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

    np.testing.assert_array_equal(original, original_data)
    assert axes.shape[0] == eigenvalues.shape[0] == n_dim

    transformed_data_rotated_back = transformed_data.dot(R.from_matrix(axes.T).as_euler('z'))
    np.testing.assert_array_almost_equal(transformed_data_rotated_back, original_data, decimal=2)

    np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(variances), decimal=2)




# def test_perform_PCA_and_get_transformed_data():
#     # Define a simple 2D data to perform PCA
#     original_data = np.array([[1, 2], [3, 4], [5, 6]])

#     # Call your function
#     original, transformed_data, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

   
#     np.testing.assert_array_equal(original, original_data)

#     # Check if axes and eigenvalues are of correct shape
#     assert axes.shape[0] == eigenvalues.shape[0] == original_data.shape[1]

#     # Add more assertions as needed to verify the correctness of transformed_data, axes, and eigenvalues
