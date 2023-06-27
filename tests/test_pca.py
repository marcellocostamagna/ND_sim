import numpy as np
import pytest
from ..source import pca

def test_perform_PCA_and_get_transformed_data():
    # Define a simple 2D data to perform PCA
    original_data = np.array([[1, 2], [3, 4], [5, 6]])

    # Call your function
    original, transformed_data, axes, eigenvalues = pca.perform_PCA_and_get_transformed_data(original_data)

    # Assertions to check if function is working correctly

    # Check if original data is unchanged
    np.testing.assert_array_equal(original, original_data)

    # Check if axes and eigenvalues are of correct shape
    assert axes.shape[0] == eigenvalues.shape[0] == original_data.shape[1]

    # Add more assertions as needed to verify the correctness of transformed_data, axes, and eigenvalues
