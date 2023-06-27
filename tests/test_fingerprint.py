# Script to test the functions in the fingerprint.py file

import numpy as np
import pytest
from ..source import fingerprint

def test_get_reference_points():
    # Call the function
    points = fingerprint.get_reference_points()

    # Check if the function returns the correct number and type of points
    assert isinstance(points, np.ndarray)
    assert points.shape == (7, 6)  # 6D centroid plus 6 6D unit vectors

def test_compute_distances():
    # Define a simple 2D molecule_data to compute distances
    molecule_data = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    # Call your function
    distances = fingerprint.compute_distances(molecule_data)

    # Check if the distances are of correct shape and type
    assert isinstance(distances, np.ndarray)
    assert distances.shape == (molecule_data.shape[0], 7)  # Number of reference points is 7

def test_compute_statistics():
    # Define a simple 2D distances to compute statistics
    distances = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Call your function
    statistics = fingerprint.compute_statistics(distances)

    # Check if the statistics are of correct length and type
    assert isinstance(statistics, list)
    assert len(statistics) == distances.shape[0] * 3  # For each row, we have 3 statistics

def test_get_fingerprint():
    # Define a simple 2D molecule_data to compute fingerprint
    molecule_data = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    # Call your function
    fingerprint_data = fingerprint.get_fingerprint(molecule_data)

    # Check if the fingerprint_data are of correct length and type
    assert isinstance(fingerprint_data, list)
    assert len(fingerprint_data) == (molecule_data.shape[1] +1) * 3  # For each row, we have 3 statistics
