# Script to test the functions in the fingerprint.py file

import numpy as np
import pytest
from ..source import fingerprint

def test_get_reference_points():
    points = fingerprint.get_reference_points()
    assert isinstance(points, np.ndarray)
    assert points.shape == (7, 6)  # 6D centroid plus 6 6D unit vectors

def test_compute_distances():
    molecule_data = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    trivial_data = np.array([[0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1],])

    distances = fingerprint.compute_distances(molecule_data)
    trivial_distances = fingerprint.compute_distances(trivial_data)

    assert isinstance(distances, np.ndarray)
    assert distances.shape == (molecule_data.shape[0], 7) 

    # Check the calculated distances against the known distances
    assert np.isclose(trivial_distances[0, 0], 0) 
    assert np.isclose(trivial_distances[1, 0], np.sqrt(6)) 

def test_compute_statistics():
    # Define a simple 2D distances to compute statistics
    distances = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    trivial_distances = np.array([    
        [1, 1, 1],   # All values are the same: mean=1, std_dev=0, skewness=0
        [-1, 0, 1],  # Symmetric around 0: mean=0, std_dev=sqrt(2/3), skewness=0
        [0, 2, 4],   # Linearly increasing values: mean=2, std_dev=sqrt(8/3), skewness=0
    ])

    statistics = fingerprint.compute_statistics(distances)
    trivial_statistics = fingerprint.compute_statistics(trivial_distances)

    assert isinstance(statistics, list)
    assert len(statistics) == distances.shape[0] * 3  # For each row, we have 3 statistics

    # Check the calculated statistics against the known values
    assert np.isclose(trivial_statistics[0], 1)  # mean of first row
    assert np.isclose(trivial_statistics[1], 0)  # std_dev of first row
    assert np.isclose(trivial_statistics[2], 0)  # skewness of first row

    assert np.isclose(trivial_statistics[3], 0)  # mean of second row
    assert np.isclose(trivial_statistics[4], np.sqrt(2/3))  # std_dev of second row
    assert np.isclose(trivial_statistics[5], 0)  # skewness of second row

    assert np.isclose(trivial_statistics[6], 2)  # mean of third row
    assert np.isclose(trivial_statistics[7], np.sqrt(8/3))  # std_dev of third row
    assert np.isclose(trivial_statistics[8], 0)  # skewness of third row

def test_get_fingerprint():
    # Define a simple 2D molecule_data to compute fingerprint
    molecule_data = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    # Call your function
    fingerprint_data = fingerprint.get_fingerprint(molecule_data)

    # Check if the fingerprint_data are of correct length and type
    assert isinstance(fingerprint_data, list)
    assert len(fingerprint_data) == (molecule_data.shape[1] +1) * 3  # For each row, we have 3 statistics
