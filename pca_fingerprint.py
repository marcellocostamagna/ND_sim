import numpy as np


def principal_components(points, masses):
    """
    Calculates the principal components (eigenvectors) of the covariance matrix of points with masses.

    Args:
        points (numpy.ndarray): A numpy array of shape (n, 3), representing n points in 3D space.
        masses (numpy.ndarray): A numpy array of shape (n,), representing the masses of each point.

    Returns:
        numpy.ndarray: A numpy array of shape (3, 3), containing the eigenvectors corresponding to the principal components.
    """
    points_with_masses = np.hstack((points, masses.reshape(-1, 1)))
    covariance_matrix = np.cov(points_with_masses, ddof=0, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, sorted_indices]

def max_distance_from_center(points, center):
    """
    Calculates the maximum distance between any point in the points array and the given center point.

    Args:
        points (numpy.ndarray): A numpy array of shape (n, 3), representing n points in 3D space.
        center (numpy.ndarray): A numpy array of shape (3,), representing the central point.

    Returns:
        float: The maximum distance between any point in the points array and the given center point.
    """
    distances = np.linalg.norm(points - center, axis=1)
    return np.max(distances)

def compute_reference_points(points, masses):
    """
    Computes the four reference points along the principal axes, each at the maximum distance from the center.

    Args:
        points (numpy.ndarray): A numpy array of shape (n, 3), representing n points in 3D space.
        masses (numpy.ndarray): A numpy array of shape (n,), representing the masses of each point.

    Returns:
        list: A list of four numpy arrays of shape (3,), each representing a reference point along the principal axes.
    """
    center = center_of_mass(points, masses)
    principal_axes = principal_components(points, masses)
    
    scale = max_distance_from_center(points, center)
    
    reference_points = [center + scale * axis for axis in principal_axes.T]
    return reference_points