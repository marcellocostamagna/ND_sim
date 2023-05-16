# Python script collecting operration to modify, perturb and change the poin clouds 

import numpy as np

def rotate_points(points, angle1_deg, angle2_deg, angle3_deg):
    """
    Rotate a set of 3D points around the x, y, and z axes by the given angles.
    
    Parameters
    ----------
    points : numpy.ndarray
        An n x 3 array of 3D points.
    angle1_deg : float
        Rotation angle around the x-axis in degrees, range: [-180, 180].
    angle2_deg : float
        Rotation angle around the y-axis in degrees, range: [-180, 180].
    angle3_deg : float
        Rotation angle around the z-axis in degrees, range: [-180, 180].

    Returns
    -------
    numpy.ndarray
        An n x 3 array of rotated 3D points.
    """

    # Convert angles from degrees to radians
    angle1 = np.radians(angle1_deg)
    angle2 = np.radians(angle2_deg)
    angle3 = np.radians(angle3_deg)

    # Rotation matrix around the x-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle1), -np.sin(angle1)],
                   [0, np.sin(angle1), np.cos(angle1)]])

    # Rotation matrix around the y-axis
    Ry = np.array([[np.cos(angle2), 0, np.sin(angle2)],
                   [0, 1, 0],
                   [-np.sin(angle2), 0, np.cos(angle2)]])

    # Rotation matrix around the z-axis
    Rz = np.array([[np.cos(angle3), -np.sin(angle3), 0],
                   [np.sin(angle3), np.cos(angle3), 0],
                   [0, 0, 1]])

    # Combine the rotation matrices
    R_combined = np.dot(Rx, np.dot(Ry, Rz))

    # Apply the combined rotation matrix
    rotated_points = np.dot(points, R_combined)

    return rotated_points


def perturb_coordinates(points, decimal_place):
    """
    Apply random perturbations to the input 3D points based on the specified decimal place.

    Parameters:
    points (numpy.ndarray): A numpy array of shape (n, 3) representing the 3D coordinates of n points.
    decimal_place (int): The decimal place where the perturbation will take effect.

    Returns:
    numpy.ndarray: A new numpy array with the perturbed coordinates.
    """

    perturbed_points = np.zeros_like(points)
    for i, point in enumerate(points):
        perturbation_range = 10 ** -decimal_place
        perturbations = np.random.uniform(-perturbation_range * 9, perturbation_range * 9, point.shape)
        perturbed_points[i] = point + perturbations

    return perturbed_points



def scale_coordinates(points, s):
    """
    Scale the input 3D points by a given factor while maintaining the relative distances among the points.

    Parameters:
    points (numpy.ndarray): A numpy array of shape (n, 3) representing the 3D coordinates of n points.
    s (float): The scaling factor.

    Returns:
    numpy.ndarray: A new numpy array with the scaled coordinates.
    """

    scaled_points = points * s
    return scaled_points