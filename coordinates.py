import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
import math as m
from rdkit import Geometry as geom

### Center of mass and geometrical center ###
def compute_center_of_mass(points, masses):
    return np.average(points, axis=0, weights=masses)

def compute_geometrical_center(points):
    return np.mean(points, axis=0)


### Coordinates transformations ###
def translate_points_to_center_of_mass(points, masses):
    # Calculate the center of mass
    center_of_mass = np.average(points, axis=0, weights=masses)
    # Translate the points so that the center of mass is at the origin
    translated_points = points - center_of_mass
    return translated_points

def translate_points_to_geometrical_center(points):
    # Calculate the geometrical center
    geometrical_center = np.mean(points, axis=0)
    # Translate the points so that the geometrical center is at the origin
    translated_points = points - geometrical_center
    return translated_points

### Tensor of inertia and principal axes ###

def compute_inertia_tensor(points, masses, center_of_mass):
    inertia_tensor = np.zeros((3, 3))
    for point, mass in zip(points, masses):
        r = point - center_of_mass
        inertia_tensor += mass * (np.eye(3) * np.dot(r, r) - np.outer(r, r))

    return inertia_tensor

def compute_inertia_tensor_no_masses(points):
    geometrical_center = compute_geometrical_center(points)
    inertia_tensor = np.zeros((3, 3))
    for point in points:
        r = point - geometrical_center
        inertia_tensor += (np.eye(3) * np.dot(r, r) - np.outer(r, r))

    return inertia_tensor

def compute_principal_axes(inertia_tensor, points):
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    principal_axes = eigenvectors.T

    # If one of the eigenvalues is zero, the corresponding eigenvector is redefined as 
    # a vector orthogonal to the other two eigenvectors with the positive direction
    # pointing towards the more massive side of the cloud of points.
    for i, eigenvalue in enumerate(eigenvalues):
        axis = principal_axes[i]
        if abs(eigenvalue) <= 1e-2:
            eigenvalues[i] = 1e-2
            axis = np.cross(principal_axes[(i+1)%3], principal_axes[(i+2)%3])
            
        # Project the coordinates of the cloud of points onto the fake axis
        # TODO: Problem with sign
        projections = np.sign(np.dot(points, axis))
        # projections without masses
        sum = np.sum(projections)
        if sum != 0:
            axis = axis * np.sign(sum)
        #axis = axis / np.linalg.norm(axis)
        principal_axes[i] = axis
    
    handedness = compute_handedness(principal_axes, eigenvalues)
    #print("Handedness: ", handedness)

    if handedness == "left-handed":
         principal_axes[0] = -principal_axes[0]

    return principal_axes, eigenvalues

def compute_handedness(principal_axes, eigenvalues):

    # Sort the principal axes based on their eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    sorted_axes = principal_axes[sorted_indices]

    triple_scalar_product = np.dot(principal_axes[0], np.cross(principal_axes[1], principal_axes[2]))
    if triple_scalar_product > 0:
        return "right-handed"
    else:
        return "left-handed"

### Covariance and principal axes ###

def covariance_principal_components(points):
    """
    Calculates the principal components (eigenvectors) of the covariance matrix of points 
    """
    covariance_matrix = np.cov(points, ddof=0, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, sorted_indices]

### Fixed reference system ###
def compute_new_coordinates(principal_axes, points):
    # Translate the points to the center of mass
    #translated_points = translate_points_to_geometrical_center(points)
    # Rotate the points so that the principal axes are aligned with the axes of the reference system
    points_new_coord = points @ principal_axes.T
    return points_new_coord

### Distances from the center of mass ###

def max_distance_from_center_of_mass(points, center_of_mass):
    distances = np.linalg.norm(points - center_of_mass, axis=1)
    return np.max(distances)

def max_distance_from_geometrical_center(points):
    distances = np.linalg.norm(points, axis=1)
    return np.max(distances)

### Distances from axis ###

def max_distance_from_axis(points, axis):
    distances = np.abs(np.dot(points, axis))
    return np.max(distances)

### Reference points ###

### methods based on the principal axes ###
def generate_reference_points(center_of_mass, principal_axes, max_distance):
    center = geom.Point3D(*center_of_mass.tolist())
    points = [center]
    
    for axis in principal_axes:
        point = center_of_mass + max_distance * (axis/np.linalg.norm(axis))
        # Point to 3dPoint
        point = point.tolist()
        point = geom.Point3D(*point)

        points.append(point)
    
    return points

### methods baseed on closest and furthest concepts ###
# TODO: Implement the methods based on the closest and furthest concepts from 
# similarity_3d.py