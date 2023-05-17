import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
import math as m

def compute_center_of_mass(points, masses):
    return np.average(points, axis=0, weights=masses)

def compute_geometrical_center(points):
    return np.mean(points, axis=0)

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

def max_distance_from_center_of_mass(points, center_of_mass):
    distances = np.linalg.norm(points - center_of_mass, axis=1)
    return np.max(distances)

def max_distance_from_geometrical_center(points, geometrical_center):
    distances = np.linalg.norm(points, axis=1)
    return np.max(distances)

def generate_reference_points(center_of_mass, principal_axes, max_distance):
    points = [center_of_mass]
    
    for axis in principal_axes:
        point = center_of_mass + max_distance * (axis/np.linalg.norm(axis))
        points.append(point)
    
    return points

def compute_inertia_tensor(points, masses, center_of_mass):
    inertia_tensor = np.zeros((3, 3))
    for point, mass in zip(points, masses):
        r = point - center_of_mass
        inertia_tensor += mass * (np.eye(3) * np.dot(r, r) - np.outer(r, r))
        #inertia_tensor *= -1

    return inertia_tensor

def compute_principal_axes(inertia_tensor, points, masses):
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    principal_axes = eigenvectors.T

    # If one of the eigenvalues is zero, the corresponding eigenvector is redefined as 
    # a vector orthogonal to the other two eigenvectors with the positive direction
    # pointing towards the more massive side of the cloud of points.
    for i, eigenvalue in enumerate(eigenvalues):
        axis = principal_axes[i]
        if abs(eigenvalue) <= 1e-2:
            eigenvalues[i] = 1e-6
            axis = np.cross(principal_axes[(i+1)%3], principal_axes[(i+2)%3])
            
        # Project the coordinates of the cloud of points onto the fake axis
        # TODO: Problem with sign
        projections = np.sign(np.dot(points, axis))
        # Compute the weighted sum of projections using the masses
        # weighted_sum = np.dot(projections, masses)
        # # projections without masses
        sum = np.sum(projections)
        # # Normalize the fake axis
        # if weighted_sum == 0:
        #     weighted_sum = 1
        # axis = axis * np.sign(weighted_sum)
        # consdiering no masses
        axis = axis * np.sign(sum)
        #axis = axis / np.linalg.norm(axis)
        principal_axes[i] = axis
    
    handedness = compute_handedness(principal_axes, eigenvalues)
    print("Handedness: ", handedness)

    if handedness == "left-handed":
         principal_axes[1] = -principal_axes[1]

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