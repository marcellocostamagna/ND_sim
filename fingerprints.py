import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
import math as m
from coordinates import *
from visualization import *

from scipy.spatial import KDTree


#### New KDTREE implementation ####

def compute_matching(points_query, points_target):
    """Compute the matching between two sets of points"""
    # Build the KDTree
    tree = KDTree(points_target)
    # Query the tree
    distances, indices = tree.query(points_query)

    # Use the indices to access the corresponding points in points_target
    corresponding_points = points_target[indices]

    # Now you can compare points_query and corresponding_points
    differences = points_query - corresponding_points
    return distances, indices, differences

def compute_distances(points, reference_points):
    """Compute the distance of each point to the 4 refernce points"""
    num_points = points.shape[0]
    num_ref_points = len(reference_points)
    distances = np.zeros((num_ref_points, num_points))
    
    for i, point in enumerate(points):
        for j, ref_point in enumerate(reference_points):
            distances[j, i] = np.linalg.norm(point - ref_point)
            
    return distances  


def compute_weighted_distances(points, masses, reference_points):
    """Compute the mass-weigthed distance of each point to the 4 refernce points"""
    num_points = points.shape[0]
    num_ref_points = len(reference_points)
    weighted_distances = np.zeros((num_ref_points, num_points))

    for i, (point, mass) in enumerate(zip(points, masses)):
        for j, ref_point in enumerate(reference_points):
            if mass == 0:
                mass = 1
            weighted_distances[j, i] = ((m.log(mass))) * np.linalg.norm(point - ref_point)

    return weighted_distances


def compute_statistics(distances):
    means = np.mean(distances, axis=1)
    std_devs = np.std(distances, axis=1)
    skewness = skew(distances, axis=1)
    # check if skewness is nan
    skewness[np.isnan(skewness)] = 0
    
    statistics_matrix = np.vstack((means, std_devs, skewness)).T 
    # add all rows to a list   
    statistics_list = [element for row in statistics_matrix for element in row]

    
    return statistics_list  

def compute_fingerprint(points, masses, n_prot, n_neut, n_elec):

    print(n_neut)
    #particles = [n_prot, n_neut, n_elec]
    fingerprints = []

    #points, center_of_mass = translate_points_to_center_of_mass(points, masses), [0,0,0]
    points, geometrical_center = translate_points_to_geometrical_center(points), [0,0,0]

    #inertia_tensor = compute_inertia_tensor(points, masses, center_of_mass)
    weights = np.ones(len(points))
    inertia_tensor = compute_inertia_tensor(points, weights, geometrical_center)

    principal_axes, eigenvalues = compute_principal_axes(inertia_tensor, points, masses)

    #max_distance = max_distance_from_center_of_mass(points, center_of_mass)
    max_distance = max_distance_from_geometrical_center(points, geometrical_center)

    #reference_points = generate_reference_points(center_of_mass, principal_axes, max_distance)
    reference_points = generate_reference_points(geometrical_center, principal_axes, max_distance)
    # compute distances
    #distances = compute_distances(points, reference_points)
    # compute weighted distances
    proton_distances = compute_weighted_distances(points, n_prot, reference_points)
    neutron_distances = compute_weighted_distances(points, n_neut, reference_points)
    electron_distances = compute_weighted_distances(points, n_elec, reference_points)
    # compute statistics
    # statistics_matrix, fingerprint_1 = compute_statistics(distances)
    # statistics_matrix, fingerprint_2 = compute_statistics(weighted_distances)
    proton_fingerprint = compute_statistics(proton_distances)
    print(proton_fingerprint)
    neutron_fingerprint = compute_statistics(neutron_distances)
    electron_fingerprint = compute_statistics(electron_distances)
    
    # print("Center of mass:", center_of_mass)
    # # print("Inertia tensor:", inertia_tensor)
    # print("Principal axes:", principal_axes)
    # print("Eigenvalues:", eigenvalues)
    # # print("Distances:", distances)
    # # print("Fingerprint of regular distances:", fingerprint_1)
    # # print("Fingerprint of weighted distances:", fingerprint_2)
    # print(f'Handedness: {compute_handedness(principal_axes, eigenvalues)}')

    # If the third eigenvalue less than 0.001, we still need to visulaize the third axis
    if np.abs(eigenvalues[2]) < 0.001:
        eigenvalues[2] = 0.5 * eigenvalues[1]

    #visualize(points, n_prot, center_of_mass, principal_axes, eigenvalues, max_distance, reference_points)
    visualize(points, n_prot, geometrical_center, principal_axes, eigenvalues, max_distance, reference_points)

    fingerprints = [proton_fingerprint, neutron_fingerprint, electron_fingerprint]

    return fingerprints