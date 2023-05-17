import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
import math as m
from coordinates import *
from visualization import *
from utils import *
from rdkit import Chem

from scipy.spatial import KDTree


#### New KDTREE fingerprint implementation ####

def molecule_info(molecule):

    elements, masses, protons, neutrons, electrons, coordinates = get_atoms_info(molecule)
    info = {'elements': elements,
            'masses': masses,
            'protons': protons, 
            'neutrons': neutrons, 
            'electrons': electrons, 
            'coordinates': coordinates}
    return info


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
    return distances, indices, differences, corresponding_points


def size_similarity(N_query, N_target):
    """Compute the size similarity between two sets of points"""
    similarity_s = min(N_query, N_target) / max(N_query, N_target)
    return similarity_s

def positional_similarity(differences):
    """Compute the positional similarity between two sets of points"""
    # Compute the mean of the distances
    mean = np.mean(differences)
    similarity_r = 1 / (1 + mean)
    return similarity_r


def formula_isotopic_charge_similarity(molecule1: dict, molecule2: dict):
    """Compute the formula similarity between two sets of points"""
    for i in range(0, len(molecule1['elements'])):
        p_diff = abs(molecule1['protons'][i] - molecule2['protons'][i])
        if p_diff == 0:
            n_diff = abs(molecule1['neutrons'][i] - molecule2['neutrons'][i])
            e_diff = molecule1['electrons'][i] - molecule2['electrons'][i]
        else:
            # Get the neutrons and elctron of the most abundant isotope of the element in molecule 2
            element = molecule2['elements'][i]
            neutrons = Chem.GetMassDifference(element)
            electrons = Chem.GetAtomicNum(element)
            n_diff = molecule2['neutrons'][i] - neutrons
            e_diff = molecule2['electrons'][i] - electrons    

    similarity_f =1/(1 + p_diff)
    similarity_n =1/(1 + n_diff)
    similarity_e =1/(1 + e_diff)
    return similarity_f, similarity_n, similarity_e

def reduced_formula_isotopic_charge_similarity(molecule1: dict, molecule2: dict):
    """Compute the reduced formula similarity between two sets of points"""
    # Mass of the moleules
    mass1 = sum(molecule1['masses'])
    mass2 = sum(molecule2['masses'])
    similarity_f = min(mass1, mass2) / max(mass1, mass2)
    # TODO:Check if there are isotopes in the molecules
    isotopes1 = []
    isotopes2 = []
    for i, neutrons in enumerate(molecule1['neutrons']):
        std_neutrons = Chem.GetMassDifference(molecule1['elements'][i])
        diff_n = abs(neutrons - std_neutrons)
        isotopes1.append(diff_n)
    for j, neutrons in enumerate(molecule2['neutrons']):
        std_neutrons = Chem.GetMassDifference(molecule2['elements'][j])
        diff_n = abs(neutrons - std_neutrons)
        isotopes2.append(diff_n)

    diff1 = abs(len(isotopes1) - len(isotopes2))
    diff2 = abs(sum(isotopes1) - sum(isotopes2))
    similarity_n = 1 / (1 + diff1 + diff2)
    
    # Charges of the molecules
    charge1 = sum(molecule1['protons']) - sum(molecule1['electrons'])
    charge2 = sum(molecule2['protons']) - sum(molecule2['electrons'])

    similarity_e = 1/ (1 + abs(charge1 - charge2)/ len(molecule1['elements']))

    return similarity_f, similarity_e, similarity_n

    


################################################

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