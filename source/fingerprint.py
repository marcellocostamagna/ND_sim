# Script that provides the fingerprints of the molecules represented as a 6D array

import numpy as np
from scipy.spatial import distance
from scipy.stats import skew

from similarity.source.pre_processing import *
from similarity.source.pca_tranform import *
from similarity.source.utils import *


def get_reference_points(dimensionality):
    """
    Returns the reference points
    """
    centroid = np.zeros(dimensionality)
    axis_points = np.eye(dimensionality)
    reference_points = np.vstack((centroid, axis_points))
    return reference_points

def compute_distances(molecule_data: np.ndarray, scaling_factor=None, scaling_matrix=None):
    """
    Computes the Euclidean distance of each point from each reference point
    """
    reference_points = get_reference_points(molecule_data.shape[1])

    # Scale the reference points based on provided scaling factor or matrix
    if scaling_factor is not None:
        reference_points *= scaling_factor
    elif scaling_matrix is not None:
        reference_points = np.dot(reference_points, scaling_matrix)

    distances = np.empty((molecule_data.shape[0], len(reference_points)))
    for i, point in enumerate(molecule_data):
        for j, ref_point in enumerate(reference_points):
            distances[i, j] = distance.euclidean(point, ref_point)
    return distances


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

def get_fingerprint(molecule_data: np.ndarray, scaling_factor=None, scaling_matrix=None):
    """Computes the fingerprint of a given data set.
    Parameters:
    - molecule_data (np.ndarray): The data set of molecules.
    - scaling_factor (float, optional): The scaling factor to use. Default is None.
    - scaling_matrix (np.ndarray, optional): The scaling matrix to use. Default is None.
    
    Returns:
    - list: The fingerprint of the given molecule data.
    """

    if scaling_factor is not None and scaling_matrix is not None:
        raise ValueError("Both scaling_factor and scaling_matrix provided. Please provide only one.")

    # Compute the Euclidean distance of each point from each reference point (which are fixed)
    distances = compute_distances(molecule_data)
    # Compute the statistics of the distances (mean, std_dev, skewness)
    fingerprint = compute_statistics(distances.T)
    
    return fingerprint


def get_nd_fingerprint(molecule, features=DEFAULT_FEATURES, scaling_method='factor'):
    """
    Computes the fingerprint for the given molecule using all the provided steps.
    
    Parameters:
    - molecule: RDKit molecule object.
    - features: Dictionary of features to be used. Default is DEFAULT_FEATURES.
    - scaling_method: 'factor' to use scaling factor or 'matrix' to use scaling matrix.

    Returns:
    - Fingerprint of the molecule.
    """
    
    # Convert molecule to n-dimensional data
    molecule_data = mol_nd_data(molecule, features)
    
    # PCA transformation
    _, transformed_data, _, _ = perform_PCA_and_get_transformed_data_cov(molecule_data)
    
    # Determine scaling
    if scaling_method == 'factor':
        scaling = compute_scaling_factor(transformed_data)
        fingerprint = get_fingerprint(transformed_data, scaling_factor=scaling)
    elif scaling_method == 'matrix':
        scaling = compute_scaling_matrix(transformed_data)
        fingerprint = get_fingerprint(transformed_data, scaling_matrix=scaling)
    else:
        raise ValueError(f"Invalid scaling method: {scaling_method}. Choose 'factor' or 'matrix'.")
    
    return fingerprint
