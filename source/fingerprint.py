# Script that provides the fingerprints of the molecules represented as a ND array

import numpy as np
from scipy.spatial import distance
from scipy.stats import skew

from similarity.source.pre_processing import *
from similarity.source.pca_tranform import *
from similarity.source.utils import *


def generate_reference_points(dimensionality):
    """
    Generate reference points in the n-dimensional space.
    
    Parameters
    ----------
    dimensionality : int
        The number of dimensions.

    Returns
    -------
    np.ndarray
        An array of reference points including the centroid and the points on each axis.
    """
    centroid = np.zeros(dimensionality)
    axis_points = np.eye(dimensionality)
    reference_points = np.vstack((centroid, axis_points))
    return reference_points

def compute_distances(molecule_data: np.ndarray, scaling_factor=None, scaling_matrix=None):
    """
    Calculate the Euclidean distance between each point in molecule_data and reference points.
    
    Parameters
    ----------
    molecule_data : np.ndarray
        Data of the molecule with each row representing a point.
    scaling_factor : float, optional
        Factor by which reference points are scaled.
    scaling_matrix : np.ndarray, optional
        Matrix by which reference points are scaled.

    Returns
    -------
    np.ndarray
        Matrix with distances between each point and each reference point.
    """
    reference_points = generate_reference_points(molecule_data.shape[1])
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
    """
    Calculate statistical moments (mean, standard deviation, skewness) for the given distances.
    
    Parameters
    ----------
    distances : np.ndarray
        Matrix with distances between each point and each reference point.

    Returns
    -------
    list
        A list of computed statistics.
    """
    means = np.mean(distances, axis=1)
    std_devs = np.std(distances, axis=1)
    skewness = skew(distances, axis=1)
    # check if skewness is NaN
    skewness[np.isnan(skewness)] = 0
    
    statistics_matrix = np.vstack((means, std_devs, skewness)).T   
    statistics_list = [element for row in statistics_matrix for element in row]

    return statistics_list  

def generate_molecule_fingerprint(molecule_data: np.ndarray, scaling_factor=None, scaling_matrix=None):
    """
    Compute a fingerprint for the provided molecular data based on distance statistics.

    Parameters
    ----------
    molecule_data : np.ndarray
        Data of the molecule with each row representing a point.
    scaling_factor : float, optional
        Factor by which reference points are scaled.
    scaling_matrix : np.ndarray, optional
        Matrix by which reference points are scaled.

    Returns
    -------
    list
        A fingerprint derived from the molecule data.
    """

    if scaling_factor is not None and scaling_matrix is not None:
        raise ValueError("Both scaling_factor and scaling_matrix provided. Please provide only one.")

    distances = compute_distances(molecule_data, scaling_factor, scaling_matrix)
    fingerprint = compute_statistics(distances.T)
    
    return fingerprint

def generate_nd_molecule_fingerprint(molecule, features=DEFAULT_FEATURES, scaling_method='factor', scaling_value=None, chirality=False, removeHs=False):
    """
    Generate a fingerprint for the given molecule.
    
    This function converts a molecule to n-dimensional data, performs PCA transformation,
    scales the data if needed, and then computes the fingerprint based on distance statistics.

    Parameters
    ----------
    molecule : RDKit Mol
        RDKit molecule object.
    features : dict, optional
        Dictionary of features to be considered. Default is DEFAULT_FEATURES.
    scaling_method : str, optional
        Specifies how to scale the data. It can be 'factor', 'matrix', or None.
    scaling_value : float or numpy.ndarray, optional
        Value used for scaling. If method is 'factor', it should be a number.
        If method is 'matrix', it should be an array. Default is None.

    Returns
    -------
    list
        A fingerprint derived from the molecule data.
    """
    
    # Convert molecule to n-dimensional data
    molecule_data = molecule_to_ndarray(molecule, features, removeHs=removeHs)
    # PCA transformation
    transformed_data = compute_pca_using_covariance(molecule_data, chirality=chirality)
    # Determine scaling
    if scaling_method == 'factor':
        if scaling_value is None:
            scaling_value = compute_scaling_factor(transformed_data)
        fingerprint = generate_molecule_fingerprint(transformed_data, scaling_factor=scaling_value)
    elif scaling_method == 'matrix':
        if scaling_value is None:
            scaling_value = compute_scaling_matrix(transformed_data)
        fingerprint = generate_molecule_fingerprint(transformed_data, scaling_matrix=scaling_value)
    elif scaling_method is None:
        fingerprint = generate_molecule_fingerprint(transformed_data)
    else:
        raise ValueError(f"Invalid scaling method: {scaling_method}. Choose 'factor' or 'matrix'.")
    
    return fingerprint
