# ND_sim
# This file is part of ND_sim, which is licensed under the
# GNU Lesser General Public License v3.0 (or any later version).
# See the LICENSE file for more details.

# Script that provides the fingerprints of the molecules

import numpy as np
from scipy.spatial import distance
from scipy.stats import skew
from nd_sim.pre_processing import *
from nd_sim.pca_transform import *
from nd_sim.utils import *


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

def compute_distances(molecule_data: np.ndarray, scaling=None):
    """
    Calculate the Euclidean distance between each point in molecule_data and scaled reference points.
    
    This function computes the distances between each data point in a molecule and a set of reference points. 
    The reference points are scaled either by a factor or by a matrix depending on the type of the 'scaling' parameter.

    Parameters
    ----------
    molecule_data : np.ndarray
        Data of the molecule with each row representing a point.
    scaling : float, np.ndarray
        The scaling applied to the reference points.

    Returns
    -------
    np.ndarray
        A matrix of distances, where each element [i, j] is the distance between the i-th molecule data point and the j-th reference point.
    """
    reference_points = generate_reference_points(molecule_data.shape[1])
    
    if scaling is not None:
        if isinstance(scaling, (float, int)):
            reference_points *= scaling
        elif isinstance(scaling, np.ndarray):
            reference_points = np.dot(reference_points, scaling)
        else:
            raise TypeError("Scaling must be either a number (factor) or a numpy array (matrix).")
        
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
    skewness = np.nan_to_num(skew(distances, axis=1))

    statistics_matrix = np.vstack((means, std_devs, skewness)).T   
    statistics_list = [element for row in statistics_matrix for element in row]

    return statistics_list  

def generate_fingerprint_from_transformed_data(molecule_data: np.ndarray, scaling):
    """
    Compute a fingerprint from transformed molecular data.

    This function generates a molecular fingerprint based on distance statistics. 
    It calculates distances between the transformed molecular data points and a set 
    of reference points that are scaled using the provided scaling parameter. 
    The fingerprint is derived from these distance measurements.

    Parameters
    ----------
    molecule_data : np.ndarray
        Transformed data of the molecule, each row representing a transformed point.
    scaling : float, np.ndarray
        The scaling applied to the reference points. 

    Returns
    -------
    list
        Fingerprint derived from the distance measurements to scaled reference points.
    """

    distances = compute_distances(molecule_data, scaling)
    fingerprint = compute_statistics(distances.T)
    
    return fingerprint

def generate_fingerprint_from_data(molecule_data: np.array, scaling='matrix', chirality=False):
    """
    Generate a fingerprint directly from molecular data.

    This function takes the data of a molecule, applies PCA transformation considering chirality if needed,
    and computes the fingerprint.

    Parameters
    ----------
    molecule_data : np.array
        Data of the molecule, with each row representing a point.
    scaling : str, float, or np.ndarray
        Specifies the scaling applied to reference points. If set to 'matrix' (default), 
        a scaling matrix is automatically computed based on the PCA-transformed data. 
        If a float is provided, it's used as a scaling factor. If a numpy.ndarray is provided, 
        it's used as a scaling matrix.
    chirality : bool, optional
        Consider chirality in PCA transformation if set to True.

    Returns
    -------
    list or tuple
        Fingerprint of the molecule, and dimensionality if chirality is considered.
    """
    if chirality:
        transformed_data, dimensionality = compute_pca_using_covariance(molecule_data, chirality=chirality)
    else:
        transformed_data = compute_pca_using_covariance(molecule_data, chirality=chirality)
    # Determine scaling
    if scaling == 'matrix': # Default behaviour
        scaling_value = compute_scaling_matrix(transformed_data)
    else:
        scaling_value = scaling
        
    fingerprint = generate_fingerprint_from_transformed_data(transformed_data, scaling_value)
   
    return (fingerprint, dimensionality) if chirality else fingerprint

def generate_fingerprint_from_molecule(molecule, features=DEFAULT_FEATURES, scaling='matrix', chirality=False, removeHs=False):
    """
    Generate a fingerprint from a molecular structure using specified features and scaling.

    This function processes an RDKit molecule object to generate its fingerprint. 
    It first converts the molecule into n-dimensional data based on the specified features, 
    optionally removing hydrogen atoms if specified. A PCA transformation is then performed, 
    with an option to consider chirality. The reference points for distance calculation are 
    scaled as per the provided scaling parameter, and the fingerprint is computed based on these distances.

    Parameters
    ----------
    molecule : RDKit Mol
        RDKit molecule object.
    features : dict, optional
        Features to consider for molecule conversion. Default is DEFAULT_FEATURES.
    scaling : str, float, or np.ndarray
        Specifies the scaling applied to reference points. If 'matrix', a scaling matrix is computed and applied.
        If a float, it is used as a scaling factor. If a numpy.ndarray, it is directly used as the scaling matrix.
    chirality : bool, optional
        If True, chirality is considered in the PCA transformation, which can be important for distinguishing chiral molecules.
    removeHs : bool, optional
        If True, hydrogen atoms are removed from the molecule before conversion, focusing on heavier atoms.

    Returns
    -------
    list or tuple
        Fingerprint of the molecule. If chirality is considered, also returns the dimensionality post-PCA transformation.
    """
    
    # Convert molecule to n-dimensional data
    molecule_data = molecule_to_ndarray(molecule, features, removeHs=removeHs)
    
    # Fingerprint
    if chirality:
        fingerprint, dimensionality = generate_fingerprint_from_data(molecule_data, scaling=scaling, chirality=chirality)
        return fingerprint, dimensionality
    else:
        fingerprint = generate_fingerprint_from_data(molecule_data, scaling=scaling, chirality=chirality)
        return fingerprint
