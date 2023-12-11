# ND_sim
# This file is part of ND_sim, which is licensed under the
# GNU Lesser General Public License v3.0 (or any later version).
# See the LICENSE file for more details.

# Script to calculate similarity scores between molecules and/or their fingerprints

from nd_sim.utils import * 
from nd_sim.fingerprint import *

def calculate_mean_absolute_difference(moments1: list, moments2:list):
    """
    Calculate the mean absolute difference between two lists.

    This function computes the mean of the absolute differences between 
    corresponding elements of two lists.

    Parameters
    ----------
    moments1 : list
        The first list of numerical values.
    moments2 : list
        The second list of numerical values, must be of the same length as moments1.

    Returns
    -------
    float
        The mean absolute difference between the two lists.
    """
    partial_score = 0
    for i in range(len(moments1)):
        partial_score += abs(moments1[i] - moments2[i])
    return partial_score / len(moments1)

def calculate_similarity_from_difference(partial_score):
    """
    Calculate similarity score from a difference score.

    This function converts a difference score into a similarity score using 
    a reciprocal function. The similarity score approaches 1 as the difference 
    score approaches 0, and it approaches 0 as the difference score increases.

    Parameters
    ----------
    partial_score : float
        The difference score, a non-negative number.

    Returns
    -------
    float
        The similarity score derived from the difference score.
    """
    return 1/(1 + partial_score)

def compute_similarity_score(fingerprint_1: list, fingerprint_2: list):
    """
    Calculate the similarity score between two fingerprints.
    
    Parameters
    ----------
    fingerprint_1 : list
        The fingerprint of the first molecule.
    fingerprint_2 : list
        The fingerprint of the second molecule.

    Returns
    -------
    float
        The computed similarity score.
    """
    partial_score = calculate_mean_absolute_difference(fingerprint_1, fingerprint_2)
    similarity = calculate_similarity_from_difference(partial_score)
    return similarity

def compute_similarity_from_ndarray(mol1_nd: np.array, mol2_nd: np.array, scaling='matrix', chirality=False):
    """
    Calculate the similarity score between two molecules represented as N-dimensional arrays.

    This function computes fingerprints for two molecules based on their N-dimensional array 
    representations and then calculates a similarity score between these fingerprints.
    
    Parameters
    ----------
    mol1_nd : numpy.ndarray
        The N-dimensional array representing the first molecule.
    mol2_nd : numpy.ndarray 
        The N-dimensional array representing the second molecule.
    scaling : str, float, or np.ndarray
        Specifies the scaling applied to reference points. If set to 'matrix' (default), 
        a scaling matrix is automatically computed based on the PCA-transformed data. 
        If a float is provided, it's used as a scaling factor. If a numpy.ndarray is provided, 
        it's used as a scaling matrix.
    chirality : bool, optional
        Consider chirality in the generation of fingerprints if set to True.

    Returns
    -------
    float
        The computed similarity score between the two molecules.
    """

    if chirality:
        f1, dimensionality1 = generate_fingerprint_from_data(mol1_nd, scaling=scaling, chirality=chirality)
        f2, dimensionality2 = generate_fingerprint_from_data(mol2_nd, scaling=scaling, chirality=chirality)
        
        if dimensionality1 != dimensionality2:
            print(f"WARNING: Comparison between molecules of different dimensionality: {dimensionality1} and {dimensionality2}.\n"
                   "The similarity score may not be accurate!")
    else:
        f1 = generate_fingerprint_from_data(mol1_nd, scaling=scaling, chirality=chirality)
        f2 = generate_fingerprint_from_data(mol2_nd, scaling=scaling, chirality=chirality)
        
    similarity_score = compute_similarity_score(f1, f2)
    return similarity_score

def compute_similarity(mol1, mol2, features=DEFAULT_FEATURES, scaling='matrix', removeHs=False, chirality=False):
    """
    Calculate the similarity score between two molecules using their n-dimensional fingerprints.
    
    This function generates fingerprints for two molecules based on their structures and a set of features, 
    and then computes a similarity score between these fingerprints.
    
    Parameters
    ----------
    mol1 : RDKit Mol
        The first RDKit molecule object.
    mol2 : RDKit Mol
        The second RDKit molecule object.
    features : dict, optional
        Dictionary of features to be considered. Default is DEFAULT_FEATURES.
    scaling : str, float, or np.ndarray
        Specifies the scaling applied to reference points. If set to 'matrix' (default), 
        a scaling matrix is automatically computed based on the PCA-transformed data. 
        If a float is provided, it's used as a scaling factor. If a numpy.ndarray is provided, 
        it's used as a scaling matrix.
    removeHs : bool, optional
        If True, hydrogen atoms are removed from the molecule before generating the fingerprint.
    chirality : bool, optional
        Consider chirality in the generation of fingerprints if set to True.

    Returns
    -------
    float
        The computed similarity score between the two molecules.
    """
    # Get molecules' fingerprints
    if chirality:
        f1, dimensionality1 = generate_fingerprint_from_molecule(mol1, features=features, scaling=scaling, removeHs=removeHs, chirality=chirality)
        f2, dimensionality2 = generate_fingerprint_from_molecule(mol2, features=features, scaling=scaling, removeHs=removeHs, chirality=chirality)
        
        # Compute similarity score
        if dimensionality1 != dimensionality2:
            print(f"WARNING: Comparison between molecules of different dimensionality: {dimensionality1} and {dimensionality2}.\n"
                   "The similarity score may not be accurate!")
    else:
        f1 = generate_fingerprint_from_molecule(mol1, features=features, scaling=scaling, removeHs=removeHs, chirality=chirality)
        f2 = generate_fingerprint_from_molecule(mol2, features=features, scaling=scaling, removeHs=removeHs, chirality=chirality)
   
    similarity_score = compute_similarity_score(f1, f2)             
    return similarity_score

