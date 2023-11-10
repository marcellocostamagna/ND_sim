# Script to calculate similarity scores between molecules and/or their fingerprints

from similarity.source.utils import * 
from similarity.source.fingerprint import *

def calculate_mean_absolute_difference(moments1: list, moments2:list):
    partial_score = 0
    for i in range(len(moments1)):
        partial_score += abs(moments1[i] - moments2[i])
    return partial_score / len(moments1)

def calculate_similarity_from_difference(partial_score):
    return 1/(1 + partial_score)

def compute_similarity_score(fingerprint_1, fingerprint_2):
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

def compute_similarity(mol1, mol2, features=DEFAULT_FEATURES, scaling_method='matrix', removeHs=False, chirality=False):
    """
    Calculate the similarity score between two molecules using their n-dimensional fingerprints.
    
    Parameters
    ----------
    mol1 : RDKit Mol
        The first RDKit molecule object.
    mol2 : RDKit Mol
        The second RDKit molecule object.
    features : dict, optional
        Dictionary of features to be considered. Default is DEFAULT_FEATURES.
    scaling_method : str, optional
        Specifies how to scale the data. It can be 'factor', 'matrix', or None.

    Returns
    -------
    float
        The computed similarity score between the two molecules.
    """
    # Get molecules' fingerprints
    f1 = generate_nd_molecule_fingerprint(mol1, features=features, scaling_method=scaling_method, removeHs=removeHs, chirality=chirality)
    f2 = generate_nd_molecule_fingerprint(mol2, features=features, scaling_method=scaling_method, removeHs=removeHs, chirality=chirality)
    # Compute similarity score
    similarity_score = compute_similarity_score(f1, f2)
    return similarity_score

