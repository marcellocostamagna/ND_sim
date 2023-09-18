# Script to compare two fingerprints and provide a single-value similarity measure
from similarity.source.utils import * 
from similarity.source.fingerprint import *

def calculate_partial_score(moments1: list, moments2:list):
    partial_score = 0
    for i in range(len(moments1)):
        partial_score += abs(moments1[i] - moments2[i])
    return partial_score / len(moments1)

def get_similarity_measure(partial_score):
    return 1/(1 + partial_score)

def get_similarity_score(fingerprint_1, fingerprint_2):
    partial_score = calculate_partial_score(fingerprint_1, fingerprint_2)
    similarity = get_similarity_measure(partial_score)
    return similarity

def compute_similarity(mol1, mol2, features=DEFAULT_FEATURES, scaling_method='matrix'):
    # Get molecules' fingerprints
    f1 = get_nd_fingerprint(mol1, features=features, scaling_method=scaling_method)
    f2 = get_nd_fingerprint(mol2, features=features, scaling_method=scaling_method)
    # Compute similarity score
    similarity_score = get_similarity_score(f1, f2)
    return similarity_score

