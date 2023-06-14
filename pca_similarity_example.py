# Script to test the capability of the pca_fingerprints and their comparison

import numpy as np
import matplotlib.pyplot as plt
from perturbations import *
from rdkit import Chem
from utils import *
from fingerprints import *
from similarity_3d import *
from copy import deepcopy
from pca_fingerprint import *


def compute_6D_similarity(query, target):
    """Compute the similarity between two 6D fingerprints"""

    # points = translate_points_to_geometrical_center(query['coordinates'])
    # points1 = translate_points_to_geometrical_center(target['coordinates'])

    data = np.hstack((query['coordinates'], 
                      np.array(query['protons']).reshape(-1, 1), 
                      np.array(query['neutrons']).reshape(-1, 1),
                      np.array(query['electrons']).reshape(-1, 1)))
    
    # rotate data
    target['coordinates'] = rotate_points(target['coordinates'], 0, 0, 0)
    
    data1 = np.hstack((target['coordinates'], 
                       np.array(target['protons']).reshape(-1, 1),
                       np.array(target['neutrons']).reshape(-1, 1),
                       np.array(target['electrons']).reshape(-1, 1)))

    fingerprint_query = get_pca_fingerprint(data)
    fingerprint_target = get_pca_fingerprint(data1)

    similarity = 1/(1 + calculate_nD_partial_score(fingerprint_query, fingerprint_target))
    return similarity


# Molecules 
#suppl = Chem.SDMolSupplier('big_mol.sdf', removeHs=False)
#suppl = Chem.SDMolSupplier('coumarins_test.sdf', removeHs=False)
#suppl = Chem.SDMolSupplier('isomers_natural.sdf', removeHs=False)
suppl = Chem.SDMolSupplier('isomers_test.sdf', removeHs=False)
#suppl = Chem.SDMolSupplier('swapping.sdf', removeHs=False)
molecules = [mol for mol in suppl if mol is not None]
#print(len(molecules))

molecules_info = {}

for i, molecule in enumerate(molecules):
    info = molecule_info(molecule)
    molecules_info[f'molecule_{i}'] = info

molecule_1 = molecules_info['molecule_4']
molecule_2 = molecules_info['molecule_5']

# similarity_4d = compute_4D_similarity(molecule_1, molecule_2)
# similarity_5d = compute_5D_similarity(molecule_1, molecule_2)
similarity_6d = compute_6D_similarity(molecule_1, molecule_2) 

# print(f'Similarity_4d: {similarity_4d}')
# print(f'Similarity_5d: {similarity_5d}')
print(f'Similarity_6d: {similarity_6d}')








#def compute_4D_similarity(query, target):
#     """Compute the similarity between two 4D fingerprints"""

#     # points = translate_points_to_geometrical_center(query['coordinates'])
#     # points1 = translate_points_to_geometrical_center(target['coordinates'])

#     # add the protons to the coordinates
#     data = np.hstack((query['coordinates'], np.array(query['protons']).reshape(-1, 1)))
#     data1 = np.hstack((target['coordinates'], np.array(target['protons']).reshape(-1, 1)))
#     fingerprint_query = get_pca_fingerprint(data)
#     fingerprint_target = get_pca_fingerprint(data1)

#     similarity = 1/(1 + calculate_nD_partial_score(fingerprint_query, fingerprint_target))
#     return similarity

# def compute_5D_similarity(query, target):
#     """Compute the similarity between two 5D fingerprints"""

#     # points = translate_points_to_geometrical_center(query['coordinates'])
#     # points1 = translate_points_to_geometrical_center(target['coordinates'])

#     data = np.hstack((query['coordinates'], np.array(query['protons']).reshape(-1, 1), np.array(query['delta_neutrons']).reshape(-1, 1)))
#     data1 = np.hstack((target['coordinates'], np.array(target['protons']).reshape(-1, 1), np.array(target['delta_neutrons']).reshape(-1, 1)))

#     fingerprint_query = get_pca_fingerprint(data)
#     fingerprint_target = get_pca_fingerprint(data1)

#     similarity = 1/(1 + calculate_nD_partial_score(fingerprint_query, fingerprint_target))
#     return similarity