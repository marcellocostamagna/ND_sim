# Script to test the capability of the pca_fingerprints and their comparison

import numpy as np
from trials.perturbations import *
from rdkit import Chem
from trials.utils import *
from trials.cov_fingerprint import *
from similarity_3d import *
from pca_fingerprint import *
from copy import deepcopy


def compute_6D_similarity_SVD(query, target):
    """Compute the similarity between two 6D fingerprints"""

    # Features normalization
    # query = taper_features(query)
    #query = normalize_features(query)
    #query = taper_features(query)

    # Delta features normalization
    query = taper_delta_features(query)
    #query = normalize_delta_features(query)
    
    # data = np.hstack((query['coordinates'], 
    #                   np.array(query['protons']).reshape(-1, 1), 
    #                   np.array(query['neutrons']).reshape(-1, 1),
    #                   np.array(query['electrons']).reshape(-1, 1)))
    
    data = np.hstack((query['coordinates'],
                        np.array(query['protons']).reshape(-1, 1),
                        np.array(query['delta_neutrons']).reshape(-1, 1),
                        np.array(query['formal_charges']).reshape(-1, 1)))
    
    
    # Rotate data
    #target['coordinates'] = rotate_points(target['coordinates'], 0, 0, 0)

    # Features normalization
    # target = taper_features(target)
    #target = normalize_features(target)
    #target = taper_features(target)

    # Delta features normalization
    target = taper_delta_features(target)
    #target = normalize_delta_features(target)
    
    # data1 = np.hstack((target['coordinates'], 
    #                    np.array(target['protons']).reshape(-1, 1),
    #                    np.array(target['neutrons']).reshape(-1, 1),
    #                    np.array(target['electrons']).reshape(-1, 1)))
    
    data1 = np.hstack((target['coordinates'],
                        np.array(target['protons']).reshape(-1, 1),
                        np.array(target['delta_neutrons']).reshape(-1, 1),
                        np.array(target['formal_charges']).reshape(-1, 1)))
    
    fingerprint_query,_ ,_ ,_ = get_pca_fingerprint(data)
    fingerprint_target, _, _ , _ = get_pca_fingerprint(data1)

    similarity = 1/(1 + calculate_nD_partial_score(fingerprint_query, fingerprint_target))
    return similarity


# Molecules 
#suppl = Chem.SDMolSupplier('coumarins_test.sdf') #, removeHs=False)
#suppl = Chem.SDMolSupplier('isomers_test.sdf', removeHs=False)
suppl = Chem.SDMolSupplier('swapping.sdf', removeHs=False)
molecules = [mol for mol in suppl if mol is not None]

molecules_info = {}

for i, molecule in enumerate(molecules):
    info = molecule_info(molecule)
    molecules_info[f'molecule_{i}'] = info

molecule_1 = molecules_info['molecule_2']
molecule_2 = molecules_info['molecule_3']

molecule_1_cov = deepcopy(molecules_info['molecule_0'])
molecule_2_cov = deepcopy(molecules_info['molecule_1'])

similarity_6d_SVD = compute_6D_similarity_SVD(molecule_1, molecule_2) 
similarity_6d_cov = compute_6D_similarity_cov(molecule_1_cov, molecule_2_cov)

print(f'Similarity_6d_SVD: {similarity_6d_SVD}')
print(f'Similarity_6d_cov: {similarity_6d_cov}')

