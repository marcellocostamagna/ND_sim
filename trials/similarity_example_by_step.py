# Scripts for calculating between all pairs of molecules in a list of molecules.
# All steps of the similarity method are comptued sequentially 

import numpy as np  
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from similarity.trials.perturbations import *
import os 

np.set_printoptions(precision=4, suppress=True)

cwd = os.getcwd()
# PRE-PROCESSING
# List of molecules from SDF file
molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/swapping_sim_2d.sdf', removeHs=False, sanitize=True)

### ROTATE MOLECULES ###
rotated_molecules = []
for molecule in molecules:
    angle1 = np.random.randint(0, 360)
    angle2 = np.random.randint(0, 360)
    angle3 = np.random.randint(0, 360)
    mol = rotate_molecule(molecule, angle1, angle2, angle3)
    rotated_molecules.append(mol)

# Get molecules representaTion based on features expressed by the user
molecules_data = [molecule_to_ndarray(mol) for mol in rotated_molecules]

fingerprints = []
for j, molecule_data in enumerate(molecules_data):
    # PCA
    # Get the PCA tranformed data 
    transformed_data = compute_pca_using_covariance(molecule_data)
    
    # FINGERPRINT
    # OPTIONAL
    # Define a scaling factor of a scaling matrix to modify the reference points
    scaling_matrix = compute_scaling_matrix(transformed_data) 

    # Get the fingerprint from the tranformed data
    fingerprint = generate_molecule_fingerprint(transformed_data, scaling_factor=None, scaling_matrix=scaling_matrix)

    fingerprints.append(fingerprint)

# Compute similarity between all pairs of fingerprints
n_molecules = len(fingerprints)
for i in range(n_molecules):
    for j in range(i+1, n_molecules):
        partial_score = calculate_mean_absolute_difference(fingerprints[i], fingerprints[j])
        similarity = calculate_similarity_from_difference(partial_score)
        print(f"{i+1}-{j+1}: {similarity:.4f}")