import rdkit 
import os 
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from similarity.trials.perturbations import *

cwd = os.getcwd()
np.set_printoptions(precision=4, suppress=True)

molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/swapping.sdf', removeHs=False)

# Equal molecules with S-O exchange (equivalent to two reflections/rotations in 3D)
mol1 = molecules[0]
mol2 = molecules[1]

mol1 = molecule_to_ndarray(mol1)
mol2 = molecule_to_ndarray(mol2)
# Center molecules
# mol1 = mol1 - np.mean(mol1, axis=0)
# mol2 = mol2 - np.mean(mol2, axis=0)

print(f"molecule 1: \n {mol1}")
print(f"molecule 2: \n {mol2}")

## Rotate the second molecule

#extract 3D coordinates (firts three coloumns of the mol array)
mol2_3D = mol2[:, :3]
rotated_points = rotate_points(mol2_3D, 0, 0, 0)
# substitute the first three coordinates of mol2 with the rotated ones
mol2[:,:3] = rotated_points
print(f"molecule 2 rotated: \n {mol2}")


# # PCA
# # Get the PCA tranformed data
transformed_data1, _ = compute_pca_using_covariance(mol1)
transformed_data2, _= compute_pca_using_covariance(mol2)

# FINGERPRINT
# OPTIONAL
# Define a scaling factor of a scaling matrix to modify the reference points
scaling_factor1 = compute_scaling_factor(transformed_data1)
scaling_factor2 = compute_scaling_factor(transformed_data2)

# Get the fingerprint from the tranformed data
fingerprint1  = generate_molecule_fingerprint(transformed_data1, scaling_factor1)
fingerprint2 = generate_molecule_fingerprint(transformed_data2, scaling_factor2)

# Compute similarity between all pairs of fingerprints
partial_score = calculate_mean_absolute_difference(fingerprint1, fingerprint2)
similarity = calculate_similarity_from_difference(partial_score)
print(f"similarity: {similarity}")

