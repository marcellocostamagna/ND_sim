import numpy as np  
from nd_sim.pre_processing import *
from nd_sim.pca_transform import * 
from nd_sim.fingerprint import *
from nd_sim.similarity import *
from nd_sim.utils import *
from trials.perturbations import *
import os 

np.set_printoptions(precision=4, suppress=True)

cwd = os.getcwd()
# PRE-PROCESSING
# List of molecules from SDF file
molecules = load_molecules_from_sdf(f'{cwd}/sd_data/simple_case.sdf', removeHs=False, sanitize=False)

fingerprints = [generate_fingerprint_from_molecule(molecule, EXAMPLE_FEATURES, scaling='matrix') for molecule in molecules]

n_molecules = len(fingerprints)
for i in range(n_molecules):
    for j in range(i+1, n_molecules):
        similarity = compute_similarity_score(fingerprints[i], fingerprints[j])
        print(f"{i+1}-{j+1}: {similarity:.4f}")