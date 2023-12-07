# Scripts for calculating between all pairs of molecules in a list of molecules.

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
molecules = load_molecules_from_sdf(f'{cwd}/sd_data/optoiso_test/rocs_problematic_mols.sdf', removeHs=True, sanitize=True)

### ROTATE MOLECULES ###
rotated_molecules = []
for molecule in molecules:
    angle1 = np.random.randint(0, 360)
    angle2 = np.random.randint(0, 360)
    angle3 = np.random.randint(0, 360)
    mol = rotate_molecule(molecule, angle1, angle2, angle3)
    rotated_molecules.append(mol)

# fingerprints = [generate_nd_molecule_fingerprint(molecule, DEFAULT_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=PROTONS_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=NEUTRONS_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=CHARGES_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=PROTONS_NEUTRONS_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=PROTONS_CHARGES_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=NEUTRONS_CHARGES_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
fingerprints = [generate_nd_molecule_fingerprint(molecule, features=None, scaling_method='matrix') for molecule in rotated_molecules]

n_molecules = len(fingerprints)
for i in range(n_molecules):
    for j in range(i+1, n_molecules):
        similarity = compute_similarity_score(fingerprints[i], fingerprints[j])
        print(f"{i+1}-{j+1}: {similarity:.4f}")
        
    