# Scrpits collectiing examples of chirality and isomerism

import numpy as np  
from nd_sim.pre_processing import *
from nd_sim.pca_transform import * 
from nd_sim.fingerprint import *
from nd_sim.similarity import *
from nd_sim.utils import *
from trials.perturbations import *
import os 

cwd = os.getcwd()

# PRE-PROCESSING
# List of molecules from SDF file
# molecules = load_molecules_from_sdf(f'{cwd}/sd_data/coordination_isomerism_3d.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/sd_data/linkage_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/sd_data/linkage_isomerism_avo.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/sd_data/fac_mer_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/sd_data/cis_trans_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/sd_data/cis_trans_isomerism_planar.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/sd_data/cis_trans_isomerisms_planar_substituted.sdf', removeHs=False, sanitize=False)
molecules = load_molecules_from_sdf(f'{cwd}/sd_data/agostic/agostic_mols.sdf', removeHs=False, sanitize=False)

### ROTATE MOLECULES ###
rotated_molecules = []
for molecule in molecules:
    angle1 = np.random.randint(0, 360)
    angle2 = np.random.randint(0, 360)
    angle3 = np.random.randint(0, 360)
    mol = rotate_molecule(molecule, angle1, angle2, angle3)
    rotated_molecules.append(mol)
    
fingerprints = [generate_fingerprint_from_molecule(molecule, DEFAULT_FEATURES, scaling='matrix', chirality=False) for molecule in rotated_molecules]


# COMPARE ALL PAIRS OF MOLECULES
# Compute similarity between all pairs of fingerprints
n_molecules = len(fingerprints)
for i in range(n_molecules):
    for j in range(i+1, n_molecules):
        similarity = compute_similarity_score(fingerprints[i], fingerprints[j])
        print(f"{i+1}-{j+1}: {similarity:.4f}")#:.4f}")