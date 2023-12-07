# Scrpits collectiing examples of chirality and isomerism

import numpy as np  
from nd_sim.pre_processing import *
from nd_sim.pca_transform import * 
from nd_sim.fingerprint import *
from nd_sim.similarity import *
from nd_sim.utils import *
from trials.perturbations import *
import os 

def print_3d_coordinates(mol):
    print(f"3D coordinates \n")
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        print(f"({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")

cwd = os.getcwd()

# # PERMUTATIONS OF ATOMS LIST
# input_sdf_file = f'{cwd}/similarity/sd_data/isomers_output_1.sdf'
# output_sdf_file = f'{cwd}/similarity/sd_data/permutated_isomers_output_1.sdf'
# permute_sdf(input_sdf_file, output_sdf_file)

# PRE-PROCESSING
# List of molecules from SDF file
### PURE FEATURES ISOMERISM ###
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_output.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_protons.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_neutrons.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_charges.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_protons_neutrons.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_protons_charges.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_neutrons_charges.sdf', removeHs=False, sanitize=False)
molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_protons_neutrons_charges.sdf', removeHs=False, sanitize=False)

### FEATURES ISOMERISM ON OPTICAL ISOMER ###
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/isomers_protons.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/isomers_neutrons.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/isomers_charges.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/isomers_protons_neutrons.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/isomers_protons_charges.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/isomers_neutrons_charges.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/isomers_protons_neutrons_charges.sdf', removeHs=False, sanitize=False)

### EFFECTS OF SYMMETRY AND ABSENCE OF FEATURES ON OPTICAL ISOMER ###
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/perfect_isomers.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/near_perfect_isomers.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/not_perfect_isomers.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/imperfect_isomers.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/3d_chiral_output.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/isomers.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/no_isomers.sdf', removeHs=False, sanitize=False)

### OTHER OPTICAL ISOMERS ###
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/lambda_delta_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/stereogenic_centers_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/allene_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/alkylidene_cycloalkanes_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/spiranes_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/atroposomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/atroposomerisms_no_charge.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/helicene_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/cyclophane_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/annulene_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/annulene_isomerism_1.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/cycloalkene_isomerism.sdf', removeHs=False, sanitize=False)

### OTHER ISOMERISM ###
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/coordination_isomerism_3d.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/linkage_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/linkage_isomerism_avo.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/fac_mer_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/cis_trans_isomerism.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/cis_trans_isomerism_planar.sdf', removeHs=False, sanitize=False)
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/cis_trans_isomerisms_planar_substituted.sdf', removeHs=False, sanitize=False)

# ## COMPUTE SIMILARITY BETWEEN FINGERPRINTS ##
# # Get the fingerprints
# fingerprints = [generate_nd_molecule_fingerprint(molecule, DEFAULT_FEATURES, scaling_method='matrix') for molecule in molecules]
# print(f'Similarity from fp: {compute_similarity_score(fingerprints[0], fingerprints[1])}')
# # print(f'Similarity from fp: {compute_similarity_score(fingerprints[0], fingerprints[2])}')
# # print(f'Similarity from fp: {compute_similarity_score(fingerprints[0], fingerprints[3])}')

# ## COMPUTE SIMILARITY DIRECTLY BETWEEN MOLECULES ##
# print(f'Similarity from molecules: {compute_similarity(molecules[0], molecules[1], DEFAULT_FEATURES, scaling_method="matrix")}')

### ROTATE MOLECULES ###
rotated_molecules = []
for molecule in molecules:
    angle1 = np.random.randint(0, 360)
    angle2 = np.random.randint(0, 360)
    angle3 = np.random.randint(0, 360)
    mol = rotate_molecule(molecule, angle1, angle2, angle3)
    rotated_molecules.append(mol)
    
fingerprints = [generate_nd_molecule_fingerprint(molecule, DEFAULT_FEATURES, scaling_method='matrix', chirality=True) for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=PROTONS_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=NEUTRONS_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=CHARGES_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=PROTONS_NEUTRONS_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=PROTONS_CHARGES_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=NEUTRONS_CHARGES_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]
# fingerprints = [generate_nd_molecule_fingerprint(molecule, features=None, scaling_method='matrix') for molecule in rotated_molecules]

# COMPARE ALL PAIRS OF MOLECULES
# Compute similarity between all pairs of fingerprints
n_molecules = len(fingerprints)
for i in range(n_molecules):
    for j in range(i+1, n_molecules):
        similarity = compute_similarity_score(fingerprints[i], fingerprints[j])
        print(f"{i+1}-{j+1}: {similarity:.4f}")#:.4f}")