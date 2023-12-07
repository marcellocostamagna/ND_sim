# Script for testing chirality and isomerism with perturbations

import numpy as np  
from nd_sim.pre_processing import *
from nd_sim.pca_transform import * 
from nd_sim.fingerprint import *
from nd_sim.similarity import *
from nd_sim.utils import *
from trials.perturbations import *
import os 

# np.set_printoptions(precision=4, suppress=True)

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
# molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/symmetric_tetrahedron_protons_neutrons_charges.sdf', removeHs=False, sanitize=False)

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
molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/helicene_isomerism.sdf', removeHs=False, sanitize=False)
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



### ROTATE MOLECULES ###
rotated_molecules = []
for molecule in molecules:
    angle1 = np.random.randint(0, 360)
    angle2 = np.random.randint(0, 360)
    angle3 = np.random.randint(0, 360)
    angle1 = 0
    angle2 = 0
    angle3 = 0
    mol = rotate_molecule(molecule, angle1, angle2, angle3)
    rotated_molecules.append(mol)
    
molecules_data = [molecule_to_ndarray(mol) for mol in rotated_molecules]

# Perturb 3D coordinates 
perturbed_molecules_data = []
# extract the 3D coordinates (first three colunns of the ndarray), perturb them and then concatenate them with the rest of the data 
for molecule_data in molecules_data:
    print(f"molecule_data: \n {molecule_data}")
    # print(f"molecule_data[:, :3]: \n {molecule_data[:, :3]}")
    print(f"molecule_data[:, :2]: \n {molecule_data[:, :2]}")
    # print(f"molecule_data[:, :1]: \n {molecule_data[:, :1]}")
    # perturbed_3d_coordinates = perturb_coordinates(molecule_data[:, :3], 8, 1)
    perturbed_2d_coordinates = perturb_coordinates(molecule_data[:, :2], 6, 0.1)
    # perturbed_1d_coordinates = perturb_coordinates(molecule_data[:, :1], 4, 0.04)
    # print(f"perturbed_3d_coordinates: \n {perturbed_3d_coordinates}")
    print(f"perturbed_3d_coordinates: \n {perturbed_2d_coordinates}")
    # print(f"perturbed_3d_coordinates: \n {perturbed_1d_coordinates}")
    # print(f"molecule_data[:, 3:]: \n {molecule_data[:, 3:]}")
    print(f"molecule_data[:, 2:]: \n {molecule_data[:, 2:]}")
    # print(f"molecule_data[:, 1:]: \n {molecule_data[:, 1:]}")
    # perturbed_molecule_data = np.concatenate((perturbed_3d_coordinates, molecule_data[:, 3:]), axis=1)
    perturbed_molecule_data = np.concatenate((perturbed_2d_coordinates, molecule_data[:, 2:]), axis=1)
    # perturbed_molecule_data = np.concatenate((perturbed_1d_coordinates, molecule_data[:, 1:]), axis=1)
    print(f"perturbed_molecule_data: \n {perturbed_molecule_data}")
    perturbed_molecules_data.append(perturbed_molecule_data)

    
fingerprints = []
for j, molecule_data in enumerate(perturbed_molecules_data):
    # PCA
    # Get the PCA tranformed data 
    transformed_data = compute_pca_using_covariance(molecule_data)
    # print(f"molecule {j+1} : \n {transformed_data}")

    # FINGERPRINT
    # OPTIONAL
    # Define a scaling factor of a scaling matrix to modify the reference points
    # scaling_factor = compute_scaling_factor(transformed_data)
    scaling_matrix = compute_scaling_matrix(transformed_data) 
    # scaling_factor = 1

    # Get the fingerprint from the tranformed data
    fingerprint = generate_molecule_fingerprint(transformed_data, scaling_factor=None, scaling_matrix=scaling_matrix)
  

    fingerprints.append(fingerprint)

# Compute similarity between all pairs of fingerprints
n_molecules = len(fingerprints)
for i in range(n_molecules):
    for j in range(i+1, n_molecules):
        partial_score = calculate_mean_absolute_difference(fingerprints[i], fingerprints[j])
        similarity = calculate_similarity_from_difference(partial_score)
        print(f"{i+1}-{j+1}: {similarity}")