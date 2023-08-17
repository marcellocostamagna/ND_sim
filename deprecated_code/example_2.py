# Python script to generate the USRE fingerprint of molecules and compute their similarity

from rdkit import Chem
import deprecated_code.similarity_3d as sim3d
import deprecated_code.coordinates as coord
import numpy as np
from trials.utils import get_atoms_info
from visualization import *
# molecules = Chem.SDMolSupplier('sample3d_optimized.sdf')
#molecules = Chem.SDMolSupplier('sample3d_optimized_switched.sdf', removeHs=False)
#molecules = Chem.SDMolSupplier('isomers_test.sdf', removeHs=False)
#molecules = Chem.SDMolSupplier('swapping.sdf', removeHs=False)
molecules = Chem.SDMolSupplier('coumarins_test.sdf', removeHs=False)

molecules_info = {}
for i, molecule in enumerate(molecules):
    masses, coordinates = get_atoms_info(molecule)
    info = {'masses': masses,
            'coordinates': coordinates}
    molecules_info[f'molecule_{i}'] = info

molecule_1 = molecules_info['molecule_0']
molecule_2 = molecules_info['molecule_1']

# Select two identical molecules from the list 
mol1 = molecules[8]
mol2 = molecules[10]
coords_1, masses_1 = sim3d.get_coordinates_and_masses(mol1)
coords_2, masses_2 = sim3d.get_coordinates_and_masses(mol2)

# masses
# center_mass_1 = coord.compute_center_of_mass(coords_1, masses_1)
# center_mass_2 = coord.compute_center_of_mass(coords_2, masses_2)
# tensor_with_masses_1 = coord.compute_inertia_tensor(coords_1, masses_1, center_mass_1)
# tensor_with_masses_2 = coord.compute_inertia_tensor(coords_2, masses_2, center_mass_2)
# tmp_axis_1, tmp_eig_1 = coord.compute_principal_axes(tensor_with_masses_1, coords_1, masses_1)
# tmp_axis_2, tmp_eig_2 = coord.compute_principal_axes(tensor_with_masses_2, coords_2, masses_2)

# # visualization check
# visualize1(molecule_1['coordinates'], molecule_1['masses'], center_mass_1, tmp_axis_1, tmp_eig_1)
# visualize1(molecule_2['coordinates'], molecule_2['masses'], center_mass_2, tmp_axis_2, tmp_eig_2)

# no masses
masses_1 = np.ones(len(coords_1))
masses_2 = np.ones(len(coords_2))
center_geom_1 = coord.compute_geometrical_center(coords_1)
center_geom_2 = coord.compute_geometrical_center(coords_2)
tensor_no_masses_1 = coord.compute_inertia_tensor_no_masses(coords_1)
tensor_no_masses_2 = coord.compute_inertia_tensor_no_masses(coords_2)
axis_1, eig_1 = coord.compute_principal_axes(tensor_no_masses_1, coords_1, masses_1)
axis_2, eig_2 = coord.compute_principal_axes(tensor_no_masses_2, coords_2, masses_2)

# # visualization check
# visualize1(molecule_1['coordinates'], molecule_1['masses'], center_geom_1, axis_1, eig_1)
# visualize1(molecule_2['coordinates'], molecule_2['masses'], center_geom_2, axis_2, eig_2)


# # Projection
# # R_1 = coord.find_minimal_rotation(tmp_axis_1, axis_1)
# # R_2 = coord.find_minimal_rotation(tmp_axis_2, axis_2)
# handedness_1 = coord.compute_handedness(tmp_axis_1)
# if handedness_1 == "left-handed":
#     axis_1[2] = -axis_1[2]
# handedness_2 = coord.compute_handedness(tmp_axis_2)
# if handedness_2 == "left-handed":
#     axis_2[2] = -axis_2[2]
# final_axis_1 = axis_1
# final_axis_2 = axis_2

# # visualization check
# visualize1(molecule_1['coordinates'], molecule_1['masses'], center_geom_1, final_axis_1, eig_1)
# visualize1(molecule_2['coordinates'], molecule_2['masses'], center_geom_2, final_axis_2, eig_2)

distance_1 = coord.max_distance_from_geometrical_center(coords_1)
distance_2 = coord.max_distance_from_geometrical_center(coords_2)

reference_points_1 = coord.generate_reference_points(center_geom_1, axis_1, distance_1)
reference_points_2 = coord.generate_reference_points(center_geom_2, axis_2, distance_2)
fingerprint1 = sim3d.generate_usre_fingerprint(mol1, reference_points_1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2, reference_points_2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity: {}'.format(similarity))

#plt.show()


# # Select two different molecules from the list with the same elements
# mol1 = molecules[0]
# mol2 = molecules[6]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarities = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2, mol1, mol2)
# coefficients, penalty = sim3d.compute_coefficients(similarities, mol1, mol2, fingerprint1, fingerprint2)
# similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
# print('Similarity between two different molecules with the same elements: {}'.format(similarity))


# # Select two different molecules from the list with different elements (diff1 != 0, diff2 = 0)
# mol1 = molecules[3]
# mol2 = molecules[6]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarities = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2, mol1, mol2)
# coefficients, penalty = sim3d.compute_coefficients(similarities, mol1, mol2, fingerprint1, fingerprint2)
# similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
# print('Similarity between two different molecules with different elements (1): {}'.format(similarity))


# # Select two different molecules from the list with different elements (diff1 = 0, diff2 != 0)
# mol1 = molecules[5]
# mol2 = molecules[6]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarities = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2, mol1, mol2)
# coefficients, penalty = sim3d.compute_coefficients(similarities, mol1, mol2, fingerprint1, fingerprint2)
# similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
# print('Similarity between two different molecules with different elements (2): {}'.format(similarity))


# # Select two different molecules from the list with different elements (diff1 & diff2 != 0)
# mol1 = molecules[4]
# mol2 = molecules[9]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarities = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2, mol1, mol2)
# coefficients, penalty = sim3d.compute_coefficients(similarities, mol1, mol2, fingerprint1, fingerprint2)
# similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
# print('Similarity between two different molecules with different elements (3): {}'.format(similarity))


# # # Select identical molecules with oxygens and sulphurs switched
# # mol1 = molecules[0]
# # mol2 = molecules_2[0]
# # fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# # fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# # similarities = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2, mol1, mol2)
# # coefficients, penalty = sim3d.compute_coefficients(similarities, mol1, mol2, fingerprint1, fingerprint2)
# # similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
# # print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

# # # Select identical molecules with oxygens and sulphurs switched
# # mol1 = molecules[0]
# # mol2 = molecules_2[1]
# # fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# # fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# # similarities = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2, mol1, mol2)
# # coefficients, penalty = sim3d.compute_coefficients(similarities, mol1, mol2, fingerprint1, fingerprint2)
# # similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
# # print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

# # # Select identical molecules with oxygens and sulphurs switched
# # mol1 = molecules_2[0]
# # mol2 = molecules_2[1]
# # fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# # fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# # similarities = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2, mol1, mol2)
# # coefficients, penalty = sim3d.compute_coefficients(similarities, mol1, mol2, fingerprint1, fingerprint2)
# # similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
# # print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))