# Python script for comparing different similarity measures 

import numpy as np
import math
import matplotlib.pyplot as plt
from trials.perturbations import *
from rdkit import Chem
from trials.utils import *
from fingerprints import *
from deprecated_code.similarity_3d import *
from copy import deepcopy


# Molecules 
#suppl = Chem.SDMolSupplier('swapping.sdf', removeHs=False)
#suppl = Chem.SDMolSupplier('coumarins_test.sdf', removeHs=False)
suppl = Chem.SDMolSupplier('isomers_test.sdf', removeHs=False)
molecules = [mol for mol in suppl if mol is not None]
#print(len(molecules))

molecules_info = {}

for i, molecule in enumerate(molecules):
    info = molecule_info(molecule)
    molecules_info[f'molecule_{i}'] = info

molecule_1 = molecules_info['molecule_4']
molecule_2 = molecules_info['molecule_5']

# Rotate molecule_2
# molecule_2['coordinates'] = rotate_points(molecule_2['coordinates'], 180, -180, 0)
# molecule_2['coordinates_no_H'] = rotate_points(molecule_2['coordinates_no_H'], 180, -180, 0)

# Perturb coordinates of molecule_2
#molecule_2['coordinates'] = perturb_coordinates(molecule_2['coordinates'], 3)

print('SIMILARITIES')
print('-------------')
# Similarities

#### 1- Similarities based on matching ####

# # Tensor of inertia
# similarities, similarity = compute_similarity_based_on_matching(molecule_1, molecule_2)
# print(f'Similarity based on matching: {similarity}')
# print(f'Similarities based on matching: {similarities}')
# print('-------------')

# #### 2- Similarities based on moments ####

# # "Objective" points based on tensor of inertia, with masses, isotopes and charges

# similarities, similarity = compute_3d_similarity(molecule_1, molecule_2)
# print(f'Similarity 3d: {similarity}')
# print(f'Similarities 3d: {similarities}')
# print('-------------')

# (Standard USR (closest, furthest atoms) with masses, isotopes and charges (It does not handle chirality))

# #### 3- Similarities ND ####

# # 4D with masses 
# similarity = compute_4D_similarity(molecule_1, molecule_2)
# print(f'Similarity 4d: {similarity}')
# print('-------------')

# # 5D with masses and charges
# similarity = compute_5D_similarity(molecule_1, molecule_2)
# print(f'Similarity 5d: {similarity}')
# print('-------------')

# 6D with masses, isotopes and charges
similarity = compute_6D_similarity(molecule_1, molecule_2)
print(f'Similarity 6d: {similarity}')
print('-------------')

plt.show()

