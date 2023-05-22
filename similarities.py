# Python script for comparing different similarity measures 

import numpy as np
import math
from similarity_3d import calculate_partial_score
import matplotlib.pyplot as plt
from perturbations import *
from rdkit import Chem
from utils import *
from fingerprints import *
from similarity_3d import *
#from pca_fingerprint import *
from copy import deepcopy

# The properties should be:
# 1. Number of protons
# 2. Number of neutrons
# 3. Number of electrons

# Molecules 
suppl = Chem.SDMolSupplier('swapping.sdf', removeHs=False)
#suppl = Chem.SDMolSupplier('isomers.sdf', removeHs=False)
molecules = [mol for mol in suppl if mol is not None]
print(len(molecules))

molecules_info = {}

for i, molecule in enumerate(molecules):
    elements, masses, protons, neutrons, electrons, coordinates = get_atoms_info(molecule)
    info = molecule_info(molecule)

    info1 = {'elements': elements, 
            'masses': masses,
            'protons': protons, 
            'neutrons': neutrons, 
            'electrons': electrons, 
            'coordinates': coordinates}
    molecules_info[f'molecule_{i}'] = info

molecule_1 = deepcopy(molecules_info['molecule_0'])
molecule_2 = deepcopy(molecules_info['molecule_0'])

# Rotate molecule_2
molecule_2['coordinates'] = rotate_points(molecule_2['coordinates'], -90, 45, 3)

# Perturb coordinates of molecule_2
molecule_2['coordinates'] = perturb_coordinates(molecule_2['coordinates'], 4)

     

print('SIMILARITIES')
print('-------------')
# Similarities

#### 1- Similarities based on matching ####

# Tensor of inertia
similarities, similarity = compute_similarity_based_on_matching(molecule_1, molecule_2)
print(f'Similarity based on matching: {similarity}')
print(f'Similarities based on matching: {similarities}')
print('-------------')

#### 2- Similarities based on moments ####

# "Objective" points based on tensor of inertia, with masses, isotopes and charges

similarities, similarity = compute_3d_similarity(molecule_1, molecule_2)
print(f'Similarity 3d: {similarity}')
print(f'Similarities 3d: {similarities}')
print('-------------')

# (Standard USR (closest, furthest atoms) with masses, isotopes and charges (It does not handle chirality))


#### 3- Similarities ND ####

# 4D with masses 

similarity = compute_4D_similarity(molecule_1, molecule_2)
print(f'Similarity 4d: {similarity}')
print('-------------')

# 5D with masses and charges

similarity = compute_5D_similarity(molecule_1, molecule_2)
print(f'Similarity 5d: {similarity}')
print('-------------')

# 6D with masses, isotopes and charges

similarity = compute_6D_similarity(molecule_1, molecule_2)
print(f'Similarity 6d: {similarity}')
print('-------------')

plt.show()