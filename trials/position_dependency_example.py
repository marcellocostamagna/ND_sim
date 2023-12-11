# Script to show the similarity value dependency on the position the same change occurs 

import numpy as np  
from nd_sim.pre_processing import *
from nd_sim.pca_transform import * 
from nd_sim.fingerprint import *
from nd_sim.similarity import *
from nd_sim.utils import *
from trials.perturbations import *
import os 
from rdkit import Chem
from copy import deepcopy

np.set_printoptions(precision=4, suppress=True)

def generate_all_single_deuterium_variants(molecule):
    hydrogens = [atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "H"]
    modified_molecules = []

    for hydrogen in hydrogens:
        modified_molecule = deepcopy(molecule)
        hydrogen_atom = modified_molecule.GetAtomWithIdx(hydrogen.GetIdx())  # Get the corresponding hydrogen in the copied molecule
        hydrogen_atom.SetIsotope(2)  # Set isotope number to 2 for deuterium
        modified_molecules.append(modified_molecule)

    return modified_molecules


cwd = os.getcwd()
# PRE-PROCESSING
# List of molecules from SDF file
molecules = load_molecules_from_sdf(f'{cwd}/sd_data/change_position_dependency.sdf', removeHs=False, sanitize=False)

original_molecule = molecules[0]

modified_molecules = generate_all_single_deuterium_variants(original_molecule)

# Print the similarities between the original molecule and its perturbed versions
for i,modified_molecule in enumerate(modified_molecules):
    similarity = compute_similarity(original_molecule, modified_molecule, DEFAULT_FEATURES, scaling='matrix', chirality=False)
    print(f'Hydrogen {i+1}: {similarity:.4f}') 