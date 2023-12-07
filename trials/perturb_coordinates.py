# Scripts for perturbing a molecule coordinates

import numpy as np  
from nd_sim.pre_processing import *
from nd_sim.pca_transform import * 
from nd_sim.fingerprint import *
from nd_sim.similarity import *
from nd_sim.utils import *
from trials.perturbations import *
import os 
from rdkit import Chem

np.set_printoptions(precision=4, suppress=True)

cwd = os.getcwd()
# PRE-PROCESSING
# List of molecules from SDF file
molecules = load_molecules_from_sdf(f'{cwd}/sd_data/change_position_dependency.sdf', removeHs=False, sanitize=False)

mol = molecules[0]

# Perturb molecule coordinates

# Extract coordinates from molecule as ndarray
conf = mol.GetConformer()
coordinates = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()])

# Perturb coordinates
perturbed_coordinates = perturb_coordinates(coordinates, 3, 1)

# Update coordinates in molecule
for i, atom in enumerate(mol.GetAtoms()):
    conf.SetAtomPosition(atom.GetIdx(), Chem.rdGeometry.Point3D(perturbed_coordinates[i, 0], perturbed_coordinates[i, 1], perturbed_coordinates[i, 2]))


# Open an SDF writer and specify the output SDF file
with Chem.SDWriter(f'{cwd}/sd_data/change_position_dependency_1.sdf') as writer:
    # Write the perturbed molecule
    writer.write(mol)

writer.close()



