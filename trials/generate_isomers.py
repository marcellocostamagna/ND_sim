# Scripts for creating (and saving) isomers from a molecule by reflecting it for each axis.

import numpy as np  
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from similarity.trials.perturbations import *
import os 
from rdkit import Chem

def print_3d_coordinates(mol):
    print(f"3D coordinates \n")
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        print(f"({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")


np.set_printoptions(precision=4, suppress=True)

cwd = os.getcwd()
# PRE-PROCESSING
# List of molecules from SDF file
molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/cis_trans_isomerism_planar_substituted.sdf', removeHs=False, sanitize=False)


## SIMILARITY BETWEEN ISOMERS CONSTRUCTED BY REFLECTION FOR EACH AXIS ##
axis = ['x', 'y', 'z']
## Reflect molecule to construct an optical isomer ##
mol = molecules[0]


# Open an SDF writer and specify the output SDF file
with Chem.SDWriter(f'{cwd}/similarity/sd_data/cis_trans_isomerisms_planar_substituted.sdf') as writer:
    # Write the original molecule
    writer.write(mol)

    # Reflect the molecule for each axis and compute similarity
    for ax in axis:
        mol1 = reflect_molecule_coordinate(mol, coordinate=ax)
        writer.write(mol1)

writer.close()



