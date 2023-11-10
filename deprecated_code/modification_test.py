import numpy as np  
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from similarity.trials.perturbations import *
import os 
from rdkit import Chem

np.set_printoptions(precision=4, suppress=True)

def replace_random_hydrogen_with_deuterium(molecule):
    hydrogens = [atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "H"]
    if hydrogens:
        random_hydrogen = np.random.choice(hydrogens)
        random_hydrogen.SetIsotope(2)  # Set isotope number to 2 for deuterium
    return molecule

cwd = os.getcwd()
# PRE-PROCESSING
# List of molecules from SDF file
molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/size_increasing_molecules/range_18.sdf', removeHs=False, sanitize=False)

mol = molecules[0]


# Open an SDF writer and specify the output SDF file
with Chem.SDWriter(f'{cwd}/similarity/sd_data/size_increasing_molecules/two_18.sdf') as writer:
    # Write the original molecule
    writer.write(mol)

    # Write the modified molecule
    modified_mol = replace_random_hydrogen_with_deuterium(mol)
    writer.write(modified_mol)

writer.close()



