# Script to collect and pre-process molecules from SDF files and 
# convert them in datastructures to compute their similarity based on 
# a PCA method considering coordinates, protons, neutrons and charges of every atom.

import numpy as np
from rdkit import Chem
from utils import DEFAULT_FEATURES

def collect_molecules_from_sdf(path, removeHs=False):
    """
    Collects molecules from a SDF file and returns a list of RDKit molecules.

    Parameters:
        path (str): Path to the SDF file.
        removeHs (bool, optional): Whether to remove hydrogens. Defaults to False.
        
    Returns:
        list: A list of RDKit molecule objects.
    """
    suppl = Chem.SDMolSupplier(path, removeHs=removeHs)
    molecules = [mol for mol in suppl if mol is not None]
    return molecules

def mol_nd_data(molecule, features=DEFAULT_FEATURES):
    """
    Generates a numpy array representing the given molecule.
    Each row of the array corresponds to an atom in the molecule:
    - First three columns are the x, y, z coordinates of the atom.
    - Subsequent columns represent values of the features specified in the features dictionary.

    Parameters:
    - molecule: The input molecule (rdkit molecule object).
    - features: Dictionary where keys are feature names and values are lists of functions to compute the feature.

    Returns:
    - Numpy array with shape (number of atoms, 3 + number of features).
    """
    
    molecule_info = {'coordinates': []}

    for atom in molecule.GetAtoms():
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        molecule_info['coordinates'].append([position.x, position.y, position.z])

        if features:
            for key, funcs in features.items():
                raw_value = funcs[0](atom)
                value = funcs[1](raw_value) if len(funcs) > 1 else raw_value

                if key not in molecule_info:
                    molecule_info[key] = []
                molecule_info[key].append(value)

    arrays = [np.array(molecule_info[key]).reshape(-1, 1) for key in molecule_info]
    mol_nd = np.hstack(arrays)
    return mol_nd

