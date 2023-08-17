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

def collect_molecule_info(molecule):
    """
    Collects information from a rdkit molecule and returns a dictionary with the following keys:

    'coordinates': array of the 3D coordinates of the atoms
    'protons': array of the number of protons of the atoms
    'neutrons': array of the number of neutrons of the atoms
    'formal_charges': array of the formal charges of the atoms
    """
    molecule_info = {
        'coordinates': [],
        'protons': [],
        'delta_neutrons': [],
        'formal_charges': []
    }
    for atom in molecule.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        mass_num = atom.GetMass()  
        neutron_num = int(round(mass_num)) - atomic_num
        delta_neutrons = neutron_num - atomic_num
        formal_charge = atom.GetFormalCharge()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())

        molecule_info['coordinates'].append((position.x, position.y, position.z))
        molecule_info['protons'].append(atomic_num)
        molecule_info['delta_neutrons'].append(delta_neutrons)
        molecule_info['formal_charges'].append(formal_charge)
        
    # convert lists to numpy arrays
    for key in molecule_info:
        molecule_info[key] = np.array(molecule_info[key])
    return molecule_info

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

