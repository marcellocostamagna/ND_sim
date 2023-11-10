# Script to collect and pre-process molecules from SDF files and 
# convert them in datastructures to compute their similarity based on 
# a PCA method considering coordinates, protons, neutrons and charges of every atom.

import numpy as np
from rdkit import Chem
from similarity.source.utils import DEFAULT_FEATURES

def load_molecules_from_sdf(path, removeHs=False, sanitize=True):
    """
    Load molecules from an SDF file.
    
    Parameters
    ----------
    path : str
        Path to the SDF file.
    removeHs : bool, optional
        Whether to remove hydrogens. Defaults to False.
    sanitize : bool, optional
        Whether to sanitize the molecules. Defaults to True.

    Returns
    -------
    list of rdkit.Chem.rdchem.Mol
        A list of RDKit molecule objects.
    """
    suppl = Chem.SDMolSupplier(path, removeHs=removeHs, sanitize=sanitize)
    molecules = [mol for mol in suppl if mol is not None]
    return molecules

def molecule_to_ndarray(molecule, features=DEFAULT_FEATURES, removeHs=False):
    """
    Generate a numpy array representing the given molecule in N dimensions.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The input RDKit molecule object.
    features : dict, optional
        Dictionary where keys are feature names and values are lists of functions to compute the feature.
        Defaults to DEFAULT_FEATURES.
    removeHs : : bool, optional
        If True, hydrogen atoms will not be included in the array representation.
        Defaults to False.

    Returns
    -------
    numpy.ndarray
        Array with shape (number of atoms, 3 spatial coordinates + number of features),
        representing the molecule.
    """
    
    molecule_info = {'coordinates': []}

    if features:
        for key in features:
            molecule_info[key] = []

    for atom in molecule.GetAtoms():
        # Skip hydrogens if removeHs is True
        if removeHs and atom.GetAtomicNum() == 1:
            continue
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        molecule_info['coordinates'].append([position.x, position.y, position.z])

        if features:
            for key, funcs in features.items():
                raw_value = funcs[0](atom)
                value = funcs[1](raw_value) if len(funcs) > 1 else raw_value
                molecule_info[key].append(value)

    arrays = []
    for key in molecule_info:
        if key == 'coordinates':
            arrays.append(np.array(molecule_info[key]))  
        else:
            arrays.append(np.array(molecule_info[key]).reshape(-1, 1))
    mol_nd = np.hstack(arrays)
    # Centering data
    mol_nd = mol_nd - np.mean(mol_nd, axis=0)
    return mol_nd


