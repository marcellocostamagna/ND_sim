# Script to collect and pre-process molecules from SDF files and 
# convert them in datastructures to compute their similarity based on 
# a PCA method considering coordinates, protons, neutrons and charges of every atom.

import numpy as np
from rdkit import Chem
from similarity.source.utils import DEFAULT_FEATURES

def collect_molecules_from_sdf(path, removeHs=False, sanitize=True):
    """
    Collect molecules from an SDF file.
    
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

# TODO: Improve function name
def mol_nd_data(molecule, features=DEFAULT_FEATURES):
    """
    Generate a numpy array representing the given molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The input RDKit molecule object.
    features : dict, optional
        Dictionary where keys are feature names and values are lists of functions to compute the feature.
        Defaults to DEFAULT_FEATURES.

    Returns
    -------
    numpy.ndarray
        Array with shape (number of atoms, 3 + number of features), representing the molecule.
    """
    
    molecule_info = {'coordinates': []}

    if features:
        for key in features:
            molecule_info[key] = []

    for atom in molecule.GetAtoms():
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        molecule_info['coordinates'].append([position.x, position.y, position.z])

        if features:
            for key, funcs in features.items():
                raw_value = funcs[0](atom)
                value = funcs[1](raw_value) if len(funcs) > 1 else raw_value

                # if key not in molecule_info:
                #     molecule_info[key] = []
                molecule_info[key].append(value)

    arrays = []
    for key in molecule_info:
        if key == 'coordinates':
            arrays.append(np.array(molecule_info[key]))  # Convert directly to numpy array without reshaping
        else:
            arrays.append(np.array(molecule_info[key]).reshape(-1, 1))
    mol_nd = np.hstack(arrays)
    # Center the data
    # print(f'mean: {np.mean(mol_nd, axis=0)}')
    mol_nd = mol_nd - np.mean(mol_nd, axis=0)
    return mol_nd

#### TEMPORARY FUNCTIONS ####

def collect_molecules_from_file(path, file_format='sdf', removeHs=False, sanitize=True):
    """
    Collect molecules from a file of the specified format.

    Parameters
    ----------
    path : str
        Path to the file.
    file_format : str, optional
        Type of the file. Accepted formats: 'sdf', 'mol', 'pdb', 'mol2', 'xyz'. Defaults to 'sdf'.
    removeHs : bool, optional
        Whether to remove hydrogens. Defaults to False.
    sanitize : bool, optional
        Whether to sanitize the molecules. Defaults to True.

    Returns
    -------
    list of rdkit.Chem.rdchem.Mol
        A list of RDKit molecule objects.
    
    Raises
    ------
    ValueError
        If an unsupported file format is provided.
    """
    
    if file_format == 'sdf':
        suppl = Chem.SDMolSupplier(path, removeHs=removeHs, sanitize=sanitize)
    elif file_format == 'mol':
        suppl = Chem.MolFromMolFile(path, removeHs=removeHs, sanitize=sanitize)
    elif file_format == 'pdb':
        suppl = Chem.MolFromPDBFile(path, removeHs=removeHs, sanitize=sanitize)
    elif file_format == 'mol2':
        suppl = Chem.MolFromMol2File(path, removeHs=removeHs, sanitize=sanitize)
    elif file_format == 'xyz':
        # RDKit doesn't natively support XYZ. You might want to use another approach/library for this.
        # For the purpose of this example, we'll return an empty list.
        return []
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    molecules = [mol for mol in suppl if mol is not None]
    return molecules
