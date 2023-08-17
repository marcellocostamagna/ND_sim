# Script to collect and pre-process molecules from SDF files and 
# convert them in datastructures to compute their similarity based on 
# a PCA method considering coordinates, protons, neutrons and charges of every atom.

import numpy as np
from rdkit import Chem
from utils import DEFAULT_FEATURES

def collect_molecules_from_sdf(path):
    """
    Collects molecules from a SDF file and returns a list of RDKit molecules
    """
    suppl = Chem.SDMolSupplier(path, removeHs=False)
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

def normalize_features(info: dict):
    """
    Normalizes the numbers of protons, delta_neutrons and formal_charges of a molecule 
    using the range of the coordinates
    """
    # Get the maximum and minimum from the coordinates array
    max = np.amax(info['coordinates'])
    min = np.amin(info['coordinates'])

    # Normalize the features
    # Protons
    p_min = np.amin(info['protons'])
    p_max = np.amax(info['protons'])
    info['protons'] = (info['protons'] - p_min) / (p_max - p_min) * (max - min) + min

    # Delta neutrons
    dn_min = np.amin(info['delta_neutrons'])
    dn_max = np.amax(info['delta_neutrons'])
    if dn_min < 0:
        dn_min = 0
    if dn_min != dn_max:
        info['delta_neutrons'] = (info['delta_neutrons'] - dn_min) / (dn_max - dn_min) * (max - min) + min

    # Formal charges
    fc_min = np.amin(info['formal_charges'])
    fc_max = np.amax(info['formal_charges'])
    if fc_min != fc_max:
        info['formal_charges'] = (info['formal_charges'] - fc_min) / (fc_max - fc_min) * (max - min) + min

    return info

def normalize_features2(info: dict):
    """
    Normalizes the numbers of protons, delta_neutrons and formal_charges of a molecule 
    using the range of the coordinates
    """
    # Calculate the range for each axis
    ranges = np.ptp(info['coordinates'], axis=0)

    # Find the index of the axis with the largest range
    #axis = np.argmax(ranges)
    # Find the index of the axis with the smallest range
    axis = np.argmin(ranges)
    # Find the index of the axis with the intermediate range
    #axis= np.argsort(ranges)[1] 


    # Get the maximum and minimum from the coordinates array for the axis with the largest range
    max = np.max(info['coordinates'][:, axis])
    min = np.min(info['coordinates'][:, axis])

    # Normalize the features
    # Protons
    p_min = np.amin(info['protons'])
    p_max = np.amax(info['protons'])
    info['protons'] = (info['protons'] - p_min) / (p_max - p_min) * (max - min) + min

    # Delta neutrons
    dn_min = np.amin(info['delta_neutrons'])
    dn_max = np.amax(info['delta_neutrons'])
    if dn_min < 0:
        dn_min = 0
    if dn_min != dn_max:
        info['delta_neutrons'] = (info['delta_neutrons'] - dn_min) / (dn_max - dn_min) * (max - min) + min

    # Formal charges
    fc_min = np.amin(info['formal_charges'])
    fc_max = np.amax(info['formal_charges'])
    if fc_min != fc_max:
        info['formal_charges'] = (info['formal_charges'] - fc_min) / (fc_max - fc_min) * (max - min) + min

    return info

def taper_features(info: dict, function: callable = np.log):
    """
    Tapers the features of a molecule
    """
    info['protons'] = function(info['protons'])
    info['delta_neutrons'] = function(info['delta_neutrons'] + 2) # +2 to avoid negative delta_neutrons and value 0
    info['formal_charges'] = function(info['formal_charges'] + 5) # +5 to avoid negative formal_charges and value 0

    return info

def get_molecule_6D_datastructure(info):
    """
    Returns a 6D datastructure of a molecule
    """
    molecule = np.hstack((info['coordinates'], 
                          info['protons'].reshape(-1, 1),
                          info['delta_neutrons'].reshape(-1, 1),
                          info['formal_charges'].reshape(-1, 1)))
    return molecule

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

