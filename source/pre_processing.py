# Script to collect and pre-process molecules from SDF files and 
# convert them in datastructures to compute their similarity based on 
# a PCA method considering coordinates, protons, neutrons and charges of every atom.

import numpy as np
from rdkit import Chem

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
        'neutrons': [],
        'formal_charges': []
    }
    for atom in molecule.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        mass_num = atom.GetMass()  
        neutron_num = int(round(mass_num)) - atomic_num
        formal_charge = atom.GetFormalCharge()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())

        molecule_info['coordinates'].append((position.x, position.y, position.z))
        molecule_info['protons'].append(atomic_num)
        molecule_info['neutrons'].append(neutron_num)
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

def taper_features(info: dict, function: np.function ):
    """
    Tapers the features of a molecule
    """
    info['protons'] = function(info['protons'])
    info['delta_neutrons'] = function(info['delta_neutrons'] + 1) # +1 to avoid negative delta_neutrons
    info['formal_charges'] = function(info['formal_charges'] + 4) # +4 to avoid negative formal_charges

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