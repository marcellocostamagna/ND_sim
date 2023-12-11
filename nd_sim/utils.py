# ND_sim
# This file is part of ND_sim, which is licensed under the
# GNU Lesser General Public License v3.0 (or any later version).
# See the LICENSE file for more details.

# Script collecting all utility functions used in the similarity package.

import numpy as np

###### PRE-PROCESSING #######

### Fetaures functions ###
def extract_proton_number(atom):
    return np.sqrt(atom.GetAtomicNum())

# Difference between the number of neutrons of the current atom
# and the number of neutrons of the most common isotope of the element (stored)
def extract_neutron_difference_from_common_isotope(atom):
    n_neutrons = int(round(atom.GetMass())) - atom.GetAtomicNum()
    n_neutrons_most_common = N_NEUTRONS[atom.GetSymbol()]
    raw_value= n_neutrons - n_neutrons_most_common
    return np.sign(raw_value) * np.sqrt(abs(raw_value))

def extract_formal_charge(atom):
    return atom.GetFormalCharge()

###### FINGERPRINT utils ########

def compute_scaling_factor(molecule_data):
    """
    Computes the largest distance between the centroid and the molecule data points
    """
    centroid = np.zeros(molecule_data.shape[1])
    distances = np.linalg.norm(molecule_data - centroid, axis=1)
    return np.max(distances)

def compute_scaling_matrix(molecule_data):
    """
    Computes a diagonal scaling matrix with the maximum absolute values 
    for each dimension of the molecule data as its diagonal entries
    """
    tolerance = 1e-6
    max_values = np.max(np.abs(molecule_data), axis=0)
    max_values[np.isclose(max_values, 0, atol=tolerance)] = 1.0
    return np.diag(max_values)

#### DEFAULTS #####

DEFAULT_FEATURES = {
    'protons' : extract_proton_number,
    'delta_neutrons' : extract_neutron_difference_from_common_isotope,
    'formal_charges' : extract_formal_charge
    }

### EXAMPLE FEATURES ###

def proton_number(atom):
    return atom.GetAtomicNum()

def neutron_difference(atom):
    n_neutrons = int(round(atom.GetMass())) - atom.GetAtomicNum()
    n_neutrons_most_common = N_NEUTRONS[atom.GetSymbol()]
    return n_neutrons - n_neutrons_most_common

def formal_charge(atom):
    return atom.GetFormalCharge()

EXAMPLE_FEATURES = {
    'protons' : proton_number,
    'neutrons' : neutron_difference,
    'charge' : formal_charge
    }

### CONSTANTS ### 
    
N_NEUTRONS = {'H': 0, 'He': 2, 'Li': 4, 'Be': 5, 'B': 6, 'C': 6, 'N': 7, 'O': 8,
              'F': 10, 'Ne': 10, 'Na': 12, 'Mg': 12, 'Al': 14, 'Si': 14, 'P': 16,
              'S': 16, 'Cl': 18, 'Ar': 22, 'K': 20, 'Ca': 20, 'Sc': 24, 'Ti': 26,
              'V': 28, 'Cr': 28, 'Mn': 30, 'Fe': 30, 'Co': 32, 'Ni': 31, 'Cu': 35,
              'Zn': 35, 'Ga': 39, 'Ge': 41, 'As': 42, 'Se': 45, 'Br': 45, 'Kr': 48,
              'Rb': 48, 'Sr': 50, 'Y': 50, 'Zr': 51, 'Nb': 52, 'Mo': 54, 'Tc': 55,
              'Ru': 57, 'Rh': 58, 'Pd': 60, 'Ag': 61, 'Cd': 64, 'In': 66, 'Sn': 69,
              'Sb': 71, 'Te': 76, 'I': 74, 'Xe': 77, 'Cs': 78, 'Ba': 81, 'La': 82,
              'Ce': 82, 'Pr': 82, 'Nd': 84, 'Pm': 84, 'Sm': 88, 'Eu': 89, 'Gd': 93,
              'Tb': 94, 'Dy': 96, 'Ho': 98, 'Er': 99, 'Tm': 100, 'Yb': 103, 'Lu': 104,
              'Hf': 106, 'Ta': 108, 'W': 110, 'Re': 111, 'Os': 114, 'Ir': 115, 'Pt': 117,
              'Au': 118, 'Hg': 121, 'Tl': 123, 'Pb': 125, 'Bi': 126, 'Po': 125, 'At': 125,
              'Rn': 136, 'Fr': 136, 'Ra': 138, 'Ac': 138, 'Th': 142, 'Pa': 140, 'U': 146,
              'Np': 144, 'Pu': 150, 'Am': 148, 'Cm': 151, 'Bk': 150, 'Cf': 153, 'Es': 153,
              'Fm': 157, 'Md': 157, 'No': 157, 'Lr': 159, 'Rf': 163, 'Db': 163, 'Sg': 163,
              'Bh': 163, 'Hs': 161, 'Mt': 169, 'Ds': 171, 'Rg': 170, 'Cn': 173, 'Nh': 171,
              'Fl': 175, 'Mc': 173, 'Lv': 177, 'Ts': 175, 'Og': 176}