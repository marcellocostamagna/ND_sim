import numpy as np
from rdkit import Chem

###### PRE-PROCESSING #######

def get_protons(atom):
    return atom.GetAtomicNum()

def get_delta_neutrons(atom):
    return int(round(atom.GetMass())) - atom.GetAtomicNum()

def get_formal_charge(atom):
    return atom.GetFormalCharge()

def taper_p(value):
    return np.log(value)

def taper_n(value):
    return np.log(value + 2)

def taper_c(value):
    return np.log(value + 5)

DEFAULT_FEATURES = {
    'protons' : [get_protons, taper_p],
    'delta_neutrons' : [get_delta_neutrons, taper_n],
    'formal_charges' : [get_formal_charge, taper_c]
    }

###### FINGERPRINT ########


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
    max_values = np.max(np.abs(molecule_data), axis=0)
    return np.diag(max_values)