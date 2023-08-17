import numpy as np
from rdkit import Chem


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
