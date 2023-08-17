import numpy
from rdkit import Chem

DEFAULT_FEATURES = {
    'protons' = [get_protons]
    'delta_neutrons' = []
    'formal_charges'= []
}

def get_protons(atom):
    return atom.GetAtomicNum()