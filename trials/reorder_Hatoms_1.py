# Scrit to reorder the hydrgens atoms in a sd file to collect them at the end of the molecule
# and order them in respect to one axis.

import numpy as np  
from nd_sim.pre_processing import *
from nd_sim.pca_transform import * 
from nd_sim.fingerprint import *
from nd_sim.similarity import *
from nd_sim.utils import *
from trials.perturbations import *
import os 
from rdkit import Chem

cwd = os.getcwd()

def reorder_hydrogens_by_z_axis(mol):
    """
    Reorders atoms in a molecule so that all hydrogen atoms are at the end,
    sorted by their z-axis coordinate.

    Parameters:
    mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
    rdkit.Chem.Mol: A new RDKit molecule with reordered atoms.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule does not have 3D coordinates.")

    conf = mol.GetConformer()

    # Separate hydrogen and non-hydrogen atoms
    hydrogens = [(atom.GetIdx(), conf.GetAtomPosition(atom.GetIdx()).z) for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    non_hydrogens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H']

    # Sort hydrogens by their z-coordinate
    hydrogens.sort(key=lambda x: x[1])

    # Combine indices
    indices = non_hydrogens + [idx for idx, _ in hydrogens]

    # Create a new molecule
    new_mol = Chem.RWMol()

    # Add atoms to new molecule in new order
    new_indices = {}
    for idx in indices:
        atom = mol.GetAtomWithIdx(idx)
        new_idx = new_mol.AddAtom(atom)
        new_indices[idx] = new_idx

    # Traverse bonds in the original molecule and add to new molecule
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        new_mol.AddBond(new_indices[begin_idx], new_indices[end_idx], bond.GetBondType())

    # Copy the conformer data
    new_conf = Chem.Conformer(conf.GetNumAtoms())
    for idx in indices:
        pos = conf.GetAtomPosition(idx)
        new_conf.SetAtomPosition(new_indices[idx], pos)

    new_mol.AddConformer(new_conf)

    # Update molecule properties
    new_mol.SetProp("_Name", mol.GetProp("_Name"))
    for prop_name in mol.GetPropNames():
        new_mol.SetProp(prop_name, mol.GetProp(prop_name))

    return new_mol.GetMol()

molecules = load_molecules_from_sdf(f'{cwd}/sd_data/change_position_dependency.sdf', removeHs=False, sanitize=False)
mol = molecules[0]

mol1 = reorder_hydrogens_by_z_axis(mol)

# Open an SDF writer and specify the output SDF file
with Chem.SDWriter(f'{cwd}/sd_data/change_position_dependency_reordered.sdf') as writer:
    # Write the perturbed molecule
    writer.write(mol1)
    