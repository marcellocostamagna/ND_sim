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
    # Check if the molecule has 3D coordinates
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule does not have 3D coordinates.")

    # Extract the conformer to access atom coordinates
    conf = mol.GetConformer()

    # Sort hydrogens by their z-coordinate
    hydrogens = [(atom, conf.GetAtomPosition(atom.GetIdx()).z) for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    hydrogens.sort(key=lambda x: x[1])

    # Separate non-hydrogen atoms
    non_hydrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() != 'H']

    # Create a new empty editable molecule
    new_mol = Chem.RWMol()

    # Add non-hydrogens first
    for atom, _ in non_hydrogens:
        new_mol.AddAtom(atom)

    # Then add sorted hydrogens
    for atom, _ in hydrogens:
        # atom = mol.GetAtomWithIdx(idx)
        new_mol.AddAtom(atom)
        
    # Add bonds with updated indices
    old_to_new_idx = {old_idx: i for i, (old_idx, _) in enumerate(non_hydrogens + hydrogens)}
    for bond in mol.GetBonds():
        start_atom_idx = old_to_new_idx[bond.GetBeginAtomIdx()]
        end_atom_idx = old_to_new_idx[bond.GetEndAtomIdx()]
        new_mol.AddBond(start_atom_idx, end_atom_idx, bond.GetBondType())
    
    # Convert back to a normal Mol object
    new_mol = new_mol.GetMol()

    # Update 3D coordinates
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())
    for i, (atom, _) in enumerate(non_hydrogens + hydrogens):
        orig_idx = atom.GetIdx()
        pos = conf.GetAtomPosition(orig_idx)
        new_conf.SetAtomPosition(i, pos)
    new_mol.AddConformer(new_conf)

    return new_mol

molecules = load_molecules_from_sdf(f'{cwd}/sd_data/change_position_dependency.sdf', removeHs=False, sanitize=False)
mol = molecules[0]

mol1 = reorder_hydrogens_by_z_axis(mol)

# Open an SDF writer and specify the output SDF file
with Chem.SDWriter(f'{cwd}/sd_data/change_position_dependency_reordered.sdf') as writer:
    # Write the perturbed molecule
    writer.write(mol1)
    