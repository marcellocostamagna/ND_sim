# Python script to change atoms in a molecule

from rdkit import Chem
from rdkit.Chem import AllChem

# Read the molecules from the sdf file
mol_list = Chem.SDMolSupplier('sample3d.sdf')
opt_mol_list = []   
count = 0
for molecule in mol_list:
    # Optimize the geometry of the molecule
    AllChem.EmbedMolecule(molecule)
    AllChem.UFFOptimizeMolecule(molecule)
    molecule.GetConformer()
    if count == 0:
        # Change all oxygens into sulphurs
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 8:
                atom.SetAtomicNum(16)
    if count == 8:
        # Change atom 13, 15, and 28 into nitrogen
        for atom in molecule.GetAtoms():
            if atom.GetIdx() == 13 or atom.GetIdx() == 15 or atom.GetIdx() == 28:
                atom.SetAtomicNum(7)

    # Store the molecule in a list 
    opt_mol_list.append(molecule)
    count += 1


# Write the optimized molecules in a sdf file
w = Chem.SDWriter('sample3d_optimized_switched.sdf')
for molecule in opt_mol_list:
    w.write(molecule)
w.close()