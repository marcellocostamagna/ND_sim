
# Python script to inspect partial charges of molecules

from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdPartialCharges as rdPC


mol_list = Chem.SDMolSupplier('charged_sample3d_optimized_switched.sdf')
charged_mol_list = []   
for molecule in mol_list:
    for atom in molecule.GetAtoms():
        print(atom.GetSymbol(), atom.GetPartialCharge())
    # Store the molecule in a list 
    charged_mol_list.append(molecule)


# Write the optimized molecules in a sdf file
w = Chem.SDWriter('charged_sample3d_optimized_switched.sdf')
for molecule in charged_mol_list:
    w.write(molecule)
w.close()