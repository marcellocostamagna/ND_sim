# Python script to optimize the geometries of molecules in a sdf file
from rdkit import Chem
from rdkit.Chem import AllChem


# Read the molecules from the sdf file, optimize their geometries and
# saved them in a sdf file called 'sample3d_optimized.sdf'

mol_list = Chem.SDMolSupplier('sample3d.sdf')
opt_mol_list = []   
for molecule in mol_list:
    # Optimize the geometry of the molecule
    AllChem.EmbedMolecule(molecule)
    AllChem.UFFOptimizeMolecule(molecule)
    molecule.GetConformer()
    # Store the molecule in a list 
    opt_mol_list.append(molecule)


# Write the optimized molecules in a sdf file
w = Chem.SDWriter('sample3d_optimized.sdf')
for molecule in opt_mol_list:
    w.write(molecule)
w.close()



    






