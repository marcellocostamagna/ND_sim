# Python scrpit to manage molecules and extract information to compute similarity

import numpy as np
from rdkit import Chem



def get_atoms_info(molecule):
    protons = []
    neutrons = []
    electrons = []
    formal_charges = []
    isotopes = []
    coordinates = []

    for atom in molecule.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        mass_num = atom.GetMass()  # Extracts the mass number (isotope) of the atom
        neutron_num = int(round(mass_num)) - atomic_num
        formal_charge = atom.GetFormalCharge()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())  # Get the atom's 3D coordinates

        protons.append(atomic_num)
        neutrons.append(neutron_num)
        electrons.append(atomic_num - formal_charge)  # Adjusted for the formal charge
        formal_charges.append(formal_charge)
        isotopes.append(mass_num)
        coordinates.append((position.x, position.y, position.z))

    coordinates = np.array(coordinates)
    
    return protons, neutrons, electrons, formal_charges, isotopes, coordinates


# Molecules 
suppl = Chem.SDMolSupplier('coumarins.sdf', removeHs=False)
molecules = [mol for mol in suppl if mol is not None]
#molecules_2 = Chem.SDMolSupplier('sample3d_optimized_switched.sdf')


for i, molecule in enumerate(molecules):
    protons, neutrons, electrons, formal_charges, isotopes, coordinates = get_atoms_info(molecule)
    print(f'Molecule {i + 1}:')
    print("Protons:", protons)
    print("Neutrons:", neutrons)
    print("Electrons:", electrons)
    print("Formal charges:", formal_charges)
    print("Isotopes:", isotopes)

