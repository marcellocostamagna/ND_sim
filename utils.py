# Python scrpit to manage molecules and extract information to compute similarity

import numpy as np
from rdkit import Chem




def get_atoms_info(molecule):
    """
    Extracts the information of the atoms of a molecule
    """
    protons = []
    neutrons = []
    electrons = []
    formal_charges = []
    masses = []
    elements = []
    coordinates = []
    coordinates_no_H = []

    for atom in molecule.GetAtoms():
        element_symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        mass_num = atom.GetMass()  # Extracts the mass number (isotope) of the atom
        neutron_num = int(round(mass_num)) - atomic_num
        formal_charge = atom.GetFormalCharge()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())  # Get the atom's 3D coordinates

        protons.append(atomic_num)
        neutrons.append(neutron_num)
        electrons.append(atomic_num - formal_charge)  # Adjusted for the formal charge
        formal_charges.append(formal_charge)
        masses.append(mass_num)
        elements.append(element_symbol)
        coordinates.append((position.x, position.y, position.z))
        if element_symbol != 'H':
            coordinates_no_H.append((position.x, position.y, position.z))

    coordinates = np.array(coordinates)

    return elements, masses, protons, neutrons, electrons, coordinates, coordinates_no_H
    #return protons, neutrons, electrons, coordinates


# # Molecules 
# suppl = Chem.SDMolSupplier('coumarins.sdf', removeHs=False)
# molecules = [mol for mol in suppl if mol is not None]
# #molecules_2 = Chem.SDMolSupplier('sample3d_optimized_switched.sdf')


# for i, molecule in enumerate(molecules):
#     protons, neutrons, electrons, formal_charges, isotopes, coordinates = get_atoms_info(molecule)
#     print(f'Molecule {i + 1}:')
#     print("Protons:", protons)
#     print("Neutrons:", neutrons)
#     print("Electrons:", electrons)
#     print("Formal charges:", formal_charges)
#     print("Isotopes:", isotopes)

