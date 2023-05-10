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
    coordinates = []
    nutrons_difference = []

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
        masses.append(mass_num)
        coordinates.append((position.x, position.y, position.z))
        if atomic_num == 1 and neutron_num == 0:
            nutrons_difference.append(0)
        else:
            nutrons_difference.append(neutron_num - atomic_num)

    coordinates = np.array(coordinates)

    return masses, protons, nutrons_difference, formal_charges, coordinates
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

