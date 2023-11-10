# Python scrpit to manage molecules and extract information to compute similarity

import numpy as np

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
    delta_neutrons = []

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
        if neutron_num - atomic_num < 0:    
            delta_neutrons.append(0)
        else:
            delta_neutrons.append(neutron_num - atomic_num)

    coordinates = np.array(coordinates)
    coordinates_no_H = np.array(coordinates_no_H)

    return elements, masses, np.array(protons), np.array(neutrons), np.array(electrons), coordinates, coordinates_no_H, np.array(formal_charges), np.array(delta_neutrons)

def molecule_info(molecule):

    elements, masses, protons, neutrons, electrons, coordinates, coordinates_no_H, formal_charges, delta_neutrons = get_atoms_info(molecule)
    info = {'elements': elements,
            'masses': masses,
            'protons': protons, 
            'neutrons': neutrons, 
            'electrons': electrons, 
            'coordinates': coordinates,
            'coordinates_no_H': coordinates_no_H,
            'formal_charges': formal_charges,
            'delta_neutrons': delta_neutrons }
    return info

def normalize_features(info: dict):
    """
    Normalizes the numbers of protons, neutrons and electrons of a molecule 
    using the range of the coordinates
    """
    # Get the maximum and minimum from the coordinates array
    max = np.amax(info['coordinates'])
    min = np.amin(info['coordinates'])

    # Normalize the features
    # Protons
    p_min = np.amin(info['protons'])
    p_max = np.amax(info['protons'])
    info['protons'] = (info['protons'] - p_min) / (p_max - p_min) * (max - min) + min
    # Neutrons
    n_min = np.amin(info['neutrons'])
    n_max = np.amax(info['neutrons'])
    info['neutrons'] = (info['neutrons'] - n_min) / (n_max - n_min) * (max - min) + min
    # Electrons
    e_min = np.amin(info['electrons'])
    e_max = np.amax(info['electrons'])
    info['electrons'] = (info['electrons'] - e_min) / (e_max - e_min) * (max - min) + min

    return info

def normalize_delta_features(info: dict):
    """
    Normalizes the numbers of protons, delta_neutrons and formal_charges of a molecule 
    using the range of the coordinates
    """
    # Get the maximum and minimum from the coordinates array
    max = np.amax(info['coordinates'])
    min = np.amin(info['coordinates'])

    # Normalize the features
    # Protons
    p_min = np.amin(info['protons'])
    p_max = np.amax(info['protons'])
    info['protons'] = (info['protons'] - p_min) / (p_max - p_min) * (max - min) + min

    # Delta neutrons
    dn_min = np.amin(info['delta_neutrons'])
    dn_max = np.amax(info['delta_neutrons'])
    if dn_min < 0:
        dn_min = 0
    if dn_min != dn_max:
        info['delta_neutrons'] = (info['delta_neutrons'] - dn_min) / (dn_max - dn_min) * (max - min) + min

    # Formal charges
    fc_min = np.amin(info['formal_charges'])
    fc_max = np.amax(info['formal_charges'])
    if fc_min != fc_max:
        info['formal_charges'] = (info['formal_charges'] - fc_min) / (fc_max - fc_min) * (max - min) + min

    return info

def taper_features(info: dict):
    """
    Tapers the numbers of protons, neutrons and electrons of a molecule using a tapering function
    """
    info['protons'] = np.sqrt(info['protons'])
    info['neutrons'] = np.sqrt(info['neutrons'])
    info['electrons'] = np.sqrt(info['electrons'])

    # info['protons'] = np.log(info['protons'] )
    # info['neutrons'] = np.log(info['neutrons'] + 1)
    # info['electrons'] = np.log(info['electrons'] + 1)

    return info

def taper_delta_features(info: dict):
    """
    Tapers the number of protons, delta_neutrons and formal_charges of a molecule using a tapering function
    """

    # info['protons'] = np.sqrt(info['protons'])
    # info['delta_neutrons'] = np.sqrt(info['delta_neutrons'])
    # info['formal_charges'] = np.sqrt(info['formal_charges'])

    info['protons'] = np.log(info['protons'])
    info['delta_neutrons'] = np.log(info['delta_neutrons'] + 1)
    info['formal_charges'] = np.log(info['formal_charges'] + 4)

    return info