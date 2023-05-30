# Ultrafast Shape Recognition with Elements 
# Python script to generate the USRE fingerprint of molecules
# and compute their similarity

from rdkit import Geometry as geom
import numpy as np

########### utils ###########

def get_number_of_atoms_of_element(element: str, mol):
    """Get the number of atoms of a given element in the molecule."""
    counter = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == element:
            counter += 1
    return counter

def get_number_of_atoms_with_label(label, mol):
    """Get the number of atoms with a given label in the molecule."""
    counter = 0
    for atom in mol.GetAtoms():
        if atom.GetProp('label') == label:
            counter += 1
    return counter

########### Functions to generate the USRE fingerprint of a molecule ###########

def calculate_centroid(molecule):
    """Calculate the centroid of a molecule."""
    centroid = [0, 0, 0]
    for atom in molecule.GetAtoms():
        coords = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        for i in range(3):
            centroid[i] += coords[i]
    for i in range(3):
        centroid[i] /= molecule.GetNumAtoms()
    return geom.Point3D(*centroid)

def calculate_fixed_points(mol):
    """
    Calculate the fixed points for the USR:
     
    ctd: centroid of the molecule
    ctd: closest atom to centroid
    fct: farthest atom from centroid
    ftf: farthest atom from fct
    """
    fixed_points = []
    # Calculate the centroid of the molecule
    centroid = calculate_centroid(mol)
    fixed_points.append(centroid)
    # Calculate the closest atom to the centroid
    min_distance = np.inf
    for atom in mol.GetAtoms():
        coords = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        distance = coords.Distance(centroid)
        if distance < min_distance:
            min_distance = distance
            ctd = coords
    fixed_points.append(ctd)

    # Calculate the farthest atom from the centroid
    max_distance = 0
    for atom in mol.GetAtoms():
        coords = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        distance = coords.Distance(centroid)
        if distance > max_distance:
            max_distance = distance
            fct = coords
    fixed_points.append(fct)

    # Calculate the farthest atom from fct
    max_distance = 0
    for atom in mol.GetAtoms():
        coords = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        distance = coords.Distance(fct)
        if distance > max_distance:
            max_distance = distance
            ftf = coords
    fixed_points.append(ftf)

    return fixed_points
    
def get_list_of_unique_atoms(mol):
    """Get the list of unique atoms in the molecule."""
    unique_atoms = []
    for atom in mol.GetAtoms():
        mass = round(atom.GetMass())
        charge = atom.GetFormalCharge()
        symbol = atom.GetSymbol()
        label = f'{symbol}_{mass}_{charge}'
        atom.SetProp('label', label)
        if label not in unique_atoms:
            unique_atoms.append(label)
    return unique_atoms

def get_distributions(coords: list, fixed_points):
    """
    Calculate the distance distributions of the a set of coordinates
    for each fixed point.
    """
    distributions = []
    for point in fixed_points:
        distribution = []
        for coord in coords:
            distribution.append(coord.Distance(point))
        distributions.append(distribution)
    return distributions

def moment1(distribution):
    """
    Calculate the first moment of the distance distribution
    aka the mean of the distribution.
    """
    mean = sum(distribution)/len(distribution)
    return mean

def moment2(distribution, mean):
    """
    Calculate the second moment of the distance distribution
    aka the variance of the distribution.
    """
    variance = sum([(x - mean)**2 for x in distribution])/len(distribution)
    if variance < 1e-2:
        variance = 0
    return variance

def moment3(distribution, mean, variance):
    """
    Calculate the third moment of the distance distribution
    aka the skewness of the distribution.
    """
    if variance == 0:
        skewness = 0
    else:
        skewness = sum([((x - mean)/variance)**3 for x in distribution])/len(distribution)
    return skewness

def calculate_usr_moments(coordinates: list, reference_points):
    """
    Calculate the USR moments for the list of coordinates with regard to the fixed points.
    """
    # Calculate the distance distributions
    distributions = get_distributions(coordinates, reference_points)
    # Calculate the moments
    moments = []
    for distribution in distributions:
        mean = moment1(distribution)
        variance = moment2(distribution, mean)
        skewness = moment3(distribution, mean, variance)
        moments.append(mean)
        moments.append(variance)
        moments.append(skewness)
    return moments

def get_element_coordinates(mol, element):
    """Get the coordinates of all the occurences of an element in the molecule."""
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == element:
            coords.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
    return coords

def get_coordinates_from_label(mol, label):
    """Get the coordinates of all the atoms with a given label in the molecule."""
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetProp('label') == label:
            coords.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
    return coords

def get_all_coordinates(mol):
    """Get the coordinates of all the atoms in the molecule."""
    coords = []
    for atom in mol.GetAtoms():
        coords.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
    return coords

def get_coordinates_and_masses(mol):
    coords = []
    masses = []
    for atom in mol.GetAtoms():
        coords.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
        masses.append(atom.GetMass())
    return coords, masses

# def generate_usre_fingerprint(mol, reference_points):
#     """Generate the USR fingerprint for a molecule."""
#     fingerprint = {}
#     # Calculate the fixed points
#     #reference_points = calculate_fixed_points(mol)
#     #(Alternative)
#     #reference_points = get_alternative_fix_points_1(reference_points)
#     #reference_points = get_alternative_fix_points_2(reference_points)
#     # Calculate the standard USR moments for the entire molecule
#     coords = get_all_coordinates(mol)
#     moments = calculate_usr_moments(coords, reference_points)
#     # Add the moments to the fingerprint (first entry of the dictionary)
#     fingerprint['molecule'] = moments
#     # Calculate the USR moments for each element in the molecule
#     # Get the list of unique atoms in the molecule
#     unique_atoms = get_list_of_unique_atoms(mol)
#     for element in unique_atoms:
#         # Get the coordinates of the element
#         coords = get_element_coordinates(mol, element)
#         # Calculate the USR moments for the element
#         moments = calculate_usr_moments(coords, reference_points)
#         # Add the moments to the fingerprint
#         fingerprint[element] = moments
#     return fingerprint

def generate_usre_fingerprint(mol, reference_points):
    """Generate the USR fingerprint for a molecule."""
    fingerprint = {}
    # Calculate the standard USR moments for the entire molecule
    coords = get_all_coordinates(mol)
    moments = calculate_usr_moments(coords, reference_points)
    # Add the moments to the fingerprint (first entry of the dictionary)
    fingerprint['molecule'] = moments
    # Calculate the USR moments for each label in the molecule
    # Get the list of unique labels in the molecule
    unique_labels = get_list_of_unique_atoms(mol)
    for label in unique_labels:
        # Get the coordinates of the element
        coords = get_coordinates_from_label(mol, label)
        # Calculate the USR moments for the element
        moments = calculate_usr_moments(coords, reference_points)
        # Add the moments to the fingerprint
        fingerprint[f'{label}'] = moments
    return fingerprint


################### Isomers Discrimination ###################

def get_alternative_fix_points_1(fix_points):
    """
    Cross product of the vectors between the ctd, the ctc and the fct
    """
    v_1 = fix_points[1] - fix_points[0]
    v_2 = fix_points[2] - fix_points[0]
    # Cross product
    v_3 = v_1.CrossProduct(v_2)
    # TODO: normalize v_3 based on the similarity of the two molecules
    #(If the molecules have exatcly the same shape, what is the minimum value of v3)
    #Normalize v_3
    v_3.Normalize()
    # Calculate the new fix point
    new_fix_point = fix_points[0] + v_3

    return [fix_points[0], fix_points[1], fix_points[2], new_fix_point]

def get_alternative_fix_points_2(fix_points):
    """
    Cross product of the vectors between the ctd, the fct and the ftf
    (As in ElctroShape)
    """
    v_1 = fix_points[2] - fix_points[0]
    v_2 = fix_points[3] - fix_points[0]
    # Cross product
    v_3 = v_1.CrossProduct(v_2)
    # TODO: normalize v_3 based on the similarity of the two molecules
    # Normalize v_3
    v_3.Normalize()
    # Calculate the new fix point
    new_fix_point = fix_points[0] + v_3

    return [fix_points[0], fix_points[2], fix_points[3], new_fix_point]


################### Functions for calculating the similarity score ###################

def calculate_partial_score(moments1: list, moments2: list):
    """Calculate the partial score between two molecules. """
    partial_score = 0
    for i in range(12):
        partial_score += abs(moments1[i] - moments2[i])
    return partial_score / 12

def get_keys_to_iterate_on(fingerprint_1, fingerprint_2):
    """Get the keys to iterate on."""
    # Find the atoms shared between the two molecules
    intersection = (fingerprint_1.keys() & fingerprint_2.keys())
    # Find the atoms unique to molecule 1
    diff1 = fingerprint_1.keys() - fingerprint_2.keys()
    # Find the atoms unique to molecule 2
    diff2 = fingerprint_2.keys() - fingerprint_1.keys()
    # List of atoms (keys) we want to iterate on
    atoms = list(intersection) + list(diff1) + list(diff2)
    return atoms, intersection, diff1, diff2

# def compute_USRE_similarity(fingerprint_target, fingerprint_query, target_mol, query_mol):
#     """Compute the USRE similarity between two molecules."""
#     # Get the keys to iterate on
#     _, intersection, _, _ = get_keys_to_iterate_on(fingerprint_target, fingerprint_query)

#     # First component (standard USR)
#     # Calculate the partial score for the entire molecule
#     partial_score = calculate_partial_score(fingerprint_target['molecule'], fingerprint_query['molecule'])
#     usr_similarity = 1/(1 + partial_score)

#     # Second component (shared elements)
#     # Calculate the partial score for the shared elements
#     partial_score_shared = 0
#     shared_atoms_similarity = 0
#     shared_atoms_target = 0
#     shared_atoms_query = 0
#     # Get the total number of shared atoms in the molecules
#     for atom in intersection - {'molecule'}:
#         shared_atoms_target += get_number_of_atoms_of_element(atom, target_mol)
#         shared_atoms_query += get_number_of_atoms_of_element(atom, query_mol)
#     total_shared_atoms = shared_atoms_target + shared_atoms_query    

#     # Iterate through the shared elements but not molecule
#     for atom in intersection - {'molecule'}:
#         partial_score_shared = calculate_partial_score(fingerprint_target[atom], fingerprint_query[atom])
#         local_coefficient = (get_number_of_atoms_of_element(atom, target_mol) + get_number_of_atoms_of_element(atom, query_mol))/total_shared_atoms
#         shared_atoms_similarity += (1/(1 + partial_score_shared)) * local_coefficient

#     # Third component (unshared elements)
#     f = (shared_atoms_target + shared_atoms_query)/(target_mol.GetNumAtoms() + query_mol.GetNumAtoms())

#     # Final similarity score
#     similarity = ( usr_similarity +  shared_atoms_similarity + f)/3  

#     return similarity

def compute_USRE_similarity(fingerprint_target, fingerprint_query, target_mol, query_mol):
    """Compute the USRE similarity between two molecules."""
    # Get the keys to iterate on
    _, intersection, _, _ = get_keys_to_iterate_on(fingerprint_target, fingerprint_query)

    # First component (standard USR)
    # Calculate the partial score for the entire molecule
    partial_score = calculate_partial_score(fingerprint_target['molecule'], fingerprint_query['molecule'])
    usr_similarity = 1/(1 + partial_score)

    # Second component (shared elements)
    # Calculate the partial score for the shared elements
    partial_score_shared = 0
    shared_atoms_similarity = 0
    shared_atoms_target = 0
    shared_atoms_query = 0
    # Get the total number of shared atoms in the molecules
    for atom in intersection - {'molecule'}:
        shared_atoms_target += get_number_of_atoms_with_label(atom, target_mol)
        shared_atoms_query += get_number_of_atoms_with_label(atom, query_mol)
    total_shared_atoms = shared_atoms_target + shared_atoms_query    

    # Iterate through the shared elements but not molecule
    for atom in intersection - {'molecule'}:
        partial_score_shared = calculate_partial_score(fingerprint_target[atom], fingerprint_query[atom])
        local_coefficient = (get_number_of_atoms_with_label(atom, target_mol) + get_number_of_atoms_with_label(atom, query_mol))/total_shared_atoms
        shared_atoms_similarity += (1/(1 + partial_score_shared)) * local_coefficient

    # Third component (unshared elements)
    f = (shared_atoms_target + shared_atoms_query)/(target_mol.GetNumAtoms() + query_mol.GetNumAtoms())

    # Final similarity score
    similarity = ( usr_similarity +  shared_atoms_similarity + f)/3  

    return similarity