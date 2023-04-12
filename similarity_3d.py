# Ultrafast Shape Recognition with Elements 
# Python script to generate the USRE fingerprint of molecules
# and compute their similarity

from rdkit import Geometry as geom

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
    min_distance = 1000000
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
        if atom.GetSymbol() not in unique_atoms:
            unique_atoms.append(atom.GetSymbol())
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
    # Description of the distributions:
    # the variable distributions is a list of 4 lists each of size equal 
    # to the number of atoms of a particular element in the molecule 
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
    return variance

def moment3(distribution, mean, variance):
    """
    Calculate the third moment of the distance distribution
    aka the skewness of the distribution.
    """
    if variance == 0:
        skewness = 0
    else:
        skewness = sum([(x - mean)**3 for x in distribution])/len(distribution)/variance**1.5
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

def get_all_coordinates(mol):
    """Get the coordinates of all the atoms in the molecule."""
    coords = []
    for atom in mol.GetAtoms():
        coords.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
    return coords

def generate_usre_fingerprint(mol):
    """Generate the USR fingerprint for a molecule."""
    fingerprint = {}
    # Calculate the fixed points
    reference_points = calculate_fixed_points(mol)
    # Calculate the standard USR moments for the entire molecule
    coords = get_all_coordinates(mol)
    moments = calculate_usr_moments(coords, reference_points)
    # Add the moments to the fingerprint (first entry of the dictionary)
    fingerprint['molecule'] = moments
    # Calculate the USR moments for each element in the molecule
    # Get the list of unique atoms in the molecule
    unique_atoms = get_list_of_unique_atoms(mol)
    for element in unique_atoms:
        # Get the coordinates of the element
        coords = get_element_coordinates(mol, element)
        # Calculate the USR moments for the element
        moments = calculate_usr_moments(coords, reference_points)
        # Add the moments to the fingerprint
        fingerprint[element] = moments
    return fingerprint

def calculate_coefficients(mol):
    """Calculate the coefficients for the similarity score."""
    # Total number of atoms in the molecule
    total_atoms = mol.GetNumAtoms()
    # Initialize the coefficients dictionary
    unique_elements = []
    coefficients = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in unique_elements:
            coefficients[atom.GetSymbol()] = 1 #/total_atoms
            unique_elements.append(atom.GetSymbol())
        # else:    
        #     coefficients[atom.GetSymbol()] += 1/total_atoms
    coefficients['molecule'] = 1     
    return coefficients

################### Functions for calculating the similarity score ###################


def calculate_partial_score(moments1: list, moments2: list):
    """Calculate the partial score between two molecules. """
    partial_score = 0
    for i in range(12):
        partial_score += abs(moments1[i] - moments2[i])
    return partial_score / 12

def calculate_dummy_partial_score(moments):
    """
    Calculate the partial score between two molecules
    for an atom not shared by the molecules. 
    """
    partial_score = 0
    for i in range(12):
        partial_score += abs(moments[i])
    return partial_score/12


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
    return atoms, intersection, diff1


def compute_USRE_similarity(fingerprint_target, fingerprint_query, coefficients):
    """
    Compute the USRE similarity between two molecules.
    """
    elements_to_iterate_on, intersection, diff1 = get_keys_to_iterate_on(fingerprint_target, fingerprint_query)
    score = 0
    for atom in elements_to_iterate_on:

        # Coefficients for the atoms
        # Description of the coefficients:
        # The coefficients are used to weight the contribution of each atom to the similarity score.
        # Each coefficient is the number of atoms of a particular element in the target molecule 
        # divided by the total number of atoms.
        # The coefficient for the entire molecule is 1.
        # The coefficients of unshared atoms are 1 to maximize their negative contribution to the similarity score.

        # If the atom is in the intersection
        if atom in intersection:
            coefficient = coefficients[atom]
            partial_score = calculate_partial_score(fingerprint_target[atom], fingerprint_query[atom])
        # If the atom is unique to molecule 1    
        elif atom in diff1:
            coefficient = coefficients[atom]
            partial_score = calculate_dummy_partial_score(fingerprint_target[atom])
        # If the atom is unique to molecule 2    
        else:
            coefficient = 1
            partial_score = calculate_dummy_partial_score(fingerprint_query[atom])
        # Add the partial score to the total score
        score += partial_score * coefficient
    # Get the similarity from the score
    similarity = 1 /(1 + score)

    return similarity
   