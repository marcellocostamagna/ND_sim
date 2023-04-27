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
    return atoms, intersection, diff1, diff2

def maximize_unshared_atoms_similarity(fingerprint_target, fingerprint_query):
    """
    Maximize the similarity of the unshared atoms by matching the unshared atoms
    with the highest similarity score. If there are more unshared atoms in one molecule,
    their number is used as a penalty.
    (Note: maximize the similarity means minimize their partial scores)
    """
    # Get the keys to iterate on
    _, _, diff1, diff2 = get_keys_to_iterate_on(fingerprint_target, fingerprint_query)

    # If both molecules have unshared atoms
    # create two dictionaries for the query and the target with only the unshared atoms
    # and put them in a list ordered from the smallest to the largest dictionary (based on the number of keys)
    unshared_atoms_target = {key: fingerprint_target[key] for key in diff1}
    unshared_atoms_query = {key: fingerprint_query[key] for key in diff2}
    unshared_atoms = [unshared_atoms_target, unshared_atoms_query]
    # sort the list of dictionaries based on the number of keys
    unshared_atoms.sort(key=len)
    # create a list of all possible combinations of keys from the two dictionaries
    # create a dictionary to store the highest value for each key in the smallest dictionary
    lowest_values = {}
    # iterate through each key in the smallest dictionary and find the highest value
    for key in unshared_atoms[0].keys():
        lowest_value = float('inf')
        for key_1 in unshared_atoms[1].keys():
            value1 = unshared_atoms[0].get(key)
            value2 = unshared_atoms[1].get(key_1)
            partial_score = calculate_partial_score(value1, value2)
            if partial_score < lowest_value:
                lowest_value = partial_score
                lowest_values[key] = (key, key_1, partial_score)
    
    #unused_keys = [key for key in unshared_atoms[1].keys() if key not in lowest_values.keys()]
    #unused_keys = abs(len(diff1)-len(diff2))
    # calculate the similarity 
    # Now we calculate the similarity of each couple and we make an average
    # Now we accept the similarity between unshared atoms moments only if the similarity is greater than 0.7
    similarity = 0
    for key in lowest_values.keys():
        if lowest_values[key][2] < 0.42:
            similarity += 1/(1 + lowest_values[key][2]) 
    similarity = similarity/len(lowest_values.keys())

    return similarity #, unused_keys

# Alternative implementation of the USRE similarity score
# The difference with the USRCAT implementation is that the contributions from the positions of the atoms
# in the all molecule (aka. standard USR), the positions of the atoms of the shared elements and the positions
# of the atoms of the unshared elements are separated and weighted differently.
def compute_USRE_similarity_components(fingerprint_target, fingerprint_query, target_mol, query_mol):
    # Get the keys to iterate on
    _, intersection, diff1, diff2 = get_keys_to_iterate_on(fingerprint_target, fingerprint_query)
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
        shared_atoms_target += get_number_of_atoms_of_element(atom, target_mol)
        shared_atoms_query += get_number_of_atoms_of_element(atom, query_mol)
    total_shared_atoms = shared_atoms_target + shared_atoms_query    

    # Iterate through the shared elements but not molecule
    for atom in intersection - {'molecule'}:
        partial_score_shared = calculate_partial_score(fingerprint_target[atom], fingerprint_query[atom])
        # Thise partial similarity should be weighted by the number of atoms of that element in the molecules
        # The coefficient for the shared atoms is the number of atoms of that element in the molecules divided by the total number of shared atoms
        local_coefficient = (get_number_of_atoms_of_element(atom, target_mol) + get_number_of_atoms_of_element(atom, query_mol))/total_shared_atoms
        shared_atoms_similarity += (1/(1 + partial_score_shared)) * local_coefficient
    #shared_atoms_similarity = 1/(1 + partial_score_shared)
    #shared_atoms_similarity = shared_atoms_similarity/len(intersection - {'molecule'})
    
    # Third component (unshared elements)
    # if there are not unshared atoms there is no such contribution to the similarity score
    # otherwise, maximize the similarity of the unshared atoms by matching the unshared atoms


    if len(diff1) != 0 and len(diff2) != 0:
        unshared_atoms_similarity = maximize_unshared_atoms_similarity(fingerprint_target, fingerprint_query)
        similarities = [usr_similarity, shared_atoms_similarity, unshared_atoms_similarity]
    # If only one molecule has unshared atoms
    elif len(diff1) == 0 and len(diff2) == 0:
        #unpaired_atoms = 0
        similarities = [usr_similarity, shared_atoms_similarity]
    elif len(diff1) == 0 or len(diff2) == 0:    
        #unpaired_atoms = max(len(diff1), len(diff2))
        similarities = [usr_similarity, shared_atoms_similarity]
    
    
    return similarities #, unpaired_atoms

def compute_coefficients(similarities, target_mol, query_mol, fingerprint_target, fingerprint_query):
    # Get the keys to iterate on
    _, intersection, diff1, diff2 = get_keys_to_iterate_on(fingerprint_target, fingerprint_query)

    # Based on the two or thre components and the molecules, we evaluate the coefficients 
    # for each contribution 

    # Coefficient for the standard USR
    total_atoms = target_mol.GetNumAtoms() + query_mol.GetNumAtoms()
    difference = abs(target_mol.GetNumAtoms() - query_mol.GetNumAtoms())
    # Calculate the coefficient
    c_usr = 1 - difference/total_atoms

    # Coefficient for the shared atoms
    # for each atom in the intersection we get its number in both molecules
    count_target = 0
    count_query = 0
    for atom in intersection - {'molecule'}:
        # count the number of atoms with symbol atom in the target molecule
        count_target += get_number_of_atoms_of_element(atom, target_mol)
        count_query += get_number_of_atoms_of_element(atom, query_mol)
    # Calculate the coefficient
    c_shared_1 = (count_target + count_query)/(target_mol.GetNumAtoms() + query_mol.GetNumAtoms())
    # TODO: this coefficient could be influenced by the value or the usr_similarity,
    # the higher the similarity the higher the coefficient
    c_shared_2 = similarities[0]
    c_shared = (c_shared_1 + c_shared_2)/2
    # c_shared = c_shared_1
    # Coefficient for the unshared atoms
    if len(similarities) > 2:
        # weight from the number of unshared atoms in both molecules
        c_unshared_1 = 1 - c_shared_1
        # weight based on the other similarities (the higher the similarity the lower the weight)
        # TODO: this overweights the similarity of the unshared atoms
        #c_unshared_2 = 1 - ((similarities[0]+similarities[1])/2)
        c_unshared_2 = (similarities[0] + similarities[1])/2 
        c_unshared = (c_unshared_1 + c_unshared_2)/2
        #c_unshared = c_unshared_1
        coefficients = [c_usr, c_shared, c_unshared]
    else: 
        coefficients = [c_usr, c_shared]
    
    # Calculate the penalty. As the number of unshared atoms increases, the penalty values decreases
    # and hence producing a further lowering of the total similarity score
    # This way the penalty overkills the total similarity score (it has to insrease in value)
    #penalty = (c_shared_1 + similarities[0] + similarities[1])/3
    # Penalty weighted by the coefficients of the similarities
    #penalty = (c_shared_1 + c_usr * similarities[0] + c_shared * similarities[1])/3
    #penalty = (c_shared_1 + similarities[0] + similarities[1])/3
    penalty = (c_shared_1 +  c_usr *similarities[0] +  similarities[1])/3
    print('penalty: ', penalty)

    return coefficients, penalty

def compute_USRE_similarity(similarities, coefficients, penalty):
    # Compute the USRE similarity as a weighted average of the similarities
    similarity = 0
    
    # weighted by the coefficients
    similarity = (sum([similarities[i]*coefficients[i] for i in range(len(similarities))])\
                  /sum(coefficients))*penalty
                  

    return similarity



#    def compute_coefficients(similarities, unused_keys, target_mol, query_mol, fingerprint_target, fingerprint_query):
#     # Get the keys to iterate on
#     _, intersection, diff1, diff2 = get_keys_to_iterate_on(fingerprint_target, fingerprint_query)

#     # Based on the two or thre components and the molecules, we evaluate the coefficients 
#     # for each contribution 

#     # Coefficient for the standard USR
#     total_atoms = target_mol.GetNumAtoms() + query_mol.GetNumAtoms()
#     difference = abs(target_mol.GetNumAtoms() - query_mol.GetNumAtoms())
#     # Calculate the coefficient
#     c_usr = 1 - difference/total_atoms

#     # Coefficient for the shared atoms
#     # for each atom in the intersection we get its number in both molecules
#     count_target = 0
#     count_query = 0
#     for atom in intersection - {'molecule'}:
#         # count the number of atoms with symbol atom in the target molecule
#         count_target += get_number_of_atoms_of_element(atom, target_mol)
#         count_query += get_number_of_atoms_of_element(atom, query_mol)
#     # Calculate the coefficient
#     c_shared_1 = (count_target + count_query)/(target_mol.GetNumAtoms() + query_mol.GetNumAtoms())
#     # TODO: this coefficient could be influenced by the value or the usr_similarity,
#     # the higher the similarity the higher the coefficient
#     c_shared_2 = similarities[0]
#     c_shared = (c_shared_1 + c_shared_2)/2
#     # c_shared = c_shared_1
#     # Coefficient for the unshared atoms
#     if len(similarities) > 2:
#         # weight from the number of unshared atoms in both molecules
#         c_unshared_1 = 1 - c_shared_1
#         # weight based on the other similarities (the higher the similarity the lower the weight)
#         # TODO: this overweights the similarity of the unshared atoms
#         #c_unshared_2 = 1 - ((similarities[0]+similarities[1])/2)
#         c_unshared_2 = (similarities[0] + similarities[1])/2 
#         c_unshared = (c_unshared_1 + c_unshared_2)/2
#         #c_unshared = c_unshared_1
        
#         # Compute the influence of the possible unused keys, hence the penalty
#         # For now the penalty will be a multiplicative factor based on the number of unused keys
#         # and the weight of the unshared atoms on the total number of atoms in the two molecules
#         # for more unused keys the c_unshared_1 will be higher
#         penalty_factor = 0
#         for i in range(len(diff1)+len(diff2)):
#             penalty_factor += c_unshared_1
#         penalty = 1 - penalty_factor
#         coefficients = [c_usr, c_shared, c_unshared]
#     else:
#         if unused_keys != 0:
#             penalty_factor = 0
#             c_unshared_1 = 1 - c_shared_1
#             for i in range(len(diff1)+len(diff2)):
#                 penalty_factor += c_unshared_1
#             penalty = 1 - penalty_factor
#             coefficients = [c_usr, c_shared]
#         else:    
#             penalty = 1  
#             coefficients = [c_usr, c_shared]

#     return coefficients, penalty