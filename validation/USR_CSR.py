
import numpy as np
from numpy.linalg import norm
from scipy.stats import skew

def usr(molecule):
    """Function used in USR calculation
    """

    coordinates = []
    
    for atom in molecule.GetAtoms():
        # Skip hydrogens
        if atom.GetAtomicNum() == 1:
            continue
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        coordinates.append([position.x, position.y, position.z])

    # if len(coordinates) == 0:
    #     return np.zeros(12), ((0., 0., 0.),) * 4
    
    # Calculate the centroid of the molecule
    ctd = np.mean(coordinates, axis=0)
    # Calculate the distance of each atom from the centroid
    distances_ctd = norm(coordinates - ctd, axis=1)

    # Calculate the closest atom to the centroid
    cst = np.array(coordinates[distances_ctd.argmin()])
    # Calculate the distance of each atom from the closest atom to the centroid
    distances_cst = norm(coordinates - cst, axis=1)

    # Calculate the farthest atom to the centroid
    fct = np.array(coordinates[distances_ctd.argmax()])
    # Calculate the distance of each atom from the farthest atom to the centroid
    distances_fct = norm(coordinates - fct, axis=1)

    
    # Calculate the farthest atom to the farthest atom to the centroid
    ftf = np.array(coordinates[distances_fct.argmax()])
    # Calculate the distance of each atom from the farthest atom to the farthest atom to the centroid
    distances_ftf = norm(coordinates - ftf, axis=1)

    distances_list = [distances_ctd, distances_cst, distances_fct, distances_ftf]

    shape_descriptor = np.zeros(12)

    for i, distances in enumerate(distances_list):
        shape_descriptor[i * 3 + 0] = np.mean(distances)
        shape_descriptor[i * 3 + 1] = np.std(distances)
        shape_descriptor[i * 3 + 2] = np.nan_to_num(skew(distances, bias=False), nan=0.0)

    return shape_descriptor 

def csr(molecule):
    """Computes shape descriptor based on CSR
    """
    coordinates = []
    
    for atom in molecule.GetAtoms():
        # Skip hydrogens
        if atom.GetAtomicNum() == 1:
            continue
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        coordinates.append([position.x, position.y, position.z])

    if len(coordinates) == 0:
        return np.zeros(12), ((0., 0., 0.),) * 4
    
    # Calculate the centroid of the molecule
    c1 = np.mean(coordinates, axis=0)
    # Calculate the distance of each atom from the centroid
    distances_c1 = norm(coordinates - c1, axis=1)

    # Calculate the furthest atom to the centroid
    c2 = np.array(coordinates[distances_c1.argmax()])
    # Calculate the distance of each atom from the furthest atom to the centroid
    distances_c2 = norm(coordinates - c2, axis=1)

    # Calculate the furthest atom from c2
    c3 = np.array(coordinates[distances_c2.argmax()])
    # Calculate the distance of each atom from the furthest atom from c2
    distances_c3 = norm(coordinates - c3, axis=1)

    vector_a = c2 - c1
    vector_b = c3 - c1
   
    vector_c = np.array(((norm(vector_a) /
                (2 * norm(np.cross(vector_a, vector_b))))
                * np.cross(vector_a, vector_b)))

    # Calculate c4 as the centroid of the molecule + vector_c
    c4 = c1 + vector_c
    # Calculate the distance of each atom from c4
    distances_c4 = norm(coordinates - c4, axis=1)
    
    distances_list = [distances_c1, distances_c2, distances_c3,
                      distances_c4,]

    shape_descriptor = np.zeros(12)

    i = 0
    for distances in distances_list:
        shape_descriptor[0 + i] = np.mean(distances)
        shape_descriptor[1 + i] = np.std(distances)
        shape_descriptor[2 + i] = np.nan_to_num(skew(distances, bias=False), nan=0.0)
        i += 3
    return shape_descriptor

def similarity(mol1_shape, mol2_shape):
    """Computes similarity between molecules
    """
    sim = 1. / (1. + (1. / 12) * np.sum(np.fabs(mol1_shape - mol2_shape)))
    return sim