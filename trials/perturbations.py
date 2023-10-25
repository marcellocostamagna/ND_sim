# Python script collecting operration to modify, perturb and change the poin clouds 

import numpy as np
import random
from rdkit import Chem

def translate_points(points, x, y, z):
    """
    Translate a set of 3D points by the given x, y, and z values.
    
    Parameters
    ----------
    points : numpy.ndarray
        An n x 3 array of 3D points.
    x : float
        Translation along the x-axis.
    y : float
        Translation along the y-axis.
    z : float
        Translation along the z-axis.

    Returns
    -------
    numpy.ndarray
        An n x 3 array of translated 3D points.
    """

    # Create a translation vector
    translation = np.array([x, y, z])

    # Translate the points
    translated_points = points + translation

    return translated_points


def rotate_points(points, angle1_deg, angle2_deg, angle3_deg):
    """
    Rotate a set of 3D points around the x, y, and z axes by the given angles.
    
    Parameters
    ----------
    points : numpy.ndarray
        An n x 3 array of 3D points.
    angle1_deg : float
        Rotation angle around the x-axis in degrees, range: [-180, 180].
    angle2_deg : float
        Rotation angle around the y-axis in degrees, range: [-180, 180].
    angle3_deg : float
        Rotation angle around the z-axis in degrees, range: [-180, 180].

    Returns
    -------
    numpy.ndarray
        An n x 3 array of rotated 3D points.
    """

    # Convert angles from degrees to radians
    angle1 = np.radians(angle1_deg)
    angle2 = np.radians(angle2_deg)
    angle3 = np.radians(angle3_deg)

    # Rotation matrix around the x-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle1), -np.sin(angle1)],
                   [0, np.sin(angle1), np.cos(angle1)]])

    # Rotation matrix around the y-axis
    Ry = np.array([[np.cos(angle2), 0, np.sin(angle2)],
                   [0, 1, 0],
                   [-np.sin(angle2), 0, np.cos(angle2)]])

    # Rotation matrix around the z-axis
    Rz = np.array([[np.cos(angle3), -np.sin(angle3), 0],
                   [np.sin(angle3), np.cos(angle3), 0],
                   [0, 0, 1]])

    # Combine the rotation matrices
    R_combined = np.dot(Ry, np.dot(Rx, Rz))

    # Apply the combined rotation matrix
    rotated_points = np.dot(R_combined, points.T).T


    return rotated_points

def rotation_matrix(axis, theta):
    """
    Generate the rotation matrix for a given axis and angle (in radians).
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def rotate_molecule(molecule, x_angle_deg, y_angle_deg, z_angle_deg):
    """
    Rotate the molecule about the x, y, and z axes by the specified angles.
    Angles should be provided in degrees.
    """
    # Convert angles to radians
    x_angle_rad = np.radians(x_angle_deg)
    y_angle_rad = np.radians(y_angle_deg)
    z_angle_rad = np.radians(z_angle_deg)

    # Get the rotation matrices
    Rx = rotation_matrix([1,0,0], x_angle_rad)
    Ry = rotation_matrix([0,1,0], y_angle_rad)
    Rz = rotation_matrix([0,0,1], z_angle_rad)

    # Apply rotations to all atom coordinates
    for atom in molecule.GetAtoms():
        pos = molecule.GetConformer().GetAtomPosition(atom.GetIdx()) 
        new_pos = np.dot(Rx, [pos.x, pos.y, pos.z])
        new_pos = np.dot(Ry, new_pos)
        new_pos = np.dot(Rz, new_pos)
        molecule.GetConformer().SetAtomPosition(atom.GetIdx(), Chem.rdGeometry.Point3D(*new_pos))

    return molecule


# def perturb_coordinates(points, decimal_place):
#     """
#     Apply random perturbations to the input 3D points based on the specified decimal place.

#     Parameters:
#     points (numpy.ndarray): A numpy array of shape (n, 3) representing the 3D coordinates of n points.
#     decimal_place (int): The decimal place where the perturbation will take effect.

#     Returns:
#     numpy.ndarray: A new numpy array with the perturbed coordinates.
#     """

#     perturbed_points = np.zeros_like(points)
#     for i, point in enumerate(points):
#         perturbation_range = 10 ** -decimal_place
#         perturbations = np.random.uniform(-perturbation_range * 9, perturbation_range * 9, point.shape)
#         perturbed_points[i] = point + perturbations

#     return perturbed_points

def perturb_coordinates(points, decimal_place, percentage=1.0):
    """
    Apply random perturbations to a subset of the input 3D points based on the specified decimal place.

    Parameters:
    points (numpy.ndarray): A numpy array of shape (n, 3) representing the 3D coordinates of n points.
    decimal_place (int): The decimal place where the perturbation will take effect.
    percentage (float): Percentage of points to perturb. Should be between 0 and 1.

    Returns:
    numpy.ndarray: A new numpy array with the perturbed coordinates.
    """

    # Ensure the percentage is between 0 and 1
    percentage = max(0.0, min(1.0, percentage))
    
    # Number of points to perturb
    num_perturb = int(percentage * len(points))

    # Randomly select a subset of indices
    indices_to_perturb = np.sort(np.random.choice(len(points), num_perturb, replace=False))
        
    perturbed_points = points.copy()
    for i in indices_to_perturb:
        perturbation_range = 10 ** -decimal_place
        perturbations = np.random.uniform(-perturbation_range * 9, perturbation_range * 9, points[i].shape)
        perturbed_points[i] += perturbations

    return perturbed_points

def scale_coordinates(points, s):
    """
    Scale the input 3D points by a given factor while maintaining the relative distances among the points.

    Parameters:
    points (numpy.ndarray): A numpy array of shape (n, 3) representing the 3D coordinates of n points.
    s (float): The scaling factor.

    Returns:
    numpy.ndarray: A new numpy array with the scaled coordinates.
    """

    scaled_points = points * s
    return scaled_points

def reflect_points(points):
    """Reflects the points with regard to the yz plane."""
    reflected_points = np.zeros_like(points)
    reflected_points[:, 0] = -points[:, 0]
    reflected_points[:, 1:] = points[:, 1:]
    return reflected_points

def rotate_points_and_get_rotation_matrix(points, angle1_deg, angle2_deg, angle3_deg):
    """
    Rotate a set of 3D points around the x, y, and z axes by the given angles.
    
    Parameters
    ----------
    points : numpy.ndarray
        An n x 3 array of 3D points.
    angle1_deg : float
        Rotation angle around the x-axis in degrees, range: [-180, 180].
    angle2_deg : float
        Rotation angle around the y-axis in degrees, range: [-180, 180].
    angle3_deg : float
        Rotation angle around the z-axis in degrees, range: [-180, 180].

    Returns
    -------
    numpy.ndarray
        An n x 3 array of rotated 3D points.
    """

    # Convert angles from degrees to radians
    angle1 = np.radians(angle1_deg)
    angle2 = np.radians(angle2_deg)
    angle3 = np.radians(angle3_deg)

    # Rotation matrix around the x-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle1), -np.sin(angle1)],
                   [0, np.sin(angle1), np.cos(angle1)]])

    # Rotation matrix around the y-axis
    Ry = np.array([[np.cos(angle2), 0, np.sin(angle2)],
                   [0, 1, 0],
                   [-np.sin(angle2), 0, np.cos(angle2)]])

    # Rotation matrix around the z-axis
    Rz = np.array([[np.cos(angle3), -np.sin(angle3), 0],
                   [np.sin(angle3), np.cos(angle3), 0],
                   [0, 0, 1]])

    # Combine the rotation matrices
    R_combined = np.dot(Ry, np.dot(Rx, Rz))

    # Apply the combined rotation matrix
    rotated_points = np.dot(R_combined, points.T).T


    return rotated_points, R_combined

def permute_atoms(mol):
    """
    Permutes the order of atoms in a molecule without breaking bond information.
    Preserves the 3D coordinates.
    """
    indices = list(range(mol.GetNumAtoms()))
    random.shuffle(indices)
    
    # Create a new molecule
    new_mol = Chem.RWMol()

    # Add atoms to new molecule in permuted order
    new_indices = {}
    for idx in indices:
        atom = mol.GetAtomWithIdx(idx)
        new_idx = new_mol.AddAtom(atom)
        new_indices[idx] = new_idx

    # Traverse bonds in the original molecule and add to new molecule
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        new_mol.AddBond(new_indices[begin_idx], new_indices[end_idx], bond.GetBondType())
        
    # Copy the conformer data, if available
    if mol.GetNumConformers() > 0:
        old_conf = mol.GetConformer()
        new_conf = Chem.Conformer(old_conf.GetNumAtoms())

        for idx in indices:
            pos = old_conf.GetAtomPosition(idx)
            new_conf.SetAtomPosition(new_indices[idx], pos)

        new_mol.AddConformer(new_conf)

    # Update molecule properties from the original molecule
    new_mol.SetProp("_Name", mol.GetProp("_Name"))
    for prop_name in mol.GetPropNames():
        new_mol.SetProp(prop_name, mol.GetProp(prop_name))

    return new_mol.GetMol()

def permute_sdf(input_filename, output_filename):
    """
    Reads an SDF, permutes the order of atoms in each molecule, and writes the result to another SDF.
    """
    supplier = Chem.SDMolSupplier(input_filename, removeHs=False)
    writer = Chem.SDWriter(output_filename)
    
    for mol in supplier:
        if mol:  # Check if molecule was read properly
            permuted_mol = permute_atoms(mol)
            writer.write(permuted_mol)

    writer.close()


def reflect_molecule_coordinate(molecule, coordinate='x'):
    # Create a deep copy of the molecule
    mol_copy = Chem.Mol(molecule)
    
    # Ensure the molecule has conformers
    if not mol_copy.GetNumConformers():
        raise ValueError("The provided molecule does not have any conformers.")
    
    # Get the first conformer; adjust if your molecule has multiple conformers
    conformer = mol_copy.GetConformer(0)
    
    # Loop through all atoms and modify the specified coordinate
    for idx in range(conformer.GetNumAtoms()):
        pos = conformer.GetAtomPosition(idx)
        if coordinate == 'x':
            conformer.SetAtomPosition(idx, (-pos.x, pos.y, pos.z))
        elif coordinate == 'y':
            conformer.SetAtomPosition(idx, (pos.x, -pos.y, pos.z))
        elif coordinate == 'z':
            conformer.SetAtomPosition(idx, (pos.x, pos.y, -pos.z))

    return mol_copy