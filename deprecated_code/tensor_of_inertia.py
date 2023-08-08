import sys
import numpy as np
import math

# Atomic mass unit (kg), Planck constant (J s), speed of light (m s-1)
u, h, c = 1.66053886e-27, 6.62606957e-34, 2.99792458e8


# TETRAHEDRON
# coordinates
xyz = np.array([
     [  1,  0, -1/math.sqrt(2)],
     [ -0.5,  0, -1/math.sqrt(2)],
     [  0,  1,  1/math.sqrt(2)],
     [  0, -1,  1/math.sqrt(2)]
 ])
# masses
masses_1 = np.array([1, 3, 5, 7 ])

### CENTER OF MASS ###
def translate_to_cofm(masses, xyz):
    # Position of centre of mass in original coordinates
    cofm = sum(masses[:,np.newaxis] * xyz) / np.sum(masses)
    # Transform to CofM coordinates and return
    xyz -= cofm
    return xyz

#### MOMENT OF INERTIA TENSOR ###
def get_inertia_matrix(masses, xyz):
    # Moment of intertia tensor
    xyz = translate_to_cofm(masses, xyz)
    x, y, z = xyz.T
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * x * y)
    Iyz = -np.sum(masses * y * z)
    Ixz = -np.sum(masses * x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    return I

##### PRINCIPAL MOMENTS OF INERTIA, AXIS #####
def get_principal_moi(I):
    Ip = np.linalg.eigvals(I)
    # Sort and convert principal moments of inertia to SI (kg.m2)
    Ip.sort()
    return Ip

##### ROTOR TYPE  #####
##TODO: build on that to compute handedness
def classify_molecule(A, B, C):
    if np.isclose(A, B):
        if np.isclose(B, C):
            return 'Spherical top'
        return 'Oblate symmetric top'
    if np.isclose(B, C):
        return 'Prolate symmetric top'
    return 'Asymmetric top'

I = get_inertia_matrix(masses_1, xyz)

Ip = get_principal_moi(I)
Ip *= u / 1.e20
A, B, C = h / 8 / np.pi**2 / c / 100 / Ip
rotor_type = classify_molecule(A, B, C)

print('{}: A={:.6f}, B={:.6f}, C={:.6f} cm-1'.format(rotor_type, A, B, C))




# def read_xyz(filename):
#     try:
#         data = np.loadtxt(filename, skiprows=2)
#     except FileNotFoundError:
#         print('No such file:', filename)
#         sys.exit(1)
#     except ValueError as e:
#         print('Malformed data in {}: {}'.format(filename, e))
#         sys.exit(1)
#     return data[:,0], data[:,1:]


# try:
#     masses, xyz = read_xyz(sys.argv[1])
# except IndexError:
#     print('Usage: {} <xyz filename>'.format(sys.argv[0]))
#     sys.exit(1)