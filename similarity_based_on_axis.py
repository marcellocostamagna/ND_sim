# Python script for computing the similarity of two mass-weight poiny clouds from fingerprints based on their 
# principle axis of inertia

import numpy as np
import math
import axis_fingerprint as afp
from similarity_3d import calculate_partial_score
import matplotlib.pyplot as plt
from perturbations import *


# The properties should be:
# 1. Number of protons
# 2. Number of neutrons
# 3. Number of electrons

# BIATOMIC MOLECULES
# point_cloud_1:
# coordinates
pc_1 = np.array([[ 1, 1, 1], [-1, -1, -1] ])
# masses
n_protons_1 = [6, 8 ]
n_neutrons_1 = [6, 8 ]
n_electrons_1 = [6, 8 ]

# point_cloud_2:
# coordinates
pc_2 = np.array([[ 1, 1, 1], [-1, -1, -1] ])
# masses
n_protons_2 = [9, 8]
n_neutrons_2 = [9, 8]
n_electrons_2 = [8, 8]


# Fingerprints
# fingerprint_1, mass_weighted_fingerprint_1 = afp.compute_fingerprint(pc_1, n_protons_1, n_neutrons_1, n_electrons_1)
# fingerprint_2, mass_weighted_fingerprint_2 = afp.compute_fingerprint(pc_2, n_protons_2, n_neutrons_2, n_electrons_2)
proton_fingerprint_1, neutron_fingerprint_1, electron_fingerprint_1 = afp.compute_fingerprint(pc_1, n_protons_1, n_neutrons_1, n_electrons_1)
proton_fingerprint_2, neutron_fingerprint_2, electron_fingerprint_2 = afp.compute_fingerprint(pc_2, n_protons_2, n_neutrons_2, n_electrons_2)

# Similarity
# Not mass-weighted similarity
# similarity = 1/(1 + calculate_partial_score(fingerprint_1, fingerprint_2))
# print(f'Similarity only based on distances: {similarity}')

# # Mass-weighted similarity
# mass_weighted_similarity = 1/(1 + calculate_partial_score(mass_weighted_fingerprint_1, mass_weighted_fingerprint_2))
# print(f'Mass-weighted similarity: {mass_weighted_similarity}')

# proton,neutron,elctron similarity
# average implementation
proton_similarity = 1/(1 + calculate_partial_score(proton_fingerprint_1, proton_fingerprint_2))
neutron_similarity = 1/(1 + calculate_partial_score(neutron_fingerprint_1, neutron_fingerprint_2))
electron_similarity = 1/(1 + calculate_partial_score(electron_fingerprint_1, electron_fingerprint_2))
similarity_mean = np.mean([proton_similarity, neutron_similarity, electron_similarity])
print(f'Similarity as mean = {similarity_mean}')

# USRCAT implementation
ps_1 = calculate_partial_score(proton_fingerprint_1, proton_fingerprint_2)
ps_2 = calculate_partial_score(neutron_fingerprint_1, neutron_fingerprint_2)
ps_3 = calculate_partial_score(electron_fingerprint_1, electron_fingerprint_2)

similarity_USRCAT = 1/(1 + np.sum([ps_1, ps_2, ps_3]))
print(f'Similarity as USRCAT = {similarity_USRCAT}')

similarity_USRCAT_mean = 1/(1 + np.mean([ps_1, ps_2, ps_3]))
print(f'Similarity as USRCAT mean = {similarity_USRCAT_mean}')

plt.show()


# # BIATOMIC MOLECULES
# # point_cloud_1:
# # coordinates
# pc_1 = np.array([[ 1, 1, 1], [-1, -1, -1] ])
# # masses
# masses_1 = [1, 3 ]

# # point_cloud_2:
# # coordinates
# pc_2 = np.array([[ 1, 1, 1], [-1, -1, -1] ])
# # masses
# masses_2 = [3, 1 ]

# # TRIATOMIC MOLECULES
# # point_cloud_1:
# # coordinates
# pc_1 = np.array([[ 1, 1, 1], [-1, -1, -1], [0, 0, 0] ])
# # masses
# masses_1 = [1, 3, 5 ]

# # point_cloud_2:
# # coordinates
# pc_2 = np.array([[ 1, 1, 1], [-1, -1, -1], [0, 0, 0] ])
# # masses
# masses_2 = [3, 1, 5 ]

# # TRIATOMIC MOLECULES in a Triangle
# # point_cloud_1:
# # coordinates
# pc_1 = np.array([[ 0, 0, 0], [1, 0, 0], [0.5, 0.5*math.sqrt(3) , 0] ])
# # masses
# masses_1 = [1, 3, 5 ]

# # point_cloud_2:
# # coordinates
# pc_2 = np.array([[ 0, 0, 0], [1, 0, 0], [0.5, 0.5*math.sqrt(3) , 0] ])
# # masses
# masses_2 = [3, 1, 5 ]


# # TETRAHEDRONS
# # coordinates
# pc_1 = np.array([
#      [  1,  0, -1/math.sqrt(2)],
#      [ -1,  0, -1/math.sqrt(2)],
#      [  0,  1,  1/math.sqrt(2)],
#      [  0, -1,  1/math.sqrt(2)]
#  ])
# # masses
# masses_1 = [1, 3, 5, 7 ]

# # point_cloud_2:
# # coordinates
# pc_2 = np.array([
#      [  1,  0, -1/math.sqrt(2)],
#      [ -1,  0, -1/math.sqrt(2)],
#      [  0,  1,  1/math.sqrt(2)],
#      [  0, -1,  1/math.sqrt(2)]
#  ])
# # masses
# masses_2 = [1, 3, 7, 5 ] 