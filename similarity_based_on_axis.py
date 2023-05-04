# Python script for computing the similarity of two mass-weight poiny clouds from fingerprints based on their 
# principle axis of inertia

import numpy as np
import math
import axis_fingerprint as afp
from similarity_3d import calculate_partial_score
import matplotlib.pyplot as plt

# point_cloud_1:
# coordinates
pc_1 = np.array([
     [  1,  0, -1/math.sqrt(2)],
     [ -1,  0, -1/math.sqrt(2)],
     [  0,  1,  1/math.sqrt(2)],
     [  0, -1,  1/math.sqrt(2)]
 ])
# masses
masses_1 = [1, 3, 5, 7 ]

# point_cloud_2:
# coordinates
pc_2 = np.array([
     [  1,  0, -1/math.sqrt(2)],
     [ -1,  0, -1/math.sqrt(2)],
     [  0,  1,  1/math.sqrt(2)],
     [  0, -1,  1/math.sqrt(2)]
 ])
# masses
masses_2 = [1, 3, 7, 5 ] 

# Fingerprints
fingerprint_1, mass_weighted_fingerprint_1 = afp.compute_fingerprint(pc_1, masses_1)
fingerprint_2, mass_weighted_fingerprint_2 = afp.compute_fingerprint(pc_2, masses_2)

# Similarity
# Not mass-weighted similarity
similarity = 1/(1 + calculate_partial_score(fingerprint_1, fingerprint_2))
print(f'Similarity only based on distances: {similarity}')

# Mass-weighted similarity
mass_weighted_similarity = 1/(1 + calculate_partial_score(mass_weighted_fingerprint_1, mass_weighted_fingerprint_2))
print(f'Mass-weighted similarity: {mass_weighted_similarity}')


plt.show()
