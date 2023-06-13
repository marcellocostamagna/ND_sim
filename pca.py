import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import math
from nD_tools import *
from perturbations import *
from fingerprints import * 
from similarity_3d import calculate_partial_score


# TETRAHEDRON
data = np.array([
    [  1,  0, -1/math.sqrt(2),  7],
    [  0,  0,  0,               6],
    [ -1,  0, -1/math.sqrt(2),  1],
    [  0,  1.1,  1/math.sqrt(2),  8],
    [  0, -1,  1/math.sqrt(2),  9],
])

print(data)

# tetrahedron rotated 90 degrees around x axis
# data_rotated = np.array([
#     [  1,  1/math.sqrt(2),  0,  7],
#     [  0,  0,               0,  6],
#     [ -1,  1/math.sqrt(2),  0,  1],
#     [  0, -1/math.sqrt(2),  1,  8],
#     [  0, -1/math.sqrt(2), -1,  9],
# ])

# tetrahedron rotated 90 degrees around y axis
# data_rotated = np.array([
#     [  1/math.sqrt(2),  0, -1, 7],
#     [  0,              0,  0, 6],
#     [  1/math.sqrt(2),  0,  1, 1],
#     [ -1/math.sqrt(2),  1,  0, 8],
#     [ -1/math.sqrt(2), -1,  0, 9]
# ])

# tetrahedron rotated 90 degrees around z axis
# data_rotated = np.array([
#     [  0,  1, -1/math.sqrt(2), 7],
#     [  0,  0,  0,              6],
#     [  0, -1, -1/math.sqrt(2), 1],
#     [ -1,  0,  1/math.sqrt(2), 8],
#     [  1,  0,  1/math.sqrt(2), 9]
# ])

# # Center the data
# data = data - np.mean(data, axis=0)

# Perform PCA on the 4D data
pca = PCA(n_components=4)
pca.fit(data)

# The principal axes in feature space, representing the directions of maximum variance in the data.
# These are your "direction vectors" for each principal axis
axes = pca.components_
eigenvalues = pca.explained_variance_


print('Principal components')
for i in np.argsort(eigenvalues)[::-1]:
    print(eigenvalues[i],'->',axes[i])


# Compute the centroid
centroid = np.mean(data, axis=0)

# Compute the 4 reference points along each axis
reference_points = [centroid + axis for axis in axes]
reference_points.append(centroid)

print(reference_points)

# Compute the Euclidean distance of each point from each reference point
distances = np.empty((data.shape[0], len(reference_points)))
for i, point in enumerate(data):
    for j, ref_point in enumerate(reference_points):
        distances[i, j] = distance.euclidean(point, ref_point)

print(distances)

fingerprint = compute_statistics(distances)

print(fingerprint)

# Rotate points
coords = data[:, :-1]
masses = data[:, -1]

coords, rotation_matrix = rotate_points(coords, 0, 0, 90)

data1 = np.c_[coords, masses]
#data1 = data_rotated

print(data1)

# # Center the data
# data1 = data1 - np.mean(data1, axis=0)

# Perform PCA on the 4D data
pca1 = PCA(n_components=4)
pca1.fit(data1)

axes1 = pca1.components_
eigenvalues1 = pca1.explained_variance_


print('Principal components')
for i in np.argsort(eigenvalues1)[::-1]:
    print(eigenvalues1[i],'->',axes1[i])

centroid1 = np.mean(data1, axis=0)

reference_points1 = [centroid1 + axis for axis in axes1]
reference_points1.append(centroid1)

print(reference_points1)

distances1 = np.empty((data1.shape[0], len(reference_points1)))
for i, point in enumerate(data1):
    for j, ref_point in enumerate(reference_points1):
        distances1[i, j] = distance.euclidean(point, ref_point)

print(distances1)

fingerprint1 = compute_statistics(distances1)

print(fingerprint1)

similarity = 1 / (1 + calculate_nD_partial_score(fingerprint, fingerprint1))
print(f'Similarity: {similarity}')