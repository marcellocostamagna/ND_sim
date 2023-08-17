import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.linalg import null_space
import math
from nD_tools import *
from trials.perturbations import *
from fingerprints import * 
from deprecated_code.similarity_3d import calculate_partial_score
from sklearn.preprocessing import StandardScaler

DIST = 1

# TETRAHEDRON
data = np.array([
    [  1,  0, -1/math.sqrt(2),  9],
    [  0,  0,  0,               6],
    [ -1,  0, -1/math.sqrt(2),  17],
    [  0,  1.01,  1/math.sqrt(2),  53],
    [  0, -1,  1/math.sqrt(2),  35],
])

# number of coloumns in data
N_AXIS_REQUIRED = np.shape(data)[1] 


# ENANTIOMER
data_enant = np.array([
    [  1,  0, -1/math.sqrt(2),  9],
    [  0,  0,  0,               6],
    [ -1,  0, -1/math.sqrt(2),  17],
    [  0,  1.01,  1/math.sqrt(2),  35],
    [  0, -1,  1/math.sqrt(2),  35],
])

# scaler = StandardScaler()

# Standardize the data
# data = scaler.fit_transform(data)

# Perform PCA on the 4D data
pca = PCA()
pca.fit(data)

# The principal axes in feature space, representing the directions of maximum variance in the data.
# These are your "direction vectors" for each principal axis
axes = pca.components_
n_axes = pca.n_components_
eigenvalues = pca.explained_variance_
if n_axes < N_AXIS_REQUIRED:
    additional_vectors = null_space(axes)
    additional_vectors = additional_vectors / np.linalg.norm(additional_vectors, axis=0)
    additional_vectors = additional_vectors.T
    axes = np.vstack((axes, additional_vectors))

print('Principal components')
for i in np.argsort(eigenvalues)[::-1]:
    print(eigenvalues[i],'->',axes[i])


# Compute the centroid
centroid = np.mean(data, axis=0)

# Compute the 4 reference points along each axis
reference_points = [centroid + DIST * axis for axis in axes]
reference_points.append(centroid)

print('Reference_points')
print(np.array(reference_points))

# Compute the Euclidean distance of each point from each reference point
distances = np.empty((data.shape[0], len(reference_points)))
for i, point in enumerate(data):
    for j, ref_point in enumerate(reference_points):
        distances[i, j] = distance.euclidean(point, ref_point)

print('Distances')
print(distances.T)

fingerprint = compute_statistics(distances.T)

print('Fingerprint')
print(fingerprint)

# Rotate points
# coords = data[:, :-1]
# masses = data[:, -1]

# # coords, rotation_matrix = rotate_points(coords, 180, 0, 0)

# # angle_x = np.random.uniform(-180, 180)
# # angle_y = np.random.uniform(-180, 180)
# # angle_z = np.random.uniform(-180, 180)

# # coords, rotation_matrix = rotate_points(coords, angle_x, angle_y, angle_z)
# # coords = perturb_coordinates(coords, 4)

# coords = reflect_points(coords)

# data1 = np.c_[coords, masses]
data1 = data_enant

# # standardize data
# data1 = scaler.fit_transform(data1)

N_AXIS_REQUIRED = np.shape(data1)[1] 

#print(data1)

# # Center the data
# data1 = data1 - np.mean(data1, axis=0)

# Perform PCA on the 4D data
pca1 = PCA()
pca1.fit(data1)

axes1 = pca1.components_
eigenvalues1 = pca1.explained_variance_

if pca1.n_components_ < N_AXIS_REQUIRED:    
    additional_vectors = null_space(axes1)
    additional_vectors = additional_vectors / np.linalg.norm(additional_vectors, axis=0)
    additional_vectors = additional_vectors.T
    axes1 = np.vstack((axes1, additional_vectors))

# # invert the oredr of axes
# axes1 = axes1[::-1]
print('Principal components')
for i in np.argsort(eigenvalues1)[::-1]:
    print(eigenvalues1[i],'->',axes1[i])



centroid1 = np.mean(data1, axis=0)

reference_points1 = [centroid1 + DIST * axis for axis in axes1]
reference_points1.append(centroid1)

print('Reference points')
print(np.array(reference_points1))

distances1 = np.empty((data1.shape[0], len(reference_points1)))
for i, point in enumerate(data1):
    for j, ref_point in enumerate(reference_points1):
        distances1[i, j] = distance.euclidean(point, ref_point)

print('Distances')
print(distances1.T)

fingerprint1 = compute_statistics(distances1.T)

print('Fingerprint')
print(fingerprint1)

similarity = 1 / (1 + calculate_nD_partial_score(fingerprint, fingerprint1))
print(f'Similarity: {similarity}')