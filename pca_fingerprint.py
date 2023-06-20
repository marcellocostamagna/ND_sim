import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.linalg import null_space
import math
from nD_tools import *
from perturbations import *
import fingerprints as fp
import sklearn.preprocessing as skp



def get_pca_fingerprint(data):
    """Computes the PCA fingerprint of a given data set."""
    N_AXIS_REQUIRED = np.shape(data)[1]
    DIST = 1
    # print('Data')   
    # print(data)

    # # Standardize the data
    # data = skp.StandardScaler().fit_transform(data)
    # # print('Standardized data')
    # # print(data)

    pca = PCA()
    pca.fit(data)

    # The principal axes in feature space, representing the directions of maximum variance in the data.
    axes = pca.components_
    n_axes = pca.n_components_
    eigenvalues = pca.explained_variance_
    
    if n_axes < N_AXIS_REQUIRED:
        additional_vectors = null_space(axes).T
        axes = np.vstack((axes, additional_vectors))
        print('Additional vectors')
        print(additional_vectors)

    # TODO:Axis convention
    for ax in axes:
        if ax[0] < 0:
            ax *= -1

    print('Principal components')
    for i in np.argsort(eigenvalues)[::-1]:
        print(eigenvalues[i],'->',axes[i])

    # Compute the centroid
    centroid = np.mean(data, axis=0)
    # print('Centroid')
    # print(centroid)

    # Compute the 4 reference points along each axis
    reference_points = [centroid + DIST * axis for axis in axes]
    reference_points.append(centroid)

    # print('Reference_points')
    # print(np.array(reference_points))

    # Compute the Euclidean distance of each point from each reference point
    distances = np.empty((data.shape[0], len(reference_points)))
    for i, point in enumerate(data):
        for j, ref_point in enumerate(reference_points):
            distances[i, j] = distance.euclidean(point, ref_point)
    # print('Distances')
    # print(distances.T)

    fingerprint = fp.compute_statistics(distances.T)
    # print('Fingerprint')
    # print(fingerprint)

    return fingerprint

# def get_pca_fingerprint1(data):
#     """Computes the PCA fingerprint of a given data set."""
#     N_AXIS_REQUIRED = np.shape(data)[1]
#     DIST = 1
#     # print('Data')   
#     # print(data)

#     # # Standardize the data
#     # data = skp.StandardScaler().fit_transform(data)
#     # # print('Standardized data')
#     # # print(data)

#     pca = PCA()
#     pca.fit(data)

#     # The principal axes in feature space, representing the directions of maximum variance in the data.
#     axes = pca.components_
#     n_axes = pca.n_components_
#     eigenvalues = pca.explained_variance_
    
#     if n_axes < N_AXIS_REQUIRED:
#         additional_vectors = null_space(axes).T
#         axes = np.vstack((axes, additional_vectors))
#         print('Additional vectors')
#         print(additional_vectors)
    
#     #axes[1] = -axes[1]

#     print('Principal components')
#     for i in np.argsort(eigenvalues)[::-1]:
#         print(eigenvalues[i],'->',axes[i])

#     # Compute the centroid
#     centroid = np.mean(data, axis=0)
#     # print('Centroid')
#     # print(centroid)

#     # Compute the 4 reference points along each axis
#     reference_points = [centroid + DIST * axis for axis in axes]
#     reference_points.append(centroid)

#     print('Reference_points')
#     print(np.array(reference_points))

#     # Compute the Euclidean distance of each point from each reference point
#     distances = np.empty((data.shape[0], len(reference_points)))
#     for i, point in enumerate(data):
#         for j, ref_point in enumerate(reference_points):
#             distances[i, j] = distance.euclidean(point, ref_point)
#     print('Distances')
#     print(distances.T)

#     fingerprint = fp.compute_statistics(distances.T)
#     print('Fingerprint')
#     print(fingerprint)

#     return fingerprint