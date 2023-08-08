import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.linalg import null_space
import math
from nD_tools import *
from trials.perturbations import *
import fingerprints as fp
import sklearn.preprocessing as skp

def get_pca_fingerprint(original_data):
    """Computes the PCA fingerprint of a given data set."""
    N_AXIS_REQUIRED = np.shape(original_data)[1]
    DIST = 1
    print('Data:' f'\n{original_data}')

    pca = PCA()
    pca.fit(original_data)

    # # Transform the data
    data = pca.transform(original_data)
    print('Transformed data:' f'\n{data}')

    # The principal axes in feature space, representing the directions of maximum variance in the data.
    axes = pca.components_
    n_axes = pca.n_components_
    eigenvalues = pca.explained_variance_

    if n_axes < N_AXIS_REQUIRED:
        additional_vectors = null_space(axes).T
        axes = np.vstack((axes, additional_vectors))
        print('Additional vectors:' f'\n{additional_vectors}' )

    print('Principal components')
    for i in np.argsort(eigenvalues)[::-1]:
        print(eigenvalues[i],'->',axes[i])

    # Compute the centroid
    centroid = np.mean(data, axis=0)
    # print('Centroid')
    # print(centroid)

    pca_1 = PCA()
    pca_1.fit(data)
    std_axis = pca_1.components_

    std_axis = np.eye(data.shape[1])

    # Compute the 4 reference points along each axis
    reference_points = [centroid + DIST * axis for axis in std_axis]
    reference_points.append(centroid)

    print('Reference_points:' f'\n{np.array(reference_points)}')

    # Compute the Euclidean distance of each point from each reference point
    distances = np.empty((data.shape[0], len(reference_points)))
    for i, point in enumerate(data):
        for j, ref_point in enumerate(reference_points):
            distances[i, j] = distance.euclidean(point, ref_point)
    print('Distances:' f'\n{distances.T}') 

    fingerprint = fp.compute_statistics(distances.T)
    print('Fingerprint:' f'\n{fingerprint}')

    return fingerprint, original_data, data, axes, std_axis, np.array(reference_points), distances