# Script to perform the PCA analysis to the n-dimensional data representing a molecule
# and reutrning the transformed data with which obtain the fingerprint


import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import null_space

def perform_PCA_and_get_transformed_data(original_data):
    """
    Performs the PCA analysis to the n-dimensional data representing a molecule 
    and reutrning the transformed data with which obtain the fingerprint
    """
    N_AXIS_REQUIRED = np.shape(original_data)[1]
    DIST = 1
    #print('Data:' f'\n{original_data}')

    pca = PCA()
    pca.fit(original_data)

    # Transform the data
    data = pca.transform(original_data)
    #print('Transformed data:' f'\n{data}')

    # The principal axes in feature space, representing the directions of maximum variance in the data.
    axes = pca.components_
    n_axes = pca.n_components_
    eigenvalues = pca.explained_variance_

    # Adds the null space vectors to the axes if the number of axes is less than the number of dimensions
    # with a corresponding eigenvalue of 0
    if n_axes < N_AXIS_REQUIRED:
        additional_vectors = null_space(axes).T
        axes = np.vstack((axes, additional_vectors))
        # Add the eigenvalues of the null space vectors
        eigenvalues = np.append(eigenvalues, np.zeros(N_AXIS_REQUIRED - n_axes))
        #print('Additional vectors:' f'\n{additional_vectors}' )

    #print('Principal components')
    # for i in np.argsort(eigenvalues)[::-1]:
    #     print(eigenvalues[i],'->',axes[i])

    return original_data, data, axes, eigenvalues


    
