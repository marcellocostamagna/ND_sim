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


    
def perform_PCA_and_get_transformed_data_cov(original_data):
    """
    Performs the PCA analysis via the eigendecomposition of the covariance 
    matrix to the n-dimensional data representing a molecule and returning 
    the transformed data with which obtain the fingerprint, 
    """
    covariance_matrix = np.cov(original_data, ddof=0, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

    transformed_data = np.dot(original_data, eigenvectors)
    
    eigenvectors, transformed_data = choose_eig_sign(eigenvectors, transformed_data)

    return original_data, transformed_data, eigenvectors, eigenvalues

def choose_eig_sign(eigenvectors, transformed_data, tolerance= 1e-4):
    for i in range(eigenvectors.shape[1]):
        remaining_indices = np.arange(transformed_data.shape[0])  # start considering all points
        max_abs_coordinate = np.max(np.abs(transformed_data[:, i]))

        while True:
            # find the points with maximum absolute coordinate among the remaining ones
            mask_max = np.isclose(np.abs(transformed_data[remaining_indices, i]), max_abs_coordinate, atol=tolerance)
            max_indices = remaining_indices[mask_max]  # indices of points with maximum absolute coordinate

            if len(max_indices) == 1:  # we found a single unambiguous point
                break
            
            # if there is a tie, ignore these points and find the maximum absolute coordinate again
            remaining_indices = remaining_indices[~mask_max]
            if len(remaining_indices) == 0: # if all points have the same component, break the loop
                break
            max_abs_coordinate = np.max(np.abs(transformed_data[remaining_indices, i]))
        
        if len(remaining_indices) > 0 and transformed_data[max_indices[0], i] < 0:
            eigenvectors[:, i] *= -1
            transformed_data[:, i] *= -1

    return eigenvectors, transformed_data