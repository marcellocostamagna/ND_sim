# Script to perform the PCA analysis to the n-dimensional data representing a molecule
# and returning the transformed data with which obtain the fingerprint

import numpy as np

# TODO: Improve name of function and return values    
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