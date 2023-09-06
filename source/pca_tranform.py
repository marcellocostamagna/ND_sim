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

    eigenvectors = choose_eig_sign(original_data, eigenvectors)

    transformed_data = np.dot(original_data, eigenvectors)

    return original_data, transformed_data, eigenvectors, eigenvalues


def choose_eig_sign(original_data, eigenvectors, tolerance=1e-4):
    for i in range(eigenvectors.shape[1]):
        # Compute the projections of the original data onto the current eigenvector
        projections = original_data.dot(eigenvectors[:, i])
        sorted_indices = np.argsort(np.abs(projections))[::-1]
        # Determine the largest magnitude projection
        largest_projection = projections[sorted_indices[0]]
        
        # Check for ambiguities and resolve them
        for idx in sorted_indices:
            if np.isclose(np.abs(projections[idx]), np.abs(largest_projection), atol=tolerance):
                if projections[idx] < 0: 
                    largest_projection = projections[idx]  
                    break
            else:
                break  
        # Update the direction of the eigenvector based on the sign of the largest magnitude projection
        if largest_projection < 0:
            eigenvectors[:, i] *= -1

    return eigenvectors
