# Script to perform the PCA analysis to the n-dimensional data representing a molecule
# and returning the transformed data with which obtain the fingerprint

import numpy as np

# TODO: Improve name of function and return values    
def perform_PCA_and_get_transformed_data_cov(original_data):
    """
    Perform PCA analysis via eigendecomposition of the covariance matrix.
    
    The function carries out PCA on n-dimensional data representing a molecule 
    and returns the original and transformed data alongside eigenvectors and eigenvalues.

    Parameters
    ----------
    original_data : numpy.ndarray
        N-dimensional array representing a molecule, where each row is a sample/point.

    Returns
    -------
    original_data : numpy.ndarray
        The input n-dimensional data.
    transformed_data : numpy.ndarray
        Data after PCA transformation.
    eigenvectors : numpy.ndarray
        Eigenvectors obtained from the PCA decomposition.
    eigenvalues : numpy.ndarray
        Eigenvalues obtained from the PCA decomposition.
    """
    covariance_matrix = np.cov(original_data, rowvar=False, ddof=0,)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

    eigenvectors = choose_eig_sign(original_data, eigenvectors)

    transformed_data = np.dot(original_data, eigenvectors)

    return original_data, transformed_data, eigenvectors, eigenvalues


def choose_eig_sign(original_data, eigenvectors, tolerance= 1e-4):
    """
    Adjust the sign of eigenvectors based on the data's projections.

    For each eigenvector, the function determines the sign by looking at 
    the direction of the data's maximum projection. If the maximum projection
    is negative, the sign of the eigenvector is flipped.

    Parameters
    ----------
    original_data : numpy.ndarray
        N-dimensional array representing a molecule, where each row is a sample/point.
    eigenvectors : numpy.ndarray
        Eigenvectors obtained from the PCA decomposition.
    tolerance : float, optional
        Tolerance used when comparing projections. Defaults to 1e-4.

    Returns
    -------
    eigenvectors : numpy.ndarray
        Adjusted eigenvectors with their sign possibly flipped.
    """
    for i in range(eigenvectors.shape[1]):
        # Compute the projections of the original data onto the current eigenvector
        projections = original_data.dot(eigenvectors[:, i])

        remaining_indices = np.arange(original_data.shape[0])  # start considering all points
        max_abs_coordinate = np.max(np.abs(projections))

        while True:
            # find the points with maximum absolute coordinate among the remaining ones
            mask_max = np.isclose(np.abs(projections[remaining_indices]), max_abs_coordinate, atol=tolerance)
            max_indices = remaining_indices[mask_max]  # indices of points with maximum absolute coordinate

            if len(max_indices) == 1:  # we found a single unambiguous point
                break
            
            # if there is a tie, ignore these points and find the maximum absolute coordinate again
            remaining_indices = remaining_indices[~mask_max]
            if len(remaining_indices) == 0: # if all points have the same component, break the loop
                break
            max_abs_coordinate = np.max(np.abs(projections[remaining_indices]))
        
        if len(remaining_indices) > 0 and projections[max_indices[0]] < 0:
            eigenvectors[:, i] *= -1

    return eigenvectors 

