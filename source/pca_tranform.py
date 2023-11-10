# Script to perform Principal Component Analysis (PCA) analysis on n-dimensional molecular data 
# and return the transformed data for fingerprint calculation

import numpy as np
from scipy.stats import skew

def compute_pca_using_covariance(original_data, chirality=False):
    """
    Perform PCA analysis via eigendecomposition of the covariance matrix.
    
    This function conducts PCA to produce a consistent reference system, 
    allowing for comparison between molecules.The emphasis is on generating 
    eigenvectors that offer deterministic outcomes and consistent orientations.
    To enable the distinction of chiral molecules, the determinant's sign is 
    explicitly considered and ensured to be positive.

    Parameters
    ----------
    original_data : numpy.ndarray
        N-dimensional array representing a molecule, where each row is a sample/point.

    Returns
    -------
    transformed_data : numpy.ndarray
        Data after PCA transformation.
    eigenvectors : numpy.ndarray
        Eigenvectors obtained from the PCA decomposition.
    """
    covariance_matrix = np.cov(original_data, rowvar=False, ddof=0,) # STEP 1: Covariance Matrix
    
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix) # STEP 2: Eigendecomposition of Covariance Matrix
    
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    
    threshold = 1e-4
    significant_indices = np.where(abs(eigenvalues) > threshold)[0]
    
    # Create the reduced eigenvector matrix by selecting both rows and columns
    reduced_eigenvectors = extract_relevant_subspace(eigenvectors, significant_indices)

    if chirality:
        determinant = np.linalg.det(reduced_eigenvectors) 
        if determinant < 0:
            eigenvectors[:, 0] *= -1
   
    adjusted_eigenvectors, n_changes, best_eigenvector_to_flip  = adjust_eigenvector_signs(original_data, eigenvectors[:, significant_indices], chirality) # STEP 4: Adjust eigenvector signs
    eigenvectors[:, significant_indices] = adjusted_eigenvectors

    if chirality:
        if n_changes % 2 == 1 and chirality:            
            eigenvectors[:, best_eigenvector_to_flip] *= -1
    
    transformed_data = np.dot(original_data, eigenvectors)
    
    return  transformed_data

def adjust_eigenvector_signs(original_data, eigenvectors, chirality=False, tolerance= 1e-4):
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
    sign_changes = 0
    symmetric_eigenvectors = []
    skewness_values = []
    best_eigenvector_to_flip = 0
    
    for i in range(eigenvectors.shape[1]):
        # Compute the projections of the original data onto the current eigenvector
        projections = original_data.dot(eigenvectors[:, i])
        
        # Compute skewness for current projections
        if chirality:
            current_skewness = skew(projections)
            skewness_values.append(abs(current_skewness))

        remaining_indices = np.arange(original_data.shape[0])  # start considering all points
        max_abs_coordinate = np.max(np.abs(projections))

        while True:
            # find the points with maximum absolute coordinate among the remaining ones
            mask_max = np.isclose(np.abs(projections[remaining_indices]), max_abs_coordinate, atol=tolerance)
            max_indices = remaining_indices[mask_max]  # indices of points with maximum absolute coordinate
            
            # If all points with the maximum absolute coordinate have the same sign, use them for a decision
            unique_signs = np.sign(projections[max_indices])
            if np.all(unique_signs == unique_signs[0]):
                break

            if len(max_indices) == 1:
                break
            
            # if there is a tie, ignore these points and find the maximum absolute coordinate again
            remaining_indices = remaining_indices[~mask_max]
            if len(remaining_indices) == 0: # if all points have the same component, break the loop
                symmetric_eigenvectors.append(i)
                break
            max_abs_coordinate = np.max(np.abs(projections[remaining_indices]))
        
        if len(remaining_indices) > 0 and projections[max_indices[0]] < 0:
            eigenvectors[:, i] *= -1
            sign_changes += 1
            
    if symmetric_eigenvectors:
        if sign_changes % 2 == 1:
            eigenvectors[:, symmetric_eigenvectors[0]] *= -1
            sign_changes = 0
            
    if chirality:
        best_eigenvector_to_flip = np.argmax(skewness_values)   
         
    return eigenvectors, sign_changes, best_eigenvector_to_flip 


def extract_relevant_subspace(eigenvectors, significant_indices, tol=1e-10):
    """
    Extracts the subset of eigenvectors that's relevant for the determinant calculation.
    
    This function prunes eigenvectors by removing rows and columns that have all zeros 
    except for a single entry close to 1 or -1 within a given tolerance (eigenvectors with
    an eigenvalue equal to 0, and relative components). Then, it further 
    reduces the matrix using the provided significant indices to give a relevant 
    subset of eigenvectors.

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        The eigenvectors matrix to prune and reduce.
    significant_indices : numpy.ndarray
        Indices of significant eigenvectors.
    tol : float, optional (default = 1e-10)
        Tolerance for determining whether a value is close to 0, 1, or -1.

    Returns
    -------
    numpy.ndarray
        The determinant-relevant subset of eigenvectors.
    """
    
    row_mask = ~np.all((np.abs(eigenvectors) < tol) | (np.abs(eigenvectors - 1) < tol) | (np.abs(eigenvectors + 1) < tol), axis=1)    
    col_mask = ~np.all((np.abs(eigenvectors.T) < tol) | (np.abs(eigenvectors.T - 1) < tol) | (np.abs(eigenvectors.T + 1) < tol), axis=1)
    
    # row_mask[significant_indices] = True
    # col_mask[significant_indices] = True
    
    pruned_eigenvectors = eigenvectors[row_mask][:, col_mask]
    reduced_eigenvectors = pruned_eigenvectors[significant_indices][:, significant_indices]
    
    return reduced_eigenvectors

