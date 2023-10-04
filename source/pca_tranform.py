# Script to perform Principal Component Analysis (PCA) analysis on n-dimensional molecular data 
# and return the transformed data for fingerprint calculation

import numpy as np
from scipy.stats import skew

def compute_pca_using_covariance(original_data):
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
    # print(f'covariance_matrix: \n{covariance_matrix}')
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix) # STEP 2: Eigendecomposition of Covariance Matrix
    # print(f'eigenvectors: \n{eigenvectors}')
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    # print(f'eigenvalues: \n{eigenvalues}')
    
    threshold = 1e-4
    significant_indices = np.where(abs(eigenvalues) > threshold)[0]
    
    # Create the reduced eigenvector matrix by selecting both rows and columns
    # reduced_eigenvectors = eigenvectors[significant_indices][:, significant_indices]
    reduced_eigenvectors = extract_relevant_subspace(eigenvectors, significant_indices)
    # print(f'reduced_eigenvectors: \n{reduced_eigenvectors}')
    
    # print(f'Initial eigenvectors: \n{eigenvectors}')
    # determinant = np.linalg.det(eigenvectors)
    determinant = np.linalg.det(reduced_eigenvectors)   # STEP 3: Impose eigenvectors' determinant to be positive
    if determinant < 0:
        eigenvectors[:, 0] *= -1
        # eigenvectors[:, 1] *= -1
        # eigenvectors[:, 2] *= -1
    # print(f'Determinant imposed eigenvectors: \n{eigenvectors}')
    # adjusted_eigenvectors, n_changes = adjust_eigenvector_signs(original_data, eigenvectors[:, significant_indices]) # STEP 4: Adjust eigenvector signs
    adjusted_eigenvectors, n_changes, best_eigenvector_to_flip  = adjust_eigenvector_signs(original_data, eigenvectors[:, significant_indices]) # STEP 4: Adjust eigenvector signs
    eigenvectors[:, significant_indices] = adjusted_eigenvectors
    # print(f"Sign adjusted eigenvectors: \n{eigenvectors}")
    
    if n_changes % 2 == 1:              # STEP 5: Flip the sign of the first eigenvector of n_changes is odd (Chiral Distinction)
        # eigenvectors[:, 0] *= -1
        # eigenvectors[:, 1] *= -1
        # eigenvectors[:, 2] *= -1
        # eigenvectors[:, 3] *= -1
        eigenvectors[:, best_eigenvector_to_flip] *= -1
        
    # Check determinant
    # print(f"Final determinant: {np.linalg.det(eigenvectors)}")
        
    # print(f"'Chiral' eigenvectors: \n{eigenvectors}")
    
    transformed_data = np.dot(original_data, eigenvectors)
    # print(f"transformed_data: \n{transformed_data}")
    
    # visualize(original_data, np.mean(original_data, axis=0), eigenvectors)
    
    return  transformed_data, eigenvectors    

def adjust_eigenvector_signs(original_data, eigenvectors, tolerance= 1e-4):
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
    
    for i in range(eigenvectors.shape[1]):
        # Compute the projections of the original data onto the current eigenvector
        projections = original_data.dot(eigenvectors[:, i])
        
        # Compute skewness for current projections
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
            
    # Check determinant of the resulting eigenvectors
    # if np.linalg.det(eigenvectors) < 0 and symmetric_eigenvectors:
    #     # Flip the sign of one of the symmetric eigenvectors
    #     eigenvectors[:, symmetric_eigenvectors[0]] *= -1
    #     sign_changes = 0
    if symmetric_eigenvectors:
        # sign_changes = 0
        # if np.linalg.det(eigenvectors) < 0:
        #     eigenvectors[:, symmetric_eigenvectors[0]] *= -1
        if sign_changes % 2 == 1:
            eigenvectors[:, symmetric_eigenvectors[0]] *= -1
            sign_changes = 0
    
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
    pruned_eigenvectors = eigenvectors[row_mask][:, col_mask]
    reduced_eigenvectors = pruned_eigenvectors[significant_indices][:, significant_indices]
    
    return reduced_eigenvectors



# def adjust_eigenvector_signs(original_data, eigenvectors, tolerance= 1e-4):
#     """
#     Adjust the sign of eigenvectors based on the data's projections.

#     For each eigenvector, the function determines the sign by looking at 
#     the direction of the data's maximum projection. If the maximum projection
#     is negative, the sign of the eigenvector is flipped.

#     Parameters
#     ----------
#     original_data : numpy.ndarray
#         N-dimensional array representing a molecule, where each row is a sample/point.
#     eigenvectors : numpy.ndarray
#         Eigenvectors obtained from the PCA decomposition.
#     tolerance : float, optional
#         Tolerance used when comparing projections. Defaults to 1e-4.

#     Returns
#     -------
#     eigenvectors : numpy.ndarray
#         Adjusted eigenvectors with their sign possibly flipped.
#     """
#     sign_changes = 0
    
#     for i in range(eigenvectors.shape[1]):
#         # Compute the projections of the original data onto the current eigenvector
#         projections = original_data.dot(eigenvectors[:, i])

#         remaining_indices = np.arange(original_data.shape[0])  # start considering all points
#         max_abs_coordinate = np.max(np.abs(projections))

#         while True:
#             # find the points with maximum absolute coordinate among the remaining ones
#             mask_max = np.isclose(np.abs(projections[remaining_indices]), max_abs_coordinate, atol=tolerance)
#             max_indices = remaining_indices[mask_max]  # indices of points with maximum absolute coordinate

#             if len(max_indices) == 1:
#                 break
            
#             # if there is a tie, ignore these points and find the maximum absolute coordinate again
#             remaining_indices = remaining_indices[~mask_max]
#             if len(remaining_indices) == 0: # if all points have the same component, break the loop
#                 break
#             max_abs_coordinate = np.max(np.abs(projections[remaining_indices]))
        
#         if len(remaining_indices) > 0 and projections[max_indices[0]] < 0:
#             eigenvectors[:, i] *= -1
#             sign_changes += 1

#     return eigenvectors, sign_changes 
