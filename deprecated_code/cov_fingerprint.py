import os 
print(f'Current working directory: {os.getcwd()}')
import numpy as np
from scipy.stats import skew
from similarity.trials.utils import *
from similarity.source.similarity import calculate_partial_score

def compute_distances(points, reference_points):
    """Compute the distance of each point to the 4 refernce points"""
    num_points = points.shape[0]
    num_ref_points = len(reference_points)
    distances = np.zeros((num_ref_points, num_points))
    
    for i, point in enumerate(points):
        for j, ref_point in enumerate(reference_points):
            distances[j, i] = np.linalg.norm(point - ref_point)

    print('Distances')
    print(distances)

    return distances 

def compute_statistics(distances):
    means = np.mean(distances, axis=1)
    std_devs = np.std(distances, axis=1)
    skewness = skew(distances, axis=1)
    # check if skewness is nan
    skewness[np.isnan(skewness)] = 0
    
    statistics_matrix = np.vstack((means, std_devs, skewness)).T 
    # add all rows to a list   
    statistics_list = [element for row in statistics_matrix for element in row]

    return statistics_list  

def principal_components(data):
    """
    Calculates the principal components (eigenvectors) of the covariance matrix of points with 
    additional info.
    """
    # print('Data')
    # print(data)

    covariance_matrix = np.cov(data, ddof=0, rowvar=False)
    # print('Data covariance matrix:')
    # print(covariance_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # eigenvectors = eigenvectors.T
    # # TODO: Axes convention
    # for vec in eigenvectors:
    #     if vec[0] < 0:
    #         vec *= -1
    # eigenvectors = eigenvectors.T

    # # Ensure the eigenvectors are sorted by eigenvalue
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Center the data
    data -= np.mean(data, axis=0)
    # Transform the data to the new coordinate system
    data = np.dot(data, eigenvectors) 
    print('Transformed data')
    print(data)

    eigenvectors = eigenvectors.T

    print('Principal components')
    for i in np.argsort(eigenvalues)[::-1]:
        print(eigenvalues[i],'->',eigenvectors[i])
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[sorted_indices, :]
    # print('Sorted principal components')
    # print(eigenvectors)
    return eigenvectors, covariance_matrix, data

def compute_reference_points(data, eigenvectors):
    centroid = np.mean(data, axis = 0)
    reference_points = [centroid + axis for axis in eigenvectors]
    reference_points.append(centroid)
    return reference_points


def compute_nD_fingerprint(data):
    pca_axis, cov_matrix, data = principal_components(data)
    #std_axis, _, _ = principal_components(data)
    std_axis = np.eye(np.shape(data)[1])
    reference_points = compute_reference_points(data, std_axis)
    #visualize_nD_3d_projection(data, pca_axis)
    distances = compute_distances(data, reference_points)
    fingerprint = compute_statistics(distances)
    print('Fingerprint')
    print(fingerprint)
    return fingerprint, data, pca_axis, std_axis

def compute_6D_similarity_cov(query, target):
    """Compute the similarity between two 6D fingerprints"""

    # points = translate_points_to_geometrical_center(query['coordinates'])
    # points1 = translate_points_to_geometrical_center(target['coordinates'])

    # Normalization and tapering
    query = taper_delta_features(query)
    #query = normalize_delta_features(query)
    target = taper_delta_features(target)
    #target = normalize_delta_features(target)

    data = np.hstack((query['coordinates'], np.array(query['protons']).reshape(-1, 1), np.array(query['delta_neutrons']).reshape(-1, 1), np.array(query['formal_charges']).reshape(-1, 1)))
    data1 = np.hstack((target['coordinates'], np.array(target['protons']).reshape(-1, 1), np.array(target['delta_neutrons']).reshape(-1, 1), np.array(target['formal_charges']).reshape(-1, 1)))

    fingerprint_query,_ ,_ ,_ = compute_nD_fingerprint(data)
    fingerprint_target,_ ,_ ,_ = compute_nD_fingerprint(data1)

    similarity = 1/(1 + calculate_partial_score(fingerprint_query, fingerprint_target))
    return similarity