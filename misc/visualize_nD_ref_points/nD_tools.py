import numpy as np
import math
import matplotlib.pyplot as plt

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

POINTTRANSPARENCY = 0.5
POINTSIZE = 50

def plot_data_and_ref_points(data):
    covariance_matrix = np.cov(data, ddof = 0, rowvar = False)
    print('Data covariance matrix:')
    print(covariance_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print('Principal components')
    for i in np.argsort(eigenvalues)[::-1]:
        print(eigenvalues[i],'->',eigenvectors[i])
        
    num_points, num_dimensions = data.shape
    x_coord = data[:, [0]]
    y_coord = np.zeros(x_coord.shape)
    z_coord = np.zeros(x_coord.shape)
    m_coord = np.zeros(x_coord.shape)
    if num_dimensions > 1:
        y_coord = data[:, [1]]
        if num_dimensions > 2:
            z_coord = data[:, [2]]
            if num_dimensions > 3:
                m_coord = data[:, [3]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_coord, y_coord, z_coord, s=POINTSIZE, c=m_coord, alpha=POINTTRANSPARENCY)
    centroid = data.mean(axis = 0)
    x_centroid = centroid[0]
    y_centroid = np.zeros(x_centroid.shape)
    z_centroid = np.zeros(x_centroid.shape)
    if num_dimensions > 1:
        y_centroid = centroid[1]
        if num_dimensions > 2:
            z_centroid = centroid[2]
    ax.scatter(x_centroid, y_centroid, z_centroid, s = POINTSIZE,
               c='red', marker='*', alpha=POINTTRANSPARENCY)
    
    reference_points = centroid + eigenvectors*(data.max(axis=0)-centroid)
    dist = []
    distances = np.linalg.norm(data - centroid, axis = 1)
    print(f'Distances to cen0',distances)
    dist.append(distances)
    for i in range(data.shape[1]):
        distances = np.linalg.norm(data - reference_points[i], axis = 1)
        dist.append(distances)
        print(f'Distances to cen{(i+1)}',distances)
    
    x_coord_refs = reference_points[:, [0]]
    y_coord_refs = np.zeros(x_coord_refs.shape)
    z_coord_refs = np.zeros(x_coord_refs.shape)
    if num_dimensions > 1:
        y_coord_refs = reference_points[:, [1]]
        if num_dimensions > 2:
            z_coord_refs = reference_points[:, [2]]
    ax.scatter(x_coord_refs, y_coord_refs, z_coord_refs, s = POINTSIZE,
               c='green', marker='+', alpha=1)
    for x,y,z,i in zip(x_coord_refs,y_coord_refs,z_coord_refs,range(
        len(x_coord_refs))):
        ax.text(x[0],y[0],z[0],i)
    plt.show(block=False)
    plt.pause(0.001)
    return dist

