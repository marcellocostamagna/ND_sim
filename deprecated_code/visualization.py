import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
import math as m

POINTTRANSPARENCY = 0.5
POINTSIZE = 50

def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])    

def visualize(points, masses, center_of_mass, principal_axes, eigenvalues, scale, four_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualize points with size based on mass
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=[mass * 20 for mass in masses], c='b', alpha=0.6)

    # Visualize the four points as yellow dots
    four_points = np.array(four_points)
    ax.scatter(four_points[:, 0], four_points[:, 1], four_points[:, 2], c='y', marker='o', s=50)


    # Visualize eigenvectors with colors based on eigenvalues
    colors = ['r', 'g', 'k']
    min_eigenvalue = np.min(eigenvalues)
    if min_eigenvalue == 0:
        min_eigenvalue = 0.001
    scaled_axes = []
    for axis, eigenvalue, color in zip(principal_axes, eigenvalues, colors):
        # deal with the case one eigenvalue is zero
        #scaled_axis = axis *  (eigenvalue / min_eigenvalue) * scale # Adjust this factor to change the length of the eigenvectors
        scaled_axis = axis * scale
        scaled_axes.append(scaled_axis)
        ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                  scaled_axis[0], scaled_axis[1], scaled_axis[2],
                  color=color, lw=2, arrow_length_ratio=0.1)

    # Adjust axes to fit all points
    scaled_axes = np.array(scaled_axes)
    min_values = np.min(points[:, None, :] - scaled_axes, axis=(0, 1))
    max_values = np.max(points[:, None, :] + scaled_axes, axis=(0, 1))
    padding = 0.1  # Increase or decrease this value to change the padding around the axes

    ax.set_xlim(min_values[0] - padding, max_values[0] + padding)
    ax.set_ylim(min_values[1] - padding, max_values[1] + padding)
    ax.set_zlim(min_values[2] - padding, max_values[2] + padding)

    set_axes_equal(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show(block=False)
    plt.pause(0.001)



def set_axes_equal1(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])    

def visualize1(points, num_protons, center_of_mass, principal_axes, eigenvalues):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualize points with size based on mass (assuming protons)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=[proton * 20 for proton in num_protons], c='b', alpha=0.6)

    # Visualize eigenvectors with colors based on eigenvalues
    colors = ['r', 'g', 'k']
    max_eigenvalue = np.max(eigenvalues)
    min_eigenvalue = np.min(eigenvalues)
    if min_eigenvalue == 0:
        min_eigenvalue = 0.001
    scaled_axes = []
    for axis, eigenvalue, color in zip(principal_axes, eigenvalues, colors):
        scaled_axis = (axis/np.linalg.norm(axis)) * np.sqrt(max_eigenvalue)  # Adjust this factor to change the length of the eigenvectors
        scaled_axes.append(scaled_axis)
        ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                  scaled_axis[0], scaled_axis[1], scaled_axis[2],
                  color=color, lw=2, arrow_length_ratio=0.1)
        # print the length of the axis
        #print(np.linalg.norm(scaled_axis))

    # Adjust axes to fit all points
    # scaled_axes = np.array(scaled_axes)
    # min_values = np.min(points[:, None, :] - scaled_axes, axis=(0, 1))
    # max_values = np.max(points[:, None, :] + scaled_axes, axis=(0, 1))
    padding = 1 # Increase or decrease this value to change the padding around the axes

    # ax.set_xlim(min_values[0] - padding, max_values[0] + padding)
    # ax.set_ylim(min_values[1] - padding, max_values[1] + padding)
    # ax.set_zlim(min_values[2] - padding, max_values[2] + padding)

    # Find the maximum arrow length
    max_arrow_length = np.max([np.linalg.norm(axis) for axis in scaled_axes])

    # Adjust axes to fit all points and arrows
    min_values = np.min(points, axis=0) - max_arrow_length - padding
    max_values = np.max(points, axis=0) + max_arrow_length + padding

    ax.set_xlim(min_values[0], max_values[0])
    ax.set_ylim(min_values[1], max_values[1])
    ax.set_zlim(min_values[2], max_values[2])

    set_axes_equal1(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show(block=False)
    plt.pause(0.001)


def visualize_nD_3d_projection(data, eigenvectors):
        
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
    
    reference_points = centroid + eigenvectors*(data.max(axis=0))
    

    
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
    


# def plot_data_and_ref_points(data):
#     covariance_matrix = np.cov(data, ddof = 0, rowvar = False)
#     print('Data covariance matrix:')
#     print(covariance_matrix)
#     eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
#     print('Principal components')
#     for i in np.argsort(eigenvalues)[::-1]:
#         print(eigenvalues[i],'->',eigenvectors[i])
        
#     num_points, num_dimensions = data.shape
#     x_coord = data[:, [0]]
#     y_coord = np.zeros(x_coord.shape)
#     z_coord = np.zeros(x_coord.shape)
#     m_coord = np.zeros(x_coord.shape)
#     if num_dimensions > 1:
#         y_coord = data[:, [1]]
#         if num_dimensions > 2:
#             z_coord = data[:, [2]]
#             if num_dimensions > 3:
#                 m_coord = data[:, [3]]
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(x_coord, y_coord, z_coord, s=POINTSIZE, c=m_coord, alpha=POINTTRANSPARENCY)
#     centroid = data.mean(axis = 0)
#     x_centroid = centroid[0]
#     y_centroid = np.zeros(x_centroid.shape)
#     z_centroid = np.zeros(x_centroid.shape)
#     if num_dimensions > 1:
#         y_centroid = centroid[1]
#         if num_dimensions > 2:
#             z_centroid = centroid[2]
#     ax.scatter(x_centroid, y_centroid, z_centroid, s = POINTSIZE,
#                c='red', marker='*', alpha=POINTTRANSPARENCY)
    
#     reference_points = centroid + eigenvectors*(data.max(axis=0)-centroid)
    
#     distances = np.linalg.norm(data - centroid, axis = 1)
#     print(f'Distances to cen0',distances)
#     for i in range(data.shape[1]):
#         distances = np.linalg.norm(data - reference_points[i], axis = 1)
#         print(f'Distances to cen{(i+1)}',distances)
    
#     x_coord_refs = reference_points[:, [0]]
#     y_coord_refs = np.zeros(x_coord_refs.shape)
#     z_coord_refs = np.zeros(x_coord_refs.shape)
#     if num_dimensions > 1:
#         y_coord_refs = reference_points[:, [1]]
#         if num_dimensions > 2:
#             z_coord_refs = reference_points[:, [2]]
#     ax.scatter(x_coord_refs, y_coord_refs, z_coord_refs, s = POINTSIZE,
#                c='green', marker='+', alpha=1)
#     for x,y,z,i in zip(x_coord_refs,y_coord_refs,z_coord_refs,range(
#         len(x_coord_refs))):
#         ax.text(x[0],y[0],z[0],i)

#     plt.show(block=False)
#     plt.pause(0.001)    
    
  
    