import numpy as np
from perturbations import *
from rdkit import Chem
from utils import *
from cov_fingerprint import *
from similarity_3d import *
from pca_fingerprint import *
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm


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

def visualize(points, center_of_mass, principal_axes, ax, labels, masses):
    unique_masses = np.unique(masses) # Get unique masses
    num_unique_masses = len(unique_masses)

    # Create a colormap
    colormap = cm.get_cmap('viridis', num_unique_masses)

    # Create a dictionary that maps each mass to a color
    mass_to_color = {mass: colormap(i) for i, mass in enumerate(unique_masses)}

    # Visualize points with size based on mass (assuming protons)
    for i, label in enumerate((labels)):
        ax.scatter(points[i, 0], points[i, 1], points[i, 2], c=np.array([mass_to_color[masses[i]]]), alpha=0.6, label = label)
        ax.text(points[i, 0], points[i, 1], points[i, 2], labels[i], fontsize=12)

    colors = ['r', 'g', 'k']
    for axis, color in zip(principal_axes, colors):
        ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                  axis[0]*5, axis[1]*5, axis[2]*5, # The scaling factor is used to make the eigenvector more visible
                  color=color, lw=2, arrow_length_ratio=0.1)
    
    padding = 1 # Increase or decrease this value to change the padding around the axes

    # Find the maximum arrow length
    max_arrow_length = np.max([np.linalg.norm(axis) for axis in principal_axes])

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

# Irregular quadrilateral
# data = np.array([[ -0.9, -1.8, 1],
#                 [-2.2, -0.9, 2],
#                 [1.7, 0.5, 2], 
#                 [0.8, 1.9, 4]]).astype(float)

coordinates = np.random.uniform(low = -10 , high = 10 , size = (5, 3))
masses = np.random.randint(low = 1 , high = 10 , size = (5, 1))

data = np.concatenate((coordinates, masses), axis = 1)

# Rotate the points
tmp_data = perturb_coordinates(coordinates, 3)
# generate random float number between 0 and 360
angle1 = np.random.uniform(0, 360)
angle2 = np.random.uniform(0, 360)
angle3 = np.random.uniform(0, 360)
coordinates1 = rotate_points(tmp_data, angle1, angle2, angle3)
data1 = np.concatenate((coordinates1, masses), axis = 1)
masses = masses.flatten()

# PCA with pca_fingerprint.py
fingerprint_pca, transformed_data_pca, principal_axis_pca, standard_axis_pca = get_pca_fingerprint(data)

fingerprint_pca1, transformed_data_pca1, principal_axis_pca1, standard_axis_pca1 = get_pca_fingerprint(data1)

similarity_pca = 1/(1 + calculate_nD_partial_score(fingerprint_pca, fingerprint_pca1))
print('similarity_pca: ', similarity_pca)

print(f'Rotation angles: X: {angle1}, Y: {angle2}, Z: {angle3}')

labels = [str(i) for i in range(len(data))]

# Pre-processing of 4D data for 3D visualization
data = data[:, :3]
data1 = data1[:, :3]
transformed_data_pca = transformed_data_pca[:, :3]
transformed_data_pca1 = transformed_data_pca1[:, :3]
principal_axis_pca = principal_axis_pca[:3, :3]
principal_axis_pca1 = principal_axis_pca1[:3, :3] 
standard_axis_pca = standard_axis_pca[:3, :3]
standard_axis_pca1 = standard_axis_pca1[:3, :3]

# 3D visulaization

fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(221, projection='3d')
ax1.set_title('Original data')
visualize(data, np.mean(data, axis=0), principal_axis_pca, ax1, labels, masses)

ax2 = fig.add_subplot(222, projection='3d')
ax2.set_title('Rotated/Perturb data')
visualize(data1, np.mean(data1, axis=0), principal_axis_pca1, ax2, labels, masses)

ax3 = fig.add_subplot(223, projection='3d')
ax3.set_title('PCA Transformed data')
visualize(transformed_data_pca, np.mean(transformed_data_pca, axis=0), standard_axis_pca, ax3, labels, masses)

ax4 = fig.add_subplot(224, projection='3d')
ax4.set_title('PCA Transformed data')
visualize(transformed_data_pca1, np.mean(transformed_data_pca1, axis=0), standard_axis_pca1, ax4, labels, masses)

# visualize(data, np.mean(data, axis=0), principal_axis_pca)

plt.show()
