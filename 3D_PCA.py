import numpy as np
from perturbations import *
from rdkit import Chem
from utils import *
from cov_fingerprint import *
from similarity_3d import *
from pca_fingerprint import *
from copy import deepcopy
import matplotlib.pyplot as plt


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

def visualize(points, center_of_mass, principal_axes, ax, labels):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # Visualize points with size based on mass (assuming protons)
    for i, label in enumerate((labels)):
        ax.scatter(points[i, 0], points[i, 1], points[i, 2], c='b', alpha=0.6, label = label)
        ax.text(points[i, 0], points[i, 1], points[i, 2], labels[i], fontsize=12)


    colors = ['r', 'g', 'k']
    for axis, color in zip(principal_axes, colors):
        ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                  axis[0], axis[1], axis[2],
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
data = np.array([[ -0.9, -1.8, 1],
                [-2.2, -0.9, 2],
                [1.7, 0.5, 2], 
                [0.8, 1.9, 4]]).astype(float)

# Rotate the points
tmp_data = perturb_coordinates(data, 3)
# generate random float number between 0 and 360
angle1 = np.random.uniform(0, 360)
angle2 = np.random.uniform(0, 360)
angle3 = np.random.uniform(0, 360)
data1 = rotate_points(tmp_data, angle1, angle2, angle3)


# PCA with pca_fingerprint.py
fingerprint_pca, transformed_data_pca, principal_axis_pca, standard_axis_pca = get_pca_fingerprint(data)

fingerprint_pca1, transformed_data_pca1, principal_axis_pca1, standard_axis_pca1 = get_pca_fingerprint(data1)

similarity_pca = 1/(1 + calculate_nD_partial_score(fingerprint_pca, fingerprint_pca1))
print('similarity_pca: ', similarity_pca)

labels = ['A', 'B', 'C', 'D']

fig = plt.figure()

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Original data')
visualize(data, np.mean(data, axis=0), principal_axis_pca, ax1, labels)

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Rotated/Perturb data')
visualize(data1, np.mean(data1, axis=0), principal_axis_pca1, ax2, labels)

# visualize(data, np.mean(data, axis=0), principal_axis_pca)

plt.show()



#print('Angle: ', angle)


# # PCA with SVD visulization

# # Visualize the points
# plt.ion()  # turn interactive mode on
# # Plot original data
# x, y = data.T 
# points = ['A', 'B', 'C', 'D']
# fig1 = plt.figure(figsize=(10, 8))
# fig1.suptitle('PCA analysis with SVD')
# plt.subplots_adjust(hspace = 0.5)
# ax1 = fig1.add_subplot(321)
# ax1.set_title('Original data')
# for i, point in enumerate(points):
#     ax1.scatter(x[i], y[i], label=point)  # plot the points
#     ax1.text(x[i], y[i], point)  # label the points
# ax1.axis('equal')
# ax1.grid(True)

# x_rot, y_rot = data1.T  
# ax2 = fig1.add_subplot(322)
# ax2.set_title('Rotated/Perturb data')
# for i, point in enumerate(points):
#     ax2.scatter(x_rot[i], y_rot[i], label=point)  # plot the points
#     ax2.text(x_rot[i], y_rot[i], point)  # label the points
# ax2.axis('equal')
# ax2.grid(True)

# # Title for second row of plots

# ax3 = fig1.add_subplot(323)
# ax3.set_title('PCA_svd')
# for i, point in enumerate(points):
#     ax3.scatter(x[i], y[i], label=point)  # plot the points
#     ax3.text(x[i], y[i], point)  # label the points
# ax3.axis('equal')
# ax3.grid(True)

# # Add the principal axes
# for axis_pca, color in zip([principal_axis_pca], ['r']):
#     # principal_axis should be an array of shape (2, 2), 
#     # where each row is a principal axis (2D vector)
#     for px, py in axis_pca:
#         ax3.quiver(0, 0, px, py, color=color, angles='xy', scale_units='xy', scale=1)

# ax4 = fig1.add_subplot(324)
# ax4.set_title('PCA_svd')
# for i, point in enumerate(points):
#     ax4.scatter(x_rot[i], y_rot[i], label=point)  # plot the points
#     ax4.text(x_rot[i], y_rot[i], point)  # label the points
# ax4.axis('equal')
# ax4.grid(True)

# # Add the principal axes
# for axis_pca, color in zip([principal_axis_pca1], ['r']):
#     # principal_axis should be an array of shape (2, 2), 
#     # where each row is a principal axis (2D vector)
#     for px, py in axis_pca:
#         ax4.quiver(0, 0, px, py, color=color, angles='xy', scale_units='xy', scale=1)


# ax5 = fig1.add_subplot(325)
# ax5.set_title('Transformed data PCA')
# x_pca, y_pca = transformed_data_pca.T
# for i, point in enumerate(points):
#     ax5.scatter(x_pca[i], y_pca[i], label=point+"'", color='r', marker='x')  # plot the points
#     ax5.text(x_pca[i], y_pca[i], point+"'")  # label the points
# ax5.axis('equal')
# ax5.grid(True)

# ax6 = fig1.add_subplot(326)
# ax6.set_title('Transformed data PCA')
# x_pca, y_pca = transformed_data_pca1.T
# for i, point in enumerate(points):
#     ax6.scatter(x_pca[i], y_pca[i], label=point+"'", color='r', marker='x')  # plot the points
#     ax6.text(x_pca[i], y_pca[i], point+"'")  # label the points
# ax6.axis('equal')
# ax6.grid(True)

# fig1.show()

# # Covariance visualization
# # Visualize the points
# # Plot original data
# x, y = data.T 
# points = ['A', 'B', 'C', 'D']
# fig2 = plt.figure(figsize=(10, 8))
# fig2.suptitle('PCA analysis with Covariance')
# plt.subplots_adjust(hspace = 0.5)
# ax1 = fig2.add_subplot(321)
# ax1.set_title('Original data')
# for i, point in enumerate(points):
#     ax1.scatter(x[i], y[i], label=point)  # plot the points
#     ax1.text(x[i], y[i], point)  # label the points
# ax1.axis('equal')
# ax1.grid(True)

# x_rot, y_rot = data1.T  
# ax2 = fig2.add_subplot(322)
# ax2.set_title('Rotated/Perturb data')
# for i, point in enumerate(points):
#     ax2.scatter(x_rot[i], y_rot[i], label=point)  # plot the points
#     ax2.text(x_rot[i], y_rot[i], point)  # label the points
# ax2.axis('equal')
# ax2.grid(True)

# # Title for second row of plots

# ax3 = fig2.add_subplot(323)
# ax3.set_title('PCA_cov')
# for i, point in enumerate(points):
#     ax3.scatter(x[i], y[i], label=point)  # plot the points
#     ax3.text(x[i], y[i], point)  # label the points
# ax3.axis('equal')
# ax3.grid(True)

# # Add the principal axes
# for axis_pca, color in zip([principal_axis_cov], ['r']):
#     # principal_axis should be an array of shape (2, 2), 
#     # where each row is a principal axis (2D vector)
#     for px, py in axis_pca:
#         ax3.quiver(0, 0, px, py, color=color, angles='xy', scale_units='xy', scale=1)

# ax4 = fig2.add_subplot(324)
# ax4.set_title('PCA_cov')
# for i, point in enumerate(points):
#     ax4.scatter(x_rot[i], y_rot[i], label=point)  # plot the points
#     ax4.text(x_rot[i], y_rot[i], point)  # label the points
# ax4.axis('equal')
# ax4.grid(True)

# # Add the principal axes
# for axis_pca, color in zip([principal_axis_cov1], ['r']):
#     # principal_axis should be an array of shape (2, 2), 
#     # where each row is a principal axis (2D vector)
#     for px, py in axis_pca:
#         ax4.quiver(0, 0, px, py, color=color, angles='xy', scale_units='xy', scale=1)


# ax5 = fig2.add_subplot(325)
# ax5.set_title('Transformed data PCA')
# x_pca, y_pca = transformed_data_cov.T
# for i, point in enumerate(points):
#     ax5.scatter(x_pca[i], y_pca[i], label=point+"'", color='r', marker='x')  # plot the points
#     ax5.text(x_pca[i], y_pca[i], point+"'")  # label the points
# ax5.axis('equal')
# ax5.grid(True)

# ax6 = fig2.add_subplot(326)
# ax6.set_title('Transformed data PCA')
# x_pca, y_pca = transformed_data_cov1.T
# for i, point in enumerate(points):
#     ax6.scatter(x_pca[i], y_pca[i], label=point+"'", color='r', marker='x')  # plot the points
#     ax6.text(x_pca[i], y_pca[i], point+"'")  # label the points
# ax6.axis('equal')
# ax6.grid(True)

# fig2.show()

# input("Press any key to exit...")