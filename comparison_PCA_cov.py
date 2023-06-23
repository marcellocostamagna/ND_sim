import numpy as np
from perturbations import *
from rdkit import Chem
from utils import *
from cov_fingerprint import *
from similarity_3d import *
from pca_fingerprint import *
from copy import deepcopy
import matplotlib.pyplot as plt


# rectangle rotatated 45 degrees around the z axis
# data = np.array([[ -1, -2],
#                 [-2, -1],
#                 [2, 1], 
#                 [1, 2]]).astype(float)

# Irregular quadrilateral
data = np.array([[ -0.9, -1.8, 0],
                [-2.2, -0.9, 0],
                [1.7, 0.5, 0], 
                [0.8, 1.9, 0]]).astype(float)

# Rotate the points
tmp_data = perturb_coordinates(data, 1)
# generate random float number between 0 and 360
angle = np.random.uniform(0, 360)
data1 = rotate_points(tmp_data, 0, 0, angle)

# Get only x and y coordinates
data = data[:, :-1]
data1 = data1[:, :-1]

# PCA with pca_fingerprint.py
fingerprint_pca, transformed_data_pca, principal_axis_pca, standard_axis_pca = get_pca_fingerprint(data)

fingerprint_pca1, transformed_data_pca1, principal_axis_pca1, standard_axis_pca1 = get_pca_fingerprint(data1)

# # PCA with cov_fingerprint.py
fingerprint_cov, transformed_data_cov, principal_axis_cov, standard_axis_cov = compute_nD_fingerprint(data)

fingerprint_cov1, transformed_data_cov1, principal_axis_cov1, standard_axis_cov1 = compute_nD_fingerprint(data1)

similarity_pca = 1/(1 + calculate_nD_partial_score(fingerprint_pca, fingerprint_pca1))
print('similarity_pca: ', similarity_pca)

similarity_cov = 1/(1 + calculate_nD_partial_score(fingerprint_cov, fingerprint_cov1))
print('similarity_cov: ', similarity_cov)

similarity = 1/(1 + calculate_nD_partial_score(fingerprint_pca, fingerprint_cov))
print('similarity: ', similarity)

print('Angle: ', angle)

# PCA with SVD visulaization

# Visualize the points
plt.ion()  # turn interactive mode on
# Plot original data
x, y = data.T 
points = ['A', 'B', 'C', 'D']
fig1 = plt.figure(figsize=(10, 8))
fig1.suptitle('PCA analysis with SVD')
plt.subplots_adjust(hspace = 0.5)
ax1 = fig1.add_subplot(321)
ax1.set_title('Original data')
for i, point in enumerate(points):
    ax1.scatter(x[i], y[i], label=point)  # plot the points
    ax1.text(x[i], y[i], point)  # label the points
ax1.axis('equal')
ax1.grid(True)

x_rot, y_rot = data1.T  
ax2 = fig1.add_subplot(322)
ax2.set_title('Rotated/Perturb data')
for i, point in enumerate(points):
    ax2.scatter(x_rot[i], y_rot[i], label=point)  # plot the points
    ax2.text(x_rot[i], y_rot[i], point)  # label the points
ax2.axis('equal')
ax2.grid(True)

# Title for second row of plots

ax3 = fig1.add_subplot(323)
ax3.set_title('PCA_svd')
for i, point in enumerate(points):
    ax3.scatter(x[i], y[i], label=point)  # plot the points
    ax3.text(x[i], y[i], point)  # label the points
ax3.axis('equal')
ax3.grid(True)

# Add the principal axes
for axis_pca, color in zip([principal_axis_pca], ['r']):
    # principal_axis should be an array of shape (2, 2), 
    # where each row is a principal axis (2D vector)
    for px, py in axis_pca:
        ax3.quiver(0, 0, px, py, color=color, angles='xy', scale_units='xy', scale=1)

ax4 = fig1.add_subplot(324)
ax4.set_title('PCA_svd')
for i, point in enumerate(points):
    ax4.scatter(x_rot[i], y_rot[i], label=point)  # plot the points
    ax4.text(x_rot[i], y_rot[i], point)  # label the points
ax4.axis('equal')
ax4.grid(True)

# Add the principal axes
for axis_pca, color in zip([principal_axis_pca1], ['r']):
    # principal_axis should be an array of shape (2, 2), 
    # where each row is a principal axis (2D vector)
    for px, py in axis_pca:
        ax4.quiver(0, 0, px, py, color=color, angles='xy', scale_units='xy', scale=1)


ax5 = fig1.add_subplot(325)
ax5.set_title('Transformed data PCA')
x_pca, y_pca = transformed_data_pca.T
for i, point in enumerate(points):
    ax5.scatter(x_pca[i], y_pca[i], label=point+"'", color='r', marker='x')  # plot the points
    ax5.text(x_pca[i], y_pca[i], point+"'")  # label the points
ax5.axis('equal')
ax5.grid(True)

ax6 = fig1.add_subplot(326)
ax6.set_title('Transformed data PCA')
x_pca, y_pca = transformed_data_pca1.T
for i, point in enumerate(points):
    ax6.scatter(x_pca[i], y_pca[i], label=point+"'", color='r', marker='x')  # plot the points
    ax6.text(x_pca[i], y_pca[i], point+"'")  # label the points
ax6.axis('equal')
ax6.grid(True)

fig1.show()

# Covariance visualization
# Visualize the points
# Plot original data
x, y = data.T 
points = ['A', 'B', 'C', 'D']
fig2 = plt.figure(figsize=(10, 8))
fig2.suptitle('PCA analysis with Covariance')
plt.subplots_adjust(hspace = 0.5)
ax1 = fig2.add_subplot(321)
ax1.set_title('Original data')
for i, point in enumerate(points):
    ax1.scatter(x[i], y[i], label=point)  # plot the points
    ax1.text(x[i], y[i], point)  # label the points
ax1.axis('equal')
ax1.grid(True)

x_rot, y_rot = data1.T  
ax2 = fig2.add_subplot(322)
ax2.set_title('Rotated/Perturb data')
for i, point in enumerate(points):
    ax2.scatter(x_rot[i], y_rot[i], label=point)  # plot the points
    ax2.text(x_rot[i], y_rot[i], point)  # label the points
ax2.axis('equal')
ax2.grid(True)

# Title for second row of plots

ax3 = fig2.add_subplot(323)
ax3.set_title('PCA_cov')
for i, point in enumerate(points):
    ax3.scatter(x[i], y[i], label=point)  # plot the points
    ax3.text(x[i], y[i], point)  # label the points
ax3.axis('equal')
ax3.grid(True)

# Add the principal axes
for axis_pca, color in zip([principal_axis_cov], ['r']):
    # principal_axis should be an array of shape (2, 2), 
    # where each row is a principal axis (2D vector)
    for px, py in axis_pca:
        ax3.quiver(0, 0, px, py, color=color, angles='xy', scale_units='xy', scale=1)

ax4 = fig2.add_subplot(324)
ax4.set_title('PCA_cov')
for i, point in enumerate(points):
    ax4.scatter(x_rot[i], y_rot[i], label=point)  # plot the points
    ax4.text(x_rot[i], y_rot[i], point)  # label the points
ax4.axis('equal')
ax4.grid(True)

# Add the principal axes
for axis_pca, color in zip([principal_axis_cov1], ['r']):
    # principal_axis should be an array of shape (2, 2), 
    # where each row is a principal axis (2D vector)
    for px, py in axis_pca:
        ax4.quiver(0, 0, px, py, color=color, angles='xy', scale_units='xy', scale=1)


ax5 = fig2.add_subplot(325)
ax5.set_title('Transformed data PCA')
x_pca, y_pca = transformed_data_cov.T
for i, point in enumerate(points):
    ax5.scatter(x_pca[i], y_pca[i], label=point+"'", color='r', marker='x')  # plot the points
    ax5.text(x_pca[i], y_pca[i], point+"'")  # label the points
ax5.axis('equal')
ax5.grid(True)

ax6 = fig2.add_subplot(326)
ax6.set_title('Transformed data PCA')
x_pca, y_pca = transformed_data_cov1.T
for i, point in enumerate(points):
    ax6.scatter(x_pca[i], y_pca[i], label=point+"'", color='r', marker='x')  # plot the points
    ax6.text(x_pca[i], y_pca[i], point+"'")  # label the points
ax6.axis('equal')
ax6.grid(True)

fig2.show()

input("Press any key to exit...")