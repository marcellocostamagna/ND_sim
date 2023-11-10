import numpy as np
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

def visualize(points, center_of_mass, principal_axes): # , ax, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # # Visualize points with size based on mass (assuming protons)
    # for i, label in enumerate((labels)):
    #     ax.scatter(points[i, 0], points[i, 1], points[i, 2], c='b', alpha=0.6, label = label)
    #     ax.text(points[i, 0], points[i, 1], points[i, 2], labels[i], fontsize=12)
    # Visualize points 
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', alpha=0.6)
    
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