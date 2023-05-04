import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Sample input data
#points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]])
points = np.array([
     [  1,  0, -1/math.sqrt(2)],
     [ -1,  0, -1/math.sqrt(2)],
     [  0,  1,  1/math.sqrt(2)],
     [  0, -1,  1/math.sqrt(2)]
 ])

masses = [1, 2, 3, 4] #, 5, 6, 7, 8]

def compute_center_of_mass(points, masses):
    return np.average(points, axis=0, weights=masses)

def compute_inertia_tensor(points, masses, center_of_mass):
    inertia_tensor = np.zeros((3, 3))
    for point, mass in zip(points, masses):
        r = point - center_of_mass
        inertia_tensor += mass * (np.outer(r, r) - np.eye(3) * np.dot(r, r))
    return inertia_tensor

def compute_principal_axes(inertia_tensor):
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    principal_axes = eigenvectors.T
    return principal_axes, eigenvalues

def visualize(points, masses, center_of_mass, principal_axes, eigenvalues):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualize points with size based on mass
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=[mass * 20 for mass in masses], c='b', alpha=0.6)

    # Visualize eigenvectors with colors based on eigenvalues
    colors = ['r', 'g', 'k']
    for axis, eigenvalue, color in zip(principal_axes, eigenvalues, colors):
        scaled_axis = axis * np.sqrt(eigenvalue)
        ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                  scaled_axis[0], scaled_axis[1], scaled_axis[2],
                  color=color, lw=2, arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    center_of_mass = compute_center_of_mass(points, masses)
    inertia_tensor = compute_inertia_tensor(points, masses, center_of_mass)
    principal_axes, eigenvalues = compute_principal_axes(inertia_tensor)

    print("Center of mass:", center_of_mass)
    print("Inertia tensor:", inertia_tensor)
    print("Principal axes:", principal_axes)

    visualize(points, masses, center_of_mass, principal_axes, eigenvalues)

if __name__ == "__main__":
    main()
