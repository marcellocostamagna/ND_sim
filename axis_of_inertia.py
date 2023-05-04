import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# Sample input data
#points = np.array([[1, 1, 3], [4, 1, 3] , [2.5 , 2 , 3]]) #, [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23 , 24]])
points = np.array([
     [  1,  0, -1/math.sqrt(2)],
     [ -1,  0, -1/math.sqrt(2)],
     [  0,  1,  1/math.sqrt(2)],
     [  0, -1,  1/math.sqrt(2)]
 ])

masses = [1, 2, 4, 3 ] #, 3, 4 , 5, 6, 7, 8]

def compute_center_of_mass(points, masses):
    return np.average(points, axis=0, weights=masses)

def max_distance_from_center_of_mass(points, center_of_mass):
    distances = np.linalg.norm(points - center_of_mass, axis=1)
    return np.max(distances)

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

def visualize(points, masses, center_of_mass, principal_axes, eigenvalues, scale):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualize points with size based on mass
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=[mass * 20 for mass in masses], c='b', alpha=0.6)

    # Visualize eigenvectors with colors based on eigenvalues
    colors = ['r', 'g', 'k']
    max_eigenvalue = np.max(np.abs(eigenvalues))
    for axis, eigenvalue, color in zip(principal_axes, eigenvalues, colors):
        scaled_axis = axis * (eigenvalue / max_eigenvalue) * scale # Adjust this factor to change the length of the eigenvectors
        ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                  scaled_axis[0], scaled_axis[1], scaled_axis[2],
                  color=color, lw=2, arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def compute_distances(points, center_of_mass, principal_axes):
    num_points = points.shape[0]
    distances = np.zeros((4, num_points))
    
    for i, point in enumerate(points):
        distances[0, i] = np.linalg.norm(point - center_of_mass)
        
        for j, axis in enumerate(principal_axes):
            point_rel = point - center_of_mass
            projection = np.dot(point_rel, axis) * axis
            distances[j + 1, i] = np.linalg.norm(point_rel - projection)
            
    return distances    

def main():
    center_of_mass = compute_center_of_mass(points, masses)
    inertia_tensor = compute_inertia_tensor(points, masses, center_of_mass)
    principal_axes, eigenvalues = compute_principal_axes(inertia_tensor)
    # compute distances
    distances = compute_distances(points, center_of_mass, principal_axes)

    print("Center of mass:", center_of_mass)
    print("Inertia tensor:", inertia_tensor)
    print("Principal axes:", principal_axes)
    print("Eigenvalues:", eigenvalues)
    print("Distances:", distances)

    # If the third eigenvalue less than 0.001, we still need to visulaize the third axis
    if np.abs(eigenvalues[2]) < 0.001:
        eigenvalues[2] = 0.5 * eigenvalues[1]


    max_distance = max_distance_from_center_of_mass(points, center_of_mass)

    visualize(points, masses, center_of_mass, principal_axes, eigenvalues, max_distance)

if __name__ == "__main__":
    main()
