import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.stats import skew

# Sample input data
#points = np.array([[1, 1, 3], [4, 1, 3] , [2.5 , 2 , 3]]) #, [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23 , 24]])
points = np.array([
     [  1,  0, -1/math.sqrt(2)],
     [ -1,  0, -1/math.sqrt(2)],
     [  0,  1,  1/math.sqrt(2)],
     [  0, -1,  1/math.sqrt(2)]
 ])

masses = [1, 3, 7, 5 ] #, 3, 4 , 5, 6, 7, 8]

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

def compute_handedness(principal_axes):
    triple_scalar_product = np.dot(principal_axes[0], np.cross(principal_axes[1], principal_axes[2]))
    if triple_scalar_product > 0:
        return "right-handed"
    else:
        return "left-handed"

def visualize(points, masses, center_of_mass, principal_axes, eigenvalues, scale):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualize points with size based on mass
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=[mass * 20 for mass in masses], c='b', alpha=0.6)

    # Visualize eigenvectors with colors based on eigenvalues
    colors = ['r', 'g', 'k']
    max_eigenvalue = np.max(np.abs(eigenvalues))
    scaled_axes = []
    for axis, eigenvalue, color in zip(principal_axes, eigenvalues, colors):
        scaled_axis = axis * (eigenvalue / max_eigenvalue) * scale # Adjust this factor to change the length of the eigenvectors
        scaled_axes.append(scaled_axis)
        ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
                  scaled_axis[0], scaled_axis[1], scaled_axis[2],
                  color=color, lw=2, arrow_length_ratio=0.1)

    # Adjust axes to fit all points
    scaled_axes = np.array(scaled_axes)
    min_values = np.min(points[:, None, :] - scaled_axes, axis=(0, 1))
    max_values = np.max(points[:, None, :] + scaled_axes, axis=(0, 1))
    padding = 0  # Increase or decrease this value to change the padding around the axes

    ax.set_xlim(min_values[0] - padding, max_values[0] + padding)
    ax.set_ylim(min_values[1] - padding, max_values[1] + padding)
    ax.set_zlim(min_values[2] - padding, max_values[2] + padding)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show(block=False)
    plt.pause(0.001)

def compute_distances(points, center_of_mass, principal_axes):
    num_points = points.shape[0]
    distances = np.zeros((4, num_points))
    
    for i, point in enumerate(points):
        distances[0, i] = np.linalg.norm(point - center_of_mass)
        
        for j, axis in enumerate(principal_axes):
            point_rel = point - center_of_mass
            projection = np.dot(point_rel, axis) * axis
            distance = np.linalg.norm(point_rel - projection)
            distances[j + 1, i] = distance
            
    return distances  

def compute_weighted_distances(points, masses, center_of_mass, principal_axes):
    num_points = points.shape[0]
    weighted_distances = np.zeros((4, num_points))
    
    for i, (point, mass) in enumerate(zip(points, masses)):
        weighted_distances[0, i] = mass * np.linalg.norm(point - center_of_mass)
        
        for j, axis in enumerate(principal_axes):
            point_rel = point - center_of_mass
            projection = np.dot(point_rel, axis) * axis
            distance = mass * np.linalg.norm(point_rel - projection)
            weighted_distances[j + 1, i] = mass * distance
            
    return weighted_distances

def compute_statistics(distances):
    means = np.mean(distances, axis=1)
    std_devs = np.std(distances, axis=1)
    skewness = skew(distances, axis=1)
    # check if skewness is nan
    skewness[np.isnan(skewness)] = 0
    
    statistics_matrix = np.vstack((means, std_devs, skewness)).T
    statistics_list = np.hstack((means, std_devs, skewness))
    
    return statistics_matrix, statistics_list  


def compute_fingerprint(points, masses):
    center_of_mass = compute_center_of_mass(points, masses)
    inertia_tensor = compute_inertia_tensor(points, masses, center_of_mass)
    principal_axes, eigenvalues = compute_principal_axes(inertia_tensor)
    # compute distances
    distances = compute_distances(points, center_of_mass, principal_axes)
    # compute weighted distances
    weighted_distances = compute_weighted_distances(points, masses, center_of_mass, principal_axes)
    # compute statistics
    statistics_matrix, fingerprint_1 = compute_statistics(distances)
    statistics_matrix, fingerprint_2 = compute_statistics(weighted_distances)

    # print("Center of mass:", center_of_mass)
    # print("Inertia tensor:", inertia_tensor)
    print("Principal axes:", principal_axes)
    # print("Eigenvalues:", eigenvalues)
    # print("Distances:", distances)
    # print("Fingerprint of regular distances:", fingerprint_1)
    # print("Fingerprint of weighted distances:", fingerprint_2)
    print(f'Handedness: {compute_handedness(principal_axes)}')

    # If the third eigenvalue less than 0.001, we still need to visulaize the third axis
    if np.abs(eigenvalues[2]) < 0.001:
        eigenvalues[2] = 0.5 * eigenvalues[1]

    max_distance = max_distance_from_center_of_mass(points, center_of_mass)
    visualize(points, masses, center_of_mass, principal_axes, eigenvalues, max_distance)

    return fingerprint_1, fingerprint_2


