import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew
import math as m

def compute_center_of_mass(points, masses):
    return np.average(points, axis=0, weights=masses)

def compute_geometrical_center(points):
    return np.mean(points, axis=0)

def translate_points_to_center_of_mass(points, masses):
    # Calculate the center of mass
    center_of_mass = np.average(points, axis=0, weights=masses)
    # Translate the points so that the center of mass is at the origin
    translated_points = points - center_of_mass
    return translated_points

def translate_points_to_geometrical_center(points):
    # Calculate the geometrical center
    geometrical_center = np.mean(points, axis=0)
    # Translate the points so that the geometrical center is at the origin
    translated_points = points - geometrical_center
    return translated_points

def max_distance_from_center_of_mass(points, center_of_mass):
    distances = np.linalg.norm(points - center_of_mass, axis=1)
    return np.max(distances)

def max_distance_from_geometrical_center(points, geometrical_center):
    distances = np.linalg.norm(points, axis=1)
    return np.max(distances)

def generate_reference_points(center_of_mass, principal_axes, max_distance):
    points = [center_of_mass]
    
    for axis in principal_axes:
        point = center_of_mass + max_distance * (axis/np.linalg.norm(axis))
        points.append(point)
    
    return points

def compute_inertia_tensor(points, masses, center_of_mass):
    inertia_tensor = np.zeros((3, 3))
    for point, mass in zip(points, masses):
        r = point - center_of_mass
        inertia_tensor += mass * (np.eye(3) * np.dot(r, r) - np.outer(r, r))
        #inertia_tensor *= -1

    return inertia_tensor

def compute_principal_axes(inertia_tensor, points, masses):
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    principal_axes = eigenvectors.T

    # If one of the eigenvalues is zero, the corresponding eigenvector is redefined as 
    # a vector orthogonal to the other two eigenvectors with the positive direction
    # pointing towards the more massive side of the cloud of points.
    for i, eigenvalue in enumerate(eigenvalues):
        axis = principal_axes[i]
        if abs(eigenvalue) <= 1e-2:
            eigenvalues[i] = 1e-6
            axis = np.cross(principal_axes[(i+1)%3], principal_axes[(i+2)%3])
            
        # Project the coordinates of the cloud of points onto the fake axis
        # TODO: Problem with sign
        projections = np.sign(np.dot(points, axis))
        # Compute the weighted sum of projections using the masses
        # weighted_sum = np.dot(projections, masses)
        # # projections without masses
        sum = np.sum(projections)
        # # Normalize the fake axis
        # if weighted_sum == 0:
        #     weighted_sum = 1
        # axis = axis * np.sign(weighted_sum)
        # consdiering no masses
        axis = axis * np.sign(sum)
        #axis = axis / np.linalg.norm(axis)
        principal_axes[i] = axis
    
    handedness = compute_handedness(principal_axes, eigenvalues)
    print("Handedness: ", handedness)

    if handedness == "left-handed":
         principal_axes[1] = -principal_axes[1]

    return principal_axes, eigenvalues

def compute_handedness(principal_axes, eigenvalues):

    # Sort the principal axes based on their eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    sorted_axes = principal_axes[sorted_indices]

    triple_scalar_product = np.dot(principal_axes[0], np.cross(principal_axes[1], principal_axes[2]))
    if triple_scalar_product > 0:
        return "right-handed"
    else:
        return "left-handed"
    
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

def compute_distances(points, reference_points):
    """Compute the distance of each point to the 4 refernce points"""
    num_points = points.shape[0]
    num_ref_points = len(reference_points)
    distances = np.zeros((num_ref_points, num_points))
    
    for i, point in enumerate(points):
        for j, ref_point in enumerate(reference_points):
            distances[j, i] = np.linalg.norm(point - ref_point)
            
    return distances  


def compute_weighted_distances(points, masses, reference_points):
    """Compute the mass-weigthed distance of each point to the 4 refernce points"""
    num_points = points.shape[0]
    num_ref_points = len(reference_points)
    weighted_distances = np.zeros((num_ref_points, num_points))

    for i, (point, mass) in enumerate(zip(points, masses)):
        for j, ref_point in enumerate(reference_points):
            if mass == 0:
                mass = 1
            weighted_distances[j, i] = ((m.log(mass))) * np.linalg.norm(point - ref_point)

    return weighted_distances


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

def compute_fingerprint(points, masses, n_prot, n_neut, n_elec):

    print(n_neut)
    #particles = [n_prot, n_neut, n_elec]
    fingerprints = []

    #points, center_of_mass = translate_points_to_center_of_mass(points, masses), [0,0,0]
    points, geometrical_center = translate_points_to_geometrical_center(points), [0,0,0]

    #inertia_tensor = compute_inertia_tensor(points, masses, center_of_mass)
    weights = np.ones(len(points))
    inertia_tensor = compute_inertia_tensor(points, weights, geometrical_center)

    principal_axes, eigenvalues = compute_principal_axes(inertia_tensor, points, masses)

    #max_distance = max_distance_from_center_of_mass(points, center_of_mass)
    max_distance = max_distance_from_geometrical_center(points, geometrical_center)

    #reference_points = generate_reference_points(center_of_mass, principal_axes, max_distance)
    reference_points = generate_reference_points(geometrical_center, principal_axes, max_distance)
    # compute distances
    #distances = compute_distances(points, reference_points)
    # compute weighted distances
    proton_distances = compute_weighted_distances(points, n_prot, reference_points)
    neutron_distances = compute_weighted_distances(points, n_neut, reference_points)
    electron_distances = compute_weighted_distances(points, n_elec, reference_points)
    # compute statistics
    # statistics_matrix, fingerprint_1 = compute_statistics(distances)
    # statistics_matrix, fingerprint_2 = compute_statistics(weighted_distances)
    proton_fingerprint = compute_statistics(proton_distances)
    print(proton_fingerprint)
    neutron_fingerprint = compute_statistics(neutron_distances)
    electron_fingerprint = compute_statistics(electron_distances)
    
    # print("Center of mass:", center_of_mass)
    # # print("Inertia tensor:", inertia_tensor)
    # print("Principal axes:", principal_axes)
    # print("Eigenvalues:", eigenvalues)
    # # print("Distances:", distances)
    # # print("Fingerprint of regular distances:", fingerprint_1)
    # # print("Fingerprint of weighted distances:", fingerprint_2)
    # print(f'Handedness: {compute_handedness(principal_axes, eigenvalues)}')

    # If the third eigenvalue less than 0.001, we still need to visulaize the third axis
    if np.abs(eigenvalues[2]) < 0.001:
        eigenvalues[2] = 0.5 * eigenvalues[1]

    #visualize(points, n_prot, center_of_mass, principal_axes, eigenvalues, max_distance, reference_points)
    visualize(points, n_prot, geometrical_center, principal_axes, eigenvalues, max_distance, reference_points)

    fingerprints = [proton_fingerprint, neutron_fingerprint, electron_fingerprint]

    return fingerprints


# VISUALIZATION OF PRINCIPAL AXES    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)

    # # draw principal axes
    # for axis in principal_axes:
    #     x, y, z = axis
    #     ax.quiver(0, 0, 0, x, y, z, length=1.0, color='r', lw=2, arrow_length_ratio=0.1)
    
    # plt.show()

# ORTHOGONALITY CHECK
    # # Check that all eigenvectors are orthogonal and print if it is or it is not 
    # for i in range(3):
    #     for j in range(i, 3):
    #         # print the angle between the two vectors
    #         print("The angle between the principal axes {} and {} is {} degrees".format(i, j, np.arccos(np.dot(principal_axes[i], principal_axes[j]))*180/np.pi))


# def compute_distances(points, center_of_mass, principal_axes):
#     """Compute the distance of each point to the center of mass and to each of the principal axes"""
#     num_points = points.shape[0]
#     distances = np.zeros((4, num_points))
    
#     for i, point in enumerate(points):
#         distances[0, i] = np.linalg.norm(point - center_of_mass)
        
#         for j, axis in enumerate(principal_axes):
#             point_rel = point - center_of_mass
#             projection = np.dot(point_rel, axis) * axis
#             distance = np.linalg.norm(point_rel - projection)
#             distances[j + 1, i] = distance
            
#     return distances  


# def compute_weighted_distances(points, masses, center_of_mass, principal_axes):
#     num_points = points.shape[0]
#     weighted_distances = np.zeros((4, num_points))
    
#     for i, (point, mass) in enumerate(zip(points, masses)):
#         weighted_distances[0, i] = mass * np.linalg.norm(point - center_of_mass)
        
#         for j, axis in enumerate(principal_axes):
#             point_rel = point - center_of_mass
#             projection = np.dot(point_rel, axis) * axis
#             distance = mass * np.linalg.norm(point_rel - projection)
#             weighted_distances[j + 1, i] = mass * distance
            
#     return weighted_distances


# for particle in particles:
#         points, center_of_mass = translate_points_to_center_of_mass(points, particle), [0,0,0]

#         inertia_tensor = compute_inertia_tensor(points, particle, center_of_mass)
#         principal_axes, eigenvalues = compute_principal_axes(inertia_tensor, points, particle)

#         max_distance = max_distance_from_center_of_mass(points, center_of_mass)

#         reference_points = generate_reference_points(center_of_mass, principal_axes, max_distance)
#         # compute distances
#         #distances = compute_distances(points, reference_points )
#         # compute weighted distances
#         distances = compute_weighted_distances(points, particle, reference_points)
#         # compute statistics
#         # statistics_matrix, fingerprint_1 = compute_statistics(distances)
#         # statistics_matrix, fingerprint_2 = compute_statistics(weighted_distances)
#         fingerprints.append(compute_statistics(distances))