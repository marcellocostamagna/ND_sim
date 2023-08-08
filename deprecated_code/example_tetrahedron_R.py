from nD_tools import *
from trials.perturbations import *
from fingerprints import * 
from similarity_3d import calculate_partial_score


# TETRAHEDRON
data = np.array([
    [  1,  0, -1/math.sqrt(2),  7],
    [  0,  0,  0,               6],
    [ -1,  0, -1/math.sqrt(2),  1],
    [  0,  1.01,  1/math.sqrt(2),  8],
    [  0, -1,  1/math.sqrt(2),  9],
])
# tetrahedron rotated 90 degrees around x axis
# data_rotated = np.array([
#     [  1,  1/math.sqrt(2),  0,  7],
#     [  0,  0,               0,  6],
#     [ -1,  1/math.sqrt(2),  0,  1],
#     [  0, -1/math.sqrt(2),  1,  8],
#     [  0, -1/math.sqrt(2), -1,  9],
# ])

# tetrahedron rotated 90 degrees around y axis
# data_rotated = np.array([
#     [  1/math.sqrt(2),  0, -1, 7],
#     [  0,              0,  0, 6],
#     [  1/math.sqrt(2),  0,  1, 1],
#     [ -1/math.sqrt(2),  1.1,  0, 8],
#     [ -1/math.sqrt(2), -1,  0, 9]
# ])

# LINEAR MOLECULE
# data = np.array([
#     [  0,  0,  1,  6],
#     [  0,  0,  -1,  1],
# ])

# distances = plot_data_and_ref_points(data)

eigenvectors, cov_a = principal_components(data)
ref_points = compute_reference_points(data, eigenvectors)
distances = compute_distances(data, ref_points)
#print(np.array(distances))
fingerprint = compute_statistics(np.array(distances))
print(fingerprint)

# # Rotate points
coords = data[:, :-1]
masses = data[:, -1]

# angle_x = np.random.uniform(-180, 180)
# angle_y = np.random.uniform(-180, 180)
# angle_z = np.random.uniform(-180, 180)

# coords, rotation_matrix = rotate_points(coords, angle_x, angle_y, angle_z)

coords, rotation_matrix = rotate_points(coords, 90, 0, 0)
# Check is some of the elements in the coords are less than 10^-9 then set them to 0
#coords = np.where(np.abs(coords) < 1e-9, 0, coords)

# col = np.array([0, 0, 0])
# rotation_matrix = np.c_[rotation_matrix, col]
# row = np.array([0, 0, 0, 1])
# rotation_matrix = np.r_[rotation_matrix, [row]]

# cov_test = rotation_matrix @ cov_a @ rotation_matrix.T
# eigenvalues_test, eigenvectors_test = np.linalg.eig(cov_test)
# eigenvectors_test = eigenvectors_test.T
# print('Principal components test')
# for i in np.argsort(eigenvalues_test)[::-1]:
#     print(eigenvalues_test[i],'->',eigenvectors_test[i])

#coords = translate_points(coords, 1000000000, 443254435434440, 57585432434310)

data1 = np.c_[coords, masses]

# # distances1 =plot_data_and_ref_points(data1)
# data1 = data_rotated

eigenvectors1, cov_b = principal_components(data1)
ref_points1 = compute_reference_points(data1, eigenvectors1)
distances1 = compute_distances(data1, ref_points1)
fingerprint1 = compute_statistics(np.array(distances1)) 
#print(np.array(distances1))

# # covariance matrix difference
# cov_diff = cov_a - cov_b
# print(f'covariance matrix difference: {cov_diff}')

# Test transformed coords:
# # center data
# data_transformed = data - np.mean(data, axis=0)
# data1_transformed = data1 - np.mean(data1, axis=0)
# data_transformed = data_transformed @ eigenvectors.T
# data1_transformed = data1_transformed @ eigenvectors1.T


print(f'finger1: {fingerprint}')
print(f'finger2: {fingerprint1}')

similarity = 1 / (1 + calculate_nD_partial_score(fingerprint, fingerprint1))
print(f'Similarity: {similarity}')

plt.show()


