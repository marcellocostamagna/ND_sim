from nD_tools import *
from trials.perturbations import *
from fingerprints import * 
from similarity_3d import calculate_partial_score



data = np.array([
    [  1,  0, -1/math.sqrt(2),  6],
    [  0,  0,  0,               1],
    [ -1,  0, -1/math.sqrt(2),  1],
    [  0,  1,  1/math.sqrt(2),  8],
    [  0, -1,  1/math.sqrt(2),  9]
])

distances = plot_data_and_ref_points(data)
print(np.array(distances))
fingerprint = compute_statistics(np.array(distances)) 

# Rotate points
coords = data[:, :-1]
masses = data[:, -1]

coords = rotate_points(data, 30, 0, 0)
print(coords)

#data1 = np.c_[coords, masses]
data1 = coords

distances1 =plot_data_and_ref_points(data1)
fingerprint1 = compute_statistics(np.array(distances1)) 
print(np.array(distances1))


print(f'finger1: {fingerprint}')
print(f'finger2: {fingerprint1}')

