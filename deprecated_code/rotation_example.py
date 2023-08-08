from nD_tools import *
from trials.perturbations import *
from fingerprints import * 
from similarity_3d import calculate_partial_score



data = np.array([
    [  1,  0, 0,  6],
    [ -1,  0, 0,  6], 
])

distances = plot_data_and_ref_points(data)

# rotate data

coords = data[:, :-1]

# rotate around x axis
coords_rotated = rotate_points(coords, 0, 90, 0)

# ASSERT RESULTS    
extpected_coords_rotated = np.array([[0, 1, 0], [ 0, -1, 0]])

plt.show()
