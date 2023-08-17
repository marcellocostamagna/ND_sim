import numpy as np
from rdkit import Chem

###### PRE-PROCESSING #######

### Fetaures fucntions ###
def get_protons(atom):
    return atom.GetAtomicNum()

def get_delta_neutrons(atom):
    return int(round(atom.GetMass())) - atom.GetAtomicNum()

def get_formal_charge(atom):
    return atom.GetFormalCharge()

### Re-scaling functions ###

## Tapering functions
def taper_p(value):
    return np.log(value)

def taper_n(value):
    return np.log(value + 2)

def taper_c(value):
    return np.log(value + 5)

# Normalization functions

def normalize_feature_using_full_range(feature_data: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Normalizes the given feature using the range of the coordinates.
    """
    if coordinates is None or len(coordinates) == 0:
        raise ValueError("Coordinates must be provided and should not be empty!")
    
    max_coord = np.amax(coordinates)
    min_coord = np.amin(coordinates)
    
    feature_min = np.amin(feature_data)
    feature_max = np.amax(feature_data)
    
    if feature_min != feature_max:
        normalized_feature = (feature_data - feature_min) / (feature_max - feature_min) * (max_coord - min_coord) + min_coord
    else:
        normalized_feature = feature_data
    return normalized_feature

def normalize_feature_using_specific_axis(feature_data: np.ndarray, coordinates: np.ndarray, axis_choice: str = "smallest") -> np.ndarray:
    """
    Normalizes the given feature using the range of a specific axis (largest, smallest, or intermediate) of the coordinates.
    """
    if coordinates is None or len(coordinates) == 0:
        raise ValueError("Coordinates must be provided and should not be empty!")
    
    ranges = np.ptp(coordinates, axis=0)
    axis = None
    if axis_choice == "smallest":
        axis = np.argmin(ranges)
    elif axis_choice == "largest":
        axis = np.argmax(ranges)
    elif axis_choice == "intermediate":
        axis = np.argsort(ranges)[1]
    else:
        raise ValueError("Invalid axis_choice. Choose from 'smallest', 'largest', or 'intermediate'.")

    max_coord = np.max(coordinates[:, axis])
    min_coord = np.min(coordinates[:, axis])

    feature_min = np.amin(feature_data)
    feature_max = np.amax(feature_data)

    if feature_min != feature_max:
        normalized_feature = (feature_data - feature_min) / (feature_max - feature_min) * (max_coord - min_coord) + min_coord
    else:
        normalized_feature = feature_data
    return normalized_feature

###### FINGERPRINT ########

def compute_scaling_factor(molecule_data):
    """
    Computes the largest distance between the centroid and the molecule data points
    """
    centroid = np.zeros(molecule_data.shape[1])
    distances = np.linalg.norm(molecule_data - centroid, axis=1)
    return np.max(distances)

def compute_scaling_matrix(molecule_data):
    """
    Computes a diagonal scaling matrix with the maximum absolute values 
    for each dimension of the molecule data as its diagonal entries
    """
    max_values = np.max(np.abs(molecule_data), axis=0)
    return np.diag(max_values)


#### DEFAULTS #####

DEFAULT_FEATURES = {
    'protons' : [get_protons, taper_p],
    'delta_neutrons' : [get_delta_neutrons, taper_n],
    'formal_charges' : [get_formal_charge, taper_c]
    }