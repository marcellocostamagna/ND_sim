import numpy as np  
from pre_processing import *
from pca_tranform import * 
from fingerprint import *
from similarity import *


# PRE-PROCESSING
# List of molecules from SDF file
molecules = collect_molecules_from_sdf('swapping.sdf')
# List of dictionaries containing information about each molecule
molecules_info = [collect_molecule_info(molecule) for molecule in molecules]
# Manipulation of molecules_info: Normalization or Tapering
# Normalization
# First normalization
#molecules_info = [normalize_features(molecule) for molecule in molecules_info] 
# Second normalization
#molecules_info = [normalize_features2(molecule) for molecule in molecules_info]
# Tapering
molecules_info = [taper_features(molecule, np.log) for molecule in molecules_info]
# Get the 6D matrix for each molecule
molecules_data = [get_molecule_6D_datastructure(molecule_info) for molecule_info in molecules_info]

# PCA
# Get the PCA tranformed data 
_, tranformed_data, _, _ = perform_PCA_and_get_transformed_data_cov(molecules_data[2])
_, transformed_data1, _, _ = perform_PCA_and_get_transformed_data_cov(molecules_data[3])

# FINGERPRINT
# Get the fingerprint from teh tranformed data
fingerprint  = get_fingerprint(tranformed_data)
fingerprint1 = get_fingerprint(transformed_data1)

# SIMILARITY
# Calculate the similarity between two fingerprints
partial_score = calculate_partial_score(fingerprint, fingerprint1) # Manhattan distance
similarity = get_similarity_measure(partial_score)

print(similarity)