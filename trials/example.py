import numpy as np  
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
import os 

cwd = os.getcwd()
# PRE-PROCESSING
# List of molecules from SDF file
molecules = collect_molecules_from_sdf(f'{cwd}/similarity/sd_data/swapping.sdf')

# OPTONAL
# Define new fetaures function and respective rescaling functions to store
# into a features dictionary

# Get molecules represenation based on features expressed by the user
molecules_data = [mol_nd_data(mol) for mol in molecules]

# PCA
# Get the PCA tranformed data 
_, transformed_data, _, _ = perform_PCA_and_get_transformed_data_cov(molecules_data[2])
_, transformed_data1, _, _ = perform_PCA_and_get_transformed_data_cov(molecules_data[3])

# FINGERPRINT
# OPTIONAL
# Define a scaling factor of a scaling matrix to modify the reference points
scaling_factor = compute_scaling_factor(transformed_data)
scaling_factor1 = compute_scaling_factor(transformed_data1)

# Get the fingerprint from the tranformed data
fingerprint  = get_fingerprint(transformed_data, scaling_factor)
fingerprint1 = get_fingerprint(transformed_data1, scaling_factor1)

# SIMILARITY
# Calculate the similarity between two fingerprints
partial_score = calculate_partial_score(fingerprint, fingerprint1) # Manhattan distance
similarity = get_similarity_measure(partial_score)

print(similarity)