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
molecules = load_molecules_from_sdf(f'{cwd}/similarity/sd_data/guac_sample_20.sdf')

# OPTONAL
# Define new fetaures function and respective rescaling functions to store
# into a features dictionary

# QUERY FINGERPRINT
query_fp = generate_nd_molecule_fingerprint(molecules[0], DEFAULT_FEATURES, scaling_method= None)

# ALL FINGERPRINTS
all_fp = [generate_nd_molecule_fingerprint(mol) for mol in  molecules]


# SIMILARITIES
# Calculate the similarity between the query and all the molecules
similarities = [get_similarity_score(query_fp, fp) for fp in all_fp]

# Sort the similarities in descending order
similarities = sorted(similarities, reverse=True)
    
print(similarities)