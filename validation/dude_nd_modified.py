import os
import numpy as np
import time
import random
from openbabel import openbabel, pybel
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *

def calculate_enrichment_factor(y_true, y_scores, percentage):
    n_molecules = len(y_true)
    n_actives = sum(y_true)
    f_actives = n_actives / n_molecules
    
    n_top_molecules = int(n_molecules * percentage)
    top_indices = np.argsort(y_scores)[-n_top_molecules:]
    n_top_actives = sum(y_true[i] for i in top_indices)
    
    expected_actives = percentage * n_molecules * f_actives
    enrichment_factor = n_top_actives / expected_actives
    return enrichment_factor

print(f'CWD: {os.getcwd()}')
root_directory = f"{os.getcwd()}/similarity/validation/all"
method = "nd_fingerprint"

enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}

results = {}

print(f"Running benchmark for method: {method}")
enrichments = {k: [] for k in enrichment_factors.keys()}

start_time_total = time.time()

for folder in sorted(os.listdir(root_directory)):
    folder_start_time = time.time()
    
    print(f"Processing folder: {folder}")
    folder_path = os.path.join(root_directory, folder)
    folder_enrichments = {k: [] for k in enrichment_factors.keys()}
    
    if os.path.isdir(folder_path):
        actives_file = os.path.join(folder_path, "actives_final.sdf")
        decoys_file = os.path.join(folder_path, "decoys_final.sdf")
        
        actives = collect_molecules_from_sdf(actives_file)
        decoys = collect_molecules_from_sdf(decoys_file)
        
        # Randomly select 10 actives
        selected_actives = random.sample(actives, 10) if len(actives) > 10 else actives
        
        for query_mol in selected_actives:
            all_mols = actives + decoys
            all_mols.remove(query_mol)  # remove the query molecule from the list
            query_fp = get_nd_fingerprint(query_mol, scaling_method='factor')
            y_scores = []

            for mol in all_mols:
                mol_fp = get_nd_fingerprint(mol, scaling_method='factor')
                partial_score = calculate_partial_score(query_fp, mol_fp)
                similarity = get_similarity_measure(partial_score)
                y_scores.append(similarity)
            
            y_true = [1 if mol in actives else 0 for mol in all_mols]
        
            for percentage in enrichment_factors.keys():
                ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                enrichments[percentage].append(ef)
                folder_enrichments[percentage].append(ef)

        # Print the average enrichment factors for this folder
        avg_folder_enrichments = {percentage: np.mean(values) for percentage, values in folder_enrichments.items()}
        print(f"\nAverage Enrichment Factors for folder {folder}:")
        for percentage, ef in avg_folder_enrichments.items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")

    folder_end_time = time.time()
    print(f"Finished processing folder {folder} in {folder_end_time - folder_start_time:.2f} seconds")

end_time_total = time.time()
print(f"\nTotal processing time for {method}: {end_time_total - start_time_total:.2f} seconds")

avg_enrichments = {percentage: np.mean(values) for percentage, values in enrichments.items()}

results[method] = {
    'enrichments': avg_enrichments
}

print(f"\nResults for {method}:")
for percentage, ef in results[method]['enrichments'].items():
    print(f"Enrichment Factor at {percentage*100}%: {ef}")