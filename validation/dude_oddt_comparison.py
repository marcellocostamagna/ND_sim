import os
import numpy as np
import time
from oddt import toolkit
from oddt import shape
from sklearn.metrics import roc_auc_score

def read_molecules_from_file(file_path):
    mols = []
    for mol in toolkit.readfile(os.path.splitext(file_path)[1][1:], file_path):
        mols.append(mol)
    return mols

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

root_directory = f"{os.getcwd()}/similarity/validation/all"
methods = ['usr', 'usr_cat', 'electroshape']

enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}

results = {}

for method in methods:
    print(f"Running benchmark for method: {method}")
    roc_scores = []
    enrichments = {k: [] for k in enrichment_factors.keys()}
    
    start_time_total = time.time()

    for folder in os.listdir(root_directory):
        folder_start_time = time.time()
        
        print(f"Processing folder: {folder}")
        folder_path = os.path.join(root_directory, folder)
        
        if os.path.isdir(folder_path):
            actives_file = os.path.join(folder_path, "actives_final.sdf")
            decoys_file = os.path.join(folder_path, "decoys_final.sdf")
            query_file = os.path.join(folder_path, "crystal_ligand.mol2")
            
            actives = read_molecules_from_file(actives_file)
            decoys = read_molecules_from_file(decoys_file)
            query_mol = read_molecules_from_file(query_file)[0]
            
            y_scores = []
            for mol in actives + decoys:
                if method == 'usr':
                    query_shape = shape.usr(query_mol)
                    score = shape.usr_similarity(query_shape, shape.usr(mol))
                elif method == 'usr_cat':
                    query_shape = shape.usr_cat(query_mol)
                    score = shape.usr_similarity(query_shape, shape.usr_cat(mol))
                elif method == 'electroshape':
                    query_shape = shape.electroshape(query_mol)
                    score = shape.usr_similarity(query_shape, shape.electroshape(mol))

                y_scores.append(score)
            
            y_true = [1]*len(actives) + [0]*len(decoys)
            
            roc_scores.append(roc_auc_score(y_true, y_scores))
            
            for percentage in enrichment_factors.keys():
                ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                enrichments[percentage].append(ef)

        folder_end_time = time.time()
        print(f"Finished processing folder {folder} in {folder_end_time - folder_start_time:.2f} seconds")

    end_time_total = time.time()
    print(f"\nTotal processing time for {method}: {end_time_total - start_time_total:.2f} seconds")

    avg_roc = np.mean(roc_scores)
    avg_enrichments = {percentage: np.mean(values) for percentage, values in enrichments.items()}

    results[method] = {
        'roc': avg_roc,
        'enrichments': avg_enrichments
    }

for method, res in results.items():
    print(f"\nResults for {method}:")
    print(f"Average ROC: {res['roc']}")
    for percentage, ef in res['enrichments'].items():
        print(f"Enrichment Factor at {percentage*100}%: {ef}")
