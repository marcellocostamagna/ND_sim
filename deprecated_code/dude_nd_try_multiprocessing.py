import os
import time
import numpy as np
import multiprocessing
from rdkit import Chem
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from oddt import toolkit

MAX_CORES = 3

def read_molecules_from_sdf(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    mols = [mol for mol in supplier if mol]
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

def process_folder(args):
    folder, root_directory, method, enrichment_factors = args
    print(f"\nProcessing folder: {folder}")
    
    folder_path = os.path.join(root_directory, folder)
    folder_enrichments = {k: [] for k in enrichment_factors.keys()}
    results = {}
    
    if os.path.isdir(folder_path):
        actives_file = os.path.join(folder_path, "actives_final.sdf")
        decoys_file = os.path.join(folder_path, "decoys_final.sdf")
        
        actives = read_molecules_from_sdf(actives_file)
        decoys = read_molecules_from_sdf(decoys_file)
        
        all_mols = actives + decoys

        # Pre-compute fingerprints for all molecules
        fingerprints = {mol: get_nd_fingerprint(mol, features=DEFAULT_FEATURES, scaling_method='factor') for mol in all_mols}

        for query_mol in actives:
            other_mols = list(all_mols)  # create a new list
            other_mols.remove(query_mol)  # remove the query molecule from the list

            query_fp = fingerprints[query_mol]
        
            y_scores = [get_similarity_score(query_fp, fingerprints[mol]) for mol in other_mols]

            y_true = [1 if mol in actives else 0 for mol in other_mols]
            
            for percentage in enrichment_factors.keys():
                ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                folder_enrichments[percentage].append(ef)

        # Return the enrichment factors for this folder
        avg_folder_enrichments = {percentage: np.mean(values) for percentage, values in folder_enrichments.items()}
        results[folder] = avg_folder_enrichments
        
        # Print the average enrichment factors for this folder
        print(f"\nAverage Enrichment Factors for folder {folder}:")
        for percentage, ef in avg_folder_enrichments.items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")

    return results

if __name__ == "__main__":
    print(f'CWD: {os.getcwd()}')
    root_directory = f"{os.getcwd()}/similarity/validation/all"
    method = 'standard method'
    enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}
    
    overall_results = {}

    print(f"\nRunning benchmark for method: {method}")
    start_time_total = time.time()

    folders = sorted(os.listdir(root_directory))
    args_list = [(folder, root_directory, method, enrichment_factors) for folder in folders]
    
    with multiprocessing.Pool(processes=MAX_CORES) as pool:
        results_list = pool.map(process_folder, args_list)

    # Combine all results
    results = {k: v for res in results_list for k, v in res.items()}
    
    # Calculate average enrichments
    avg_enrichments = {percentage: np.mean([res[percentage] for res in results.values()]) for percentage in enrichment_factors.keys()}
    overall_results[method] = avg_enrichments

    end_time_total = time.time()
    print(f"\nTotal processing time for {method}: {end_time_total - start_time_total:.2f} seconds")

    # Print the overall enrichment factors for the different methods
    print("\nOverall Enrichment Factors for Different Methods:")
    for method, results in overall_results.items():
        print(f"\nResults for {method}:")
        for percentage, ef in results.items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")

