import os
import numpy as np
import time
from oddt import toolkit
from oddt import shape
from multiprocessing import Pool
from rdkit import Chem
import rdkit

MAX_CORES = 4


# def save_to_sd(query_mol, target_mol, folder_name, method, base_file_name="similar_molecules"):
#     # Convert ODDT molecules to RDKit molecules
#     query_mol_rdkit = oddt_to_rdkit(query_mol)
#     target_mol_rdkit = oddt_to_rdkit(target_mol)
    
#     # Determine the next available filename
#     index = 1
#     while os.path.exists(f"similar_molecules_oddt/{base_file_name}{index}.sdf"):
#         index += 1
#     file_path = f"similar_molecules_oddt/{base_file_name}{index}.sdf"

#     w = Chem.SDWriter(file_path)
#     for mol in [query_mol_rdkit, target_mol_rdkit]:
#         mol.SetProp("Folder", folder_name)
#         mol.SetProp("Method", method)
#         mol.SetProp("Score", "1.0")  # Assuming a default score of 1.0; adjust as needed
#         w.write(mol)
#     w.close()

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

def get_fingerprint(mol, method):
    if method == 'usr':
        return shape.usr(mol)
    elif method == 'usr_cat':
        return shape.usr_cat(mol)
    elif method == 'electroshape':
        return shape.electroshape(mol)

def process_folder(args):
    folder, root_directory, method, enrichment_factors = args
    print(f"\nProcessing folder: {folder}")
    folder_path = os.path.join(root_directory, folder)
    enrichments = {k: [] for k in enrichment_factors.keys()}
    results = {}
    
    if os.path.isdir(folder_path):
        actives_file = os.path.join(folder_path, "actives_final.sdf")
        decoys_file = os.path.join(folder_path, "decoys_final.sdf")
        
        actives = read_molecules_from_file(actives_file)
        decoys = read_molecules_from_file(decoys_file)

        # Precompute fingerprints
        actives_fps = [get_fingerprint(mol, method) for mol in actives]
        decoys_fps = [get_fingerprint(mol, method) for mol in decoys]
        similarity_counter = 0
        for i, query_mol in enumerate(actives):
            query_fp = actives_fps[i]
            other_mols = list(actives) + list(decoys)
            other_fps = actives_fps + decoys_fps
            other_mols.remove(query_mol)
            other_fps.pop(i)
            
            y_scores = [shape.usr_similarity(query_fp, mol_fp) for mol_fp in other_fps]
            y_true = [1 if mol in actives else 0 for mol in other_mols]
            
            #         # Save similar molecules
            # similarity_threshold = 0.9999  # Modify this value as needed
            # for mol, score in zip(other_mols, y_scores):
            #     if score > similarity_threshold:
            #         similarity_counter += 1
                    # save_to_sd(query_mol, mol, folder, method)
            
            for percentage in enrichment_factors.keys():
                ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                enrichments[percentage].append(ef)

    avg_enrichments = {percentage: np.mean(values) for percentage, values in enrichments.items()}

    results[folder] = {
        'enrichments': avg_enrichments,
        'counter': similarity_counter
    }

    # Print the average enrichment factors for this folder
    print(f"\nAverage Enrichment Factors for folder {folder}:")
    for percentage, ef in avg_enrichments.items():
        print(f"Enrichment Factor at {percentage*100}%: {ef}")
        
    return results

if __name__ == "__main__":
    print(f'CWD: {os.getcwd()}')
    root_directory = f"{os.getcwd()}/similarity/validation/all"
    methods = ['usr']#['usr', 'usr_cat', 'electroshape']
    enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}
    overall_results = {}

    for method in methods:
        print(f"Running benchmark for method: {method}")
        start_time_total = time.time()
        
        folders = sorted(os.listdir(root_directory))
        args_list = [(folder, root_directory, method, enrichment_factors) for folder in folders]
        total_counter = 0
        with Pool(processes=MAX_CORES) as pool:
            results_list = pool.map(process_folder, args_list)
        
        # Combine all results
        results = {k: v for res in results_list for k, v in res.items()}
        total_counter = sum(res['counter'] for res in results.values())
        
        # Calculate average enrichments
        avg_enrichments = {percentage: np.mean([res['enrichments'][percentage] for res in results.values()]) for percentage in enrichment_factors.keys()}
        
        overall_results[method] = {
            'enrichments': avg_enrichments
        }
        print(f"Total molecules with score > similarity_threshold: {total_counter}")
        end_time_total = time.time()
        print(f"\nTotal processing time for {method}: {end_time_total - start_time_total:.2f} seconds")

    for method, res in overall_results.items():
        print(f"\nResults for {method}:")
        for percentage, ef in res['enrichments'].items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")
