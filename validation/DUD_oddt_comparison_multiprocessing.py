import os
import numpy as np
import time
from oddt import toolkit
from oddt import shape
from multiprocessing import Pool

MAX_CORES = 4

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

def process_target(args):
    target_name, ligands_dir, decoys_dir, method, enrichment_factors = args
    print(f"\nProcessing target: {target_name}")
    
    results = {}
    
    actives_file = os.path.join(ligands_dir, f"{target_name}_ligands.sdf")
    decoys_file = os.path.join(decoys_dir, f"{target_name}_decoys.sdf")
    
    if os.path.isfile(actives_file) and os.path.isfile(decoys_file):
        actives = read_molecules_from_file(actives_file)
        decoys = read_molecules_from_file(decoys_file)

        # Precompute fingerprints
        actives_fps = [get_fingerprint(mol, method) for mol in actives]
        decoys_fps = [get_fingerprint(mol, method) for mol in decoys]
        # Initialize the results dictionary
        results = {percentage: [] for percentage in enrichment_factors.keys()}
        for i, query_mol in enumerate(actives):
            query_fp = actives_fps[i]
            other_mols = list(actives) + list(decoys)
            other_fps = actives_fps + decoys_fps
            other_mols.remove(query_mol)
            other_fps.pop(i)
            
            y_scores = [shape.usr_similarity(query_fp, mol_fp) for mol_fp in other_fps]
            y_true = [1 if mol in actives else 0 for mol in other_mols]
            
            for percentage in enrichment_factors.keys():
                ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                results[percentage].append(ef)

    avg_enrichments = {percentage: np.mean(values) for percentage, values in results.items()}

    # Print the average enrichment factors for this target
    print(f"\nAverage Enrichment Factors for target {target_name}:")
    for percentage, ef in avg_enrichments.items():
        print(f"Enrichment Factor at {percentage*100}%: {ef}")
        
    return avg_enrichments

if __name__ == "__main__":
    print(f'CWD: {os.getcwd()}')
    ligands_dir = f"{os.getcwd()}/similarity/validation/DUD/dud_ligands2006_sdf"
    decoys_dir = f"{os.getcwd()}/similarity/validation/DUD/dud_decoys2006_sdf"
    methods = ['electroshape']#['usr', 'usr_cat', 'electroshape']
    enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}
    
    overall_results = {}

    # Generate target names based on ligand files
    target_names = [f.replace('_ligands.sdf', '') for f in sorted(os.listdir(ligands_dir)) if f.endswith('_ligands.sdf')]

    for method in methods:
        print(f"Running benchmark for method: {method}")
        start_time_total = time.time()
        
        args_list = [(target_name, ligands_dir, decoys_dir, method, enrichment_factors) for target_name in target_names]
        
        with Pool(processes=MAX_CORES) as pool:
            results_list = pool.map(process_target, args_list)
        
        # Combine all results
        avg_enrichments = {percentage: np.mean([res[percentage] for res in results_list]) for percentage in enrichment_factors.keys()}
        
        overall_results[method] = avg_enrichments
        
        end_time_total = time.time()
        print(f"\nTotal processing time for method {method}: {end_time_total - start_time_total:.2f} seconds")

    # Print the overall enrichment factors for the different methods
    for method, avg_enrichments in overall_results.items():
        print(f"\nResults for method {method}:")
        for percentage, ef in avg_enrichments.items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")
