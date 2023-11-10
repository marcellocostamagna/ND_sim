import os
import numpy as np
import time
from rdkit import Chem
from similarity.validation.USR_CSR import *
from multiprocessing import Pool

MAX_CORES = 4


def read_molecules_from_sdf(sdf_file):
    # supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    # supplier from mol2 file
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

def process_target(args):
    target_name, ligands_dir, decoys_dir, enrichment_factors = args
    print(f"\nProcessing target: {target_name}")
    
    results = {}
    
    # Define paths to active and decoy files based on the target name
    actives_file = os.path.join(ligands_dir, f"{target_name}_ligands.sdf")
    decoys_file = os.path.join(decoys_dir, f"{target_name}_decoys.sdf")
    
    if os.path.isfile(actives_file) and os.path.isfile(decoys_file):
        actives = read_molecules_from_sdf(actives_file)
        decoys = read_molecules_from_sdf(decoys_file)
        all_mols = actives + decoys
        
        # Pre-compute fingerprints
        fingerprints = {mol: usr(mol) for mol in all_mols}
        # fingerprints = {mol: csr(mol) for mol in all_mols}

        target_enrichments = {k: [] for k in enrichment_factors.keys()}
        for query_mol in actives:
            other_mols = list(all_mols)  # create a new list
            other_mols.remove(query_mol)  # remove the query molecule from the list

            query_fp = fingerprints[query_mol]
            y_scores = [similarity(query_fp, fingerprints[mol]) for mol in other_mols]

            y_true = [1 if mol in actives else 0 for mol in other_mols]
            
            for percentage in enrichment_factors.keys():
                ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                target_enrichments[percentage].append(ef)

        # Return the enrichment factors for this target
        avg_target_enrichments = {percentage: np.mean(values) for percentage, values in target_enrichments.items()}
        results[target_name] = avg_target_enrichments
        
        # Print the average enrichment factors for this target
        print(f"\nAverage Enrichment Factors for target {target_name}:")
        for percentage, ef in avg_target_enrichments.items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")

    return results

if __name__ == "__main__":
    print(f'CWD: {os.getcwd()}')
    ligands_dir = f"{os.getcwd()}/similarity/validation/DUD_mol/dud_ligands2006"
    decoys_dir = f"{os.getcwd()}/similarity/validation/DUD_mol/dud_decoys2006"
    enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}
    
    overall_results = {}

    # Generate target names based on ligand files
    target_names = [f.replace('_ligands.mol2', '') for f in sorted(os.listdir(ligands_dir)) if f.endswith('_ligands.mol2')]

    method = 'usr'
    
    print(f"\nRunning benchmark for method: {method}")
    start_time_total = time.time()

    args_list = [(target_name, ligands_dir, decoys_dir, enrichment_factors) for target_name in target_names]
    
    with Pool(processes=MAX_CORES) as pool:
        results_list = pool.map(process_target, args_list)

    # Combine all results
    results = {k: v for res in results_list for k, v in res.items()}
    
    # Calculate average enrichments
    avg_enrichments = {percentage: np.mean([res[percentage] for res in results.values()]) for percentage in enrichment_factors.keys()}
    overall_results[method] = avg_enrichments

    end_time_total = time.time()
    print(f"\nTotal processing time for method {method}: {end_time_total - start_time_total:.2f} seconds")

    # Print the overall enrichment factors for the different methods
    print("\nOverall Enrichment Factors for Different Methods:")
    for method, results in overall_results.items():
        print(f"\nResults for method {method}:")
        for percentage, ef in results.items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")
