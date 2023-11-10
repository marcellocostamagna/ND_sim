import os
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetUSRCAT, GetUSRScore, GetUSR
from multiprocessing import Pool

MAX_CORES = 4

# def save_to_sd(query_mol, target_mol, target_name, method, base_file_name="similar_molecules"):
#     # Determine the next available filename
#     index = 1
#     while os.path.exists(f"similar_molecules_rdkit/{base_file_name}_{target_name}_{index}.sdf"):
#         index += 1
#     file_path = f"similar_molecules_rdkit/{base_file_name}_{target_name}_{index}.sdf"

#     w = Chem.SDWriter(file_path)
#     for mol in [query_mol, target_mol]:
#         mol.SetProp("Target", target_name)
#         mol.SetProp("Method", method)
#         mol.SetProp("Score", "1.0")  # Assuming a default score of 1.0; adjust as needed
#         w.write(mol)
#     w.close()

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

def compute_fingerprints(molecules, method):
    if method == "usr":
        return {mol: GetUSR(mol) for mol in molecules}
    elif method == "usrcat":
        return {mol: GetUSRCAT(mol) for mol in molecules}
    return {}

def process_target(args):
    target_name, ligands_dir, decoys_dir, method, enrichment_factors = args
    print(f"\nProcessing target: {target_name}")
    
    # # Ensure the directory for saving similar molecules exists
    # if not os.path.exists("similar_molecules_rdkit"):
    #     os.makedirs("similar_molecules_rdkit")

    results = {}
    
    # Define paths to active and decoy files based on the target name
    actives_file = os.path.join(ligands_dir, f"{target_name}_ligands.sdf")
    decoys_file = os.path.join(decoys_dir, f"{target_name}_decoys.sdf")
    
    if os.path.isfile(actives_file) and os.path.isfile(decoys_file):
        actives = read_molecules_from_sdf(actives_file)
        decoys = read_molecules_from_sdf(decoys_file)
        all_mols = actives + decoys
        
        # Pre-compute fingerprints
        fingerprints = compute_fingerprints(all_mols, method)

        target_enrichments = {k: [] for k in enrichment_factors.keys()}
        for query_mol in actives:
            other_mols = list(all_mols)  # create a new list
            other_mols.remove(query_mol)  # remove the query molecule from the list

            query_fp = fingerprints[query_mol]
            y_scores = [GetUSRScore(query_fp, fingerprints[mol]) for mol in other_mols]

            # # Save similar molecules
            # similarity_threshold = 0.9999  # Modify this value as needed
            # for mol, score in zip(other_mols, y_scores):
            #     if score > similarity_threshold:
            #         save_to_sd(query_mol, mol, target_name, method)

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
    ligands_dir = f"{os.getcwd()}/similarity/validation/DUD/dud_ligands2006_sdf"
    decoys_dir = f"{os.getcwd()}/similarity/validation/DUD/dud_decoys2006_sdf"
    methods = ['usr']  # Add 'usrcat' if needed
    enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}
    
    overall_results = {}

    # Generate target names based on ligand files
    target_names = [f.replace('_ligands.sdf', '') for f in sorted(os.listdir(ligands_dir)) if f.endswith('_ligands.sdf')]

    for method in methods:
        print(f"\nRunning benchmark for method: {method}")
        start_time_total = time.time()

        args_list = [(target_name, ligands_dir, decoys_dir, method, enrichment_factors) for target_name in target_names]
        
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
