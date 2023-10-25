import os
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetUSRCAT, GetUSRScore, GetUSR
from multiprocessing import Pool

MAX_CORES = 5

def save_to_sd(query_mol, target_mol, folder_name, method, similarity_score, file_index_counter, base_file_name="similar_molecules"):

    # Create the folder if it doesn't exist
    full_folder_path = os.path.join("similar_molecules_rdkit_1", folder_name)
    os.makedirs(full_folder_path, exist_ok=True)
    
    file_path = os.path.join(full_folder_path, f"{base_file_name}{file_index_counter}.sdf")

    w = Chem.SDWriter(file_path)
    
    # Write the query molecule with modified name
    query_mol_name = query_mol.GetProp("_Name") + "_query"
    query_mol.SetProp("_Name", query_mol_name)
    w.write(query_mol)

    # Write the target molecule with modified name
    target_mol_name = target_mol.GetProp("_Name") + "_target"
    target_mol.SetProp("_Name", target_mol_name)
    target_mol.SetProp("Folder", folder_name)
    target_mol.SetProp("Method", method)
    target_mol.SetProp("Score", f"{similarity_score:.4f}")
    w.write(target_mol)

    w.close()
    file_index_counter += 1
    return file_index_counter # Return the incremented counter
    
def read_molecules_from_sdf(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    mols = []
    for mol in supplier:
        if mol:
            mol_name = mol.GetProp("_Name")  # Capture the original name
            mol.SetProp("_Name", mol_name)  # Retain the original name
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

def compute_fingerprints(molecules, method):
    if method == "usr":
        return {mol: GetUSR(mol) for mol in molecules}
    elif method == "usrcat":
        return {mol: GetUSRCAT(mol) for mol in molecules}
    return {}

def process_folder(args):
    folder, root_directory, method, enrichment_factors = args
    print(f"\nProcessing folder: {folder}")
    
    # Ensure the directory for saving similar molecules exists
    if not os.path.exists("similar_molecules_rdkit_1"):
        os.makedirs("similar_molecules_rdkit_1", exist_ok=True)

    folder_path = os.path.join(root_directory, folder)
    folder_enrichments = {k: [] for k in enrichment_factors.keys()}
    results = {}
    saved_pairs = 0  # Counter for saved molecule pairs
    
    
    if os.path.isdir(folder_path):
        actives_file = os.path.join(folder_path, "actives_final.sdf")
        decoys_file = os.path.join(folder_path, "decoys_final.sdf")
        
        actives = read_molecules_from_sdf(actives_file)
        decoys = read_molecules_from_sdf(decoys_file)
        all_mols = actives + decoys
        
        # # Check for duplicates in actives and decoys
        # active_smiles = [Chem.MolToSmiles(mol) for mol in actives]
        # decoy_smiles = [Chem.MolToSmiles(mol) for mol in decoys]
        # duplicates = set(active_smiles).intersection(decoy_smiles)

        # if duplicates:
        #     print(f"Warning in {folder}: Found {len(duplicates)} duplicate molecules in both actives and decoys!")
        
        # Pre-compute fingerprints
        fingerprints = compute_fingerprints(all_mols, method)

        file_index_counter = 0 # Counter for the number of files saved
        processed_pairs = set()
        for query_mol in actives:
            other_mols = list(all_mols)  # create a new list
            other_mols.remove(query_mol)  # remove the query molecule from the list

            query_fp = fingerprints[query_mol]
            y_scores = [GetUSRScore(query_fp, fingerprints[mol]) for mol in other_mols]

            # # Save similar molecules
            # similarity_threshold = 0.9999  # Modify this value as needed
            # for mol, score in zip(other_mols, y_scores):
            #     if score > similarity_threshold:
            #         file_index_counter = save_to_sd(query_mol, mol, folder, method, score, file_index_counter)
            #         saved_pairs += 1
            similarity_threshold = 0.9999         
            for mol, score in zip(other_mols, y_scores):
                if score > similarity_threshold:
                    pair_key = frozenset([query_mol.GetProp("_Name"), mol.GetProp("_Name")])
                    if pair_key not in processed_pairs:
                        file_index_counter = save_to_sd(query_mol, mol, folder, method, score, file_index_counter)
                        saved_pairs += 1
                        processed_pairs.add(pair_key)

            y_true = [1 if mol in actives else 0 for mol in other_mols]
            
            for percentage in enrichment_factors.keys():
                ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                folder_enrichments[percentage].append(ef)

        # Return the enrichment factors for this folder
        avg_folder_enrichments = {percentage: np.mean(values) for percentage, values in folder_enrichments.items()}
        results[folder] = avg_folder_enrichments
        
        # Return results and the folder data string
        folder_data = f"Folder: {folder}\n"
        for percentage, ef in avg_folder_enrichments.items():
            folder_data += f"Enrichment Factor at {percentage*100}%: {ef}\n"
        if saved_pairs == 0:
            folder_data += 'No chirality detected\n'
        else:
            folder_data += f"Chirality detected in {saved_pairs} pairs of molecules\n"
    
    return results, folder_data


if __name__ == "__main__":
    print(f'CWD: {os.getcwd()}')
    root_directory = f"{os.getcwd()}/similarity/validation/all"
    methods = ['usr'] #['usr', 'usrcat']
    enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}
    
    overall_results = {}

    for method in methods:
        print(f"\nRunning benchmark for method: {method}")
        start_time_total = time.time()

        folders = sorted(os.listdir(root_directory))
        args_list = [(folder, root_directory, method, enrichment_factors) for folder in folders]
        
        with Pool(processes=MAX_CORES) as pool:
            results_list = pool.map(process_folder, args_list)

        # Combine all results
        results = {k: v for res, _ in results_list for k, v in res.items()}    
            
        folder_data_strings = [data_string for _, data_string in results_list]
        
        # Write results to file
        with open(f"similar_molecules_rdkit_1/results.txt", 'a') as file:
            for folder_data in folder_data_strings:
                file.write(folder_data)
                file.write("\n")  # separate data of different folders with a newline
        
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

