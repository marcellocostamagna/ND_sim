import os
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetUSRCAT, GetUSRScore, GetUSR

def read_molecules_from_sdf(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
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

print(f'CWD: {os.getcwd()}')
root_directory = f"{os.getcwd()}/similarity/validation/all"
methods = ['usr', 'usrcat']

enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}
overall_enrichments = {k: [] for k in enrichment_factors.keys()}  

for method in methods:
    print(f"Running benchmark for method: {method}")
    start_time_total = time.time()

    for folder in sorted(os.listdir(root_directory)):
        folder_start_time = time.time()
        
        print(f"\nProcessing folder: {folder}")
        folder_path = os.path.join(root_directory, folder)
        
        if os.path.isdir(folder_path):
            actives_file = os.path.join(folder_path, "actives_final.sdf")
            decoys_file = os.path.join(folder_path, "decoys_final.sdf")
            
            actives = read_molecules_from_sdf(actives_file)
            decoys = read_molecules_from_sdf(decoys_file)
            
            if not actives:
                continue
            
            enrichments_per_active = {k: [] for k in enrichment_factors.keys()}

            for query_mol in actives:
                all_mols = actives + decoys
                all_mols.remove(query_mol)  # remove the query molecule from the list
                
                if method == 'usr':
                    query_fingerprint = GetUSR(query_mol)
                    y_scores = [GetUSRScore(query_fingerprint, GetUSR(mol)) for mol in all_mols]
                elif method == 'usrcat':
                    query_fingerprint = GetUSRCAT(query_mol)
                    y_scores = [GetUSRScore(query_fingerprint, GetUSRCAT(mol)) for mol in all_mols]
                
                y_true = [1 if mol in actives else 0 for mol in all_mols]
                
                for percentage in enrichment_factors.keys():
                    ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                    enrichments_per_active[percentage].append(ef)

            avg_enrichments = {percentage: np.mean(values) for percentage, values in enrichments_per_active.items()}

            for percentage, ef in avg_enrichments.items():
                overall_enrichments[percentage].append(ef)  

            print(f"\nAverage Enrichment Factors for folder {folder}:")
            for percentage, ef in avg_enrichments.items():
                print(f"Enrichment Factor at {percentage*100}%: {ef}")

        folder_end_time = time.time()
        print(f"\nFinished processing folder {folder} in {folder_end_time - folder_start_time:.2f} seconds")

    end_time_total = time.time()

    # Calculate and print overall average enrichments
    print(f"\nOverall Average Enrichment Factors for {method}:")
    for percentage, values in overall_enrichments.items():
        overall_avg = np.mean(values)
        print(f"Enrichment Factor at {percentage*100}%: {overall_avg}")
        
    print(f"\nTotal processing time for {method}: {end_time_total - start_time_total:.2f} seconds")
