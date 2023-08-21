import os
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetUSRCAT, GetUSRScore, GetUSR
from sklearn.metrics import roc_auc_score

def read_molecules_from_sdf(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
    mols = [mol for mol in supplier if mol]
    return mols

def read_molecule_from_mol2(mol2_file):
    return Chem.MolFromMol2File(mol2_file, removeHs=False)

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

root_directory = "all"
methods = ['usr', 'usrcat']

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
            
            actives = read_molecules_from_sdf(actives_file)
            decoys = read_molecules_from_sdf(decoys_file)
            query_mol = read_molecule_from_mol2(query_file)
            
            if not query_mol:
                continue

            if method == 'usr':
                query_fingerprint = GetUSR(query_mol)  # Modify this to match the rdkit function to compute USR
                y_scores = [GetUSRScore(query_fingerprint, GetUSR(mol)) for mol in actives + decoys]
            elif method == 'usrcat':
                query_fingerprint = GetUSRCAT(query_mol)
                y_scores = [GetUSRScore(query_fingerprint, GetUSRCAT(mol)) for mol in actives + decoys]
            
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