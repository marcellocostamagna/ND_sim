import os
import numpy as np
import time
from openbabel import openbabel, pybel
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from sklearn.metrics import roc_auc_score


def collect_molecules_from_mol2(path, removeHs=False):
    """
    Collects molecules from a MOL2 file and returns a list of RDKit molecules.

    Parameters:
        path (str): Path to the MOL2 file.
        removeHs (bool, optional): Whether to remove hydrogens. Defaults to False.
        
    Returns:
        list: A list of RDKit molecule objects.
    """
    
    # Read the .mol2 file with pybel
    mol2_molecules = [mol for mol in pybel.readfile('mol2', path)]
    
    # Convert each molecule to .sdf format and then to RDKit molecule
    rdkit_molecules = []
    for mol in mol2_molecules:
        sdf_data = mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_data, removeHs=removeHs, sanitize=False)
        if rdkit_mol is not None:
            rdkit_molecules.append(rdkit_mol)

    return rdkit_molecules

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
roc_scores = []
enrichments = {k: [] for k in enrichment_factors.keys()}


start_time_total = time.time()

for folder in sorted(os.listdir(root_directory)):
    folder_start_time = time.time()
    
    print(f"Processing folder: {folder}")
    folder_path = os.path.join(root_directory, folder)
    
    if os.path.isdir(folder_path):
        actives_file = os.path.join(folder_path, "actives_final.sdf")
        decoys_file = os.path.join(folder_path, "decoys_final.sdf")
        query_file = os.path.join(folder_path, "crystal_ligand.mol2")
        
        actives = collect_molecules_from_sdf(actives_file)
        decoys = collect_molecules_from_sdf(decoys_file)
        query_mol = collect_molecules_from_mol2(query_file)[0]
        
        query_fp = get_nd_fingerprint(query_mol, scaling_method='factor')
        y_scores = []

        for mol in actives + decoys:
            mol_fp = get_nd_fingerprint(mol, scaling_method='factor')
            partial_score = calculate_partial_score(query_fp, mol_fp)
            similarity = get_similarity_measure(partial_score)
            y_scores.append(similarity)
        
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

print(f"\nResults for {method}:")
print(f"Average ROC: {results[method]['roc']}")
for percentage, ef in results[method]['enrichments'].items():
    print(f"Enrichment Factor at {percentage*100}%: {ef}")
