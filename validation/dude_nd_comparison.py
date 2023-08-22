import os
import numpy as np
import time
from openbabel import pybel
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit.Chem import rdPartialCharges


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

# PSEUDO_USRCAT PARAMETERS & FUNCTIONS
USRCAT_SMARTS = {'hydrophobic' : "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",        
                  'aromatic' : "[a]",                                           
                  'acceotor' :"[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]",   
                  'donor' : "[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]",                             
                }
def get_pseudo_usrcat_fingerprint(mol):
    mol_nd = mol_nd_data(mol, features=None)
    _, mol_nd_pca, _, _ = perform_PCA_and_get_transformed_data_cov(mol_nd)
    pseudo_usrcat_fingerprint = []
    pseudo_usrcat_fingerprint.append(get_fingerprint(mol_nd_pca, scaling_factor=None, scaling_matrix=None)) # Standard USR fingerprint
    for smarts in USRCAT_SMARTS.values():
        # Collect the atoms indexes that match the SMARTS pattern in the query molecule
        query_atoms_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if not query_atoms_matches:
            pseudo_usrcat_fingerprint.append(np.zeros(12))
            continue
        query_atoms = [idx for match in query_atoms_matches for idx in match]
        # Construct a 'sub-molecule_3d' by getting from the query_3d_pca only the rows corresponding to the atoms in query_atoms
        sub_molecule_3d = mol_nd_pca[query_atoms]
        # Compute the fingerprint of the sub-molecule_3d
        pseudo_usrcat_fingerprint.append(get_fingerprint(sub_molecule_3d, scaling_factor=None, scaling_matrix=None))
    return pseudo_usrcat_fingerprint

def get_pseudo_usrcat_similarity(query_usrcat_fingerprint, target_usrcat_fingerprint):
    partial_scores = []
    for i in range(len(query_usrcat_fingerprint)):
        partial_scores.append(calculate_partial_score(query_usrcat_fingerprint[i], target_usrcat_fingerprint[i]))
    final_partial_score = np.sum(partial_scores)
    return get_similarity_measure(final_partial_score)

# PSEUDO_ELECTROSHAPE PARAMETERS & FUNCTIONS
def compute_partial_charges(mol):
    # Compute the partial charges of the molecule
    rdPartialCharges.ComputeGasteigerCharges(mol)
    # Add partial_charges as a property of each atom
    for atom in mol.GetAtoms():
        atom.SetProp('partial_charge', str(atom.GetDoubleProp('_GasteigerCharge')))
    return mol

def get_partial_charges(atom):
    partial_charge = float(atom.GetProp('partial_charge'))
    # Handle the case where the partial charge is NaN or Inf with np.nan_to_num
    if np.isnan(partial_charge) or np.isinf(partial_charge):
        partial_charge = np.nan_to_num(partial_charge)
    return partial_charge

def scaling_fn(value):
    result = value * 25
    # Handle possible overflows of maximum value allowed for float
    if result > np.finfo(np.float32).max:
        result = np.finfo(np.float32).max
    return result

PSEUDO_ELECTROSHAPE_FEATURES = { 'partial_charge' : [ get_partial_charges, scaling_fn] }

# START
print(f'CWD: {os.getcwd()}')
root_directory = f"{os.getcwd()}/similarity/validation/all"
methods =  ['pseudo_usr', 'pseudo_usr_cat', 'pseudo_electroshape']

enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}

results = {}

for method in methods:
    print(f"Running benchmark for method: {method}")
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
            
            if method == 'pseudo_usr':
                query = get_nd_fingerprint(query_mol, features=None, scaling_method=None)
                y_scores = [get_similarity_score(query, get_nd_fingerprint(mol, features=None, scaling_method=None)) for mol in actives + decoys]
            elif method == 'pseudo_usr_cat':
                query = get_pseudo_usrcat_fingerprint(query_mol)
                y_scores = [get_pseudo_usrcat_similarity(query, get_pseudo_usrcat_fingerprint(mol)) for mol in actives + decoys]
            elif method == 'pseudo_electroshape':
                actives = [compute_partial_charges(mol) for mol in actives]
                decoys = [compute_partial_charges(mol) for mol in decoys]
                query_mol = compute_partial_charges(query_mol)
                query = get_nd_fingerprint(query_mol, features=PSEUDO_ELECTROSHAPE_FEATURES, scaling_method=None)
                y_scores = [get_similarity_score(query, get_nd_fingerprint(mol, features=PSEUDO_ELECTROSHAPE_FEATURES, scaling_method=None)) for mol in actives + decoys]
            
            y_true = [1]*len(actives) + [0]*len(decoys)
            
            for percentage in enrichment_factors.keys():
                ef = calculate_enrichment_factor(y_true, y_scores, percentage)
                enrichments[percentage].append(ef)

        folder_end_time = time.time()
        print(f"Finished processing folder {folder} in {folder_end_time - folder_start_time:.2f} seconds")

        avg_enrichments = {percentage: np.mean(values) for percentage, values in enrichments.items()}

        results[method] = {
        'enrichments': avg_enrichments
        }

        # Print results for this method immediately after its processing
        print(f"\nResults for {method}:")
        for percentage, ef in avg_enrichments.items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")
        print("\n" + "-"*50 + "\n")  # Separator line for clarity
    
    end_time_total = time.time()
    print(f"\nTotal processing time for {method}: {end_time_total - start_time_total:.2f} seconds")

    avg_enrichments = {percentage: np.mean(values) for percentage, values in enrichments.items()}

    results[method] = {
        'enrichments': avg_enrichments
    }

for method, res in results.items():
    print(f"\nResults for {method}:")
    for percentage, ef in res['enrichments'].items():
        print(f"Enrichment Factor at {percentage*100}%: {ef}")