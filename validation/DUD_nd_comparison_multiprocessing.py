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

MAX_CORES = 4

def read_molecules_from_file(file_path):
    mols = []
    for mol in toolkit.readfile(os.path.splitext(file_path)[1][1:], file_path):
        mols.append(mol)
    return mols

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

def compute_fingerprints(molecules, method, actives, decoys):
    if method == "pseudo_usr":
        # return {mol: get_nd_fingerprint(mol, features=None, scaling_method='matrix') for mol in molecules}
        # return {mol: generate_nd_molecule_fingerprint_no_chirality(mol, features=None, scaling_method='matrix') for mol in molecules}
        return {mol: generate_nd_molecule_fingerprint(mol, features=None, scaling_method='matrix') for mol in molecules}
    elif method == "pseudo_usr_cat":
        # return {mol: get_pseudo_usrcat_fingerprint(mol) for mol in molecules}
        return {mol: get_pseudo_usrcat_fingerprint(mol) for mol in molecules}
    elif method == "pseudo_electroshape":
        # return {mol: get_nd_fingerprint(mol, features=PSEUDO_ELECTROSHAPE_FEATURES, scaling_method='matrix') for mol in molecules}
        # return {mol: generate_nd_molecule_fingerprint_no_chirality(mol, features=PSEUDO_ELECTROSHAPE_FEATURES, scaling_method='matrix') for mol in molecules}
        return {mol: generate_nd_molecule_fingerprint(mol, features=PSEUDO_ELECTROSHAPE_FEATURES, scaling_method='matrix') for mol in molecules}

# PSEUDO_USRCAT PARAMETERS & FUNCTIONS
USRCAT_SMARTS = {'hydrophobic' : "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",        
                  'aromatic' : "[a]",                                           
                  'acceotor' :"[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]",   
                  'donor' : "[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]",                             
                }

def get_pseudo_usrcat_fingerprint(mol):
    mol_3d = molecule_to_ndarray(mol, features=None)
    mol_3d_pca, _ = compute_pca_using_covariance(mol_3d)
    pseudo_usrcat_fingerprint = []
    # scaling = compute_scaling_factor(mol_3d_pca)
    scaling_matrix = compute_scaling_matrix(mol_3d_pca)
    pseudo_usrcat_fingerprint.append(generate_molecule_fingerprint(mol_3d_pca, scaling_factor=None, scaling_matrix=scaling_matrix)) # Standard USR fingerprint
    for smarts in USRCAT_SMARTS.values():
        # Collect the atoms indexes that match the SMARTS pattern in the query molecule
        query_atoms_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if not query_atoms_matches:
            pseudo_usrcat_fingerprint.append(np.zeros(12))
            continue
        query_atoms = [idx for match in query_atoms_matches for idx in match]
        # Construct a 'sub-molecule_3d' by getting from the query_3d_pca only the rows corresponding to the atoms in query_atoms
        sub_molecule_3d = mol_3d_pca[query_atoms]
        # Compute the fingerprint of the sub-molecule_3d
        # scaling = compute_scaling_factor(sub_molecule_3d)
        scaling_matrix = compute_scaling_matrix(sub_molecule_3d)
        pseudo_usrcat_fingerprint.append(generate_molecule_fingerprint(sub_molecule_3d, scaling_factor=None, scaling_matrix=scaling_matrix))
    return pseudo_usrcat_fingerprint

def get_pseudo_usrcat_similarity(query_usrcat_fingerprint, target_usrcat_fingerprint):
    partial_scores = []
    for i in range(len(query_usrcat_fingerprint)):
        partial_scores.append(calculate_mean_absolute_difference(query_usrcat_fingerprint[i], target_usrcat_fingerprint[i]))
    final_partial_score = np.sum(partial_scores)
    return calculate_similarity_from_difference(final_partial_score)

# PSEUDO_ELECTROSHAPE PARAMETERS & FUNCTIONS
def partial_charges(mol, mol_charges):
    # Set atomic property partial charge from pybel molecule
    charges = mol_charges.atom_dict['charge']
    for atom in mol.GetAtoms():
        # Set partial charge = 0 if hydrogen
        if atom.GetAtomicNum() == 1:
            atom.SetProp('partial_charge', str(0.0))
        else:
            atom.SetProp('partial_charge', str(charges[atom.GetIdx()]))   
    return mol

def get_partial_charges(atom):
    try:
        partial_charge = float(atom.GetProp('partial_charge'))
    except KeyError:
        print('property not found')
        return 0.0
    # Handle the case where the partial charge is NaN or Inf with np.nan_to_num
    partial_charge = np.nan_to_num(partial_charge)
    return partial_charge

def scaling_fn(value):
    result = value * 25
    # Handle possible overflows of maximum value allowed for float
    if result > np.finfo(np.float32).max:
        result = np.finfo(np.float32).max
    return result

PSEUDO_ELECTROSHAPE_FEATURES = { 'partial_charge' : [ get_partial_charges, scaling_fn] }


def process_target(args):
    target_name, ligands_dir, decoys_dir, method, enrichment_factors = args
    print(f"\nProcessing target: {target_name}")
    
    actives_file = os.path.join(ligands_dir, f"{target_name}_ligands.sdf")
    decoys_file = os.path.join(decoys_dir, f"{target_name}_decoys.sdf")
    
    actives = read_molecules_from_sdf(actives_file)
    decoys = read_molecules_from_sdf(decoys_file)
    # Set partial charges as in the oddt implementation (Pybel molecules)
    if method == 'pseudo_electroshape':
        actives_charges = read_molecules_from_file(actives_file)
        decoys_charges = read_molecules_from_file(decoys_file)
        actives = [partial_charges(mol, mol_charge) for mol, mol_charge in zip(actives, actives_charges)]
        decoys = [partial_charges(mol, mol_charge) for mol, mol_charge in zip(decoys, decoys_charges)]

    all_mols = actives + decoys
    fingerprints = compute_fingerprints(all_mols, method, actives, decoys)
    
    target_enrichments = {k: [] for k in enrichment_factors.keys()}
    
    for query_mol in actives:
        other_mols = list(all_mols)
        other_mols.remove(query_mol)
        
        query_fp = fingerprints[query_mol]
        # y_scores = [compute_similarity_score_no_chirality(query_fp, fingerprints[mol]) for mol in other_mols]
        y_scores = [compute_similarity_score(query_fp, fingerprints[mol]) for mol in other_mols]
        y_true = [1 if mol in actives else 0 for mol in other_mols]
        
        for percentage in enrichment_factors.keys():
            ef = calculate_enrichment_factor(y_true, y_scores, percentage)
            target_enrichments[percentage].append(ef)
    
    avg_target_enrichments = {percentage: np.mean(values) for percentage, values in target_enrichments.items()}
    results = {target_name: avg_target_enrichments}
    
    print(f"\nAverage Enrichment Factors for target {target_name}:")
    for percentage, ef in avg_target_enrichments.items():
        print(f"Enrichment Factor at {percentage*100}%: {ef}")
    
    return results

if __name__ == "__main__":
    print(f'CWD: {os.getcwd()}')
    ligands_dir = f"{os.getcwd()}/similarity/validation/DUD/dud_ligands2006_sdf"
    decoys_dir = f"{os.getcwd()}/similarity/validation/DUD/dud_decoys2006_sdf"
    methods =  ['pseudo_electroshape'] #['pseudo_usr', 'pseudo_usr_cat', 'pseudo_electroshape'] 
    enrichment_factors = {0.0025: [], 0.005: [], 0.01: [], 0.02: [], 0.03: [], 0.05: []}
    
    overall_results = {}
    
    target_names = [f.replace('_ligands.sdf', '') for f in sorted(os.listdir(ligands_dir)) if f.endswith('_ligands.sdf')]

    for method in methods:
        print(f"\nRunning benchmark for method: {method}")
        start_time_total = time.time()

        args_list = [(target_name, ligands_dir, decoys_dir, method, enrichment_factors) for target_name in target_names]
        
        with multiprocessing.Pool(processes=MAX_CORES) as pool:
            results_list = pool.map(process_target, args_list)

        results = {k: v for res in results_list for k, v in res.items()}
        
        avg_enrichments = {percentage: np.mean([res[percentage] for res in results.values()]) for percentage in enrichment_factors.keys()}
        overall_results[method] = avg_enrichments
        
        end_time_total = time.time()
        print(f"\nTotal processing time for method {method}: {end_time_total - start_time_total:.2f} seconds")

    print("\nOverall Enrichment Factors for Different Methods:")
    for method, results in overall_results.items():
        print(f"\nResults for method {method}:")
        for percentage, ef in results.items():
            print(f"Enrichment Factor at {percentage*100}%: {ef}")
