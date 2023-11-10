import numpy as np  
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import * 
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from similarity.trials.perturbations import *
import os 
from rdkit import Chem
from copy import deepcopy
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

cwd = os.getcwd()
N = 21  # Change this to the number of files you have

def replace_random_hydrogen_with_deuterium(molecule, num_hydrogens_to_replace=1):
    hydrogens = [atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "H"]
    num_hydrogens_available = len(hydrogens)
    
    if num_hydrogens_to_replace > num_hydrogens_available:
        print(f"Requested {num_hydrogens_to_replace} hydrogens to replace, but only found {num_hydrogens_available}. Replacing all available hydrogens.")
        num_hydrogens_to_replace = num_hydrogens_available
        
    hydrogens_to_replace = np.random.choice(hydrogens, size=num_hydrogens_to_replace, replace=False)
    
    for hydrogen in hydrogens_to_replace:
        hydrogen.SetIsotope(2)  # Set isotope number to 2 for deuterium

    return molecule

similarities = []

for file_num in range(1, N+1):
    file_name = f'{cwd}/similarity/sd_data/size_increasing_molecules/range_{file_num}.sdf'
    original_molecule = load_molecules_from_sdf(file_name, removeHs=False, sanitize=False)

    modified_molecule = deepcopy(original_molecule[0])
    modified_molecule = replace_random_hydrogen_with_deuterium(modified_molecule, 1)    

    molecules_to_compare = [original_molecule[0], modified_molecule]

    ### ROTATE MOLECULES ###
    rotated_molecules = []
    for molecule in molecules_to_compare:
        angle1 = np.random.randint(0, 360)
        angle2 = np.random.randint(0, 360)
        angle3 = np.random.randint(0, 360)
        # angle1 = 0
        # angle2 = 0
        # angle3 = 0
        mol = rotate_molecule(molecule, angle1, angle2, angle3)
        rotated_molecules.append(mol)
        
    fingerprints = [generate_nd_molecule_fingerprint(molecule, DEFAULT_FEATURES, scaling_method='matrix') for molecule in rotated_molecules]

    # COMPARE ALL PAIRS OF MOLECULES
    n_molecules = len(fingerprints)
    trial_similarities = []
    for i in range(n_molecules):
        for j in range(i+1, n_molecules):
            # partial_score = calculate_mean_absolute_difference(fingerprints[i], fingerprints[j])
            # similarity = calculate_similarity_from_difference(partial_score)
            similarity = compute_similarity_score(fingerprints[i], fingerprints[j])
            trial_similarities.append(similarity)
    
    avg_similarity = sum(trial_similarities) / len(trial_similarities)
    similarities.append(avg_similarity)

print('Similarities:')
print('\n'.join(map(str, similarities)))

# Plotting the average similarities
plt.figure(figsize=(10, 5))
plt.plot(range(1, N+1), similarities, marker='o', linestyle='-')
plt.xlabel('File Number')
plt.ylabel('Average Similarity')
plt.title('Similarity between Original and Modified Molecules')
plt.xticks(range(1, N+1))
plt.grid(True)
plt.tight_layout()
plt.show()
