import numpy as np
from itertools import combinations
import random
from similarity.source.pre_processing import *
from similarity.source.pca_tranform import *
from similarity.source.fingerprint import *
from similarity.source.similarity import *
from similarity.source.utils import *
from similarity.trials.perturbations import *
import os
from copy import deepcopy
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

cwd = os.getcwd()
N = 21  # Change this to the number of files you have
MAX_COMBINATIONS = 70

def generate_deuterium_variants(molecule, k=1):
    hydrogens = [atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "H"]
    modified_molecules = []

    all_combinations = list(combinations(hydrogens, k))
    selected_combinations = random.sample(all_combinations, min(MAX_COMBINATIONS, len(all_combinations)))

    # For selected combinations of k hydrogens
    for hydrogen_combination in selected_combinations:
        modified_molecule = deepcopy(molecule)
        for hydrogen in hydrogen_combination:
            hydrogen_atom = modified_molecule.GetAtomWithIdx(hydrogen.GetIdx())
            hydrogen_atom.SetIsotope(2)  # Set isotope number to 2 for deuterium
        modified_molecules.append(modified_molecule)

    return modified_molecules

# Plotting section
plt.figure(figsize=(12, 7))

# Looping through each k value
for k in range(1, 5):
    # Store the number of comparisons
    num_comparisons_list = []
    similarities = []

    for file_num in range(1, N+1):
        file_name = f'{cwd}/similarity/sd_data/size_increasing_molecules/range_{file_num}.sdf'
        original_molecule = load_molecules_from_sdf(file_name, removeHs=False, sanitize=False)

        modified_molecules = generate_deuterium_variants(original_molecule[0], k)

        trial_similarities = []

        # Compare the original molecule with each modified molecule
        original_fingerprint = generate_nd_molecule_fingerprint(original_molecule[0], DEFAULT_FEATURES, scaling_method='matrix')

        for modified_molecule in modified_molecules:
            angle1, angle2, angle3 = np.random.randint(0, 360, 3)
            rotated_modified_molecule = rotate_molecule(modified_molecule, angle1, angle2, angle3)
            modified_fingerprint = generate_nd_molecule_fingerprint(rotated_modified_molecule, DEFAULT_FEATURES, scaling_method='matrix')

            similarity = compute_similarity_score(original_fingerprint, modified_fingerprint)
            trial_similarities.append(similarity)

        avg_similarity = sum(trial_similarities) / len(trial_similarities)
        similarities.append(avg_similarity)

        # Store the number of comparisons
        num_comparisons_list.append(len(modified_molecules))

    # Plotting the data for current k
    plt.plot(range(1, N+1), similarities, marker='o', linestyle='-', label=f"{k} Hydrogen(s) Changed")

# Finalizing the plot
plt.xlabel('File Number')
plt.ylabel('Average Similarity')
plt.title('Similarity between Original and Molecules with Modified Deuteriums')
plt.xticks(range(1, N+1))
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig(f'{cwd}/example_sensitivity_neutrons.svg', format='svg')
plt.show()
