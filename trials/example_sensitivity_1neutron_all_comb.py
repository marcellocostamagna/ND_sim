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


def generate_all_single_deuterium_variants(molecule):
    hydrogens = [atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "H"]
    modified_molecules = []

    for hydrogen in hydrogens:
        modified_molecule = deepcopy(molecule)
        hydrogen_atom = modified_molecule.GetAtomWithIdx(hydrogen.GetIdx())  # Get the corresponding hydrogen in the copied molecule
        hydrogen_atom.SetIsotope(2)  # Set isotope number to 2 for deuterium
        modified_molecules.append(modified_molecule)

    return modified_molecules


similarities = []
num_comparisons_list = []
for file_num in range(1, N+1):
    file_name = f'{cwd}/similarity/sd_data/size_increasing_molecules/range_{file_num}.sdf'
    original_molecule = load_molecules_from_sdf(file_name, removeHs=False, sanitize=False)

    modified_molecules = generate_all_single_deuterium_variants(original_molecule[0])
    num_comparisons = len(modified_molecules)
    num_comparisons_list.append(num_comparisons)
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

print('Similarities:')
print('\n'.join(map(str, similarities)))

# Plotting the average similarities
plt.figure(figsize=(10, 5))
plt.plot(range(1, N+1), similarities, marker='o', linestyle='-')

# Annotate each point with the number of comparisons
for i, (x, y, num_comparisons) in enumerate(zip(range(1, N+1), similarities, num_comparisons_list)):
    plt.annotate(str(num_comparisons), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

plt.xlabel('File Number')
plt.ylabel('Average Similarity')
plt.title('Similarity between Original and Modified Molecules')
plt.xticks(range(1, N+1))
plt.grid(True)
plt.tight_layout()
# plt.savefig(f'{cwd}/sensitivity_1H_all_combinations.svg', format="svg")
plt.show()
