# Scripts for generating similarity tables for the aminoacids (L and D) datasets

import numpy as np
import os
from nd_sim.pre_processing import load_molecules_from_sdf
from nd_sim.fingerprint import generate_nd_molecule_fingerprint
from nd_sim.similarity import *
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

cwd = os.getcwd()

def visualize_subsection(data, row_names, col_names, title, n=5, save_path=None):
    visualize_table(data[:n, :n], row_names[:n], col_names[:n], title, scale=3, save_path=save_path)

def visualize_table(data, row_names, col_names, title, scale=30, save_path=None):
    # Create a DataFrame
    df = pd.DataFrame(data, index=row_names, columns=col_names)
    
    # Create a heatmap using matplotlib
    fig_width, fig_height = 8, 8  
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cax = ax.matshow(df, cmap="YlGnBu", vmax=1)
    plt.title(title, pad=20)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_xticklabels(col_names, rotation=45, ha='left', rotation_mode='anchor')
    ax.set_yticklabels(row_names)
    
    # Adjust tick position for better alignment
    ax.tick_params(axis='x', which='both', length=0, pad=10)  # Adjust the 'pad' parameter if needed for better alignment
    
    # Determine the font size 
    font_size = min(fig_width, fig_height) / scale * len(row_names)
    
    # Display the values in each cell
    for i in range(len(row_names)):
        for j in range(len(col_names)):
            ax.text(j, i, f"{data[i][j]:.4f}", ha='center', va='center', color='black', fontsize=font_size)
    
    plt.colorbar(cax, label='Similarity')
    plt.tight_layout()
    
    if save_path:
        # Create the folder if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Convert title to valid filename
        filename = ''.join(c for c in title if c.isalnum() or c in [" ", "-", "_", ".", "(", ")"])
        filename = filename.replace(' ', '_') + ".svg"

        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path, format='svg')
        plt.close() 
    else:
        plt.show()


def get_max_column_widths(names, table):
    # Initial widths are set to name lengths
    max_widths = [len(name) for name in names]
    
    for col_index in range(len(names)):
        for row in table:
            max_widths[col_index] = max(max_widths[col_index], len("{:.4f}".format(row[col_index])))
    
    return max_widths

def generate_similarity_table(molecules_1, molecules_2):
    fingerprints_1 = [generate_nd_molecule_fingerprint(molecule, DEFAULT_FEATURES, scaling_method='matrix', chirality=True) for molecule in molecules_1]
    fingerprints_2 = [generate_nd_molecule_fingerprint(molecule, DEFAULT_FEATURES, scaling_method='matrix', chirality=True) for molecule in molecules_2]
    
    n_molecules_1 = len(fingerprints_1)
    n_molecules_2 = len(fingerprints_2)
    
    table = np.zeros((n_molecules_1, n_molecules_2))
    
    for i in range(n_molecules_1):
        for j in range(n_molecules_2):
            # similarity = compute_similarity_score(fingerprints_1[i], fingerprints_2[j])
            similarity = compute_similarity_score(fingerprints_1[i], fingerprints_2[j])
            table[i][j] = similarity
    
    return table

def load_molecules_and_names(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".sdf")])
    molecules = [load_molecules_from_sdf(os.path.join(folder, f), removeHs=False, sanitize=False)[0] for f in files]  # Assuming one molecule per file
    names = [f.split('.')[0] for f in files]
    return molecules, names

# Load the molecules and their names
L_molecules, L_names = load_molecules_and_names(f'{cwd}/similarity/sd_data/Aminoacids/L')
D_molecules, D_names = load_molecules_and_names(f'{cwd}/similarity/sd_data/Aminoacids/D')

# Compute similarity tables
L_vs_L = generate_similarity_table(L_molecules, L_molecules)
D_vs_D = generate_similarity_table(D_molecules, D_molecules)
L_vs_D = generate_similarity_table(L_molecules, D_molecules)

# Write the tables to a text file
with open(f'{cwd}/aminoacids_similarity_results_with_flip.txt', 'w') as f:
    # L vs L
    f.write('L vs L Similarities:\n')
    max_widths = get_max_column_widths(L_names, L_vs_L)
    # Shifted over by one column width
    f.write(' '.ljust(max(max_widths)) + '\t' + '\t'.join([name.ljust(width) for name, width in zip(L_names, max_widths)]) + '\n')
    for i, row in enumerate(L_vs_L):
        f.write(L_names[i].ljust(max(max_widths)) + '\t' + '\t'.join(["{:.4f}".format(val).ljust(width) for val, width in zip(row, max_widths)]) + '\n')
    
    # D vs D
    f.write('\nD vs D Similarities:\n')
    max_widths = get_max_column_widths(D_names, D_vs_D)
    # Shifted over by one column width
    f.write(' '.ljust(max(max_widths)) + '\t' + '\t'.join([name.ljust(width) for name, width in zip(D_names, max_widths)]) + '\n')
    for i, row in enumerate(D_vs_D):
        f.write(D_names[i].ljust(max(max_widths)) + '\t' + '\t'.join(["{:.4f}".format(val).ljust(width) for val, width in zip(row, max_widths)]) + '\n')
    
    # L vs D
    f.write('\nL vs D Similarities:\n')
    max_widths = get_max_column_widths(D_names, L_vs_D)
    # Shifted over by one column width
    f.write(' '.ljust(max(max_widths)) + '\t' + '\t'.join([name.ljust(width) for name, width in zip(D_names, max_widths)]) + '\n')
    for i, row in enumerate(L_vs_D):
        f.write(L_names[i].ljust(max(max_widths)) + '\t' + '\t'.join(["{:.4f}".format(val).ljust(width) for val, width in zip(row, max_widths)]) + '\n')

# Specify the folder to save the SVG plots
folder_name = f"{cwd}/aminoacids_manual_chirality_with_flip_plots"  
        
# Visualize the tables
visualize_table(L_vs_L, L_names, L_names, "L vs L Similarities", save_path=folder_name)
visualize_table(D_vs_D, D_names, D_names, "D vs D Similarities", save_path=folder_name)
visualize_table(L_vs_D, L_names, D_names, "L vs D Similarities", save_path=folder_name)

# Visualize the subsections (first 5 aminoacids)
visualize_subsection(L_vs_L, L_names, L_names, "L vs L Similarities (First 5 Aminoacids)",save_path=folder_name )
visualize_subsection(D_vs_D, D_names, D_names, "D vs D Similarities (First 5 Aminoacids)", save_path=folder_name)
visualize_subsection(L_vs_D, L_names, D_names, "L vs D Similarities (First 5 Aminoacids)", save_path=folder_name)
     
plt.show()