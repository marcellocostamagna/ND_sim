import re
import numpy as np
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()

# Get the file path from user
file_path = f"{cwd}/dude_pseudo_results_no_chirality.txt"

# Read the data from the file
with open(file_path, 'r') as file:
    data = file.read()

# Extract data for each method
methods = ["pseudo_usr", "pseudo_usr_cat", "pseudo_electroshape"]
method_data = {}
for method in methods:
    method_data[method] = re.findall(r"Results for {}:(.*?)Results".format(method), data + "Results", re.S)[0]

# Compute averages
averages = {}
for method, values in method_data.items():
    averages[method] = {}
    print(f"\nAverages for {method}:")
    for percent in ["0.25%", "0.5%", "1.0%", "2.0%", "3.0%", "5.0%"]:
        values_for_percent = re.findall(r"Enrichment Factor at {}: ([-+]?\d*\.\d+|\d+)".format(percent), values)
        values_for_percent = [float(val) for val in values_for_percent]
        average_value = np.mean(values_for_percent)
        averages[method][percent] = average_value
        print(f"Enrichment Factor at {percent}: {average_value}")

# Save averages to a file
with open('averages.txt', 'w') as f:
    for method, values in averages.items():
        f.write(f"Averages for {method}:\n")
        for percent, average_value in values.items():
            f.write(f"Enrichment Factor at {percent}: {average_value}\n")
        f.write("\n")

# # Create plots
# for method, values in averages.items():
#     folders = re.findall(r"Folder: (\w+)", method_data[method])
#     enrichment_05 = [float(value) for value in re.findall(r"Enrichment Factor at 0.5%: ([-+]?\d*\.\d+|\d+)", method_data[method])]
    
#     plt.figure(figsize=(20, 6))  # Increase the figure width even more
#     plt.plot(range(len(folders)), enrichment_05, marker='o', linestyle='-')
    
#     plt.title(f"Enrichment Factor at 0.5% for {method}")
#     plt.xlabel("Folder")
#     plt.ylabel("Enrichment Factor at 0.5%")
    
#     # Custom x-axis ticks and labels with a greater rotation
#     plt.xticks(range(len(folders)), folders, rotation=90, fontsize=8)  # Increase the rotation to 90 degrees and reduce font size
    
#     plt.tight_layout()
#     # plt.savefig(f"{method}_plot.png")
#     plt.show()
# Prepare the figure and axes
plt.figure(figsize=(20, 8))

# Define a color map for the methods
colors = {
    'pseudo_usr': 'blue',
    'pseudo_usr_cat': 'green',
    'pseudo_electroshape': 'red'
}

# Extract the folder names just once (assuming the folder names are the same across methods)
folders = re.findall(r"Folder: (\w+)", method_data['pseudo_usr'])

# Loop through each method and plot on the same figure
for method, values in averages.items():
    enrichment_05 = [float(value) for value in re.findall(r"Enrichment Factor at 0.5%: ([-+]?\d*\.\d+|\d+)", method_data[method])]
    plt.plot(range(len(folders)), enrichment_05, marker='o', linestyle='-', color=colors[method], label=method)

# Set title, labels, and legend
plt.suptitle("No chirality", fontsize=16, fontweight="bold")
plt.title("Enrichment Factor at 0.5%")
plt.xlabel("Folder")
plt.ylabel("Enrichment Factor at 0.5%")
plt.xticks(range(len(folders)), folders, rotation=90, fontsize=8)
plt.legend(loc='upper right')

# Save and display the combined plot
plt.tight_layout()
plt.savefig("combined_plot.png")
plt.show()


# Save data to a file
with open('no_chirality_enrichments.txt', 'w') as f:
    for method, values in averages.items():
        enrichment_05 = [value for value in re.findall(r"Enrichment Factor at 0.5%: ([-+]?\d*\.\d+|\d+)", method_data[method])]
        f.write(f"Results for {method}:\n")
        for folder, value in zip(folders, enrichment_05):
            f.write(f"Folder: {folder}, Enrichment Factor at 0.5%: {value}\n")
        f.write("\n")
