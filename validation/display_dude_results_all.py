import re
import numpy as np
import matplotlib.pyplot as plt

def extract_data(content):
    methods = ["pseudo_usr", "pseudo_usr_cat", "pseudo_electroshape"]
    method_data = {}
    for method in methods:
        method_data[method] = re.findall(r"Results for {}:(.*?)Results".format(method), content + "Results", re.S)[0]

    folders = sorted(list(set(re.findall(r"Folder: (\w+)", content))))
    return method_data, folders

def compute_and_print_averages(method_data):
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
    return averages

colors_methods = ['firebrick', 'olivedrab', 'royalblue']

def plot_graph_for_file(file_key, method_data, folders):
    simple_name = file_key.rsplit('.', 1)[0]  # Remove the .txt extension
    plt.figure(figsize=(15, 8))
    for index, (method, folders_data) in enumerate(method_data.items()):
        enrichment_05 = [float(value) for value in re.findall(r"Enrichment Factor at 0.5%: ([-+]?\d*\.\d+|\d+)", folders_data)]
        plt.plot(range(len(folders)), enrichment_05, marker='o', linestyle='--', linewidth = 0.8, color=colors_methods[index], label=method)
    plt.title(f"Enrichment Factor at 0.5% for file {simple_name}", fontweight="bold")
    plt.xlabel("Folder")
    plt.ylabel("Enrichment Factor at 0.5%")
    plt.xticks(range(len(folders)), folders, rotation=90, fontsize=8)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"plot_{file_key}.svg", format='svg')
    # plt.show()
    
colors_files = ['aqua', 'salmon', 'darkblue']

def plot_graph_for_method(method, all_data_by_file, folders):
    plt.figure(figsize=(15, 8))
    for index, (file_key, method_data) in enumerate(all_data_by_file.items()):
        enrichment_05 = [float(value) for value in re.findall(r"Enrichment Factor at 0.5%: ([-+]?\d*\.\d+|\d+)", method_data[method])]
        simple_name = file_key.rsplit('.', 1)[0]  # Remove the .txt extension
        plt.plot(range(len(folders)), enrichment_05, marker='o', linestyle='--', linewidth = 0.8, color=colors_files[index], label=simple_name)
    plt.title(f"Enrichment Factor at 0.5% for method {method}", fontweight="bold")
    plt.xlabel("Folder")
    plt.ylabel("Enrichment Factor at 0.5%")
    plt.xticks(range(len(folders)), folders, rotation=90, fontsize=8)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"plot_{method}.svg", format='svg')
    # plt.show()


def main():
    files = [
        'dude_pseudo_results_no_chirality.txt',
        'dude_pseudo_results_chirality_with_flip.txt',
        'dude_pseudo_results_manual_chirality.txt'
    ]
    
    all_data_by_file = {}
    
    for file in files:
        with open(file, 'r') as f:
            content = f.read()

        method_data, folders = extract_data(content)
        
        # Compute averages and save to text files
        averages = compute_and_print_averages(method_data)
        simple_name = file.rsplit('.', 1)[0]  # Remove the .txt extension
        with open(f"averages_{simple_name}.txt", 'w') as outfile:
            for method, avg_data in averages.items():
                outfile.write(f"Method: {method}\n")
                for percent, value in avg_data.items():
                    outfile.write(f"Enrichment Factor at {percent}: {value}\n")
                outfile.write("\n")
        
        all_data_by_file[file] = method_data
        
    # # For each file
    for file_key, method_data in all_data_by_file.items():
        plot_graph_for_file(file_key, method_data, folders)
    for method in ["pseudo_usr", "pseudo_usr_cat", "pseudo_electroshape"]:
        plot_graph_for_method(method, all_data_by_file, folders)    
        
    plt.show()
if __name__ == "__main__":
    main()
    
  