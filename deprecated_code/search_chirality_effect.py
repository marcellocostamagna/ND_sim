import re

def extract_enrichment_data(filename):
    with open(filename, 'r') as f:
        content = f.read()

    methods = ["pseudo_usr", "pseudo_usr_cat", "pseudo_electroshape"]
    method_data = {}
    for method in methods:
        method_data[method] = {}
        method_section = re.findall(r"Results for {}:(.*?)Results".format(method), content + "Results", re.S)[0]
        for folder in re.findall(r"Folder: (\w+)", method_section):
            method_data[method][folder] = {}
            for percent in ["0.25%", "0.5%", "1.0%", "2.0%", "3.0%", "5.0%"]:
                value = re.search(r"Enrichment Factor at {} for {}: ([-+]?\d*\.\d+|\d+)".format(percent, folder), method_section)
                if value:
                    method_data[method][folder][percent] = float(value.group(1))
    return method_data

def compare_data(data1, data2):
    discrepancies_found = False  # Flag to track if discrepancies are found

    for method in data1.keys():
        for folder in data1[method].keys():
            for percent, value1 in data1[method][folder].items():
                value2 = data2[method][folder].get(percent)
                if value2 is not None and abs(value1 - value2) > 0.000000001:
                    discrepancies_found = True
                    print(f"Discrepancy found in method: {method}, folder: {folder}, for {percent}. Values: {value1} vs {value2}")

    if not discrepancies_found:
        print("No discrepancies found.")
        
def main():
    file1_data = extract_enrichment_data('dude_pseudo_results_no_chirality.txt')
    file2_data = extract_enrichment_data('dude_pseudo_results_manual_chirality.txt')
    
    compare_data(file1_data, file2_data)

if __name__ == "__main__":
    main()
