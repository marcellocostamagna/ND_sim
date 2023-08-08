# Preliminary script to compute the 5D moments of a molecule

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from scipy.stats import skew

# Read the SDF file
sdf_file = 'sample3d_optimized_switched.sdf'
suppl = Chem.SDMolSupplier(sdf_file)

# Store all the molecules in a list
molecules = [mol for mol in suppl if mol is not None]
mol_list = [molecules[2], molecules[1]]

fingerprints = []
for molecule in mol_list:
    # Get atom coordinates, masses, and formal charges
    atom_information = []
    for atom in molecule.GetAtoms():
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        mass = atom.GetMass()
        formal_charge = atom.GetFormalCharge()
        atom_information.append([position.x, position.y, position.z, mass, formal_charge])

    # Create the initial matrix
    initial_matrix = np.array(atom_information)
    print(initial_matrix)

    #### Compute the 5D moments ####

    data = initial_matrix

    # Standardize the data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=5)
    pca.fit(data_standardized)

    # Transform the data to the new axes
    transformed_data = pca.transform(data_standardized)

    # Calculate the center of the cloud in the original data space
    center_original_space = np.mean(data, axis=0)

    # # Find unique extreme points (max_index) along each axis in the principal component space
    # extreme_point_indices = set()
    # for i in range(5):
    #     sorted_indices = np.argsort(transformed_data[:, i])[::-1]  # Sort indices in descending order
    #     for index in sorted_indices:
    #         if index not in extreme_point_indices:
    #             extreme_point_indices.add(index)
    #             break  # Move on to the next axis once a unique extreme point is found
    # extreme_points_pca_space = transformed_data[list(extreme_point_indices)]

    # Find 5 (not unique) extreme points (max_index) along each axis in the principal component space
    extreme_point_indices = []
    for i in range(5):
        max_index = np.argmax(transformed_data[:, i])
        extreme_point_indices.append(max_index)        

    extreme_points_pca_space = transformed_data[extreme_point_indices]

    # Transform the unique extreme points (max_index) back to the original data space
    extreme_points_original_space = pca.inverse_transform(extreme_points_pca_space)

    # Un-standardize the extreme points (max_index) to match the original data scale
    extreme_points = scaler.inverse_transform(extreme_points_original_space)

    # Add the center of the cloud to the extreme points
    points_of_interest = np.vstack((center_original_space, extreme_points))

    # Calculate distances between each point in the original data and the 6 points of interest
    distances = {}
    for i, point in enumerate(points_of_interest):
        distances[f"point_{i}"] = [distance.euclidean(point, data_point) for data_point in data]

    # Compute the mean, standard deviation, and skewness for each distance list
    moments = {}
    for key, distance_list in distances.items():
        mean = np.mean(distance_list)
        std_dev = np.std(distance_list)
        skewness = skew(distance_list)
        moments[key] = {"mean": mean, "std_dev": std_dev, "skewness": skewness}

    # Create the fingerprint list from the moments dictionary
    fingerprint = []
    for key in moments.keys():
        fingerprint.extend([moments[key]["mean"], moments[key]["std_dev"], moments[key]["skewness"]])

    fingerprints.append(fingerprint)
    print(fingerprint)

#Calculate similarity

sum = np.sum([np.abs(fingerprints[0][i] - fingerprints[1][i]) for i in range(17)])

similarity = 1/(1 + (1/18)* sum)

print(similarity)