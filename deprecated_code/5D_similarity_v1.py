# Preliminary code for the 5D similarity fingerprint

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from scipy.stats import skew

# Generate random data (a cloud of points in 5D space as a matrix of 100 rows and 5 columns)
n = 100
np.random.seed(42)
data = np.random.randn(n, 5)

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

# Find unique extreme points (max_index) along each axis in the principal component space
extreme_point_indices = set()
for i in range(5):
    max_index = np.argmax(transformed_data[:, i])
    extreme_point_indices.add(max_index)

extreme_points_pca_space = transformed_data[list(extreme_point_indices)]

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

