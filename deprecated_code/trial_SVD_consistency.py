import numpy as np
from .perturbations import *
from source.pca_tranform import perform_PCA_and_get_transformed_data
from source.fingerprint import get_fingerprint
from source.similarity import get_similarity_measure, calculate_partial_score
import time

stating_time = time.time()
for i in range(1000000):
    # Generate random 3D points
    num_points = np.random.randint(3, 10000)
    data = np.random.uniform(low = -1000 , high = 1000 , size = (num_points, 3))

    # Rotate the points
    tmp_data = perturb_coordinates(data, 5)
    #tmp_data = data
    # generate random float number between 0 and 360
    angle1 = np.random.uniform(0, 360)
    angle2 = np.random.uniform(0, 360)
    angle3 = np.random.uniform(0, 360)
    data1 = rotate_points(tmp_data, angle1, angle2, angle3)

    ###### PCA with SVD#########

    _, tranformed_data, _, _ = perform_PCA_and_get_transformed_data(data)
    _, transformed_data1, _, _ = perform_PCA_and_get_transformed_data(data1)

    fingerprint  = get_fingerprint(tranformed_data)
    fingerprint1 = get_fingerprint(transformed_data1)

    partial_score = calculate_partial_score(fingerprint, fingerprint1)
    similarity = get_similarity_measure(partial_score)

    print(f'{i}- Similarity: {similarity}')

    if similarity < 0.9999:
        print(f'Data: \n{data}')
        print(f'Data1: \n{data1}')
        # STOP the program
        break

ending_time = time.time()
print(f'Time taken: {ending_time - stating_time}')



