from sklearn.decomposition import PCA
import numpy as np
import perturbations as ptb
for i in range(1):
    data = np.array([[-3, 0, 0],
                    [-1, 0, 0],
                    [ 1, 0, 0],
                    [ 3, 0, 0],
                    [ 0, 2, 0],
                    [ 0, 1, 0],
                    [ 0,-1, 0],
                    [ 0,-2, 0],
                    [ 0, 0, 1],
                    [ 0, 0,-1]]).astype(float)

    # pca and inverse_transform of the data
    pca = PCA()
    pca.fit(data)
    U = pca.transform(data) 
    Vt = pca.components_

    print(f'Original data: \n{data}')
    print(f'Principal components: \n{Vt}')
    reconstructed_data = U @ Vt
    reconstructed_data += pca.mean_
    reconstructed_data_second_method = pca.inverse_transform(U)

    # Get another set of data by rotating and perturbing the original data
    tmp_data = ptb.perturb_coordinates(data, 8)
    angle1 = np.random.uniform(0, 360)
    angle2 = np.random.uniform(0, 360)
    angle3 = np.random.uniform(0, 360)
    data1, rotation = ptb.rotate_points_and_get_rotation_matrix(tmp_data, angle1, angle2, angle3)

    # pca and inverse_transform of the rotated data
    pca1 = PCA()
    pca1.fit(data1)
    U1 = pca1.transform(data1)
    Vt1 = pca1.components_
    print(f'Data1 : \n{data1}')
    print(f'Principal components: \n{Vt1}')
    reconstructed_data1 = U1 @ Vt1
    reconstructed_data1 += pca1.mean_

    reconstructed_data1_second_method = pca1.inverse_transform(U1)

    # Compare data and data1 after inverse_transform and the inverse rotation.
    # The inverse rotation is the transpose of the rotation matrix.

    # First Method
    unrotated_data1 = rotation.T @ reconstructed_data1.T
    data_to_check = unrotated_data1.T

    # unrotated principal components
    unrotated_principal_components = (rotation.T @ Vt1.T).T
    print(f'Unrotated principal components: \n{unrotated_principal_components}')

    # check is reconstructed_data = unrotated_data1?
    
    print(f'First Method {np.allclose(reconstructed_data, data_to_check, atol = 1e-7 ) }')
    # print the maximum difference between the two arrays
    print(f'Maximum difference between the two arrays: {np.max(np.abs(reconstructed_data - data_to_check))}')

    # Second Method
    unrotated_data1_second_method = rotation.T @ reconstructed_data1_second_method.T
    data_to_check_second_method = unrotated_data1_second_method.T

    # check is reconstructed_data = unrotated_data1?
    print(f'Second Method {np.allclose(reconstructed_data_second_method, data_to_check_second_method, atol=1e-1)}')
    # print the maximum difference between the two arrays
    print(f'Maximum difference between the two arrays: {np.max(np.abs(reconstructed_data_second_method - data_to_check_second_method))}')



