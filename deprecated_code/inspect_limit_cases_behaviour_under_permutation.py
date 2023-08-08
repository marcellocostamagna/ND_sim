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
    reconstructed_data = U @ Vt
    reconstructed_data += pca.mean_
    reconstructed_data_second_method = pca.inverse_transform(U)
    print(f'Original data: \n{data}')
    print(f'Principal components: \n{Vt}')

    # Get another set of data by permuting the original data
    # permute the order of the data keeping track of the permutation
    permutation = np.random.permutation(data.shape[0])
    inverse_permutation = np.argsort(permutation)
    data1 = data[permutation]

    print(f'Data1 : \n{data1}')
    
   

    # data1 = data.copy()
    # np.random.shuffle(data1)
    # print(data1)

    # # pca and inverse_transform of the permutated data
    pca1 = PCA()
    pca1.fit(data1)
    U1 = pca1.transform(data1)
    Vt1 = pca1.components_
    print(f'Principal components: \n{Vt1}')
    print(f'Transformed data1: \n{U1}')

    reconstructed_data1 = U1 @ Vt1
    reconstructed_data1 += pca1.mean_
    print(f'Reconstructed data1: \n{reconstructed_data1}')
    reconstructed_data1_second_method = pca1.inverse_transform(U1)

    # # Compare data and data1 after inverse_transform and the inverse permutation.

    # First Method
    unpermutated_data1 =  reconstructed_data1[inverse_permutation]
    print(f'Unpermutated data1: \n{unpermutated_data1}')
    data_to_check = unpermutated_data1

    # check is reconstructed_data = inpermutated_data1?
    
    print(f'First Method {np.allclose(reconstructed_data, data_to_check, atol = 1e-10 ) }')
    # print the maximum difference between the two arrays
    print(f'Maximum difference between the two arrays: {np.max(np.abs(reconstructed_data - data_to_check))}')

    # # Second Method
    # unpermutated_data1_second_method = reconstructed_data1_second_method[inverse_permutation]
    # data_to_check_second_method = unpermutated_data1_second_method

    # # check is reconstructed_data = unrotated_data1?
    # print(f'Second Method {np.allclose(reconstructed_data_second_method, data_to_check_second_method, atol=1e-10)}')
    # # print the maximum difference between the two arrays
    # print(f'Maximum difference between the two arrays: {np.max(np.abs(reconstructed_data_second_method - data_to_check_second_method))}')



# # invert permutation
#     inverse_permutation = np.argsort(permutation)
#     print(data1[inverse_permutation])
    