from sklearn.decomposition import PCA
import numpy as np

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

pca = PCA()
pca.fit(data)
U = pca.transform(data) 
Vt = pca.components_
# Display U and Vt matrices
print(f'U: \n{U}')
print(f'Vt: \n{Vt}')

# Check the validity of the SVD decomposition
reconstructed_data = U @ Vt
reconstructed_data += pca.mean_
print(f'U*Vt: \n{reconstructed_data}')

reconstructed_data1 = pca.inverse_transform(U)
print(f'U*Vt (second method): \n{reconstructed_data1}')
# is data = U*Vt? 
print(f'Is data = U*Vt? {np.allclose(data, np.dot(U, Vt))}')
print(f'Variances: {pca.explained_variance_}')
print(f'Singular values: {pca.singular_values_}')

# Since we are not using the argument whiten=True in the PCA constructor, 
# (which "When True (False by default) the components_ vectors are multiplied by the square root
#  of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit
#  component-wise variances."),
# there is no 'scaling' of the components_ vectors by the square root of n_samples and the 
# decomposition results in a simple rotation of the data. Hence, to retrieve the original data
# (such as the inverse_transform method does), we need to multiply the components_ vectors only 
# by the principal components and neglect the singular values.
