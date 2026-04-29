import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create a sample dataset
data = np.array([[11.0, 20.0, 32.0],
                 [14.0, 53.0, 64.0],
                 [17.0, 86.0, 19.0]])
# Standardize the data
scaler = StandardScaler()
data_std = scaler.fit_transform(data)
# Create a PCA instance
pca = PCA(n_components=2)  # Specify the number of components
# Fit PCA to the standardized data
pca.fit(data_std)
# Transform the data into the reduced dimensionality
data_pca = pca.transform(data_std)
# The principal components
components = pca.components_
# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
# Print the results
print("Original Data:")
print(data)
print("\nStandardized Data:")
print(data_std)
print("\nPrincipal Components:")
print(components)
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)
print("\nData in Reduced Dimensionality:")
print(data_pca)