import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

"""path = 'data/R-L_80/no_cues/random_W/'
W = np.load(path+'W.npy')
print(W)

all_W = []
for i in range(0, 100):
    flat_W = W.flatten()
    np.random.shuffle(flat_W)
    W_shuffled = flat_W.reshape(W.shape)
    all_W.append(W_shuffled.flatten())

    #np.save(path + f'W_{i}.npy', W_shuffled)"""

path = "data/R-L/no_cues/random_W/"

def generate_connectivity_matrix(connectivity, n_units, sr):
    # Generate a random connectivity matrix
    matrix = np.random.rand(n_units, n_units)
    random_values = np.random.normal(loc=0, scale=1,
                                     size=(n_units, n_units))  # Use n_units for correct shape
    matrix = np.where(matrix > (1 - connectivity), random_values, 0)

    # Compute the spectral radius
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_radius = np.max(np.abs(eigenvalues))

    # Scale the matrix to achieve the desired spectral radius
    if spectral_radius > 0:
        scaling_factor = sr / spectral_radius
        W = matrix * scaling_factor
    else:
        W = matrix  # If spectral radius is 0, just return the original matrix

    return W


connectivity = 0.1
n_units = 1000
sr = 0.99
all_W = []
print('Start')
for i in range(20):
    print(i)
    W = generate_connectivity_matrix(connectivity, n_units, sr)
    all_W.append(W.flatten())
    np.save(path + f'W_{i}.npy',W)

# Initialize t-SNE for 3D projection
"""tsne = TSNE(n_components=3, random_state=42)

# Fit and transform the data
X_tsne = tsne.fit_transform(np.array(all_W))

# Plotting in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the t-SNE components
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c='b', marker='o')

# Add axis labels
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')

# Show plot
plt.show()"""
# Perform PCA
"""pca = PCA(n_components=3)
print(np.shape(all_W))
X_pca = pca.fit_transform(all_W)

# Plotting in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the PCA components
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='b', marker='o')

# Add axis labels
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Show plot
plt.show()"""
# Initialize UMAP for 3D projection
umap_3d = umap.UMAP(n_components=3)

# Fit and transform the data
X_umap = umap_3d.fit_transform(all_W)

# Plotting in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the UMAP components
ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c='b', marker='o')

# Add axis labels
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')

# Show plot
plt.show()