import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from umap import UMAP
import matplotlib.pyplot as plt

def read_tsv_file(file_path):
    return pd.read_csv(file_path, header=None, delimiter='\t').values.tolist()

file_path = 'seeds.tsv'
data = read_tsv_file(file_path)

features = [row[:-1] for row in data]
labels = [row[-1] for row in data]

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

normalized_data = [list(normalized_features[i]) + [labels[i]] for i in range(len(normalized_features))]

for row in normalized_data:
    print(row)

column_4 = [row[3] for row in normalized_data]
column_5 = [row[4] for row in normalized_data]
column_8 = [row[7] for row in normalized_data]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(column_4, column_5, column_8, c=column_8, cmap='viridis')

ax.set_xlabel('Column 4, Length of kernel')
ax.set_ylabel('Column 5, Width of kernel')
ax.set_zlabel('Column 8, Numerical class label')

cbar = fig.colorbar(scatter)
cbar.set_label('Column 8, Numerical class label')

plt.title('3D Scatter Plot')
plt.show()

random_projection = GaussianRandomProjection(n_components=2)
projected_features = random_projection.fit_transform(normalized_features)

projected_data = [list(projected_features[i]) + [labels[i]] for i in range(len(projected_features))]

column_1 = [row[0] for row in projected_data]
column_2 = [row[1] for row in projected_data]

plt.figure(figsize=(10, 8))
plt.scatter(column_1, column_2, c=labels, cmap='viridis')
plt.xlabel('Column 4, Length of kernel (Projected Dimension 1)')
plt.ylabel('Column 5, Width of kernel (Projected Dimension 2)')
plt.title('2D Scatter Plot with Gaussian Random Projection')

plt.colorbar(label='Column 8, Numerical class label')
plt.show()

umap_reducer = UMAP(n_components=2)
umap_features = umap_reducer.fit_transform(normalized_features)

umap_data = [list(umap_features[i]) + [labels[i]] for i in range(len(umap_features))]

umap_column_1 = [row[0] for row in umap_data]
umap_column_2 = [row[1] for row in umap_data]

plt.figure(figsize=(10, 8))
plt.scatter(umap_column_1, umap_column_2, c=labels, cmap='viridis')
plt.xlabel('Column 4, Length of kernel (Dimension 1)')
plt.ylabel('Column 5, Width of kernel (Dimension 2)')
plt.title('2D Scatter Plot with UMAP Dimensionality Reduction')

plt.colorbar(label='Column 8, Numerical class label')
plt.show()
