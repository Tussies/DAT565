import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def read_tsv_file(file_path):
    return pd.read_csv(file_path, header=None, delimiter='\t').values.tolist()

file_path = 'seeds.tsv'
data = read_tsv_file(file_path)

features = [row[:-1] for row in data]

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

normalized_data = [list(normalized_features[i]) for i in range(len(normalized_features))]

inertias = []
rand_indices = []

for k in range(1, 4):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(normalized_features)

    inertia = kmeans.inertia_
    inertias.append(inertia)

    true_labels = [row[-1] for row in data]
    rand_index = adjusted_rand_score(true_labels, clusters)
    rand_indices.append(rand_index)

    print(f"Results for k={k}:")
    print(f"Inertia: {inertia}")
    for i, row in enumerate(normalized_data):
        print(f"Data point {i + 1} - Cluster: {clusters[i]}")

print("All Rand Indices:", rand_indices)

plt.plot(range(1, 4), inertias, marker='o')
plt.title('Inertia as a function of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()
