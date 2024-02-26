import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from itertools import permutations

def read_tsv_file(file_path):
    return pd.read_csv(file_path, header=None, delimiter='\t').values.tolist()

file_path = 'seeds.tsv'
data = read_tsv_file(file_path)

features = [row[:-1] for row in data]

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

normalized_data = [list(normalized_features[i]) for i in range(len(normalized_features))]

true_labels = [row[-1] for row in data]

inertias = []
rand_indices = []
accuracies = []

for k in range(1, 4):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(normalized_features)

    inertia = kmeans.inertia_
    inertias.append(inertia)

    best_accuracy = 0.0
    best_permutation = None

    for perm_mapping in permutations(range(k)):
        mapped_clusters = [perm_mapping[c] for c in clusters]
        accuracy = accuracy_score(true_labels, mapped_clusters)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_permutation = perm_mapping

    accuracies.append(best_accuracy)

    print(f"Results for k={k}:")
    print(f"Inertia: {inertia}")
    print(f"Best Accuracy: {best_accuracy} (Best Permutation: {best_permutation})\n")

print("All Accuracies:", accuracies)

plt.plot(range(1, 4), inertias, marker='o')
plt.title('Inertia as a function of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()
