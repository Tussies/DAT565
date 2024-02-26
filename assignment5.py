import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import accuracy_score

def read_tsv_file(file_path):
    return pd.read_csv(file_path, header=None, delimiter='\t').values.tolist()

file_path = 'seeds.tsv'
data = read_tsv_file(file_path)

features = [row[:-1] for row in data]

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

normalized_data = [list(normalized_features[i]) for i in range(len(normalized_features))]

true_labels = [row[-1] for row in data]

linkage_options = ['ward', 'complete', 'average', 'single']
best_linkage = None
best_accuracy = 0.0

for linkage_option in linkage_options:
    agg_cluster = AgglomerativeClustering(n_clusters=3, linkage=linkage_option)
    clusters = agg_cluster.fit_predict(normalized_features)

    accuracy = accuracy_score(true_labels, clusters)

    print(f"Results for linkage={linkage_option}:")
    print(f"Accuracy: {accuracy}\n")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_linkage = linkage_option

best_linkage_clusters = AgglomerativeClustering(n_clusters=3, linkage=best_linkage)
best_clusters = best_linkage_clusters.fit_predict(normalized_features)

linkage_matrix = linkage(normalized_features, method=best_linkage)
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=true_labels, orientation='top', leaf_rotation=90, color_threshold=1.5, leaf_font_size=3)
plt.title(f'Hierarchical Clustering Dendrogram (Best Linkage: {best_linkage})')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()