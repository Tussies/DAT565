import pandas as pd
from sklearn.preprocessing import StandardScaler

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
