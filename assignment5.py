import pandas as pd
from sklearn.preprocessing import StandardScaler
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
column_1 = [row[3] for row in normalized_data]
column_2 = [row[4] for row in normalized_data]
column_8 = [row[7] for row in normalized_data]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(column_1, column_2, column_8, c=column_8, cmap='viridis')

ax.set_xlabel('Column 1')
ax.set_ylabel('Column 2')
ax.set_zlabel('Column 8')

cbar = fig.colorbar(scatter)
cbar.set_label('Column 8')

plt.show()
