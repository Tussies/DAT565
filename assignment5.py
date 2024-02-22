import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def read_tsv_file(file_path):
    return pd.read_csv(file_path, header=None, delimiter='\t').values.tolist()

file_path = 'seeds.tsv'
data = read_tsv_file(file_path)

df = pd.DataFrame(data)

scaler = MinMaxScaler()
data_normalized_minmax = scaler.fit_transform(df)

data_normalized_minmax = data_normalized_minmax.tolist()

for row in data_normalized_minmax:
    print(row)

