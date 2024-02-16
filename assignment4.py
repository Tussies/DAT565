import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

file_path = 'life_expectancy.csv'
life_expectancy = read_csv_file(file_path)

header = life_expectancy[0]
life_expectancy = life_expectancy[1:]
life_expectancy_numeric = np.array([[np.nan if val == '' else float(val) for val in row[1:]] for row in life_expectancy], dtype=np.float64)

column_means = np.nanmean(life_expectancy_numeric, axis=0)
life_expectancy_numeric = np.where(np.isnan(life_expectancy_numeric), column_means, life_expectancy_numeric)

print(len(header))
print(len(life_expectancy_numeric))

y = life_expectancy_numeric[:, -4]
X = np.hstack([life_expectancy_numeric[:, :-4], life_expectancy_numeric[:, -3:]])


column_names = header[1:-4] + header[-3:]
print(column_names)
print(y[0])

print("Length of X[0]:", (X[0]).shape)
print("Length of y:", len(y))
print("Length of column_names:", len(column_names))
print("Length of life_expectancy_numeric:", len(life_expectancy_numeric))
print("Number of columns in life_expectancy_numeric:", life_expectancy_numeric.shape[1])

if len(X) != len(y) or len(X[0]) != len(column_names):
    raise ValueError("Length mismatch in arrays")

df = pd.DataFrame(X, columns=column_names)

df['Target'] = y

correlation_df = df.corr()['Target'].drop('Target')

correlation_df = correlation_df.sort_values(ascending=False)

print(correlation_df)

header = life_expectancy[0]

correlation_coefficients = [pearsonr(life_expectancy_numeric[:, i], y)[0] for i in range(1, life_expectancy_numeric.shape[1])]

correlation_df = pd.DataFrame({'Feature': column_names, 'Correlation': correlation_coefficients})

correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

print(correlation_df)
