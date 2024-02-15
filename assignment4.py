import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

X = life_expectancy_numeric[:, 1:-1]
y = life_expectancy_numeric[:, -1]

LifeTrain, LifeTest, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

columns = header[2:-4] + header[-3:]

print("Shape of LifeTrain:", LifeTrain.shape)
print("Number of columns specified:", len(columns))

df_train = pd.DataFrame(data=np.hstack((LifeTrain, y_train.reshape(-1, 1))), columns=columns + ['Life Expectancy at Birth, both sexes (years)'])

correlation_coefficients = df_train.corr()['Life Expectancy at Birth, both sexes (years)'][:-1]
max_correlation_variable = correlation_coefficients.abs().idxmax()
max_correlation_coefficient = correlation_coefficients.abs().max()

print(f"The variable '{max_correlation_variable}' has the strongest linear relationship with the target variable.")
print(f"Pearson correlation coefficient: {max_correlation_coefficient}")
