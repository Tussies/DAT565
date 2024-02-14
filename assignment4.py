import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import r_regression

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
life_expectancy_numeric = np.array([[0 if val == '' else float(val) for val in row[1:]] for row in life_expectancy], dtype=np.float64)

X = life_expectancy_numeric[:, 1:-1]
y = life_expectancy_numeric[:, -1]

LifeTrain, LifeTest = train_test_split(X, test_size=0.25, random_state=42)

r_values, p_values = r_regression(X, y)
strongest_feature_index = np.argmax(np.abs(r_values))

strongest_feature_name = header[strongest_feature_index + 1]
strongest_r_value = r_values[strongest_feature_index]

print(f"The variable '{strongest_feature_name}' has the strongest linear relationship with the target variable.")
print(f"R-value: {strongest_r_value}")
