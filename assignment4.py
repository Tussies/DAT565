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


LifeTrain, LifeTest = train_test_split(X, test_size=0.25, random_state=42)

print(train_test_split)
