import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import r_regression

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

file_path = 'life_expectancy.csv'
life_expectancy = read_csv_file(file_path)

LifeTrain, LifeTest = train_test_split(life_expectancy, test_size=0.25, random_state=42)

print(LifeTest)

plt.scatter(X, y)
plt.show()