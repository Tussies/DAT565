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

header = life_expectancy[0]

print(X.shape)

correlation_coefficients = [pearsonr(X[:, i], y)[0] for i in range(0, X.shape[1])]

correlation_df = pd.DataFrame({'Feature': column_names, 'Correlation': correlation_coefficients})

correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

print(correlation_df)

X_negative = X[:, column_names.index('Crude Birth Rate (births per 1,000 population)')]
X_negative = X_negative.reshape(-1, 1)

# Separate training and testing sets for negative case
X_train_negative, X_test_negative, y_train_negative, y_test_negative = train_test_split(X_negative, y, test_size=0.25, random_state=42)
model_negative = LinearRegression()
model_negative.fit(X_test_negative, y_test_negative)
y_pred_negative = model_negative.predict(X_test_negative)

# Plot for negative case
plt.scatter(X_test_negative, y_test_negative, color='black', label='Actual Data')
plt.plot(X_test_negative, y_pred_negative, color='blue', linewidth=3, label='Linear Regression Model')
plt.xlabel('Crude Birth Rate (births per 1,000 population)')
plt.ylabel('Life Expectancy')
plt.title('Linear Regression Model (test set)')
plt.legend()

# Train the negative model
model_negative = LinearRegression()
model_negative.fit(X_test_negative, y_test_negative)

# Calculate R² for the negative model
r_squared_negative = model_negative.score(X_test_negative, y_test_negative)

# Get the coefficients and intercept for the negative model
coefficients_negative = model_negative.coef_
intercept_negative = model_negative.intercept_

# Print the results for the negative case

print("Coefficient of determination (R²):", r_squared_negative)
print("Coefficient of model:", coefficients_negative)
print("Intercept:", intercept_negative)

plt.show()



