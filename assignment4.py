import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# Convert to numpy array
life_expectancy_numeric = np.array([[None if val == '' else float(val) for val in row[1:]] for row in life_expectancy[1:]], dtype=np.float64)

# Extract features (X) and target variable (y)
X = life_expectancy_numeric[:, 1:-1]
y = life_expectancy_numeric[:, -1]

# Choose the column index for "Life Expectancy at Birth, both sexes (years)"
life_expectancy_index = -3  # Assuming it's the third-to-last column

# Select the specific column for the x-axis
X_column = X[:, life_expectancy_index]

# Reshape the column for compatibility with LinearRegression
X_column = X_column.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_column, y, test_size=0.25, random_state=42)

# Plot the scatter plot
plt.scatter(X_train, y_train, label='Training Set')
plt.xlabel('Life Expectancy at Birth')
plt.ylabel('Life Expectancy')
plt.title('Scatter Plot and Linear Regression')
plt.legend()

# Perform linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Plot the regression line
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Linear Regression')

plt.legend()
plt.show()