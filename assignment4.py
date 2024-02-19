import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE

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

variable_to_exclude = 'Crude Birth Rate (births per 1,000 population)'
index_to_exclude = column_names.index(variable_to_exclude)
X_excluded = np.delete(X, index_to_exclude, axis=1)
column_names_excluded = column_names[:index_to_exclude] + column_names[index_to_exclude+1:]

print("Shape of X (after exclusion):", X_excluded.shape)
print("Column names (after exclusion):", column_names_excluded)

X_negative = X[:, column_names.index('Median Age, as of 1 July (years)')]
X_negative = X_negative.reshape(-1, 1)

X_train_negative, X_test_negative, y_train_negative, y_test_negative = train_test_split(X_negative, y, test_size=0.25, random_state=42)
model_negative = LinearRegression()
model_negative.fit(X_test_negative, y_test_negative)
y_pred_negative = model_negative.predict(X_test_negative)


plt.scatter(X_test_negative, y_test_negative, color='black', label='Actual Data')
plt.plot(X_test_negative, y_pred_negative, color='blue', linewidth=3, label='Linear Regression Model')
plt.xlabel('Median Age, as of 1 July (years)')
plt.ylabel('Life Expectancy')
plt.title('Linear Regression Model (test set)')
plt.legend()

model_negative = LinearRegression()
model_negative.fit(X_test_negative, y_test_negative)
r_squared_negative = model_negative.score(X_test_negative, y_test_negative)

coefficients_negative = model_negative.coef_
intercept_negative = model_negative.intercept_

print("Coefficient of determination (R²):", r_squared_negative)
print("Coefficient of model:", coefficients_negative)
print("Intercept:", intercept_negative)

plt.show()

selected_variable_name = 'Median Age, as of 1 July (years)'
selected_variable_index = column_names.index(selected_variable_name)

plt.scatter(X_test_negative, y_test_negative, color='black', label='Actual Data')
plt.xlabel(selected_variable_name)
plt.ylabel('Life Expectancy')
plt.title('Original Relationship')
plt.show()

X_transformed = np.sqrt(X_test_negative)

plt.scatter(X_transformed, y_test_negative, color='black', label='Actual Data')
plt.xlabel(selected_variable_name + ' (Square Root Transformed)')
plt.ylabel('Life Expectancy')
plt.title('Transformed Relationship (Square Root Transformation)')
plt.show()

correlation_before_transformation_sqrt_list = [pearsonr(X_test_negative[:, i], y_test_negative)[0] for i in range(X_test_negative.shape[1])]
correlation_after_transformation_sqrt_list = [pearsonr(X_transformed[:, i], y_test_negative)[0] for i in range(X_transformed.shape[1])]

print("Pearson correlation coefficients before transformation (square root):", correlation_before_transformation_sqrt_list)
print("Pearson correlation coefficients after transformation (square root):", correlation_after_transformation_sqrt_list)

estimator_rfe = LinearRegression()
k_features = 9
selector_rfe = RFE(estimator_rfe, n_features_to_select=k_features)
X_selected_rfe = selector_rfe.fit_transform(X, y)
selected_feature_indices_rfe = np.where(selector_rfe.support_)[0]

X_train_selected_rfe, X_test_selected_rfe, y_train_selected_rfe, y_test_selected_rfe = train_test_split(X_selected_rfe, y, test_size=0.25, random_state=42)

model_selected_rfe = LinearRegression()
model_selected_rfe.fit(X_train_selected_rfe, y_train_selected_rfe)
y_pred_selected_rfe = model_selected_rfe.predict(X_test_selected_rfe)

print("\nModel with Selected Features (RFE):")
print("Selected Feature Indices (RFE):", selected_feature_indices_rfe)
print("Coefficient of Determination (R²):", model_selected_rfe.score(X_test_selected_rfe, y_test_selected_rfe))
print("Coefficients:", model_selected_rfe.coef_)
print("Intercept:", model_selected_rfe.intercept_)
print("Pearson Correlation Coefficient:", pearsonr(y_pred_selected_rfe, y_test_selected_rfe)[0])
print("MSE:", mean_squared_error(y_test_selected_rfe, y_pred_selected_rfe))

plt.show()

selected_feature_names = [column_names[i - 1] for i in selected_feature_indices_rfe]

print("Selected Feature Names (RFE):", selected_feature_names)
