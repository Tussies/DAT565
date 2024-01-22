
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import linregress
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split



population = pd.read_csv('swedish_population_by_year_and_sex_1860-2022.csv')

population.head()


population_array = np.array(population)

#print(population_array)

modified_population_array = np.delete(population_array, 1, 1)

modified_population_array2 = np.delete(modified_population_array, 0, 1)

#print(modified_population_array2)

children_rows_to_add = slice(0, 30)
workforce_rows_to_add = slice(31, 130)
elderly_rows_to_add = slice(131, 222)
sum_of_children_rows = np.sum(modified_population_array2[children_rows_to_add], axis=0)
sum_of_workforce_rows = np.sum(modified_population_array2[workforce_rows_to_add], axis=0)
sum_of_elderly_rows = np.sum(modified_population_array2[elderly_rows_to_add], axis=0)

print(sum_of_children_rows[0])
print(sum_of_elderly_rows[0])
print(sum_of_workforce_rows[0])

print(len(sum_of_children_rows))


sum_of_all = []
#loop
for i in range(len(sum_of_children_rows)):
    sum_of_all.append(100*(sum_of_children_rows[i] + sum_of_elderly_rows[i])/sum_of_workforce_rows)

print(sum_of_all[0])

year_array = np.arange(1860, 2023)

print(year_array)




