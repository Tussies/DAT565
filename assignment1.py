
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




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

#print(sum_of_children_rows[0])
#print(sum_of_elderly_rows[0])
#print(sum_of_workforce_rows[0])

#print(len(sum_of_children_rows))


sum_of_all = []
#loop
for i in range(len(sum_of_children_rows)):
    sum_of_all.append(100*(sum_of_children_rows[i] + sum_of_elderly_rows[i])/sum_of_workforce_rows)

#print(sum_of_all[0])

year_array = np.arange(1860, 2023)

#print(len(sum_of_all))
#print(len(year_array))


#print(year_array)


plt.scatter(year_array, sum_of_all)


# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Dependency ratio')
plt.title('Dependency ratio of Sweden between 1860 to 2022')

# Display the plot
plt.show()


