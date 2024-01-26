import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# task i:

file_path = '/Users/atosa/Downloads/swedish_population_by_year_and_sex_1860-2022.csv'

df = pd.read_csv(file_path)

df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
df['age_group'] = pd.cut(df['age'], bins=[-1, 14, 64, np.inf], labels=['0-14', '15-64', '65+'])

unpivoted_df = pd.melt(df, id_vars=['age_group', 'sex'], var_name='Year', value_name='Population')

unpivoted_df['Year'] = pd.to_numeric(unpivoted_df['Year'], errors='coerce')


grouped_df = unpivoted_df.groupby(['Year', 'age_group', 'sex'])['Population'].sum().reset_index()
dependency_df = grouped_df.pivot_table(index='Year', columns='age_group', values='Population', aggfunc='sum')

condition_0_14 = dependency_df['0-14']
condition_65_plus = dependency_df['65+']
condition_15_64 = dependency_df['15-64']

dependency_df['dependency_ratio'] = ((condition_0_14 + condition_65_plus) / condition_15_64) * 100

fig, ax = plt.subplots()
dependency_df['dependency_ratio'].plot(kind='bar', color='r', ax=ax)
ax.set_ylabel('Total Dependency Ratio (%)')
ax.set_xlabel('Year')
ax.set_title('Total Dependency Ratio in Sweden from 1860 to 2022')

plt.show()

# task ii:

# file_path = '/Users/atosa/Downloads/swedish_population_by_year_and_sex_1860-2022.csv'

# df = pd.read_csv(file_path)


# df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
# df['age_group'] = pd.cut(df['age'], bins=[-1, 14, 64, np.inf], labels=['0-14', '15-64', '65+'])

# years_to_plot = [str(year) for year in range(1860, 2023)]
# years = []
# children_fraction = []
# elderly_fraction = []
# total_dependency_fraction = []

# for year in years_to_plot:
#     grouped_df = df.groupby('age_group').sum()[year]

#     population_0_14 = grouped_df.loc['0-14']
#     population_15_64 = grouped_df.loc['15-64']
#     population_65_plus = grouped_df.loc['65+']

#     total_population = population_0_14 + population_15_64 + population_65_plus

#     children_fraction.append(population_0_14 / total_population)
#     elderly_fraction.append(population_65_plus / total_population)
#     total_dependency_fraction.append((population_0_14 + population_65_plus) / total_population)

#     years.append(int(year))

# result_df = pd.DataFrame({
#     'year': years,
#     'children_fraction': children_fraction,
#     'elderly_fraction': elderly_fraction,
#     'total_dependency_fraction': total_dependency_fraction
# })

# fig, ax = plt.subplots()
# result_df.plot(x='year', y=['children_fraction', 'elderly_fraction', 'total_dependency_fraction'],
#                kind='bar', stacked=True, colormap='viridis', ax=ax)
# ax.set_ylabel('Fraction of Population')
# ax.set_xlabel('Year')
# ax.set_title('Fraction of Children, Elderly, and Total Dependency Population in Sweden (1860-2022)')
# plt.legend(title='Age Group', loc='upper left')
# plt.show()
