import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# task i:

# file_path = '/Users/atosa/Downloads/swedish_population_by_year_and_sex_1860-2022.csv'
# df = pd.read_csv(file_path)

# df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
# df['age_group'] = pd.cut(df['age'], bins=[-1, 14, 64, np.inf], labels=['0-14', '15-64', '65+'])

# unpivoted_df = pd.melt(df, id_vars=['age_group', 'sex'], var_name='Year', value_name='Population')
# unpivoted_df['Year'] = pd.to_numeric(unpivoted_df['Year'], errors='coerce')
# grouped_df = unpivoted_df.groupby(['Year', 'age_group', 'sex'])['Population'].sum().reset_index()
# dependency_df = grouped_df.pivot_table(index='Year', columns='age_group', values='Population', aggfunc='sum')

# condition_0_14 = dependency_df['0-14']
# condition_65_plus = dependency_df['65+']
# condition_15_64 = dependency_df['15-64']

# dependency_df['dependency_ratio'] = ((condition_0_14 + condition_65_plus) / condition_15_64) * 100

# dependency_df.index = dependency_df.index.astype(int)

# fig, ax = plt.subplots()
# dependency_df['dependency_ratio'].plot(kind='bar', color='r', ax=ax)
# ax.tick_params(axis='x', labelsize=5)
# ax.set_ylabel('Dependency Ratio (%)')
# ax.set_xlabel('Year')
# ax.set_title('Dependency Ratio in Sweden from 1860 to 2022')

# plt.show()

# task ii:

file_path = '/Users/atosa/Downloads/swedish_population_by_year_and_sex_1860-2022.csv'
df = pd.read_csv(file_path)

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df.dropna(subset=['age'])
df = pd.melt(df, id_vars=['age', 'sex'], var_name='Year', value_name='Population')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['age_group'] = pd.cut(df['age'], bins=[-1, 14, 64, np.inf], labels=['0-14', '15-64', '65+'])


years_to_plot = list(range(1860, 2023))
filtered_df = df[df['Year'].isin(years_to_plot)]
grouped_df = filtered_df.groupby(['Year', 'age_group'])['Population'].sum().unstack()

total_population = grouped_df.sum(axis=1)
children_fraction = (grouped_df['0-14'] / total_population) * 100
elderly_fraction = (grouped_df['65+'] / total_population) * 100
total_dependency_fraction = ((grouped_df['0-14'] + grouped_df['65+']) / total_population) * 100

result_df = pd.DataFrame({
    'Year': years_to_plot,
    'Children fraction': children_fraction.values,
    'Elderly fraction': elderly_fraction.values,
    'Total dependency fraction': total_dependency_fraction.values
})

fig, ax = plt.subplots()
result_df.plot(x='Year', y=['Children fraction', 'Elderly fraction', 'Total dependency fraction'],
               kind='bar', stacked=True, colormap='viridis', ax=ax)
ax.set_ylabel('Fraction of Population (%)')
ax.set_xlabel('Year')
ax.set_title('Fraction of Children, Elderly, and Total Dependency Population in Sweden from 1860 to 2022')
ax.tick_params(axis='x', labelsize=5)
plt.legend(title='Age Group', loc='upper left')
plt.show()


