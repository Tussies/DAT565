import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '/Users/atosa/Downloads/swedish_population_by_year_and_sex_1860-2022.csv'

df = pd.read_csv(file_path)

df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
df['age_group'] = pd.cut(df['age'], bins=[-1, 14, 64, np.inf], labels=['0-14', '15-64', '65+'])
years_to_plot = [str(year) for year in range(1860, 2023)]
years = []
dependency_ratios = []

for year in years_to_plot:
    grouped_df = df.groupby(['age_group', 'sex']).sum()[year].unstack()

    population_0_14 = grouped_df.loc['0-14'].sum()
    population_15_64 = grouped_df.loc['15-64'].sum()
    population_65_plus = grouped_df.loc['65+'].sum()

    total_dependency_ratio = ((population_0_14 + population_65_plus) / population_15_64) * 100

    years.append(int(year))
    dependency_ratios.append(total_dependency_ratio)

result_df = pd.DataFrame({'year': years, 'dependency_ratio': dependency_ratios})

fig, ax = plt.subplots()
result_df.plot(x='year', y='dependency_ratio', kind='bar', color='r', ax=ax)
ax.set_ylabel('Total Dependency Ratio (%)')
ax.set_xlabel('Year')
ax.set_title('Total Dependency Ratio in Sweden from 1860 to 2022')

ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)

plt.show()
