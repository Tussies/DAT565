import os
import glob
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import locale

locale.setlocale(locale.LC_TIME, 'sv_SE')

html_folder = 'kungalv_slutpriser'

date_of_sale = []
address = []
location = []
living_area = []
room = []
ancillary_areas = []
plot = []
closing_price = []

html_files = glob.glob(os.path.join(html_folder, '*.html'))

for html_file_path in html_files:
    
    with open(html_file_path, 'r') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    for cell in soup.find_all('span', class_='hcl-label hcl-label--state hcl-label--sold-at', string=lambda t: re.compile(r'Såld(.*)').search(t)):
        date_str = re.compile(r'Såld(.*)').search(cell.text).group(1).strip()

        try:
            date_pd = pd.to_datetime(date_str, format='%d %B %Y', errors='coerce')
            date_of_sale.append(date_pd)
        except ValueError:
            date_of_sale.append(None)
    
    for cell in soup.find_all('h2', class_='sold-property-listing__heading qa-selling-price-title hcl-card__title'):
        address.append(cell.text.strip())

    for cell in soup.find_all('div', class_='sold-property-listing__location'): 
        location_text = cell.text.strip()
        index = location_text.find("VillaVilla")
        if index != -1:
            result = location_text[index + len("VillaVilla"):].strip()
            location_result = ' '.join(result.split())
            location.append(location_result)
        else:
            location_result = ' '.join(location_text.split())
            location.append(location_result)

    for cell in soup.find_all('div', class_='sold-property-listing__subheading sold-property-listing__area'):
        area_match = re.search(r'(\d+)', cell.text)
        if area_match:
            living_area.append(area_match.group(1))
        else:
            living_area.append(None)
    
    for cell in soup.find_all('div', class_='sold-property-listing__subheading sold-property-listing__area'):
        room_match = re.search(r'(\d+)\s+rum', cell.text)
        if room_match:
            room.append(room_match.group(1))
        else:
            room.append(None)
    
    for cell in soup.find_all('div', class_='sold-property-listing__subheading sold-property-listing__area'):
        ancillary_area_match = re.search(r'\+\s*(\d+)', cell.text)
        if ancillary_area_match:
            ancillary_areas.append(ancillary_area_match.group(1))
        else:
            ancillary_areas.append(None)
    
    for cell in soup.find_all('div', class_='hcl-flex--container hcl-flex--gap-2 hcl-flex--justify-space-between hcl-flex--md-justify-flex-start'):
        cell2 = cell.find('div', class_='sold-property-listing__land-area')
        cell3 = cell.find('div', class_='hcl-text hcl-text--medium')
        cell4 = cell.find('div', class_='sold-property-listing__subheading sold-property-listing__area')

        if cell2:
            plot_text = ''.join(filter(str.isdigit, cell2.text.strip())).replace("²", "")
            plot.append(plot_text if plot_text else None)
        elif cell3:
            continue
        elif cell4 and not cell2:
            plot.append(None)
    
    for cell in soup.find_all('span', class_='hcl-text hcl-text--medium'):
        closing_price_text = cell.text.strip()

        cleaned_price = re.sub(r'Slutpris|kr|\D', '', closing_price_text)

        if "%" not in closing_price_text:
            closing_price.append(float(cleaned_price) if cleaned_price else None)

max_length = max(len(date_of_sale), len(address), len(location), len(living_area), len(room), len(ancillary_areas), len(plot), len(closing_price))

lists_to_fill = [date_of_sale, address, location, living_area, room, ancillary_areas, plot, closing_price]

for lst in lists_to_fill:
    lst += [None] * (max_length - len(lst))

data_list = [
    {'Date of Sale': dte, 'Address': addr, 'Location': loc, 'Living Area (m²)': are, 'Rooms': rm, 'Ancillary Areas (m²)': anc_area, 'Plot (m²)': plt, 'Closing Price (kr)': pr}
    for dte, addr, loc, are, rm, anc_area, plt, pr in zip(date_of_sale, address, location, living_area, room, ancillary_areas, plot, closing_price)
]

df = pd.DataFrame({
    'Date of Sale': date_of_sale,
    'Address': address,
    'Location': location,
    'Living Area (m²)': living_area,
    'Rooms': room,
    'Ancillary Areas (m²)': ancillary_areas,
    'Plot (m²)': plot,
    'Closing Price (kr)': closing_price
})

numeric_columns = ['Living Area (m²)', 'Rooms', 'Ancillary Areas (m²)', 'Plot (m²)', 'Closing Price (kr)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

df['Total Area (m²)'] = df['Living Area (m²)'] + df['Ancillary Areas (m²)']

csv_file_path = 'output_data.csv'
df.to_csv(csv_file_path, index=False)

print(f'Data has been saved to {csv_file_path}')

df_2022 = df[df['Date of Sale'].dt.year == 2022]

closing_price_summary_2022 = df_2022['Closing Price (kr)'].describe()
pd.set_option('display.float_format', lambda x: f'{x:.0f}')

print("Five-Number Summary of Closing Prices for 2022:")
print(closing_price_summary_2022)

plt.figure()
plt.hist(df_2022['Closing Price (kr)'], bins=50, color='red', edgecolor='black')
plt.xscale('linear')
plt.ticklabel_format(style='plain', axis='x')
plt.title('Histogram of Closing Prices Year 2022')
plt.xlabel('Closing Price (kr)')
plt.ylabel('Frequency')
plt.xticks(np.arange(0, df_2022['Closing Price (kr)'].max() + 1000000, 2000000))
plt.yticks(np.arange(0, 16, 2))
plt.savefig('histogram_plot.pdf', format='pdf')

plt.figure(figsize=(10, 6))
plt.scatter(df_2022['Living Area (m²)'], df_2022['Closing Price (kr)'], alpha=0.5)
plt.yscale('linear')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Closing Price and Living Area Year 2022')
plt.xlabel('Living Area (m²)')
plt.ylabel('Closing Price (kr)')
plt.yticks(np.arange(0, df_2022['Closing Price (kr)'].max() + 1000000, 1000000))
plt.savefig('scatter_plot.pdf', format='pdf')

plt.figure(figsize=(10, 6))
ax = plt.axes()
ax.set_facecolor("gray")
scatter_plot_2022 = plt.scatter(
    df_2022['Living Area (m²)'],
    df_2022['Closing Price (kr)'],
    c=df_2022['Rooms'],
    cmap='hot',  
    alpha=0.8
)
cbar = plt.colorbar(scatter_plot_2022)
cbar.set_label('Number of Rooms')
plt.yscale('linear')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Closing Price and Living Area Colorized by the Number of Rooms Year 2022')
plt.xlabel('Living Area (m²)')
plt.ylabel('Closing Price (kr)')
plt.yticks(np.arange(0, df_2022['Closing Price (kr)'].max() + 1000000, 1000000))
plt.savefig('colorized_scatter_plot.pdf', format='pdf')

plt.show()
