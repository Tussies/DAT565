import os
import glob
from bs4 import BeautifulSoup
from itertools import zip_longest
import pandas as pd
import re

# Directory containing HTML files
html_folder = 'kungalv_slutpriser'

# Lists to store all information
date_of_sale = []
address = []
location = []
living_area = []
room = []
ancillary_areas = []
plot = []
closing_price = []

# Use glob to get a list of all HTML files in the directory
html_files = glob.glob(os.path.join(html_folder, '*.html'))

# Iterate through each HTML file
for html_file_path in html_files:
    # Read the HTML content
    with open(html_file_path, 'r') as file:
        html_content = file.read()

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all dates and append to the list
    for cell in soup.find_all('span', class_='hcl-label hcl-label--state hcl-label--sold-at', string=lambda t: re.compile(r'Såld(.*)').search(t)):
        date_of_sale.append(re.compile(r'Såld(.*)').search(cell.text).group(1).strip())
    
    # Find all addresses and append to the list
    for cell in soup.find_all('h2', class_='sold-property-listing__heading qa-selling-price-title hcl-card__title'):
        address.append(cell.text.strip())

    # Find all locations and append to the list
    for cell in soup.find_all('div', class_='sold-property-listing__location'): 
        location_text = cell.text.strip()
        index = location_text.find("VillaVilla")
        if index != -1:
            result = location_text[index + len("VillaVilla"):].strip()
            location.append(result)
        else:
            location.append(location_text)

    # Find all living areas and append to the list
    for cell in soup.find_all('div', class_='sold-property-listing__subheading sold-property-listing__area'):
        area_match = re.search(r'(\d+)', cell.text)
        if area_match:
            living_area.append(area_match.group(1))
        else:
            living_area.append(None)
    
    # Find all rooms and append to the list
    for cell in soup.find_all('div', class_='sold-property-listing__subheading sold-property-listing__area'):
        room_match = re.search(r'(\d+)\s+rum', cell.text)
        if room_match:
            room.append(room_match.group(1))
        else:
            room.append(None)
    
    # Find all ancillary areas and append to the list
    for cell in soup.find_all('span', class_='listing-card__attribute--normal-weight'):
        ancillary_areas.append(cell.text.strip())
    
    # Find all plots and append to the list
    for cell in soup.find_all('div', class_='sold-property-listing__land-area'):
        plot.append(cell.text.strip())
    
    # Find all closing prices and append to the list
    for cell in soup.find_all('span', class_='hcl-text hcl-text--medium'):
        closing_price.append(cell.text.strip())

# Find the maximum length among all lists
max_length = max(len(date_of_sale), len(address), len(location), len(living_area), len(room), len(ancillary_areas), len(plot), len(closing_price))

# Use zip_longest to create a list of tuples with missing values filled with NaN
data_tuples = zip_longest(date_of_sale, address, location, living_area, room, ancillary_areas, plot, closing_price, fillvalue=None)

# Create a DataFrame using the list of tuples
df = pd.DataFrame(data_tuples, columns=['Date of Sale', 'Address', 'Location', 'Living Area', 'Room', 'Ancillary Areas()', 'Plot', 'Closing Price'])

# Save DataFrame to CSV file
csv_file_path = 'output_data.csv'
df.to_csv(csv_file_path, index=False)

# Print a message indicating successful CSV creation
print(f'Data has been saved to {csv_file_path}')


#html_map.find('hcl-label hcl-label--state hcl-label--sold-at')
#html_map.find_all('hcl-label hcl-label--state hcl-label--sold-at')

# date of sale class name = "hcl-label hcl-label--state hcl-label--sold-at"
# Adress class name = "sold-property-listing__heading qa-selling-price-title hcl-card__title"
# Location of the estate class name = "property-icon property-icon--result"
# Area in the form of boarea och rum class name = "sold-property-listing__subheading sold-property-listing__area"
# Area in the form of biarea class name = "listing-card__attribute--normal-weight"
# Area of the plot class name = "sold-property-listing__land-area"
# Closing price class name = "hcl-text hcl-text--medium"

#hello

