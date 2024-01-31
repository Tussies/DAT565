import tarfile
import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

#How
#tar_file = 'kungalv_slutpriser.tar.gz'
#extract_folder = 'extracted_html'

#with tarfile.open(tar_file, 'r') as tar:
#    tar.extractall(extract_folder)

html_folder = 'kungalv_slutpriser'

# List to store all dates
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
    for cell in soup.find_all('span', class_='hcl-label hcl-label--state hcl-label--sold-at'):
        date_of_sale.append(cell.text.strip())
    
    # Find all addresses and append to the list
    for cell in soup.find_all('h2', class_='sold-property-listing__heading qa-selling-price-title hcl-card__title'):
        address.append(cell.text.strip())

    # Find all locations and append to the list
    for cell in soup.find_all('div', class_='sold-property-listing__location'):
        location.append(cell.text.strip())

    # Find all living areas and append to the list
    for cell in soup.find_all('div', class_='sold-property-listing__subheading sold-property-listing__area'):
        living_area.append(cell.text.strip())
    
    # Find all rooms and append to the list
    for cell in soup.find_all('div', class_='sold-property-listing__subheading sold-property-listing__area'):
        room.append(cell.text.strip())
    
    # Find all ancillary areas and append to the list
    for cell in soup.find_all('span', class_='listing-card__attribute--normal-weight'):
        ancillary_areas.append(cell.text.strip())
    
    # Find all plots and append to the list
    for cell in soup.find_all('div', class_='sold-property-listing__land-area'):
        plot.append(cell.text.strip())
    
    # Find all closing prices and append to the list
    for cell in soup.find_all('span', class_='hcl-text hcl-text--medium'):
        closing_price.append(cell.text.strip())



# Print the list of all dates
print(closing_price)

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

