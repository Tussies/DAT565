import tarfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

#How
#tar_file = 'kungalv_slutpriser.tar.gz'
#extract_folder = 'extracted_html'

#with tarfile.open(tar_file, 'r') as tar:
#    tar.extractall(extract_folder)

html_map = 'kungalv_slutpriser/kungalv_slutpris_page_01.html'


with open(html_map, 'r') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')

date_list = list()
for cell in soup.find_all('span', class_='hcl-label hcl-label--state hcl-label--sold-at'):
    date_list.append(cell.text.strip())

print(date_list)

#html_map.find('hcl-label hcl-label--state hcl-label--sold-at')
#html_map.find_all('hcl-label hcl-label--state hcl-label--sold-at')

# date of sale class name = "hcl-label hcl-label--state hcl-label--sold-at"
# Adress class name = "sold-property-listing__heading qa-selling-price-title hcl-card__title"
# Location of the estate class name = "property-icon property-icon--result"
# Area in the form of boarea och rum class name = "sold-property-listing__subheading sold-property-listing__area"
# Area in the form of biarea class name = "listing-card__attribute--normal-weight"
# Area of the plot class name = "sold-property-listing__land-area"
# Closing price class name = "hcl-text hcl-text--medium"

