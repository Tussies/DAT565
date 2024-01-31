import tarfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


html_map = 'kungalv_slutpriser/kungalv_slutpris_page_01.html'


with open(html_map, 'r') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')

dates = list()
for cell in soup.find_all('span', class_='hcl-label hcl-label--state hcl-label--sold-at'):
    dates.append(cell.text.strip())

print(dates)

soup.find('span', class_='row')

import re
element = soup.find(string=re.compile(r'.*1423,00.*'))

table = element.parent.parent.parent.parent.parent
table.dates

table.find('span', class_='row')

for cell in table.find('span', class_='row').select('span'):
    print(cell.text.strip())

values = list()
for row in table.find_all('span', class_='row'):
    values.append([cell.text.strip() for cell in row.select('td')])

values = [list(map(lambda s: s.replace('\xa0','').replace(',','.'),row)) for row in values]
values = [list(map(float,row[:3])) + [int(row[3])] + \
          list(map(float,row[4:6])) + [int(row[6])] + [row[7]] for row in values]

data = list()
for (name,val) in zip(dates,values):
    row = { 'Date' : date,
            'Adress' : val[0],
            'Location' : val[1],
            'Boarea' : val[2],
            'Biarea' : val[3],
            'Rum' : val[4],
            'Plot' : val[5],
            'Closing price' : val[6]
          }
    data.append(row)
data = pd.DataFrame(data)


# date of sale class name = "hcl-label hcl-label--state hcl-label--sold-at"
# Adress class name = "sold-property-listing__heading qa-selling-price-title hcl-card__title"
# Location of the estate class name = "property-icon property-icon--result"
# Area in the form of boarea och rum class name = "sold-property-listing__subheading sold-property-listing__area"
# Area in the form of biarea class name = "listing-card__attribute--normal-weight"
# Area of the plot class name = "sold-property-listing__land-area"
# Closing price class name = "hcl-text hcl-text--medium"

