import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# Example: Fetching HTML content from a URL
import requests

url = 'kungalv_slutpriser.tar.gz'
response = requests.get(url)
html_content = response.text

print(html_content)