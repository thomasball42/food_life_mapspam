import rasterio
import os
import json
from PixelAreaCalc.main import get_pixel_area

years = ["2000", "2010", "2015", "2020"]
years = ["2000"]

with open("data_index.json", 'r') as f:
    data_index = json.load(f)

year = "2000"

year_data = data_index[year]

spam_data = year_data['mapspam']

crops = spam_data.keys()
