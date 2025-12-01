import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import rasterio.features
import os
import json
from PixelAreaCalc.main import get_areas
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm # Import tqdm

# years = ["2000", "2005", "2010", "2020"]
years = ["2000"]

# global params
target_shape = (2160, 4320)
global_bounds = (-180.0, -90.0, 180.0, 90.0)
global_transform = from_bounds(*global_bounds, target_shape[1], target_shape[0])
pixel_areas = get_areas(res=(global_transform.a, -global_transform.e),
                                R = 6731.0,
                                bounds = {
                                            "left": global_bounds[0],
                                            "bottom": global_bounds[1],
                                            "right": global_bounds[2],
                                            "top": global_bounds[3]
                                         })

# load global data
with open("data_index.json", 'r') as f:
    data_index = json.load(f)
countries_shapefile = os.path.join("data", "inputs", "country_data", "WB_GAD_ADM0.shp")
countries_data = gpd.read_file(countries_shapefile)
country_isos = countries_data['ISO_A3'].unique() # Get list of countries once

# functions
def process_country(iso3, normalised_data, deltap_dataset):
    country_geom = countries_data.loc[countries_data.ISO_A3 == iso3.upper(), 'geometry'].values[0]
    mask = rasterio.features.geometry_mask([country_geom],
                                           out_shape=target_shape,
                                           transform=global_transform,
                                           invert=True)
    country_masked = np.where(mask, normalised_data, np.nan)
    country_masked_normalised = country_masked / np.nansum(country_masked)
    deltap_masked = np.where(mask, deltap_dataset, np.nan)
    if len(deltap_masked[~np.isnan(deltap_masked)]) == 0 or len(country_masked_normalised[~np.isnan(country_masked_normalised)]) == 0:
        return np.nan, np.nan
    mean_value = np.nanmean(country_masked_normalised * deltap_masked)
    mean_err = np.nanstd(country_masked_normalised * deltap_masked) / np.sqrt(np.sum(~np.isnan(country_masked_normalised * deltap_masked)))   
    return mean_value, mean_err

def normalise_spam_data_01(data_array, pixel_areas, target_shape, unit_conv=100):
             pixel_areas = pixel_areas[np.newaxis, :].T # turn into a column array
             pixel_areas = np.repeat(pixel_areas, target_shape[1], axis=1)
             proportional_output = (data_array / unit_conv) / pixel_areas # (Hectares / 100 = km2) / Area_km2 * 100 = % pixel
             proportional_output = np.where(proportional_output < 0, -1, proportional_output)
             return proportional_output

# run the thing!
for year in years:

    output_data = pd.DataFrame()
    output_file = os.path.join("data", "outputs", year, f"processed_results_{year}.csv")

    if not os.path.exists(os.path.join("data", "outputs", year)):
        os.makedirs(os.path.join("data", "outputs", year), exist_ok=True)

    deltap_data = os.path.join("data", "data_dirs", year, "deltap_final", "scaled_restore_agriculture_0.25.tif")
    deltap_dataset = rasterio.open(deltap_data)
    band_names = deltap_dataset.descriptions
    band_count = deltap_dataset.count

    total_items = len(data_index[year]['mapspam']) * country_isos.shape[0] * band_count

    with tqdm(total=total_items, desc=f"Calculating delta-p for crops in {year}", unit="item") as pbar:
         
        for band_idx in range(1, band_count + 1):

            # allows the processing of different taxa
            band_name = band_names[band_idx - 1]
            band_data = np.zeros(target_shape, dtype=np.float64)
            reproject(
                source=rasterio.band(deltap_dataset, band_idx),
                destination=band_data,
                src_transform=deltap_dataset.transform,
                src_crs=deltap_dataset.crs,
                dst_transform=global_transform,
                dst_crs=deltap_dataset.crs,
                resampling=Resampling.nearest,
                src_nodata=deltap_dataset.nodata,
            )
            
            spam_data = data_index[year]['mapspam']
        
            for item_name, item_index in spam_data.items():
                
                # print(f"Processing year:{year}, taxa:{band_name}, item:{item_name}...")

                item_path = item_index['path']
                
                with rasterio.open(item_path) as src:
                    item_dataset = np.zeros(target_shape, dtype=np.float64)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=item_dataset,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=global_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest,
                        src_nodata=src.nodata,
                    )

                normalised_data = normalise_spam_data_01(item_dataset, pixel_areas, target_shape, unit_conv=100)
                
                for iso3 in country_isos: 
                    mean_value, mean_err = process_country(iso3, normalised_data, band_data)
                    output_data.loc[len(output_data), ["ISO3", "item_name", "band_name", "deltap_mean", "deltap_mean_sem", "unit"]] = [iso3, item_name, band_name, mean_value, mean_err, "deltaP per km2"]
                    
                    pbar.update(1)

    output_data.to_csv(output_file, index=False)
