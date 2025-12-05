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
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings("ignore", category=RuntimeWarning)

years = ["2000", "2005", "2010", "2020"]
data_dirs_path = "data/data_dirs"

if len(sys.argv) > 1:
    if sys.argv[1] in years:
        years = [sys.argv[1]]
    else:
        print(f"Year {sys.argv[1]} not recognised, defaulting to all years: {years}")

def main(data_dirs_path=data_dirs_path, years = years):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
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

    # countries_shapefile = os.path.join("data", "inputs", "country_data_X", "WB_GAD_ADM0.shp")
    countries_shapefile = os.path.join("data", "inputs", "country_data", "geoBoundariesCGAZ_ADM0.shp")
    # countries_shapefile = os.path.join("data", "inputs", "country_data", "gadm_410.gpkg")

    if not os.path.isfile(countries_shapefile):
        import _get_data
        _get_data.get_country_data()

    countries_data = gpd.read_file(countries_shapefile)

    isoa3_str = "shapeGroup"
    country_isos = countries_data[isoa3_str].unique()


    # functions
    def process_country(country_geom, normalised_data, deltap_dataset): 
        mask = rasterio.features.geometry_mask([country_geom],
                                            out_shape=target_shape,
                                            transform=global_transform,
                                            invert=True)
        raw_weights = np.where(mask, normalised_data, np.nan)
        raw_vals = np.where(mask, deltap_dataset, np.nan)

        valid_indices = ~np.isnan(raw_weights) & ~np.isnan(raw_vals)
        
        if not np.any(valid_indices):
            return np.nan, np.nan, np.nan
        
        weights = raw_weights[valid_indices]
        vals = raw_vals[valid_indices]

        weights_normalised = weights / np.nansum(weights)
        
        mean_value = np.nansum(vals * weights_normalised) # weighted mean

        pixel_count = np.sum(~np.isnan(vals * weights_normalised))

        variance = np.var(vals)
        mean_sem = np.sqrt(variance * np.sum(weights_normalised ** 2))

        return mean_value, mean_sem, pixel_count


    def normalise_spam_data_01(data_array, pixel_areas, target_shape, unit_conv=100):
                """
                # in this instance I use a ones array (cf the pixel areas), the delta-p array is 
                # calculated at 'per-km2'
                # """
                pixel_areas = pixel_areas[np.newaxis, :].T # turn into a column array
                pixel_areas = np.repeat(pixel_areas, target_shape[1], axis=1)
                proportional_output = (data_array / unit_conv) / pixel_areas # (Hectares / 100 = km2) / Area_km2 * 100 = % pixel
                proportional_output = np.where(proportional_output < 0, -1, proportional_output)
                return proportional_output

    # run the thing!
    for year in years:

        output_data = pd.DataFrame()
        output_file = os.path.join(data_dirs_path, "outputs", year, f"processed_results_{year}.csv")

        if not os.path.exists(os.path.join(data_dirs_path, "outputs", year)):
            os.makedirs(os.path.join(data_dirs_path, "outputs", year), exist_ok=True)

        deltap_data = os.path.join(data_dirs_path, year, "deltap_final", "scaled_restore_agriculture_0.25.tif")
        deltap_dataset = rasterio.open(deltap_data)
        band_names = deltap_dataset.descriptions
        band_count = deltap_dataset.count

        total_items = len(data_index[year]['mapspam']) * len(country_isos) * band_count

        with tqdm(total=total_items, desc=f"Calculating delta-p for crops in {year} for {len(country_isos)} countries", unit="item") as pbar:
            
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
                
                sp_totals = pd.read_csv(os.path.join(data_dirs_path, year, "deltap", "restore_agriculture", "0.25", "totals.csv"))
                sp_count = sp_totals.loc[sp_totals['taxa'] == band_name, 'count'].values[0]

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

                    const_array = np.ones_like(pixel_areas)
                    normalised_data = normalise_spam_data_01(item_dataset, const_array, target_shape, unit_conv=100) # using ones here - deltap is already in per-km2, so we just need km2 for the crop vals
                    
                    for iso3 in country_isos:

                        country_geom = countries_data.loc[countries_data[isoa3_str] == iso3.upper(), 'geometry'].values[0]

                        mean_value, mean_sem, pixel_count = process_country(country_geom, normalised_data, band_data)

                        output_data.loc[len(output_data), ["ISO3", "item_name", "band_name", "deltaE_mean", "deltaE_mean_sem", "unit", "pixel_count", "sp_count"]] = [
                            iso3, item_name, band_name, mean_value, mean_sem, "deltaE per km2 per sp.", pixel_count, sp_count]
                        
                        pbar.update(1)

        output_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    main(data_dirs_path=data_dirs_path, years=years)