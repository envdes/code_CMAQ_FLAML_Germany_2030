"""
ERA5_nc_to_csv.py
=================

Extract a daily ERA5 variable at the modelling grid cells and store it as a CSV
with the columns: index (grid id), <variable>, time.
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import datetime

import pandas as pd
import xarray as xr
import numpy as np


# ----------------------------------------------------------------------------
# Nearest-neighbour lookup
# ----------------------------------------------------------------------------
def find_nearest_index(array, value):
    """Return the index of the element of `array` closest to `value`."""
    idx = np.abs(array - value).argmin()
    return idx


# ----------------------------------------------------------------------------
# Configuration: one ERA5 file per day
# ----------------------------------------------------------------------------
nc_file_path_template = '/home/shared/Data/ERA5_daily/ERA5_Daily_{}.nc'
data_variable_name = 'blh'
start_date = '2018-01-04'
end_date = '2018-12-31'
date_format = '%Y-%m-%d'

# ----------------------------------------------------------------------------
# Modelling grid (stations)
# ----------------------------------------------------------------------------
stations_df = pd.read_csv('../grid_data/grids.csv', index_col=0)
lon_stations = stations_df['lon'].values
lat_stations = stations_df['lat'].values
station_names = stations_df['index'].values

# ----------------------------------------------------------------------------
# Loop over the days and extract the variable at every site
# ----------------------------------------------------------------------------
all_data_df = pd.DataFrame()

start = datetime.datetime.strptime(start_date, date_format)
end = datetime.datetime.strptime(end_date, date_format)
current_date = start

while current_date <= end:
    current_date_str = current_date.strftime('%Y-%m-%d')

    # Open the NetCDF file of the current day.
    nc_file_path = nc_file_path_template.format(current_date_str)
    ds = xr.open_dataset(nc_file_path)

    lon_nc = ds['longitude']
    lat_nc = ds['latitude']
    data_variable = ds[data_variable_name]

    # Nearest grid indices of every site.
    nearest_lon_indices = [find_nearest_index(lon_nc.values, lon) for lon in lon_stations]
    nearest_lat_indices = [find_nearest_index(lat_nc.values, lat) for lat in lat_stations]

    # Extract the value of every site.
    extracted_data = []
    for i in range(len(nearest_lon_indices)):
        lon_idx = nearest_lon_indices[i]
        lat_idx = nearest_lat_indices[i]
        data_value = data_variable.isel(longitude=lon_idx, latitude=lat_idx).values.flatten()
        extracted_data.append(data_value)

    # Close the dataset to release the resources.
    ds.close()

    # One row per site, with the current day as the time column.
    df = pd.DataFrame(extracted_data, index=station_names, columns=[data_variable_name])
    df['time'] = current_date_str
    df = df.reset_index()

    # Append the day to the total table.
    all_data_df = pd.concat([all_data_df, df], ignore_index=True)

    # Next day.
    current_date += datetime.timedelta(days=1)

print(all_data_df)

# ----------------------------------------------------------------------------
# Write the result
# ----------------------------------------------------------------------------
all_data_df.to_csv('../grid_data/basic_data/blh_2018.csv')
