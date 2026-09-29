"""
CMIP6_nc_to_csv.py
==================

Extract a CMIP6 monthly variable at the modelling grid cells and store it as a
long-format CSV (sites, time, <variable>).
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import pandas as pd
import xarray as xr
import cftime
import numpy as np


# ----------------------------------------------------------------------------
# Variable and year to extract
# ----------------------------------------------------------------------------
spec = 'bldep'
year = '2030'

# ----------------------------------------------------------------------------
# Open the CMIP6 file
# ----------------------------------------------------------------------------
ds = xr.open_dataset(
    f'/home/shared/Data/CMIP6/Meteorology/{spec}/{spec}_AERmon_CanESM5_ssp126_r1i1p1f1_gn_201501-210012.nc'
)

# ----------------------------------------------------------------------------
# Keep the requested year and convert the cftime axis to datetime64
# ----------------------------------------------------------------------------
ds_2018 = ds.sel(time=ds.time.dt.year == int(year))

# Original (cftime) time coordinate.
cftime_array = ds_2018['time'].values

# Convert the cftime objects to numpy datetime64 objects.
datetime_array = np.array([np.datetime64(dt) for dt in cftime_array])

# Assign the new datetime64 array to the time coordinate.
ds_2018['time'] = datetime_array

# Convert to pandas datetime type.
ds_2018['time'] = xr.DataArray(pd.to_datetime(ds_2018['time'].values), dims='time')

print(ds_2018)

# ----------------------------------------------------------------------------
# Modelling grid (stations)
# ----------------------------------------------------------------------------
station = pd.read_csv('../grid_data/grids.csv', index_col=0)

stations_df = station
# Site (grid cell) ids, longitudes and latitudes.
station_names = stations_df['index'].values
lon_stations = stations_df['lon'].values
lat_stations = stations_df['lat'].values


# ----------------------------------------------------------------------------
# Nearest-neighbour lookup
# ----------------------------------------------------------------------------
def find_nearest_index(array, value):
    """Return the index of the element of `array` closest to `value`."""
    idx = np.abs(array - value).argmin()
    return idx


# ----------------------------------------------------------------------------
# Extract the variable at the nearest grid cell of every site
# ----------------------------------------------------------------------------
lon_nc = ds_2018['lon']
lat_nc = ds_2018['lat']
data_variable = ds_2018[spec]

# Nearest grid indices of every site.
nearest_lon_indices = [find_nearest_index(lon_nc.values, lon) for lon in lon_stations]
nearest_lat_indices = [find_nearest_index(lat_nc.values, lat) for lat in lat_stations]

# Number of time steps and number of sites.
time_steps = len(ds_2018['time'])
station_num = len(station_names)

# Container of the extracted data, shape = (sites, time steps).
extracted_data = np.empty((station_num, time_steps))

# Loop over time steps and sites.
for t in range(time_steps):
    for i in range(station_num):
        lon_idx = nearest_lon_indices[i]
        lat_idx = nearest_lat_indices[i]
        data_value = data_variable.isel(time=t, lon=lon_idx, lat=lat_idx).values
        extracted_data[i, t] = data_value

# Wide table: rows = sites, columns = time.
df_spec = pd.DataFrame(extracted_data, index=station_names,
                       columns=pd.to_datetime(ds_2018['time'].values))

ds_2018.close()

print(df_spec)

# ----------------------------------------------------------------------------
# Reshape the wide table into the long format (sites, time, variable)
# ----------------------------------------------------------------------------
import pandas as pd

# Column names of the wide table are the time steps.
time_columns = df_spec.columns.tolist()

# Store the reshaped records.
new_data = []
for index, row in df_spec.iterrows():
    for col_idx, time in enumerate(time_columns):
        value = row[col_idx]
        new_data.append([index, time, value])

# Build the long-format table.
new_df = pd.DataFrame(new_data, columns=['sites', 'time', f'{spec}'])

print(new_df)

# ----------------------------------------------------------------------------
# Write the result
# ----------------------------------------------------------------------------
new_df.to_csv(f'../grid_data/ssp1/{spec}_{year}.csv')
