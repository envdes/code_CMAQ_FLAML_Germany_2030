"""
pop_lu_nc_to_csv.py
===================

Extract the population and the land-use fractions at the modelling grid cells
and store them as CSV.

Population: read from a scenario file, or interpolated between the 2010 and 2020
maps (0.1 / 0.9 weighting for the year 2019, executed in the original notebook).

Land use: the raw plant functional types of the input4MIPs land-state file are
aggregated into five classes:

    pasture   = pastr + range
    primary   = primf + primn
    secondary = secdf + secdn + secmb + secma
    arable    = c3ann + c3nfx + c4ann + c3per + c4per
    urban     = urban
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import pandas as pd
import xarray as xr
import cftime
import numpy as np


# ----------------------------------------------------------------------------
# Population of the target year
# ----------------------------------------------------------------------------
ds = xr.open_dataset('/home/qinjialing/SSP2/Total/NetCDF/ssp2_2030.nc')

# Alternative used in the notebook: interpolate between the 2010 and 2020 maps
# (0.1 / 0.9 weighting gives the year 2019).
# ds1 = xr.open_dataset('/home/qinjialing/SSP2/Total/NetCDF/ssp2_2010.nc')
# ds2 = xr.open_dataset('/home/qinjialing/SSP2/Total/NetCDF/ssp2_2020.nc')
# var_name_1 = list(ds1.data_vars)[0]
# var_name_2 = list(ds2.data_vars)[0]
# result = 0.1 * ds1[var_name_1] + 0.9 * ds2[var_name_2]
# ds = xr.Dataset({'ssp3_2019': result})

# ----------------------------------------------------------------------------
# Modelling grid (stations)
# ----------------------------------------------------------------------------
stations_df = pd.read_csv('../grid_data/grids.csv', index_col=0)
print(stations_df)
lon_stations = stations_df['lon'].values
lat_stations = stations_df['lat'].values
station_names = stations_df['index'].values


# ----------------------------------------------------------------------------
# Nearest-neighbour lookup
# ----------------------------------------------------------------------------
def find_nearest_index(array, value):
    """Return the index of the element of `array` closest to `value`."""
    idx = np.abs(array - value).argmin()
    return idx


# ----------------------------------------------------------------------------
# Extract the population at the nearest grid cell of every site
# ----------------------------------------------------------------------------
lon_nc = ds['lon']
lat_nc = ds['lat']
data_variable = ds['ssp5_2030']

nearest_lon_indices = [find_nearest_index(lon_nc.values, lon) for lon in lon_stations]
nearest_lat_indices = [find_nearest_index(lat_nc.values, lat) for lat in lat_stations]

extracted_data = []
for i in range(len(nearest_lon_indices)):
    lon_idx = nearest_lon_indices[i]
    lat_idx = nearest_lat_indices[i]
    data_value = data_variable.isel(lon=lon_idx, lat=lat_idx).values.flatten()
    extracted_data.append(data_value)

ds.close()

# One row per site.
extracted_data_df = pd.DataFrame(extracted_data, index=station_names, columns=['population'])

print(extracted_data_df)
extracted_data_df.reset_index(inplace=True)

extracted_data_df.to_csv('../grid_data/ssp5/pop_2030.csv')

# ----------------------------------------------------------------------------
# Land use: open the land-state file and aggregate the plant functional types
# ----------------------------------------------------------------------------
ds = xr.open_dataset(
    '/home/qinjialing/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc',
    decode_times=False
)

# The file starts in 2015, so time index 15 is the year 2030.
ds = ds.isel(time=15)
ds['pasture'] = ds['pastr'] + ds['range']
ds['primary'] = ds['primf'] + ds['primn']
ds['secondary'] = ds['secdf'] + ds['secdn'] + ds['secmb'] + ds['secma']
ds['arable'] = ds['c3ann'] + ds['c3nfx'] + ds['c4ann'] + ds['c3per'] + ds['c4per']

# ----------------------------------------------------------------------------
# Urban fraction
# ----------------------------------------------------------------------------
lon_nc = ds['lon']
lat_nc = ds['lat']
data_variable = ds['urban']

nearest_lon_indices = [find_nearest_index(lon_nc.values, lon) for lon in lon_stations]
nearest_lat_indices = [find_nearest_index(lat_nc.values, lat) for lat in lat_stations]

extracted_data = []
for i in range(len(nearest_lon_indices)):
    lon_idx = nearest_lon_indices[i]
    lat_idx = nearest_lat_indices[i]
    data_value = data_variable.isel(lon=lon_idx, lat=lat_idx).values.flatten()
    extracted_data.append(data_value)

ds.close()

df_urban = pd.DataFrame(extracted_data, index=station_names, columns=['urban'])
print(df_urban)

# ----------------------------------------------------------------------------
# Pasture fraction
# ----------------------------------------------------------------------------
lon_nc = ds['lon']
lat_nc = ds['lat']
data_variable = ds['pasture']

nearest_lon_indices = [find_nearest_index(lon_nc.values, lon) for lon in lon_stations]
nearest_lat_indices = [find_nearest_index(lat_nc.values, lat) for lat in lat_stations]

extracted_data = []
for i in range(len(nearest_lon_indices)):
    lon_idx = nearest_lon_indices[i]
    lat_idx = nearest_lat_indices[i]
    data_value = data_variable.isel(lon=lon_idx, lat=lat_idx).values.flatten()
    extracted_data.append(data_value)

ds.close()

df_pasture = pd.DataFrame(extracted_data, index=station_names, columns=['pasture'])
print(df_pasture)

# ----------------------------------------------------------------------------
# Arable fraction
# ----------------------------------------------------------------------------
lon_nc = ds['lon']
lat_nc = ds['lat']
data_variable = ds['arable']

nearest_lon_indices = [find_nearest_index(lon_nc.values, lon) for lon in lon_stations]
nearest_lat_indices = [find_nearest_index(lat_nc.values, lat) for lat in lat_stations]

extracted_data = []
for i in range(len(nearest_lon_indices)):
    lon_idx = nearest_lon_indices[i]
    lat_idx = nearest_lat_indices[i]
    data_value = data_variable.isel(lon=lon_idx, lat=lat_idx).values.flatten()
    extracted_data.append(data_value)

ds.close()

df_arable = pd.DataFrame(extracted_data, index=station_names, columns=['arable'])
print(df_arable)

# ----------------------------------------------------------------------------
# Secondary land fraction
# ----------------------------------------------------------------------------
lon_nc = ds['lon']
lat_nc = ds['lat']
data_variable = ds['secondary']

nearest_lon_indices = [find_nearest_index(lon_nc.values, lon) for lon in lon_stations]
nearest_lat_indices = [find_nearest_index(lat_nc.values, lat) for lat in lat_stations]

extracted_data = []
for i in range(len(nearest_lon_indices)):
    lon_idx = nearest_lon_indices[i]
    lat_idx = nearest_lat_indices[i]
    data_value = data_variable.isel(lon=lon_idx, lat=lat_idx).values.flatten()
    extracted_data.append(data_value)

ds.close()

df_secondary = pd.DataFrame(extracted_data, index=station_names, columns=['secondary'])
print(df_secondary)

# ----------------------------------------------------------------------------
# Primary land fraction
# ----------------------------------------------------------------------------
lon_nc = ds['lon']
lat_nc = ds['lat']
data_variable = ds['primary']

nearest_lon_indices = [find_nearest_index(lon_nc.values, lon) for lon in lon_stations]
nearest_lat_indices = [find_nearest_index(lat_nc.values, lat) for lat in lat_stations]

extracted_data = []
for i in range(len(nearest_lon_indices)):
    lon_idx = nearest_lon_indices[i]
    lat_idx = nearest_lat_indices[i]
    data_value = data_variable.isel(lon=lon_idx, lat=lat_idx).values.flatten()
    extracted_data.append(data_value)

ds.close()

df_primary = pd.DataFrame(extracted_data, index=station_names, columns=['primary'])
print(df_primary)

# ----------------------------------------------------------------------------
# Combine the land-use classes and write the result
# ----------------------------------------------------------------------------
combined_df = pd.concat([df_arable, df_pasture, df_primary, df_secondary, df_urban], axis=1)
combined_df = combined_df.reset_index()
combined_df.to_csv('../grid_data/ssp1/landuse_2030.csv')

combined_df
