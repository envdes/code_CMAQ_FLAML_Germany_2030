import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from rasterio.io import MemoryFile
from rasterio.mask import mask
from shapely.ops import unary_union


# Paths
GRIDS_CSV = '/home/qinjialing/grid_data/grids.csv'
NEW_MODEL_DIR = '/home/qinjialing/new_model'
MAP_DIR = '/home/qinjialing/map'
SSP2_TOTAL_NC = '/home/qinjialing/SSP2/Total/NetCDF'
SSP5_TOTAL_NC = '/home/qinjialing/SSP5/Total/NetCDF'

# Health impact parameters
BASE_MORTALITY = 12.55 / 1000      # Baseline mortality per person
THRESHOLD = 70.0                   # O3 MDA8 threshold (ug/m3)
RR_PER_10 = 1.01                   # Relative risk per 10 ug/m3 increase
RR_MIN_PER_10 = 1.00               # Lower bound
RR_MAX_PER_10 = 1.02               # Upper bound
SUMMER_MONTHS = (4, 9)              # Summer months range (April-September)
O3_COL = 'prediction_1'            # Use model prediction column uniformly

# Scenario config: csv -> O3 concentration; pop_nc/pop_var -> population grid
SCENARIOS = {
    '2018 (baseline)': {
        'csv': f'{NEW_MODEL_DIR}/2019_pred.csv',
        'pop_nc': None,                 # Weighted blend of ssp2_2020 / ssp2_2010
        'pop_var': 'ssp2_2018',
    },
    'SSP1-2.6 (2030)': {
        'csv': f'{NEW_MODEL_DIR}/ssp126_pred.csv',
        'pop_nc': f'{SSP2_TOTAL_NC}/ssp2_2030.nc',
        'pop_var': 'ssp2_2030',
    },
    'SSP2-4.5 (2030)': {
        'csv': f'{NEW_MODEL_DIR}/ssp245_pred.csv',
        'pop_nc': f'{SSP2_TOTAL_NC}/ssp2_2030.nc',
        'pop_var': 'ssp2_2030',
    },
    'SSP5-8.5 (2030)': {
        'csv': f'{NEW_MODEL_DIR}/ssp585_pred.csv',
        'pop_nc': f'{SSP5_TOTAL_NC}/ssp5_2030.nc',
        'pop_var': 'ssp5_2030',
    },
}

# Region config: germany.shp is actually the DE+BE+NL union; onlygermany.shp is Germany proper
REGIONS = {
    'All (DE+BE+NL)': [f'{MAP_DIR}/germany.shp'],
    'Germany':        [f'{MAP_DIR}/onlygermany.shp'],
    'Belgium':        [f'{MAP_DIR}/Belgium.shp'],
    'Netherlands':    [f'{MAP_DIR}/Netherlands.shp'],
}

# Indicator output order
INDICATORS = ['pop_weighted_O3', 'total_pop',
              'RR', 'RR_min', 'RR_max',
              'PAF', 'PAF_min', 'PAF_max',
              'PD', 'PD_min', 'PD_max']


def load_o3_mean(csv_path, o3_col=O3_COL, summer=True):
    """Read prediction CSV -> summer (or annual) site MDA8 mean -> lat/lon grid DataArray."""
    df = pd.read_csv(csv_path, index_col=0)
    df['time'] = pd.to_datetime(df['time'])
    # If CSV lacks lat/lon, merge from grids.csv
    if 'lat' not in df.columns or 'lon' not in df.columns:
        grids = pd.read_csv(GRIDS_CSV, index_col=0).reset_index()
        df = df.merge(grids, on='index', how='left')
    if summer:
        df = df[df['time'].dt.month.between(*SUMMER_MONTHS)]
    annual = df.groupby('index')[o3_col].mean().reset_index()
    grids = pd.read_csv(GRIDS_CSV, index_col=0).reset_index()
    annual = annual.merge(grids, on='index', how='left').dropna(subset=['lat', 'lon'])
    annual = annual.set_index(['lat', 'lon'])
    da = annual[o3_col].to_xarray()
    da.name = 'O3'
    return da


def load_population(cfg):
    """Load population DataArray according to scenario config."""
    if cfg['pop_nc'] is None:
        # 2018 baseline: 0.9 * ssp2_2020 + 0.1 * ssp2_2010
        ds2020 = xr.open_dataset(f'{SSP2_TOTAL_NC}/ssp2_2020.nc')
        ds2010 = xr.open_dataset(f'{SSP2_TOTAL_NC}/ssp2_2010.nc')
        v2020 = list(ds2020.data_vars)[0]
        v2010 = list(ds2010.data_vars)[0]
        pop = 0.9 * ds2020[v2020] + 0.1 * ds2010[v2010]
    else:
        ds = xr.open_dataset(cfg['pop_nc'])
        pop = ds[cfg['pop_var']]
    pop.name = 'pop'
    return pop


def load_region(shp_paths):
    """Read region shapefile(s) (union if multiple), return single-row GeoDataFrame."""
    gdfs = [gpd.read_file(p) for p in shp_paths]
    geoms = [g for gd in gdfs for g in gd.geometry]
    return gpd.GeoDataFrame(geometry=[unary_union(geoms)], crs=gdfs[0].crs)


def clip_pop_to_region(pop_da, region_gdf):
    """Clip population grid to region polygon using rasterio mask."""
    data = pop_da.values
    lat = pop_da.lat.values
    lon = pop_da.lon.values
    dx = (lon[-1] - lon[0]) / (len(lon) - 1)
    dy = (lat[0] - lat[-1]) / (len(lat) - 1)
    transform = Affine(dx, 0, lon.min(), 0, -dy, lat.max())
    meta = {
        'driver': 'GTiff', 'height': data.shape[0], 'width': data.shape[1],
        'count': 1, 'dtype': data.dtype, 'crs': 'EPSG:4326', 'transform': transform,
    }
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(data, 1)
            out_image, out_transform = mask(dataset, shapes=region_gdf.geometry.values, crop=True)
    h, w = out_image.shape[1], out_image.shape[2]
    lon_min = out_transform[2]
    lon_max = out_transform[2] + out_transform[0] * w
    lat_min = out_transform[5] + out_transform[4] * h
    lat_max = out_transform[5]
    new_lon = np.linspace(lon_min, lon_max, w)
    new_lat = np.linspace(lat_max, lat_min, h)
    out = xr.DataArray(
        data=out_image[0, :, :], dims=('lat', 'lon'),
        coords={'lat': new_lat, 'lon': new_lon}, name='pop',
    )
    # Replace out-of-region zeros with NaN
    out = out.where(out != 0, other=np.nan)
    return out


def compute_indicators(o3_da, pop_da_full, region_gdf):
    """Compute all health indicators for (O3, population, region); return scalar dict."""
    # 1. Clip population to region
    pop = clip_pop_to_region(pop_da_full, region_gdf)
    # 2. Reindex O3 site grid to region population grid via nearest neighbour
    o3 = o3_da.reindex(lat=pop.lat, lon=pop.lon, method='nearest')
    ds = xr.merge([o3.rename('O3'), pop.rename('pop')])

    # 3. Relative risk RR (with lower / upper bounds)
    x = (ds['O3'] - THRESHOLD) / 10.0
    rr = RR_PER_10 ** x
    rr_min = RR_MIN_PER_10 ** x
    rr_max = RR_MAX_PER_10 ** x

    # 4. Population attributable fraction PAF
    paf = (rr - 1) / rr
    paf_min = (rr_min - 1) / rr_min
    paf_max = (rr_max - 1) / rr_max

    # 5. Attributable deaths PD = baseline mortality * population * PAF
    pd_ = BASE_MORTALITY * ds['pop'] * paf
    pd_min = BASE_MORTALITY * ds['pop'] * paf_min
    pd_max = BASE_MORTALITY * ds['pop'] * paf_max

    # 6. Population-weighted O3 and total population
    pop_sum = ds['pop'].sum(dim=('lat', 'lon'), skipna=True)
    o3_pw = (ds['pop'] * ds['O3']).sum(dim=('lat', 'lon'), skipna=True) / pop_sum

    def agg(da, op):
        if op == 'sum':
            return float(da.sum(dim=('lat', 'lon'), skipna=True).values)
        return float(da.mean(dim=('lat', 'lon'), skipna=True).values)

    return {
        'pop_weighted_O3': float(o3_pw.values),
        'total_pop': float(pop_sum.values),
        'RR': agg(rr, 'mean'), 'RR_min': agg(rr_min, 'mean'), 'RR_max': agg(rr_max, 'mean'),
        'PAF': agg(paf, 'mean'), 'PAF_min': agg(paf_min, 'mean'), 'PAF_max': agg(paf_max, 'mean'),
        'PD': agg(pd_, 'sum'), 'PD_min': agg(pd_min, 'sum'), 'PD_max': agg(pd_max, 'sum'),
    }


# Population grid per scenario
pop_data = {name: load_population(cfg) for name, cfg in SCENARIOS.items()}

# Region GeoDataFrames
region_gdfs = {rname: load_region(paths) for rname, paths in REGIONS.items()}

# Summer O3 mean per scenario
o3_data = {name: load_o3_mean(cfg['csv']) for name, cfg in SCENARIOS.items()}

print('Scenarios:', list(SCENARIOS.keys()))
print('Regions:', list(REGIONS.keys()))
print('O3 grid sample:', dict(o3_data['SSP2-4.5 (2030)'].sizes))


records = []
for sname in SCENARIOS:
    for rname in REGIONS:
        res = compute_indicators(o3_data[sname], pop_data[sname], region_gdfs[rname])
        res['scenario'] = sname
        res['region'] = rname
        records.append(res)

df_long = pd.DataFrame(records)
df_long = df_long[['scenario', 'region'] + INDICATORS]
print(df_long)


region_order = list(REGIONS.keys())
scenario_order = list(SCENARIOS.keys())

# Pivot to scenario (rows) x (region, indicator) (columns)
df_wide = df_long.pivot(index='scenario', columns='region', values=INDICATORS)
df_wide = df_wide.reindex(index=scenario_order)
# Reorder columns to (indicator, region) sorted by INDICATORS x region_order
df_wide = df_wide.reorder_levels([1, 0], axis=1)
col_order = pd.MultiIndex.from_product([INDICATORS, region_order], names=[None, 'region'])
df_wide = df_wide.reindex(columns=col_order)


def fmt_table(df):
    out = df.copy()
    int_inds = {'total_pop', 'PD', 'PD_min', 'PD_max'}
    for col in out.columns:
        ind = col[0]
        if ind in int_inds:
            out[col] = out[col].round(0).astype('Int64')
        else:
            out[col] = out[col].round(4)
    return out


print(fmt_table(df_wide))


def view(indicator):
    """Single indicator: rows=scenario, columns=region."""
    sub = df_long.pivot(index='scenario', columns='region', values=indicator)
    return sub.reindex(index=scenario_order)[region_order]


print('=== Population-weighted O3 (ug/m3) ===')
print(view('pop_weighted_O3'))


print('=== RR (relative risk, with bounds) ===')
print(pd.concat({'RR': view('RR'), 'RR_min': view('RR_min'), 'RR_max': view('RR_max')}, axis=1))


print('=== PAF (population attributable fraction, with bounds) ===')
print(pd.concat({'PAF': view('PAF'), 'PAF_min': view('PAF_min'), 'PAF_max': view('PAF_max')}, axis=1))


print('=== PD (attributable deaths, with bounds) ===')
print(pd.concat({'PD': view('PD'), 'PD_min': view('PD_min'), 'PD_max': view('PD_max')}, axis=1))


print('=== Total population ===')
print(view('total_pop'))


out_csv = f'{NEW_MODEL_DIR}/fig_7_health_impact_results.csv'
df_long.to_csv(out_csv, index=False)
print('Saved:', out_csv)
print(df_long.head(16))
