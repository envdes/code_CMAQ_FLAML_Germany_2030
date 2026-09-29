"""
metro_ratio.py
==============

Compute the monthly future / base ratios of the CMIP6 meteorology that are used
to scale the base-year meteorology to a future year.

For every meteorological variable the tables `<variable>_<base year>.csv` and
`<variable>_<future year>.csv` are merged per site and month, and
`ratio_<variable> = future / base` is stored in `<future year>_met_ratio.csv`.

Note: change `input_dir` / `base_year` to process another scenario
(e.g. ssp3, ssp2_n, ssp5_n, cmip6_met).
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------
# Scenario / years
# ----------------------------------------------------------------------------
input_dir = '../grid_data/ssp5'
base_year = '2019'
future_year = '2030'

# ----------------------------------------------------------------------------
# Read the base-year and future-year tables of every variable
# ----------------------------------------------------------------------------
df1 = pd.read_csv(f'{input_dir}/bldep_{base_year}.csv', index_col=0)
df1.rename(columns={'bldep': f'bldep_{base_year}'}, inplace=True)
df1['month'] = pd.to_datetime(df1['time']).dt.month

df2 = pd.read_csv(f'{input_dir}/bldep_{future_year}.csv', index_col=0)
df2.rename(columns={'bldep': f'bldep_{future_year}'}, inplace=True)
df2['month'] = pd.to_datetime(df2['time']).dt.month

df3 = pd.read_csv(f'{input_dir}/evspsbl_{base_year}.csv', index_col=0)
df3.rename(columns={'evspsbl': f'evspsbl_{base_year}'}, inplace=True)
df3['month'] = pd.to_datetime(df3['time']).dt.month

df4 = pd.read_csv(f'{input_dir}/evspsbl_{future_year}.csv', index_col=0)
df4.rename(columns={'evspsbl': f'evspsbl_{future_year}'}, inplace=True)
df4['month'] = pd.to_datetime(df4['time']).dt.month

df5 = pd.read_csv(f'{input_dir}/ps_{base_year}.csv', index_col=0)
df5.rename(columns={'ps': f'ps_{base_year}'}, inplace=True)
df5['month'] = pd.to_datetime(df5['time']).dt.month

df6 = pd.read_csv(f'{input_dir}/ps_{future_year}.csv', index_col=0)
df6.rename(columns={'ps': f'ps_{future_year}'}, inplace=True)
df6['month'] = pd.to_datetime(df6['time']).dt.month

df7 = pd.read_csv(f'{input_dir}/rlds_{base_year}.csv', index_col=0)
df7.rename(columns={'rlds': f'rlds_{base_year}'}, inplace=True)
df7['month'] = pd.to_datetime(df7['time']).dt.month

df8 = pd.read_csv(f'{input_dir}/rlds_{future_year}.csv', index_col=0)
df8.rename(columns={'rlds': f'rlds_{future_year}'}, inplace=True)
df8['month'] = pd.to_datetime(df8['time']).dt.month

df9 = pd.read_csv(f'{input_dir}/pr_{base_year}.csv', index_col=0)
df9.rename(columns={'pr': f'pr_{base_year}'}, inplace=True)
df9['month'] = pd.to_datetime(df9['time']).dt.month

df10 = pd.read_csv(f'{input_dir}/pr_{future_year}.csv', index_col=0)
df10.rename(columns={'pr': f'pr_{future_year}'}, inplace=True)
df10['month'] = pd.to_datetime(df10['time']).dt.month

df11 = pd.read_csv(f'{input_dir}/tas_{base_year}.csv', index_col=0)
df11.rename(columns={'tas': f'tas_{base_year}'}, inplace=True)
df11['month'] = pd.to_datetime(df11['time']).dt.month

df12 = pd.read_csv(f'{input_dir}/tas_{future_year}.csv', index_col=0)
df12.rename(columns={'tas': f'tas_{future_year}'}, inplace=True)
df12['month'] = pd.to_datetime(df12['time']).dt.month

df13 = pd.read_csv(f'{input_dir}/tdps_{base_year}.csv', index_col=0)
df13.rename(columns={'tdps': f'tdps_{base_year}'}, inplace=True)
df13['month'] = pd.to_datetime(df13['time']).dt.month

df14 = pd.read_csv(f'{input_dir}/tdps_{future_year}.csv', index_col=0)
df14.rename(columns={'tdps': f'tdps_{future_year}'}, inplace=True)
df14['month'] = pd.to_datetime(df14['time']).dt.month

df15 = pd.read_csv(f'{input_dir}/uas_{base_year}.csv', index_col=0)
df15.rename(columns={'uas': f'uas_{base_year}'}, inplace=True)
df15['month'] = pd.to_datetime(df15['time']).dt.month

df16 = pd.read_csv(f'{input_dir}/uas_{future_year}.csv', index_col=0)
df16.rename(columns={'uas': f'uas_{future_year}'}, inplace=True)
df16['month'] = pd.to_datetime(df16['time']).dt.month

df17 = pd.read_csv(f'{input_dir}/vas_{base_year}.csv', index_col=0)
df17.rename(columns={'vas': f'vas_{base_year}'}, inplace=True)
df17['month'] = pd.to_datetime(df17['time']).dt.month

df18 = pd.read_csv(f'{input_dir}/vas_{future_year}.csv', index_col=0)
df18.rename(columns={'vas': f'vas_{future_year}'}, inplace=True)
df18['month'] = pd.to_datetime(df18['time']).dt.month

# ----------------------------------------------------------------------------
# Merge all variables on month and site
# ----------------------------------------------------------------------------
dataframes = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10,
              df11, df12, df13, df14, df15, df16, df17, df18]

# Start from the first table and merge the remaining ones.
merged_df = dataframes[0]
for df in dataframes[1:]:
    merged_df = pd.merge(merged_df, df, on=['month', 'sites'])

print(merged_df)

# The merge duplicates the time column, drop it.
merged_df = merged_df.drop(columns=['time_x', 'time_y'])

merged_df.columns

# ----------------------------------------------------------------------------
# Future / base ratios of every meteorological variable
# ----------------------------------------------------------------------------
merged_df['ratio_bldep'] = merged_df[f'bldep_{future_year}'] / merged_df[f'bldep_{base_year}']
merged_df['ratio_evspsbl'] = merged_df[f'evspsbl_{future_year}'] / merged_df[f'evspsbl_{base_year}']
merged_df['ratio_ps'] = merged_df[f'ps_{future_year}'] / merged_df[f'ps_{base_year}']
merged_df['ratio_rlds'] = merged_df[f'rlds_{future_year}'] / merged_df[f'rlds_{base_year}']
merged_df['ratio_pr'] = merged_df[f'pr_{future_year}'] / merged_df[f'pr_{base_year}']
merged_df['ratio_tas'] = merged_df[f'tas_{future_year}'] / merged_df[f'tas_{base_year}']
merged_df['ratio_tdps'] = merged_df[f'tdps_{future_year}'] / merged_df[f'tdps_{base_year}']
merged_df['ratio_uas'] = merged_df[f'uas_{future_year}'] / merged_df[f'uas_{base_year}']
merged_df['ratio_vas'] = merged_df[f'vas_{future_year}'] / merged_df[f'vas_{base_year}']

merged_df.columns

# ----------------------------------------------------------------------------
# Keep only the ratios and the keys
# ----------------------------------------------------------------------------
new_df = merged_df[['ratio_bldep',
                    'ratio_evspsbl', 'ratio_ps', 'ratio_rlds', 'ratio_pr', 'ratio_tas',
                    'ratio_tdps', 'ratio_uas', 'ratio_vas', 'sites', 'month']]
new_df

# ----------------------------------------------------------------------------
# Write the result
# ----------------------------------------------------------------------------
new_df.to_csv('../grid_data/ssp1/2030_met_ratio.csv')

# ----------------------------------------------------------------------------
# Optional diagnostics: per-site statistics of the base-year boundary layer depth
# ----------------------------------------------------------------------------
bldep = pd.merge(df1, df2, on=['month', 'sites'])
bldep = bldep.drop(columns=['time_x', 'time_y'])

# Mean of the base-year values per site.
mean = bldep.groupby('sites')[f'bldep_{base_year}'].mean()
bldep['mean_2018'] = bldep['sites'].map(mean)

# Variance of the base-year values per site.
variance_series = bldep.groupby('sites')[f'bldep_{base_year}'].var()
bldep['var_2018'] = bldep['sites'].map(variance_series)

# Standard deviation.
bldep['var_2018_sqrt'] = np.sqrt(bldep['var_2018'])
