"""
merge_multisource.py
====================

Merge the base-year predictor table with the scenario deltas of the future year
and produce the pre-ML input table of that scenario.

    2030_met_ratio.csv   monthly future/base ratios of the meteorology
    2030_emis_ratio.csv  monthly future/base ratios of the emissions
    pop_2030.csv         population of the future year
    landuse_2030.csv     land-use fractions of the future year
    2019_preML.csv       base-year daily meteorology and CMAQ simulation

The base-year meteorology is scaled with the monthly ratios and the predictor
table of the future year is written to `2030_ssp5_preML.csv`.
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Scenario tables of the future year (change the directory for another scenario)
# ----------------------------------------------------------------------------
df1 = pd.read_csv('./ssp5/2030_met_ratio.csv', index_col=0)
df2 = pd.read_csv('./ssp5/2030_emis_ratio.csv', index_col=0)
df3 = pd.read_csv('./ssp5/pop_2030.csv', index_col=0)
df4 = pd.read_csv('./ssp5/landuse_2030.csv', index_col=0)

# ----------------------------------------------------------------------------
# Base-year daily predictor table
# ----------------------------------------------------------------------------
df = pd.read_csv('./2019_preML.csv', index_col=0)
df['month'] = pd.to_datetime(df['time']).dt.month

df1.columns

# ----------------------------------------------------------------------------
# Merge the population and land use with the base-year table
# ----------------------------------------------------------------------------
df34 = pd.merge(df3, df4, on='index', how='inner')

df = pd.merge(df34, df, on='index', how='outer')

# ----------------------------------------------------------------------------
# Merge the meteorological and emission ratios
# ----------------------------------------------------------------------------
df12 = pd.merge(df1, df2, on=['sites', 'month'], how='inner')

# Grid id column is renamed to `sites` so that the tables can be joined.
df = df.rename(columns={'index': 'sites'})

df_com = pd.merge(df12, df, on=['sites', 'month'], how='outer')

# ----------------------------------------------------------------------------
# Scale the base-year meteorology with the monthly future/base ratios
# ----------------------------------------------------------------------------
df_com['blh'] = df_com['ratio_bldep'] * df_com['blh']
df_com['d2m'] = df_com['ratio_tdps'] * df_com['d2m']
df_com['e'] = df_com['ratio_evspsbl'] * df_com['e']
df_com['msdwlwrfcs'] = df_com['ratio_rlds'] * df_com['msdwlwrfcs']
df_com['sp'] = df_com['ratio_ps'] * df_com['sp']
df_com['t2m'] = df_com['ratio_tas'] * df_com['t2m']
df_com['tp'] = df_com['ratio_pr'] * df_com['tp']
df_com['u10'] = df_com['ratio_uas'] * df_com['u10']
df_com['v10'] = df_com['ratio_vas'] * df_com['v10']

# ----------------------------------------------------------------------------
# Keep the predictors used by the model
# ----------------------------------------------------------------------------
df_final = df_com[['sites', 'month', 'population', 'arable', 'pasture',
                   'primary', 'secondary', 'urban', 'blh', 'time', 'd2m', 'e',
                   'msdwlwrfcs', 'sp', 't2m', 'tp', 'u10', 'v10']]

df_final.to_csv('2030_ssp5_preML.csv')
