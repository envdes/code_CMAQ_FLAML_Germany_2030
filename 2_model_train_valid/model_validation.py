"""
model_validation.py
===================

Validate the ML prediction against the observations and compare it with the raw
CMAQ simulation.

Every monitoring site is matched to the nearest grid cell of the modelling grid,
the daily ML prediction of that cell is attached to the observed / simulated MDA8
ozone, and the residuals, the skill scores and the scatter-density plots of both
products are produced.
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score


# ----------------------------------------------------------------------------
# Inputs: modelling grid, observed / simulated MDA8 table, monitoring sites
# ----------------------------------------------------------------------------
df1 = pd.read_csv('./grids.csv', index_col=0)
df2 = pd.read_csv('../data_process/MDA8_2019.csv', index_col=0)
df3 = pd.read_csv('../data_fusion/sites.csv', index_col=0)

# Convert the CMAQ mixing ratio (ppb) to a mass concentration.
df2['cmaq_8_max'] = df2['cmaq_8_max'] * 0.467

# Attach the coordinates of every site.
df23 = pd.merge(df2, df3, left_on='sites', right_on='0', how='left')

# ----------------------------------------------------------------------------
# Match every site to the nearest grid cell
# ----------------------------------------------------------------------------
# Latitude / longitude of the sites and of the grid cells.
df23_lat_lon = df23[['1', '2']].values
df1_lat_lon = df1[['lat', 'lon']].values

# Distance matrix between every site and every grid cell.
distance_matrix = cdist(df23_lat_lon, df1_lat_lon, metric='euclidean')

# Nearest grid cell of every site.
nearest_indices = np.argmin(distance_matrix, axis=1)

df23['nearest_index'] = nearest_indices

print(df23)

# ----------------------------------------------------------------------------
# ML predictions of the grid cells
# ----------------------------------------------------------------------------
df5 = pd.read_csv('../new_model/2019_pred.csv', index_col=0)
df5 = df5[['time', 'index', 'prediction_1']]

df5

# ----------------------------------------------------------------------------
# Time columns are needed for the merge
# ----------------------------------------------------------------------------
df5['time'] = pd.to_datetime(df5['time'])
df23['time'] = pd.to_datetime(df23['time'])

df23

# ----------------------------------------------------------------------------
# Attach the prediction of the nearest grid cell to every observation
# ----------------------------------------------------------------------------
merged_df = pd.merge(df23, df5, how='left', left_on=['time', 'nearest_index'], right_on=['time', 'index'])

merged_df

# ----------------------------------------------------------------------------
# Residuals of CMAQ (resid1) and of the ML model (resid2)
# ----------------------------------------------------------------------------
df_results = merged_df.dropna()

df_results['resid1'] = df_results['cmaq_8_max'] - df_results['obs_8_max']
df_results['resid2'] = df_results['prediction_1'] - df_results['obs_8_max']

df_results

# ----------------------------------------------------------------------------
# Mean residuals per site (with the coordinates of the site)
# ----------------------------------------------------------------------------
average_residuals = df_results.groupby('sites').agg(
    {'1': 'first', '2': 'first', 'resid1': 'mean', 'resid2': 'mean'}
).reset_index()

print(average_residuals)

average_residuals.to_csv('../new_model/resid2019nocmaq_sites.csv')

# ----------------------------------------------------------------------------
# Keep only strictly positive observations and predictions
# ----------------------------------------------------------------------------
condition = (df_results['obs_8_max'] <= 0) | (df_results['prediction_1'] <= 0)
df_results = df_results[~condition]

# ----------------------------------------------------------------------------
# Skill scores of the ML model
# ----------------------------------------------------------------------------
rmse_value_ml = np.round(((df_results['obs_8_max'] - df_results['prediction_1']) ** 2).mean() ** 0.5, 2)
r2_value_ml = r2_score(df_results['obs_8_max'], df_results['prediction_1'])
pearson_corr_ml = np.round((df_results['obs_8_max'].corr(df_results['prediction_1'])), 2)
mae_value_ml = np.round((df_results['obs_8_max'] - df_results['prediction_1']).abs().mean(), 2)

print(rmse_value_ml, r2_value_ml, pearson_corr_ml, mae_value_ml)

# ----------------------------------------------------------------------------
# Skill scores of the CMAQ simulation
# ----------------------------------------------------------------------------
rmse_value_cmaq = np.round(((df_results['obs_8_max'] - df_results['cmaq_8_max']) ** 2).mean() ** 0.5, 2)
r2_value_cmaq = r2_score(df_results['obs_8_max'], df_results['cmaq_8_max'])
pearson_corr_cmaq = np.round((df_results['obs_8_max'].corr(df_results['cmaq_8_max'])), 2)
mae_value_cmaq = np.round((df_results['obs_8_max'] - df_results['cmaq_8_max']).abs().mean(), 2)

df_results

# ----------------------------------------------------------------------------
# Scatter-density plot: observations vs. CMAQ
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Point density of the scatter plot.
a = df_results['obs_8_max']
b = df_results['cmaq_8_max']
xy = np.vstack([a, b])
z = gaussian_kde(xy)(xy)

plt.figure(figsize=(8, 8))
scatter_plot = plt.scatter(a, b, c=z, s=10, cmap='turbo', alpha=0.7)
plt.title("CMAQ validation", fontsize=14)

# 1:1 line.
plt.plot([0, 200], [0, 200], color="black", linestyle='--')

plt.xlim(0, 200)
plt.ylim(0, 200)

plt.xlabel(r'Observed $O_{3}$ ($\mu g/m^{3}$)', fontsize=14)
plt.ylabel(r'Predicted $O_{3}$ ($\mu g/m^{3}$)', fontsize=14)

plt.text(0.05, 0.95, f"$R^2$: {r2_value_cmaq:.2f}", transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.9, f"$RMSE$: {rmse_value_cmaq:.2f}", transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.85, f"$R$: {pearson_corr_cmaq:.2f}", transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.8, f"$MAE$: {mae_value_cmaq:.2f}", transform=plt.gca().transAxes, fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

cbar = plt.colorbar(scatter_plot)
cbar.set_label('Density')

sns.despine()
plt.grid()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Scatter-density plot: observations vs. ML prediction
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Point density of the scatter plot.
a = df_results['obs_8_max']
b = df_results['prediction_1']
xy = np.vstack([a, b])
z = gaussian_kde(xy)(xy)

plt.figure(figsize=(6, 6))
scatter_plot = plt.scatter(a, b, c=z, s=10, cmap='turbo', alpha=0.7)
plt.title("ML validation", fontsize=14)

# 1:1 line.
plt.plot([0, 200], [0, 200], color="black", linestyle='--')

plt.xlim(0, 200)
plt.ylim(0, 200)

plt.xlabel(r'Observed $O_{3}$ ($\mu g/m^{3}$)', fontsize=14)
plt.ylabel(r'Predicted $O_{3}$ ($\mu g/m^{3}$)', fontsize=14)

plt.text(0.05, 0.95, f"$R^2$: {r2_value_ml:.2f}", transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.9, f"$RMSE$: {rmse_value_ml:.2f}", transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.85, f"$R$: {pearson_corr_ml:.2f}", transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.8, f"$MAE$: {mae_value_ml:.2f}", transform=plt.gca().transAxes, fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

cbar = plt.colorbar(scatter_plot)
cbar.set_label('Density', fontsize=14)

sns.despine()
plt.grid()

plt.tight_layout()
plt.show()
