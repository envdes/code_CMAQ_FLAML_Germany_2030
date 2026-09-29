"""
model_training.py
=================

Train the FLAML AutoML regression model that corrects the CMAQ MDA8 ozone
simulation towards the observations.

Predictors: ERA5 meteorology, land-use fractions, population and the CMAQ base
simulation.  Target: the observed MDA8 ozone (`nearest_obs_8_max`).
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
from flaml import AutoML
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------
# Training table
# ----------------------------------------------------------------------------
df = pd.read_csv('./2018_preML.csv', index_col=0)
df.rename(columns={'O3_8h_avg': 'cmaq_MDA8', 'nearest_obs_8_max': 'MDA8'}, inplace=True)
df = df.dropna()
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month
df['weekday'] = df['time'].dt.weekday
train_data = df

# ----------------------------------------------------------------------------
# Predictors and target
# ----------------------------------------------------------------------------
X_train_pre = train_data[['blh', 'd2m', 'e', 'msdwlwrfcs', 'sp', 't2m', 'tp', 'u10', 'v10', 'arable', 'pasture', 'primary',
                          'secondary', 'urban', 'population', 'cmaq_MDA8']]
y_train_pre = train_data['MDA8']

# ----------------------------------------------------------------------------
# Run the FLAML AutoML search
# ----------------------------------------------------------------------------
automl1 = AutoML()
settings = {
    "time_budget": 1800,  # Time budget of the automatic tuning, in seconds.
    "metric": 'r2',
    "task": "regression",
    "estimator_list": ['lgbm', 'rf', 'xgboost', 'extra_tree', 'catboost']
}
automl1.fit(X_train=X_train_pre, y_train=y_train_pre, **settings)

# ----------------------------------------------------------------------------
# Save the trained model
# ----------------------------------------------------------------------------
import pickle

model_path = "../new_model/automl1.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(automl1, f)
print(f"Model has been saved to {model_path}")
