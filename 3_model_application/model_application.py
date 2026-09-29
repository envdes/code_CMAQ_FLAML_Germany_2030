"""
model_application.py
====================

Apply the trained FLAML AutoML model to a pre-ML predictor table and store the
predicted MDA8 ozone.

The pickled AutoML model is loaded, the predictor matrix is built from the
pre-ML table and the prediction is written to `2019_pred.csv`.
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
from flaml import AutoML
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------
# Load the trained model
# ----------------------------------------------------------------------------
import pickle

model_path = "../new_model/automl1.pkl"
with open(model_path, 'rb') as f:
    automl1 = pickle.load(f)

# ----------------------------------------------------------------------------
# Predictor table
# ----------------------------------------------------------------------------
df = pd.read_csv('./2019_preML.csv', index_col=0)
df.rename(columns={'O3_8h_avg': 'cmaq_MDA8'}, inplace=True)
df = df.dropna()

# ----------------------------------------------------------------------------
# Predict and store the result
# ----------------------------------------------------------------------------
X_test_1 = df[['blh', 'd2m', 'e', 'msdwlwrfcs', 'sp', 't2m', 'tp', 'u10','v10', 'arable', 'pasture', 'primary',
               'secondary', 'urban', 'population', 'cmaq_MDA8']]
df['prediction_1'] = automl1.predict(X_test_1)

df.to_csv('../new_model/2019_pred.csv')
