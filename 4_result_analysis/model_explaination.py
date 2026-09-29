from flaml import AutoML
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import make_regression
import shap
df =pd.read_csv('./grid_data/2018_preML.csv',index_col =0)
df.rename(columns={'O3_8h_avg': 'cmaq_MDA8', 'nearest_obs_8_max': 'MDA8'}, inplace=True)
df =df.dropna()
df['time']=pd.to_datetime(df['time'])
train_data =df
X_train_pre = train_data[['blh', 'd2m', 'e', 'msdwlwrfcs', 'sp', 't2m', 'tp', 'u10','v10','arable', 'pasture', 'primary',
'secondary', 'urban','population','cmaq_MDA8']]
y_train_pre = train_data['MDA8']
model = ExtraTreesRegressor(n_estimators=66,max_features=0.9867883466503118,max_leaf_nodes=7012,random_state=42)
model.fit(X_train_pre, y_train_pre)
df2 = pd.read_csv('./grid_data/2019_preML.csv', index_col=0)
df2.rename(columns={'O3_8h_avg': 'cmaq_MDA8'}, inplace=True)
df2 =df2.dropna()
X_test_1 = df2[['blh', 'd2m', 'e', 'msdwlwrfcs', 'sp', 't2m', 'tp', 'u10','v10','arable', 'pasture', 'primary',
    'secondary', 'urban','population','cmaq_MDA8']]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_1)
shap_df = pd.DataFrame(shap_values, columns=X_test_1.columns)
shap_df.to_csv("shap_values.csv", index=False)

importance = pd.DataFrame({
    'Feature': X_test_1.columns,
    'SHAP Importance': np.mean(np.abs(shap_values), axis=0)
})
importance = importance.sort_values('SHAP Importance', ascending=False)
importance.to_csv("feature_importance.csv", index=False)
