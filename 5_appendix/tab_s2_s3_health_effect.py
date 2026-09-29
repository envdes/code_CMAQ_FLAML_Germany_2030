import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ----------------------
# Step 1: prepare the data
# ----------------------
print("=== Step 1: prepare the data ===")

# Extract the summer months (1 April = day 91, 30 September = day 273).
summer_days = range(90, 273)
o3_ssp1_summer = o3_ssp1[summer_days, :, :]
o3_ssp2_summer = o3_ssp2[summer_days, :, :]
o3_ssp5_summer = o3_ssp5[summer_days, :, :]
o3_obs_2019_summer = o3_obs_2019[summer_days, :, :]
o3_pred_2019_summer = o3_pred_2019[summer_days, :, :]

# Dimensions of the data (all grid cells are valid land cells).
n_days_summer, n_lat, n_lon = o3_ssp1_summer.shape
n_total_grids = n_lat * n_lon

print(f"Total number of grid cells (all land): {n_total_grids}")
print(f"Number of summer days: {n_days_summer}")
print(f"Total number of observations: {n_total_grids * n_days_summer * 3}")

# Convert the arrays into a long-format DataFrame.
data = []
grid_ids = np.arange(n_total_grids).reshape(n_lat, n_lon)
date_ids = np.arange(n_days_summer)[:, np.newaxis, np.newaxis]

for scenario, o3 in zip(['ssp1', 'ssp2', 'ssp5'], [o3_ssp1_summer, o3_ssp2_summer, o3_ssp5_summer]):
    df = pd.DataFrame({
        'o3': o3.flatten(),
        'scenario': scenario,
        'grid_id': grid_ids.flatten(),
        'date_id': date_ids.flatten()
    })
    data.append(df)

df_all = pd.concat(data, ignore_index=True)
print(f"Data prepared, {len(df_all)} records in total")

# ----------------------
# Step 2: fit the mixed-effects model
# ----------------------
print("\n=== Step 2: fit the mixed-effects model ===")

# Fit the mixed-effects model with random effects of grid cell and date.
model = smf.mixedlm(
    "o3 ~ scenario",
    data=df_all,
    groups=df_all["grid_id"],
    re_formula="~1",
    random_effects={"date_id": "~1"}
)

result = model.fit()
print("Mixed-effects model fitted")
print(result.summary())

# ----------------------
# Step 3: pairwise comparisons of the three scenarios
# ----------------------
print("\n=== Step 3: pairwise comparisons of the three scenarios ===")

# Multiple comparisons with the t_test_pairwise method of MixedLMResults.
# This method is designed for mixed-effects models and is more accurate than
# the generic Tukey HSD test.
pairwise_results = result.t_test_pairwise(
    term_name="scenario",
    method="bonferroni",  # Bonferroni correction controls the family-wise error rate.
    alpha=0.05
)

print("Multiple comparison results (Bonferroni corrected):")
print(pairwise_results.summary())

# Extract the results of the three pairwise comparisons.
comparisons = {
    'ssp2_vs_ssp1': {
        'diff': pairwise_results.effect[0],
        'se_natural': pairwise_results.std_err[0],
        'p_value_natural': pairwise_results.pvalue[0]
    },
    'ssp5_vs_ssp1': {
        'diff': pairwise_results.effect[1],
        'se_natural': pairwise_results.std_err[1],
        'p_value_natural': pairwise_results.pvalue[1]
    },
    'ssp5_vs_ssp2': {
        'diff': pairwise_results.effect[2],
        'se_natural': pairwise_results.std_err[2],
        'p_value_natural': pairwise_results.pvalue[2]
    }
}

# ----------------------
# Step 4: estimate the prediction uncertainty of the model
# ----------------------
print("\n=== Step 4: estimate the prediction uncertainty of the model ===")

# Error variance of the model on the 2019 test set.
error_2019 = o3_pred_2019_summer - o3_obs_2019_summer
sigma_model_squared = np.nanmean(error_2019 ** 2)
rmse = np.sqrt(sigma_model_squared)

n_total_per_scenario = len(df_all) // 3  # Number of observations per scenario.
se_model_per_comparison = np.sqrt(2 * sigma_model_squared / n_total_per_scenario)

print(f"Model RMSE: {rmse:.2f} μg/m³")
print(f"Model error variance: {sigma_model_squared:.2f} (μg/m³)²")
print(f"Model uncertainty standard error per comparison: {se_model_per_comparison:.4f} μg/m³")

# ----------------------
# Step 5: total uncertainty and final statistics
# ----------------------
print("\n=== Step 5: final statistics ===")

final_results = []
for comp_name, comp_data in comparisons.items():
    diff = comp_data['diff']
    se_natural = comp_data['se_natural']

    # Total standard error (natural variability + model error).
    se_total = np.sqrt(se_natural ** 2 + se_model_per_comparison ** 2)

    # 95% confidence interval.
    ci_lower = diff - 1.96 * se_total
    ci_upper = diff + 1.96 * se_total

    # z statistic and p value.
    z_stat = diff / se_total
    p_value_total = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    # Apply the Bonferroni correction.
    p_value_corrected = min(p_value_total * 3, 1.0)

    # Significance level.
    if p_value_corrected < 0.0001:
        sig = '***'
    elif p_value_corrected < 0.001:
        sig = '**'
    elif p_value_corrected < 0.0167:  # Bonferroni-corrected level (0.05 / 3).
        sig = '*'
    else:
        sig = '不显著'

    final_results.append({
        '对比组': comp_name.replace('_vs_', ' vs '),
        '差异(μg/m³)': round(diff, 2),
        '自然不确定性标准误': round(se_natural, 4),
        '模型不确定性标准误': round(se_model_per_comparison, 4),
        '总标准误': round(se_total, 4),
        '95%置信区间下限': round(ci_lower, 2),
        '95%置信区间上限': round(ci_upper, 2),
        '原始p值': round(p_value_total, 6),
        'Bonferroni校正后p值': round(p_value_corrected, 6),
        '显著性': sig
    })

# Convert to a DataFrame and save it.
df_final = pd.DataFrame(final_results)
df_final.to_csv('表S2_三组情景两两对比综合统计结果.csv', index=False, encoding='utf-8-sig')
print("\nSaved: 表S2_三组情景两两对比综合统计结果.csv")
print(df_final.to_string(index=False))

# ----------------------
# Step 6: variance decomposition
# ----------------------
print("\n=== Step 6: variance decomposition ===")

# Extract the variance components.
var_grid = result.random_effects_var['grid_id']
var_date = result.random_effects_var['date_id']
var_residual = result.scale

total_var = var_grid + var_date + var_residual
var_scenario = np.var([result.params['Intercept'],
                       result.params['Intercept'] + result.params['scenario[T.ssp2]'],
                       result.params['Intercept'] + result.params['scenario[T.ssp5]']])

variance_decomposition = [
    {'方差来源': '网格细胞（空间变异）', '方差估计值': round(var_grid, 2), '占总方差比例(%)': round(var_grid / total_var * 100, 1)},
    {'方差来源': '日期（时间变异）', '方差估计值': round(var_date, 2), '占总方差比例(%)': round(var_date / total_var * 100, 1)},
    {'方差来源': '残差（情景内部变异）', '方差估计值': round(var_residual, 2), '占总方差比例(%)': round(var_residual / total_var * 100, 1)},
    {'方差来源': '情景效应', '方差估计值': round(var_scenario, 2), '占总方差比例(%)': round(var_scenario / total_var * 100, 1)},
    {'方差来源': '模型预测不确定性', '方差估计值': round(sigma_model_squared, 2), '占总方差比例(%)': round(sigma_model_squared / total_var * 100, 1)}
]

df_variance = pd.DataFrame(variance_decomposition)
df_variance.to_csv('表S3_方差分解结果.csv', index=False, encoding='utf-8-sig')
print("\nSaved: 表S3_方差分解结果.csv")
print(df_variance.to_string(index=False))

# ----------------------
# Step 7: health impact differences
# ----------------------
print("\n=== Step 7: health impact differences ===")

# WHO 2021 health impact parameters.
beta = 0.00456  # Concentration-response coefficient (μg⁻¹·m³).
baseline_conc = 70  # Baseline concentration (μg/m³).
total_population = 102000000  # Total population of the study region (Germany + Belgium + Netherlands).
baseline_mortality = 0.011  # Baseline all-cause mortality rate (per year).

health_results = []
for comp_name, comp_data in comparisons.items():
    diff_o3 = comp_data['diff']
    se_o3 = np.sqrt(comp_data['se_natural'] ** 2 + se_model_per_comparison ** 2)

    # Difference of the relative risks.
    rr1 = np.exp(beta * (result.params['Intercept'] - baseline_conc))
    if 'ssp1' in comp_name:
        rr2 = np.exp(beta * (result.params['Intercept'] + comp_data['diff'] - baseline_conc))
    else:
        rr2 = np.exp(beta * (result.params['Intercept'] + result.params['scenario[T.ssp5]'] - baseline_conc))

    diff_rr = rr2 - rr1
    se_rr = np.abs(rr2 * beta * se_o3)

    # Difference of the population attributable fractions.
    paf1 = (rr1 - 1) / rr1
    paf2 = (rr2 - 1) / rr2
    diff_paf = paf2 - paf1
    se_paf = se_rr / (rr2 ** 2)

    # Difference of the number of premature deaths.
    diff_pd = diff_paf * baseline_mortality * total_population
    se_pd = se_paf * baseline_mortality * total_population

    # 95% confidence interval.
    ci_pd_lower = diff_pd - 1.96 * se_pd
    ci_pd_upper = diff_pd + 1.96 * se_pd

    # Statistical significance.
    z_stat_pd = diff_pd / se_pd
    p_value_pd = 2 * (1 - stats.norm.cdf(np.abs(z_stat_pd)))
    p_value_pd_corrected = min(p_value_pd * 3, 1.0)

    if p_value_pd_corrected < 0.0001:
        sig = '***'
    elif p_value_pd_corrected < 0.001:
        sig = '**'
    elif p_value_pd_corrected < 0.0167:
        sig = '*'
    else:
        sig = '不显著'

    health_results.append({
        '对比组': comp_name.replace('_vs_', ' vs '),
        '臭氧浓度差异(μg/m³)': round(diff_o3, 2),
        'RR差异': round(diff_rr, 4),
        'PAF差异': round(diff_paf, 4),
        '过早死亡人数差异': round(diff_pd, 0),
        '过早死亡人数95%CI下限': round(ci_pd_lower, 0),
        '过早死亡人数95%CI上限': round(ci_pd_upper, 0),
        '校正后p值': round(p_value_pd_corrected, 6),
        '显著性': sig
    })

df_health = pd.DataFrame(health_results)
df_health.to_csv('表S4_健康影响差异统计结果.csv', index=False, encoding='utf-8-sig')
print("\nSaved: 表S4_健康影响差异统计结果.csv")
print(df_health.to_string(index=False))

print("\n=== All analyses finished ===")
