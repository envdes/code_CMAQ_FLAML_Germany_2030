#!/usr/bin/env python3
"""
1. 用训练好的 automl1_2018.pkl 模型应用于 2019_12km_preML.csv, 输出包含 pred 的结果.
2. 对 O3_sites.csv 中的站点, 计算 2019 年 obs MDA8, 匹配最近 12km 网格,
   输出 pred 和 obs 的验证结果 (R², RMSE, MAE).

输入:
  - /home/qinjialing/validation_12km/automl1_2018.pkl  (训练好的模型)
  - /home/qinjialing/trae_code/appendix/2019_12km_preML.csv  (2019 preML)
  - /home/qinjialing/obs_sites/O3_sites.csv  (验证用站点列表, 347 个)
  - /home/qinjialing/obs_sites/europe_sites.lonlat.txt  (站点经纬度)
  - /home/qinjialing/obs_sites/2019/<site>.O3.ts.txt  (站点小时 O3 数据)

输出:
  - /home/qinjialing/validation_12km/2019_12km_preML_pred.csv  (preML + pred 列)
  - /home/qinjialing/validation_12km/validation_2019_pred_obs.csv  (站点 × 日期验证表)
  - /home/qinjialing/validation_12km/validation_2019_metrics.csv  (验证指标)
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 路径
MODEL_PATH = '/home/qinjialing/validation_12km/automl1_2018.pkl'
PREML_2019 = '/home/qinjialing/trae_code/appendix/2019_12km_preML.csv'
SITES_FILE = '/home/qinjialing/obs_sites/O3_sites.csv'
LONLAT_FILE = '/home/qinjialing/obs_sites/europe_sites.lonlat.txt'
OBS_DIR = '/home/qinjialing/obs_sites/2019'
OUTPUT_DIR = '/home/qinjialing/validation_12km'
OUTPUT_PREML_PRED = os.path.join(OUTPUT_DIR, '2019_12km_preML_pred.csv')
OUTPUT_VAL = os.path.join(OUTPUT_DIR, 'validation_2019_pred_obs.csv')
OUTPUT_METRICS = os.path.join(OUTPUT_DIR, 'validation_2019_metrics.csv')

# 模型特征 (与训练时一致)
FEATURE_COLS = [
    'blh', 'd2m', 'e', 'msdwlwrfcs', 'sp', 't2m', 'tp', 'u10', 'v10',
    'arable', 'pasture', 'primary', 'secondary', 'urban',
    'population', 'cmaq_MDA8',
]

EARTH_RADIUS_KM = 6371.0


def latlon_to_xyz(lat, lon):
    """lat/lon (度) → 单位球面 3D 笛卡尔坐标 (haversine cKDTree)"""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])


def compute_mda8_for_site(file_path):
    """
    按 EPA 标准计算单站 MDA8 (参考 match_mda8_nearest.py).
    返回: pd.Series, 索引为日期 (Timestamp 午夜), 值为 MDA8 (μg/m³).
    """
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None,
                         names=['timestamp', 'o3'])
    except Exception:
        return None
    if len(df) == 0:
        return None

    # 解析 YYYYMMDDHH
    df['datetime'] = pd.to_datetime(df['timestamp'].astype(str), format='%Y%m%d%H')
    df = df.set_index('datetime').sort_index()

    # 聚合重复时间戳
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()

    # 重采样到小时频率
    df = df.resample('1H').asfreq()

    # 8 小时滑动平均 (trailing, 至少 6 个有效小时)
    df['ma8'] = df['o3'].rolling(window=8, min_periods=6).mean()

    # EPA: 每天 17 个窗口对应 rolling 在小时 7..23 (起始小时 0..16)
    df['date'] = df.index.normalize()
    df['hour'] = df.index.hour
    mda8 = df[df['hour'] >= 7].groupby('date')['ma8'].max()
    return mda8


def main():
    print("=" * 60)
    print("1. 应用模型到 2019 preML  2. pred vs obs 验证")
    print("=" * 60)

    # ---------- 1. 加载模型 ----------
    print("\n[1/6] 加载训练好的模型...")
    with open(MODEL_PATH, 'rb') as f:
        automl1 = pickle.load(f)
    print(f"  模型: {automl1.best_estimator}, "
          f"训练 R²={1 - automl1.best_loss:.4f}")

    # ---------- 2. 加载 2019 preML 并预测 ----------
    print("\n[2/6] 加载 2019 preML 并预测...")
    df = pd.read_csv(PREML_2019)
    print(f"  preML: {len(df)} 行, {df['index'].nunique()} 网格, "
          f"{df['time'].nunique()} 天")
    print(f"  日期范围: {df['time'].min()} ~ {df['time'].max()}")

    # 检查特征列
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  错误: 缺少特征列 {missing}")
        return

    # 预测
    X = df[FEATURE_COLS]
    print(f"  预测 {len(X)} 行...")
    pred = automl1.predict(X)
    df['pred'] = pred
    print(f"  pred 统计 (μg/m³): min={pred.min():.2f}, "
          f"mean={pred.mean():.2f}, max={pred.max():.2f}")

    # 保存 preML + pred
    df.to_csv(OUTPUT_PREML_PRED, index=False)
    sz = os.path.getsize(OUTPUT_PREML_PRED) / (1024 * 1024)
    print(f"  已保存: {OUTPUT_PREML_PRED} ({sz:.1f} MB)")

    # ---------- 3. 加载验证站点 ----------
    print("\n[3/6] 加载验证站点...")
    sites_df = pd.read_csv(SITES_FILE, index_col=0)
    site_ids = sites_df['station'].tolist()
    print(f"  O3_sites.csv: {len(site_ids)} 个站点")

    # 加载经纬度
    lonlat = {}
    with open(LONLAT_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                lonlat[parts[0]] = (float(parts[1]), float(parts[2]))

    # 筛选: 有经纬度 + 有 2019 obs 文件
    valid_sites = []
    for s in site_ids:
        if s in lonlat and os.path.exists(os.path.join(OBS_DIR, f'{s}.O3.ts.txt')):
            valid_sites.append({
                'site': s,
                'lat': lonlat[s][0],
                'lon': lonlat[s][1],
            })
    sites_df = pd.DataFrame(valid_sites)
    print(f"  有效站点 (有经纬度 + 有 2019 obs): {len(sites_df)}")

    # ---------- 4. 计算各站点 obs MDA8 ----------
    print("\n[4/6] 计算各站点 obs MDA8 (EPA 标准)...")
    site_mda8 = {}  # site -> pd.Series (date -> MDA8)
    for i, row in sites_df.iterrows():
        site = row['site']
        mda8 = compute_mda8_for_site(os.path.join(OBS_DIR, f'{site}.O3.ts.txt'))
        if mda8 is not None and len(mda8) > 0:
            site_mda8[site] = mda8
        if (i + 1) % 100 == 0 or (i + 1) == len(sites_df):
            print(f"  已处理 {i + 1}/{len(sites_df)} 个站点, "
                  f"有效 {len(site_mda8)}")
    print(f"  有效 MDA8 站点: {len(site_mda8)}")

    # ---------- 5. 匹配站点到最近 12km 网格 ----------
    print("\n[5/6] 匹配站点到最近 12km 网格...")
    # 从 2019 preML 提取唯一网格 (index, lat, lon)
    grids = df[['index', 'lat', 'lon']].drop_duplicates(subset='index').reset_index(drop=True)
    print(f"  2019 preML 网格数: {len(grids)}")

    # KDTree: 网格 → 站点最近邻
    grids_xyz = latlon_to_xyz(grids['lat'].values, grids['lon'].values)
    sites_xyz = latlon_to_xyz(sites_df['lat'].values, sites_df['lon'].values)
    tree = cKDTree(grids_xyz)
    dist_chord, idx_grid = tree.query(sites_xyz, k=1)
    dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(dist_chord / 2, 0, 1))

    sites_df['nearest_grid'] = grids['index'].values[idx_grid]
    sites_df['dist_km'] = dist_km
    print(f"  站点 → 最近网格距离 (km): min={dist_km.min():.2f}, "
          f"median={np.median(dist_km):.2f}, max={dist_km.max():.2f}")

    # ---------- 6. 构建 pred vs obs 验证表 ----------
    print("\n[6/6] 构建验证表并计算指标...")
    # pred 透视表: index × time → pred
    pred_pivot = df.pivot_table(index='index', columns='time', values='pred', aggfunc='first')
    print(f"  pred 透视表: {pred_pivot.shape[0]} 网格 × {pred_pivot.shape[1]} 天")

    # 构建验证记录
    val_records = []
    for _, row in sites_df.iterrows():
        site = row['site']
        if site not in site_mda8:
            continue
        grid_idx = row['nearest_grid']
        obs_series = site_mda8[site]
        if grid_idx not in pred_pivot.index:
            continue
        pred_series = pred_pivot.loc[grid_idx]
        # 按日期对齐
        obs_df = obs_series.reset_index()
        obs_df.columns = ['date', 'obs_MDA8']
        obs_df['time'] = obs_df['date'].dt.strftime('%Y-%m-%d')
        pred_df = pred_series.reset_index()
        pred_df.columns = ['time', 'pred']
        merged = obs_df.merge(pred_df, on='time', how='inner')
        for _, m in merged.iterrows():
            if pd.notna(m['obs_MDA8']) and pd.notna(m['pred']):
                val_records.append({
                    'site': site,
                    'site_lat': row['lat'],
                    'site_lon': row['lon'],
                    'nearest_grid': int(grid_idx),
                    'dist_km': row['dist_km'],
                    'time': m['time'],
                    'obs_MDA8': m['obs_MDA8'],
                    'pred': m['pred'],
                })
    val_df = pd.DataFrame(val_records)
    print(f"  验证记录: {len(val_df)} 行, {val_df['site'].nunique()} 个站点")

    # 计算指标
    obs = val_df['obs_MDA8'].values
    pred = val_df['pred'].values
    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae = mean_absolute_error(obs, pred)
    # MNB, MNE (用户之前要求不显示, 但计算备用)
    mnb = np.mean((pred - obs) / np.where(obs != 0, obs, np.nan)) * 100
    mne = np.mean(np.abs(pred - obs) / np.where(obs != 0, obs, np.nan)) * 100

    print(f"\n  验证指标 (pred vs obs_MDA8, μg/m³):")
    print(f"    样本数:  {len(val_df)}")
    print(f"    站点数:  {val_df['site'].nunique()}")
    print(f"    R²:      {r2:.4f}")
    print(f"    RMSE:    {rmse:.4f}")
    print(f"    MAE:     {mae:.4f}")
    print(f"    obs 均值:  {obs.mean():.2f}")
    print(f"    pred 均值: {pred.mean():.2f}")
    print(f"    obs 范围:  {obs.min():.2f} ~ {obs.max():.2f}")
    print(f"    pred 范围: {pred.min():.2f} ~ {pred.max():.2f}")

    # 保存验证表
    val_df.to_csv(OUTPUT_VAL, index=False)
    sz = os.path.getsize(OUTPUT_VAL) / (1024 * 1024)
    print(f"\n  验证表已保存: {OUTPUT_VAL} ({sz:.1f} MB)")

    # 保存指标
    metrics_df = pd.DataFrame([{
        'n_samples': len(val_df),
        'n_sites': val_df['site'].nunique(),
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_obs': obs.mean(),
        'mean_pred': pred.mean(),
        'min_obs': obs.min(),
        'max_obs': obs.max(),
        'min_pred': pred.min(),
        'max_pred': pred.max(),
    }])
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"  指标已保存: {OUTPUT_METRICS}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"  preML + pred: {OUTPUT_PREML_PRED}")
    print(f"  验证表:       {OUTPUT_VAL}")
    print(f"  验证指标:     {OUTPUT_METRICS}")


if __name__ == '__main__':
    main()
