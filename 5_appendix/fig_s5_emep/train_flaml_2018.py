#!/usr/bin/env python3
"""
参考 /home/qinjialing/grid_data/model.ipynb 的 FLAML 部分 (cells 0-4),
用 2018_12km_preML.csv 训练 automl1 模型, 输出训练好的模型.

参考 notebook 的训练逻辑:
  - 加载 preML CSV, dropna, 从 time 提取 month/weekday
  - 特征: ['blh','d2m','e','msdwlwrfcs','sp','t2m','tp','u10','v10',
           'arable','pasture','primary','secondary','urban','population','cmaq_MDA8']
  - 目标: obs_MDA8 (对应 notebook 中的 MDA8)
  - FLAML settings: time_budget=150, metric='r2', task='regression',
                    estimator_list=['lgbm','rf','xgboost','extra_tree','catboost']
  - pickle.dump(automl1, f) 保存模型
"""

import os
import pickle
import pandas as pd
import numpy as np
from flaml import AutoML
import warnings
warnings.filterwarnings('ignore')

# 路径
INPUT_FILE = '/home/qinjialing/validation_12km/2018_12km_preML.csv'
OUTPUT_DIR = '/home/qinjialing/validation_12km'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'automl1_2018.pkl')

# 特征列 (与 notebook cell 3/4 完全一致)
FEATURE_COLS = [
    'blh', 'd2m', 'e', 'msdwlwrfcs', 'sp', 't2m', 'tp', 'u10', 'v10',
    'arable', 'pasture', 'primary', 'secondary', 'urban',
    'population', 'cmaq_MDA8',
]
TARGET_COL = 'obs_MDA8'  # 对应 notebook 中的 MDA8


def main():
    print("=" * 60)
    print("训练 FLAML automl1 模型 (参考 model.ipynb cells 0-4)")
    print("=" * 60)

    # ---------- 1. 加载数据 (参考 cell 1) ----------
    print("\n[1/4] 加载数据...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"  列: {df.columns.tolist()}")

    # dropna (参考 cell 1: df = df.dropna())
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f"  dropna: {n_before} → {n_after} 行 (删除 {n_before - n_after} 行)")

    # 从 time 提取 month/weekday (参考 cell 1)
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['weekday'] = df['time'].dt.weekday
    print(f"  日期范围: {df['time'].min().date()} ~ {df['time'].max().date()}")

    # ---------- 2. 准备特征和目标 (参考 cell 3/4) ----------
    print("\n[2/4] 准备特征和目标...")
    X_train = df[FEATURE_COLS].copy()
    y_train = df[TARGET_COL].copy()
    print(f"  特征 ({len(FEATURE_COLS)} 个): {FEATURE_COLS}")
    print(f"  目标: {TARGET_COL}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_train 统计: min={y_train.min():.2f}, mean={y_train.mean():.2f}, "
          f"max={y_train.max():.2f}, std={y_train.std():.2f}")
    print(f"  cmaq_MDA8 统计 (ppbV): min={X_train['cmaq_MDA8'].min():.2f}, "
          f"mean={X_train['cmaq_MDA8'].mean():.2f}, max={X_train['cmaq_MDA8'].max():.2f}")

    # ---------- 3. 训练 FLAML AutoML (参考 cell 4) ----------
    print("\n[3/4] 训练 FLAML AutoML...")
    automl1 = AutoML()
    settings = {
        "time_budget": 150,  # 与 notebook 一致
        "metric": 'r2',
        "task": "regression",
        "estimator_list": ['lgbm', 'rf', 'xgboost', 'extra_tree', 'catboost'],
    }
    print(f"  settings: {settings}")
    automl1.fit(X_train=X_train, y_train=y_train, **settings)

    # 打印训练结果
    print("\n  训练完成!")
    print(f"  最佳模型: {automl1.best_estimator}")
    print(f"  最佳配置: {automl1.best_config}")
    print(f"  最佳 R² (训练): {automl1.best_loss:.6f} (loss, 越小越好)")
    # best_loss 是 1 - R² (对于 r2 metric)
    best_r2 = 1 - automl1.best_loss if automl1.best_loss is not None else None
    if best_r2 is not None:
        print(f"  最佳 R²: {best_r2:.6f}")

    # ---------- 4. 保存模型 (参考 cell 19) ----------
    print("\n[4/4] 保存模型...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(automl1, f)
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  模型已保存: {MODEL_PATH} ({file_size_mb:.1f} MB)")

    # ---------- 额外: 特征重要性 ----------
    print("\n" + "=" * 60)
    print("特征重要性")
    print("=" * 60)
    try:
        best_model = automl1.model.estimator
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': FEATURE_COLS,
                'Importance': importances,
            }).sort_values('Importance', ascending=False)
            print(feat_imp.to_string(index=False))
        else:
            print(f"  最佳模型 {type(best_model).__name__} 无 feature_importances_ 属性")
    except Exception as e:
        print(f"  获取特征重要性失败: {e}")

    # ---------- 额外: 训练集预测统计 ----------
    print("\n" + "=" * 60)
    print("训练集预测统计")
    print("=" * 60)
    try:
        y_pred = automl1.predict(X_train)
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.4f} μg/m³")
        print(f"  MAE:  {mae:.4f} μg/m³")
        print(f"  obs 均值:   {y_train.mean():.2f} μg/m³")
        print(f"  pred 均值: {y_pred.mean():.2f} μg/m³")
    except Exception as e:
        print(f"  预测统计失败: {e}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"  模型文件: {MODEL_PATH}")
    print(f"  最佳模型: {automl1.best_estimator}")
    print(f"  训练集 R²: {best_r2:.4f}" if best_r2 is not None else "")


if __name__ == '__main__':
    main()
