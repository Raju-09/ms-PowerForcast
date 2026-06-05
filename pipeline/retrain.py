"""
Automated Model Retraining Pipeline
==================================
Runs in background to retrain models, compare performance, and promote the best model.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

print("=" * 80)
print("🌀 RUNNING AUTOMATED RETRAINING PIPELINE")
print("=" * 80)

# Check paths
DATA_PATH = "data/aep_enriched.csv"
METADATA_PATH = "models/metadata.pkl"
BEST_MODEL_PATH = "models/best_model.pkl"

if not os.path.exists(DATA_PATH):
    print(f"❌ Data file not found: {DATA_PATH}. Ingestion required first.")
    sys.exit(1)

# 1. Load Data
print("\n📂 Loading enriched grid dataset...")
data = pd.read_csv(DATA_PATH)
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.sort_values('Datetime').reset_index(drop=True)
print(f"✅ Loaded {len(data):,} records")

# Features & Target
feature_columns = [
    'temperature', 'humidity', 'precipitation', 'windspeed', 'hour', 'day', 'month', 'year', 'day_of_year',
    'week', 'quarter', 'season', 'is_weekend', 'is_peak', 'is_morning_peak', 'is_evening_peak', 'is_night', 'is_holiday',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'cooling_degree', 'heating_degree',
    'feels_like', 'wind_chill', 'lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 'lag_168',
    'rolling_6', 'rolling_24', 'rolling_168', 'rolling_720', 'rolling_24_std', 'rolling_168_std', 'demand_delta_1w'
]
feature_columns = [col for col in feature_columns if col in data.columns]

X = data[feature_columns]
y = data['demand']

# Split train/test (last 20% is holdout)
split_idx = int(len(data) * 0.8)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

# 2. Load current metadata
current_r2 = 0.0
current_best_model = "None"
if os.path.exists(METADATA_PATH):
    try:
         meta = joblib.load(METADATA_PATH)
         current_r2 = meta.get('r2_score', 0.0)
         current_best_model = meta.get('best_model', 'None')
         print(f"📊 Current Active Model: {current_best_model} | Current Holdout R²: {current_r2:.6f}")
    except Exception as e:
         print(f"⚠️ Failed to load existing metadata: {e}")

# 3. Train Candidate Models
print("\n🔄 Training candidate models on training set...")

# Model A: LightGBM
print("   Training LightGBM candidate...")
lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_r2 = r2_score(y_test, lgb_preds)
lgb_mae = mean_absolute_error(y_test, lgb_preds)
print(f"   ✓ LightGBM candidate R²: {lgb_r2:.6f} | MAE: {lgb_mae:.2f} MW")

# Model B: XGBoost
print("   Training XGBoost candidate...")
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_preds)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
print(f"   ✓ XGBoost candidate R²: {xgb_r2:.6f} | MAE: {xgb_mae:.2f} MW")

# Select best candidate
if lgb_r2 >= xgb_r2:
    best_candidate_name = "LightGBM"
    best_candidate_model = lgb_model
    best_candidate_r2 = lgb_r2
    best_candidate_mae = lgb_mae
    best_candidate_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    best_candidate_mape = np.mean(np.abs((y_test - lgb_preds) / y_test)) * 100
else:
    best_candidate_name = "XGBoost"
    best_candidate_model = xgb_model
    best_candidate_r2 = xgb_r2
    best_candidate_mae = xgb_mae
    best_candidate_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    best_candidate_mape = np.mean(np.abs((y_test - xgb_preds) / y_test)) * 100

print(f"\n🏆 Best Candidate: {best_candidate_name} with R²: {best_candidate_r2:.6f}")

# 4. Evaluation and Promotion
is_promoted = False
# We promote the candidate if it outperforms the active model's R2, or if no active model exists
if best_candidate_r2 > current_r2 or current_best_model == "None":
    print(f"\n🚀 PROMOTING CANDIDATE! New R² ({best_candidate_r2:.6f}) > Active R² ({current_r2:.6f})")
    
    # Save best model
    joblib.dump(best_candidate_model, BEST_MODEL_PATH)
    if best_candidate_name == "LightGBM":
        joblib.dump(best_candidate_model, "models/lgbm_model.pkl")
    else:
        joblib.dump(best_candidate_model, "models/xgboost_model.pkl")
        
    # Update Metadata
    meta = {
        'best_model': best_candidate_name,
        'r2_score': float(best_candidate_r2),
        'mae': float(best_candidate_mae),
        'rmse': float(best_candidate_rmse),
        'mape': float(best_candidate_mape),
        'features': feature_columns,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    joblib.dump(meta, METADATA_PATH)
    
    # Retrain Quantiles for promoted best model context (in case distributions shifted)
    print("   Retraining LightGBM Quantile regression uncertainty bands...")
    lgb_q10 = lgb.LGBMRegressor(objective='quantile', alpha=0.1, n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_q10.fit(X_train, y_train)
    joblib.dump(lgb_q10, "models/lgbm_quantile_10.pkl")
    
    lgb_q90 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_q90.fit(X_train, y_train)
    joblib.dump(lgb_q90, "models/lgbm_quantile_90.pkl")
    
    is_promoted = True
    print("✅ Model Promotion Complete!")
else:
    print(f"\n❌ KEEPING ACTIVE MODEL. Candidate R² ({best_candidate_r2:.6f}) <= Active R² ({current_r2:.6f})")

# Log retrain event in logs folder
os.makedirs("outputs", exist_ok=True)
with open("outputs/retrain_log.txt", "a") as f:
    f.write(f"{pd.Timestamp.now()} - Retrained: best_candidate={best_candidate_name} ({best_candidate_r2:.6f}), promoted={is_promoted}\n")

print("\n" + "=" * 80)
print("✅ RETRAINING PIPELINE RUN COMPLETE")
print("=" * 80)
