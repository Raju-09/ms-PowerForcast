"""
Electricity Demand Forecasting - PROFESSIONAL Model Training
Industry-grade ML with time-series validation, advanced features,
quantile regression confidence intervals, and SHAP explainability.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("🔋 POWER GRID INTELLIGENCE PLATFORM — MODEL TRAINING")
print("=" * 80)

# Load dataset
print("\n📂 Loading enriched dataset...")
if not os.path.exists("data/aep_enriched.csv"):
    print("❌ Error: data/aep_enriched.csv not found! Run ingest_data.py first.")
    exit(1)

data = pd.read_csv("data/aep_enriched.csv")
print(f"✅ Loaded {len(data):,} records")
print(f"Dataset shape: {data.shape}")

# Parse Datetime
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.sort_values('Datetime').reset_index(drop=True)

# Select features
feature_columns = [
    'temperature', 'humidity', 'precipitation', 'windspeed', 'hour', 'day', 'month', 'year', 'day_of_year',
    'week', 'quarter', 'season', 'is_weekend', 'is_peak', 'is_morning_peak', 'is_evening_peak', 'is_night', 'is_holiday',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'cooling_degree', 'heating_degree',
    'feels_like', 'wind_chill', 'lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 'lag_168',
    'rolling_6', 'rolling_24', 'rolling_168', 'rolling_720', 'rolling_24_std', 'rolling_168_std', 'demand_delta_1w'
]

# Ensure all feature columns exist in data
feature_columns = [col for col in feature_columns if col in data.columns]
print(f"\n📊 Features selected for training ({len(feature_columns)}):")
for col in feature_columns:
    print(f"  ✓ {col}")

X = data[feature_columns]
y = data['demand']

# Calculate drift baseline on the entire feature set
print("\n⚙️ Calculating drift baseline stats (mean & std)...")
drift_baseline = {}
for col in feature_columns:
    drift_baseline[col] = {
        'mean': float(X[col].mean()),
        'std': float(X[col].std()) if X[col].std() > 0 else 1.0
    }
os.makedirs("models", exist_ok=True)
joblib.dump(drift_baseline, 'models/drift_baseline.pkl')
print("✅ Saved models/drift_baseline.pkl")

# ========================================
# TIME-BASED TRAIN-TEST SPLIT
# ========================================
print("\n" + "=" * 80)
print("📊 TIME-BASED TRAIN-TEST SPLIT (80% Train, 20% Test)")
print("=" * 80)

split_idx = int(len(data) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_test  = y.iloc[split_idx:]

print(f"✅ Training samples: {len(X_train):,} ({X_train.index.min()} to {X_train.index.max()})")
print(f"✅ Testing samples:  {len(X_test):,} ({X_test.index.min()} to {X_test.index.max()})")

# ========================================
# TIME-SERIES CROSS-VALIDATION
# ========================================
print("\n" + "=" * 80)
print("🔄 TIME-SERIES CROSS-VALIDATION (5 Splits)")
print("=" * 80)

tscv = TimeSeriesSplit(n_splits=5)
models_to_evaluate = {
    'Linear Regression': LinearRegression(),
    'LightGBM': lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
}

cv_results = {model_name: [] for model_name in models_to_evaluate}

for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
    print(f"\n🌀 Fold {fold+1}:")
    cv_X_train, cv_X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    cv_y_train, cv_y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    print(f"   Train size: {len(cv_X_train):,}, Val size: {len(cv_X_val):,}")
    
    for name, model in models_to_evaluate.items():
        model.fit(cv_X_train, cv_y_train)
        pred = model.predict(cv_X_val)
        mae = mean_absolute_error(cv_y_val, pred)
        rmse = np.sqrt(mean_squared_error(cv_y_val, pred))
        r2 = r2_score(cv_y_val, pred)
        mape = np.mean(np.abs((cv_y_val - pred) / cv_y_val)) * 100
        
        cv_results[name].append({
            'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape
        })
        print(f"   {name:<20} | MAE: {mae:7.2f} MW | MAPE: {mape:5.2f}% | R²: {r2:6.4f}")

print("\n📊 Average CV Performance:")
for name in models_to_evaluate:
    avg_mae = np.mean([r['mae'] for r in cv_results[name]])
    avg_mape = np.mean([r['mape'] for r in cv_results[name]])
    avg_r2 = np.mean([r['r2'] for r in cv_results[name]])
    print(f"   {name:<20} | Avg MAE: {avg_mae:7.2f} MW | Avg MAPE: {avg_mape:5.2f}% | Avg R²: {avg_r2:6.4f}")

# ========================================
# FINAL MODEL TRAINING
# ========================================
print("\n" + "=" * 80)
print("🏆 FINAL MODEL TRAINING ON WHOLE TRAIN SET")
print("=" * 80)

trained_models = {}
metrics = {}

for name, model in models_to_evaluate.items():
    print(f"Training final {name} model...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    
    metrics[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
    trained_models[name] = model
    print(f"   {name:<20} | Test MAE: {mae:7.2f} MW | Test MAPE: {mape:5.2f}% | Test R²: {r2:6.4f}")

# Train Quantile Regression for Confidence Intervals
print("\n🔮 Training Quantile Models for Uncertainty (Confidence Intervals)...")
print("   Training LightGBM 10th percentile model...")
lgb_q10 = lgb.LGBMRegressor(objective='quantile', alpha=0.1, n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
lgb_q10.fit(X_train, y_train)

print("   Training LightGBM 90th percentile model...")
lgb_q90 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
lgb_q90.fit(X_train, y_train)

# Select best main model (based on R2 score)
best_model_name = 'XGBoost' if metrics['XGBoost']['r2'] >= metrics['LightGBM']['r2'] else 'LightGBM'
best_model = trained_models[best_model_name]
print(f"\n🏆 Best Main Model Selected: {best_model_name}")

# ========================================
# SHAP EXPLAINABILITY
# ========================================
print("\n" + "=" * 80)
print("🧠 SHAP EXPLAINABILITY")
print("=" * 80)

# Create TreeExplainer on the best model
try:
    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(best_model)
    print("Computing SHAP values for a sample of 200 test instances...")
    # Sample test data for SHAP summary
    shap_sample_X = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_values = explainer(shap_sample_X)
    
    # Save SHAP assets
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    # Save the explainer and precomputed shap values
    joblib.dump({'shap_values': shap_values, 'sample_X': shap_sample_X}, 'models/shap_summary.pkl')
    print("✅ SHAP explainer and summary saved successfully")
except Exception as e:
    print(f"⚠️ Warning: SHAP generation failed ({e}). Will fall back to on-the-fly SHAP or simplified explanations.")

# ========================================
# SAVE ALL MODEL ARTIFACTS
# ========================================
print("\n" + "=" * 80)
print("💾 SAVING ARTIFACTS")
print("=" * 80)

joblib.dump(trained_models['Linear Regression'], 'models/linear_regression.pkl')
joblib.dump(trained_models['XGBoost'], 'models/xgboost_model.pkl')
joblib.dump(trained_models['LightGBM'], 'models/lgbm_model.pkl')
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(lgb_q10, 'models/lgbm_quantile_10.pkl')
joblib.dump(lgb_q90, 'models/lgbm_quantile_90.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')

# Save model metadata
metadata = {
    'best_model': best_model_name,
    'r2_score': float(metrics[best_model_name]['r2']),
    'mae': float(metrics[best_model_name]['mae']),
    'mape': float(metrics[best_model_name]['mape']),
    'rmse': float(metrics[best_model_name]['rmse']),
    'features': feature_columns,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}
joblib.dump(metadata, 'models/metadata.pkl')

print("✅ All pickle models saved in models/")

# ========================================
# VISUALIZATIONS
# ========================================
print("\n" + "=" * 80)
print("📈 GENERATING PROFESSIONAL VISUALIZATIONS")
print("=" * 80)

os.makedirs('outputs', exist_ok=True)

# 1. Actual vs Predicted (Best Model)
plt.figure(figsize=(16, 7))
sample_size = min(500, len(y_test))
x_axis = range(sample_size)
best_preds = best_model.predict(X_test.iloc[:sample_size])
lower_preds = lgb_q10.predict(X_test.iloc[:sample_size])
upper_preds = lgb_q90.predict(X_test.iloc[:sample_size])

plt.plot(x_axis, y_test.iloc[:sample_size].values, label='Actual Demand', linewidth=2.0, color='#4CC9F0')
plt.plot(x_axis, best_preds, label=f'Predicted ({best_model_name})', linewidth=2.0, color='#F72585', linestyle='--')
plt.fill_between(x_axis, lower_preds, upper_preds, color='#7209B7', alpha=0.15, label='80% Confidence Interval')

plt.xlabel('Sample Index (Hourly)', fontsize=12, fontweight='bold')
plt.ylabel('Electricity Demand (MW)', fontsize=12, fontweight='bold')
plt.title('Power Grid Demand Forecasting: Actual vs Predicted with Uncertainty Bands', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png', dpi=150)
print("✅ Saved: outputs/actual_vs_predicted.png")
plt.close()

# 2. Model Comparison
comparison_df = pd.DataFrame([
    {'Model': name, 'MAE': details['mae'], 'MAPE (%)': details['mape'], 'R²': details['r2']}
    for name, details in metrics.items()
])
print("\n" + comparison_df.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=comparison_df, palette='viridis')
plt.title('Model MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Mean Absolute Error (MW)', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150)
print("✅ Saved: outputs/model_comparison.png")
plt.close()

# 3. Feature Importance (Best Model)
if best_model_name == 'XGBoost':
    importances = best_model.feature_importances_
else:
    importances = best_model.feature_importances_

feat_imp_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feat_imp_df.head(15), palette='magma')
plt.title(f'Top 15 Feature Importances ({best_model_name})', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
print("✅ Saved: outputs/feature_importance.png")
plt.close()

print("\n🎉 MODEL TRAINING PROCESS COMPLETE!")
print("=" * 80)
