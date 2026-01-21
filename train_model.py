"""
Electricity Demand Forecasting - PROFESSIONAL Model Training
Industry-grade ML with time-series validation and advanced features
MS Elevate Capstone Project
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("🔋 PROFESSIONAL ELECTRICITY DEMAND FORECASTING SYSTEM")
print("=" * 80)

# Load dataset
print("\n📂 Loading professional dataset...")
data = pd.read_csv("data/electricity_demand.csv")
print(f"✅ Loaded {len(data):,} records")
print(f"\nDataset shape: {data.shape}")

# Display columns
print(f"\n📋 Available columns:")
for i, col in enumerate(data.columns, 1):
    print(f"   {i:2d}. {col}")

# Display basic statistics
print("\n📊 Dataset Statistics:")
print(data[['demand', 'temperature', 'humidity']].describe())

# Check for missing values
missing = data.isnull().sum()
print(f"\n🔍 Missing values:")
if missing.sum() == 0:
    print("   ✅ No missing values!")
else:
    print(missing[missing > 0])

# ========================================
# PROFESSIONAL FEATURE ENGINEERING
# ========================================
print("\n" + "=" * 80)
print("⚙️ PROFESSIONAL FEATURE ENGINEERING")
print("=" * 80)

# Handle missing values from lag/rolling features
print("\n🔧 Handling missing values from lag features...")
# Forward fill for first few rows where lag_24 is NaN
data = data.ffill()
# Any remaining NaNs, fill with mean
for col in data.columns:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].mean())

print(f"✅ Missing values handled: {data.isnull().sum().sum()} remaining")

# Select professional features
feature_columns = [
    # Time features
    'hour', 'day', 'month', 'season',
    # Binary indicators
    'is_weekend', 'is_peak', 'is_morning_peak', 'is_evening_peak', 'is_holiday',
    # Weather features
    'temperature', 'humidity', 'cooling_index', 'heating_index',
    # Time-series features (LAG - CRITICAL!)
    'lag_1', 'lag_24',
    # Trend features (ROLLING AVERAGE)
    'rolling_24', 'rolling_168'
]

print(f"\n📊 Professional Features ({len(feature_columns)}):")
feature_categories = {
    "Time Features": ['hour', 'day', 'month', 'season'],
    "Peak Indicators": ['is_weekend', 'is_peak', 'is_morning_peak', 'is_evening_peak'],
    "Event Detection": ['is_holiday'],
    "Weather": ['temperature', 'humidity', 'cooling_index', 'heating_index'],
    "Time-Series Intelligence": ['lag_1', 'lag_24'],
    "Trend Analysis": ['rolling_24', 'rolling_168']
}

for category, features in feature_categories.items():
    print(f"\n   {category}:")
    for feat in features:
        print(f"      ✓ {feat}")

X = data[feature_columns]
y = data['demand']

print(f"\n✅ Feature matrix: {X.shape}")
print(f"✅ Target vector: {y.shape}")

# ========================================
# TIME-BASED TRAIN-TEST SPLIT (PROFESSIONAL!)
# ========================================
print("\n" + "=" * 80)
print("📊 TIME-BASED TRAIN-TEST SPLIT (No Random Shuffling!)")
print("=" * 80)

# ✅ CORRECT: Time-based split (simulates real deployment)
split_index = int(len(data) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print(f"\n✅ Training samples: {len(X_train):,} ({len(X_train)/len(data)*100:.1f}%)")
print(f"✅ Testing samples:  {len(X_test):,} ({len(X_test)/len(data)*100:.1f}%)")
print(f"\n🎯 Why time-based? Future data should never train past predictions!")
print(f"   This simulates REAL-WORLD deployment scenario.")

# ========================================
# MODEL 1: Linear Regression (Baseline)
# ========================================
print("\n" + "=" * 80)
print("🔹 MODEL 1: Linear Regression (Baseline)")
print("=" * 80)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print(f"\n📈 Performance:")
print(f"   MAE:  {lr_mae:.2f} kWh")
print(f"   RMSE: {lr_rmse:.2f} kWh")
print(f"   R² Score: {lr_r2:.4f}")

# ========================================
# MODEL 2: Random Forest (Primary Model)
# ========================================
print("\n" + "=" * 80)
print("🔹 MODEL 2: Random Forest Regressor (Primary)")
print("=" * 80)

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("🔄 Training Random Forest with 500 trees...")
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"\n📈 Performance:")
print(f"   MAE:  {rf_mae:.2f} kWh")
print(f"   RMSE: {rf_rmse:.2f} kWh")
print(f"   R² Score: {rf_r2:.4f}")
print(f"   OOB Score: {rf_model.oob_score_:.4f}")

# ========================================
# MODEL 3: Gradient Boosting (Advanced)
# ========================================
print("\n" + "=" * 80)
print("🔹 MODEL 3: Gradient Boosting (Advanced)")
print("=" * 80)

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    random_state=42,
    verbose=0
)

print("🔄 Training Gradient Boosting...")
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print(f"\n📈 Performance:")
print(f"   MAE:  {gb_mae:.2f} kWh")
print(f"   RMSE: {gb_rmse:.2f} kWh")
print(f"   R² Score: {gb_r2:.4f}")

# ========================================
# FEATURE IMPORTANCE ANALYSIS
# ========================================
print("\n" + "=" * 80)
print("📊 FEATURE IMPORTANCE (Random Forest)")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:20s} → {row['importance']:.4f}")

# ========================================
# MODEL COMPARISON
# ========================================
print("\n" + "=" * 80)
print("📊 MODEL PERFORMANCE COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MAE': [lr_mae, rf_mae, gb_mae],
    'RMSE': [lr_rmse, rf_rmse, gb_rmse],
    'R²': [lr_r2, rf_r2, gb_r2]
})

print("\n" + comparison.to_string(index=False))

# Best model selection
best_idx = comparison['R²'].idxmax()
best_model_name = comparison.loc[best_idx, 'Model']
best_model = rf_model if best_idx == 1 else (gb_model if best_idx == 2 else lr_model)

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² Score: {comparison.loc[best_idx, 'R²']:.4f}")

# ========================================
# SAVE MODELS
# ========================================
print("\n" + "=" * 80)
print("💾 SAVING MODELS")
print("=" * 80)

os.makedirs('models', exist_ok=True)

joblib.dump(lr_model, 'models/linear_regression.pkl')
joblib.dump(rf_model, 'models/random_forest.pkl')
joblib.dump(gb_model, 'models/gradient_boosting.pkl')
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')

# Save model metadata
metadata = {
    'best_model': best_model_name,
    'r2_score': float(comparison.loc[best_idx, 'R²']),
    'mae': float(comparison.loc[best_idx, 'MAE']),
    'rmse': float(comparison.loc[best_idx, 'RMSE']),
    'features': feature_columns,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}
joblib.dump(metadata, 'models/metadata.pkl')

print("✅ Saved models:")
print("   → models/linear_regression.pkl")
print("   → models/random_forest.pkl")
print("   → models/gradient_boosting.pkl")
print("   → models/best_model.pkl")
print("   → models/feature_columns.pkl")
print("   → models/metadata.pkl")

# ========================================
# PROFESSIONAL VISUALIZATIONS
# ========================================
print("\n" + "=" * 80)
print("📈 GENERATING PROFESSIONAL VISUALIZATIONS")
print("=" * 80)

os.makedirs('outputs', exist_ok=True)

# 1. Actual vs Predicted (Best Model)
plt.figure(figsize=(16, 7))
sample_size = min(500, len(y_test))
x_axis = range(sample_size)

plt.plot(x_axis, y_test.values[:sample_size], label='Actual Demand', 
         linewidth=2.5, alpha=0.8, color='#4CC9F0')
plt.plot(x_axis, rf_pred[:sample_size], label=f'Predicted ({best_model_name})', 
         linewidth=2.5, alpha=0.8, color='#F72585', linestyle='--')

plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
plt.ylabel('Electricity Demand (kWh)', fontsize=12, fontweight='bold')
plt.title('Professional Demand Forecasting: Actual vs Predicted', 
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/actual_vs_predicted.png")
plt.close()

# 2. Model Comparison
plt.figure(figsize=(12, 7))
x = np.arange(len(comparison))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width, comparison['MAE'], width, label='MAE', 
               color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x, comparison['RMSE'], width, label='RMSE', 
               color='#4ECDC4', alpha=0.8)
bars3 = ax.bar(x + width, comparison['R²'] * 1000, width, label='R² (×1000)', 
               color='#45B7D1', alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comparison['Model'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/model_comparison.png")
plt.close()

# 3. Feature Importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

plt.barh(top_features['feature'], top_features['importance'], color=colors, alpha=0.8)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 15 Feature Importances (Random Forest)', 
          fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/feature_importance.png")
plt.close()

# 4. Residual Plot
plt.figure(figsize=(12, 7))
residuals = y_test - rf_pred
plt.scatter(rf_pred, residuals, alpha=0.5, color='#A29BFE', s=30)
plt.axhline(y=0, color='#FF006E', linestyle='--', linewidth=2.5)
plt.xlabel('Predicted Demand (kWh)', fontsize=12, fontweight='bold')
plt.ylabel('Residuals (kWh)', fontsize=12, fontweight='bold')
plt.title('Residual Analysis (Random Forest)', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('outputs/residual_plot.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/residual_plot.png")
plt.close()

# 5. Peak vs Off-Peak Performance
plt.figure(figsize=(12, 7))
test_data = data[split_index:].copy()
test_data['predicted'] = rf_pred
test_data['actual'] = y_test.values

peak_data = test_data[test_data['is_peak'] == 1]
offpeak_data = test_data[test_data['is_peak'] == 0]

metrics = pd.DataFrame({
    'Period': ['Peak Hours', 'Off-Peak Hours'],
    'MAE': [
        mean_absolute_error(peak_data['actual'], peak_data['predicted']),
        mean_absolute_error(offpeak_data['actual'], offpeak_data['predicted'])
    ],
    'R²': [
        r2_score(peak_data['actual'], peak_data['predicted']),
        r2_score(offpeak_data['actual'], offpeak_data['predicted'])
    ]
})

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, metrics['MAE'], width, label='MAE', 
               color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x + width/2, metrics['R²'] * 1000, width, label='R² (×1000)', 
               color='#4CC9F0', alpha=0.8)

ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Peak vs Off-Peak Performance Analysis', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics['Period'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('outputs/peak_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: outputs/peak_analysis.png")
plt.close()

# ========================================
# SUMMARY
# ========================================
print("\n" + "=" * 80)
print("✅ PROFESSIONAL ML TRAINING COMPLETE!")
print("=" * 80)

print(f"\n🎯 Key Achievements:")
print(f"   ✓ Time-based validation (no data leakage)")
print(f"   ✓ Lag features for time-series intelligence")
print(f"   ✓ Rolling averages for trend detection")
print(f"   ✓ Peak/off-peak analysis")
print(f"   ✓ Holiday impact modeling")
print(f"   ✓ Temperature indices (human behavior)")

print(f"\n📊 Best Model Performance:")
print(f"   Model: {best_model_name}")
print(f"   R² Score: {comparison.loc[best_idx, 'R²']:.4f}")
print(f"   RMSE: {comparison.loc[best_idx, 'RMSE']:.2f} kWh")
print(f"   MAE: {comparison.loc[best_idx, 'MAE']:.2f} kWh")

print(f"\n📁 Outputs:")
print(f"   Models: models/")
print(f"   Visualizations: outputs/")

print(f"\n🚀 Ready for production deployment!")
print("=" * 80)
