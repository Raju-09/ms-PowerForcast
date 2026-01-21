"""
Generate realistic electricity demand sample data with PROFESSIONAL FEATURES
Industry-grade features for real-world power utility systems
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("🔋 PROFESSIONAL ELECTRICITY DEMAND DATA GENERATOR")
print("=" * 70)

# India Public Holidays 2023
INDIA_HOLIDAYS = [
    '2023-01-26', '2023-03-08', '2023-03-22', '2023-04-04', '2023-04-07',
    '2023-04-14', '2023-05-05', '2023-06-29', '2023-08-15', '2023-08-30',
    '2023-09-19', '2023-10-02', '2023-10-24', '2023-11-12', '2023-11-13',
    '2023-11-27', '2023-12-25',
]

# Generate date range (1 year of hourly data)
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 1)
date_range = pd.date_range(start=start_date, end=end_date, freq='H')

print(f"\n📅 Generating data from {start_date.date()} to {end_date.date()}")
print(f"📊 Total records: {len(date_range)}")

# Create DataFrame directly
df = pd.DataFrame({
    'datetime': date_range,
})

# Extract time features
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)

# ========================================
# FEATURE 1: Peak/Off-Peak Indicator
# ========================================
print("\n✅ Feature 1: Peak/Off-Peak Indicators")
df['is_peak'] = ((df['hour'] >= 6) & (df['hour'] <= 10) | 
                  (df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)
print(f"   Peak hours: 6-10 AM, 6-11 PM")
print(f"   Used for: Load scheduling, dynamic pricing, grid balance")

# ========================================
# FEATURE 2: Temperature & Temperature Index
# ========================================
print("\n✅ Feature 2: Temperature Index (Human Behavior Model)")
# Generate realistic temperature data
base_temp = 20 + 10 * np.sin(2 * np.pi * df['month'] / 12)
daily_variation = 5 * np.sin(2 * np.pi * df['hour'] / 24)
noise = np.random.normal(0, 2, len(df))
df['temperature'] = base_temp + daily_variation + noise

# Cooling Index: AC usage starts after 24°C
df['cooling_index'] = np.maximum(0, df['temperature'] - 24)

# Heating Index: Heater usage starts below 18°C
df['heating_index'] = np.maximum(0, 18 - df['temperature'])

print(f"   Cooling threshold: >24°C (AC demand)")
print(f"   Heating threshold: <18°C (Heater demand)")
print(f"   Models: Human behavior, not just raw temperature")

# ========================================
# FEATURE 3: Holiday Detection
# ========================================
print("\n✅ Feature 3: Holiday & Event Impact")
df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d')
df['is_holiday'] = df['date_str'].isin(INDIA_HOLIDAYS).astype(int)
df = df.drop('date_str', axis=1)  # Remove temporary column

print(f"   Holidays included: {len(INDIA_HOLIDAYS)} Indian public holidays")
print(f"   Impact: 15-30% demand reduction on holidays")

# ========================================
# WEATHER FEATURES
# ========================================
# Generate humidity
df['humidity'] = 70 - (df['temperature'] - 20) * 1.5 + np.random.normal(0, 5, len(df))
df['humidity'] = np.clip(df['humidity'], 30, 95)

# ========================================
# GENERATE REALISTIC ELECTRICITY DEMAND
# ========================================
print("\n⚡ Generating realistic electricity demand patterns...")

# Base demand
base_demand = 5000

# Hour of day pattern
hour_factor = 1 + 0.3 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)

# Day of week pattern
day_factor = np.where(df['is_weekend'] == 1, 0.85, 1.0)

# Seasonal pattern
seasonal_factor = 1 + 0.2 * np.abs(np.sin(2 * np.pi * df['month'] / 12))

# Temperature effect
temp_factor = 1 + 0.02 * np.abs(df['temperature'] - 20)

# Peak hour effect
peak_factor = np.where(df['is_peak'] == 1, 1.2, 1.0)

# Holiday effect
holiday_factor = np.where(df['is_holiday'] == 1, 0.7, 1.0)

# Calculate demand
demand = base_demand * hour_factor * day_factor * seasonal_factor * temp_factor * peak_factor * holiday_factor
demand = demand + np.random.normal(0, 200, len(df))
demand = np.clip(demand, 2000, 12000)
df['demand'] = demand.astype(int)

# ========================================
# FEATURE 4: Lag Features (TIME-SERIES INTELLIGENCE)
# ========================================
print("\n✅ Feature 4: Lag Features (Time-Series Awareness)")
df['lag_1'] = df['demand'].shift(1)
df['lag_24'] = df['demand'].shift(24)

print(f"   Lag 1: Demand 1 hour ago")
print(f"   Lag 24: Demand same time yesterday")
print(f"   Makes model: Time-series aware")

# ========================================
# FEATURE 5: Rolling Average Features
# ========================================
print("\n✅ Feature 5: Rolling Average (Trend Detection)")
df['rolling_24'] = df['demand'].rolling(window=24, min_periods=1).mean()
df['rolling_168'] = df['demand'].rolling(window=168, min_periods=1).mean()

print(f"   Rolling 24h: Smoothed demand over past 24 hours")
print(f"   Rolling 7d: Weekly trend analysis")

# ========================================
# ADDITIONAL FEATURES
# ========================================
# Season encoding
df['season'] = 4  # Default Fall
df.loc[df['month'].isin([12, 1, 2]), 'season'] = 1  # Winter
df.loc[df['month'].isin([3, 4, 5]), 'season'] = 2  # Spring
df.loc[df['month'].isin([6, 7, 8]), 'season'] = 3  # Summer

# More precise peak indicators
df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 20)).astype(int)

# ========================================
# SAVE DATASET
# ========================================
os.makedirs('data', exist_ok=True)
df.to_csv('data/electricity_demand.csv', index=False)

print("\n" + "=" * 70)
print("✅ PROFESSIONAL DATASET GENERATED SUCCESSFULLY!")
print("=" * 70)

print(f"\n📊 Dataset Statistics:")
print(f"   Total records: {len(df):,}")
print(f"   Demand range: {df['demand'].min():.0f} - {df['demand'].max():.0f} kWh")
print(f"   Average demand: {df['demand'].mean():.0f} kWh")
print(f"   Peak hours count: {df['is_peak'].sum():,}")
print(f"   Holiday hours: {df['is_holiday'].sum():,}")
print(f"   Weekend hours: {df['is_weekend'].sum():,}")

print(f"\n💾 Saved to: data/electricity_demand.csv")

print(f"\n📋 Features Included:")
features = [
    "✓ Basic: hour, day, month, is_weekend, season",
    "✓ Weather: temperature, humidity, cooling_index, heating_index",
    "✓ Time-Series: lag_1, lag_24, rolling_24, rolling_168",
    "✓ Peak Indicators: is_peak, is_morning_peak, is_evening_peak",
    "✓ Events: is_holiday (Indian public holidays)",
    "✓ Target: demand (kWh)"
]
for feat in features:
    print(f"   {feat}")

print("\n🎯 Ready for professional ML training!")
print("=" * 70)
