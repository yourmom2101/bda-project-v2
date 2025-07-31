#!/usr/bin/env python3
"""
🏠 Property Price Predictor - BDA V2 (Simple Test Version)
==========================================================

A simple working version to test the project setup.
Perfect for beginners and first-time users!

What this does:
1. Loads house data from CSV files
2. Cleans and prepares the data
3. Trains a Random Forest model
4. Tests how well it predicts house prices
5. Creates a beautiful visualization

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

print('🏠 PROPERTY PRICE PREDICTOR - BDA V2')
print('=' * 50)

# Load data
print('📂 Loading data...')
data = pd.read_csv('data/house_data.csv')
print(f'   📊 Loaded {len(data)} houses')

# Clean data
clean_data = data[
    (data['Sale_price'] > 10000) & 
    (data['Sale_price'] < 5000000) &
    (data['Fin_sqft'] > 100) &
    (data['Fin_sqft'] < 10000)
].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])

print(f'   🧹 Cleaned data: {len(clean_data)} houses')

# Create features
clean_data['House_Age'] = 2024 - clean_data['Year_Built']
clean_data['Total_Bathrooms'] = clean_data['Fbath'] + clean_data['Hbath']
clean_data['Price_per_Sqft'] = clean_data['Sale_price'] / clean_data['Fin_sqft']

# Select features
features = ['Fin_sqft', 'Lotsize', 'House_Age', 'Total_Bathrooms', 'Bdrms', 'Stories']
X = clean_data[features]
y = clean_data['Sale_price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print('🤖 Training Random Forest model...')
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f'🏆 Model Performance:')
print(f'   📈 R² Score: {r2:.4f}')
print(f'   🏠 Houses analyzed: {len(y_test)}')

# Create plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.grid(True, alpha=0.3)

# Save plot
if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig('plots/results.png', dpi=300, bbox_inches='tight')
plt.close()

print('✅ Plot saved to plots/results.png')
print('🎉 Analysis complete!')

# Print feature importance
print('\n📊 Feature Importance:')
importance = model.feature_importances_
for feature, imp in zip(features, importance):
    print(f'   {feature}: {imp:.3f}')

print('\n🌟 Thank you for using Property Price Predictor - BDA V2!')
print('📁 Check the plots/ folder for beautiful visualizations!') 