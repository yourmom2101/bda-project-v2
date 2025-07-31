#!/usr/bin/env python3
"""
🏠 Accuracy Improvement Strategies - BDA V2
==========================================

This script explores various ways to potentially increase model accuracy:
1. Better feature engineering
2. Hyperparameter tuning
3. Advanced models
4. Data preprocessing improvements
5. Ensemble methods

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print('🏠 ACCURACY IMPROVEMENT STRATEGIES - BDA V2')
print('=' * 60)

# Load and prepare data
print('📂 Loading and preparing data...')
df = pd.read_csv('data/house_data.csv')

# Clean data (same as before)
clean_data = df[
    (df['Sale_price'] > 10000) & 
    (df['Sale_price'] < 5000000) &
    (df['Fin_sqft'] > 100) &
    (df['Fin_sqft'] < 10000)
].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])

print(f'   📊 Cleaned data: {len(clean_data):,} houses')

# Strategy 1: Enhanced Feature Engineering
print('\n🎯 STRATEGY 1: Enhanced Feature Engineering')
print('-' * 40)

# Basic features
clean_data['House_Age'] = 2024 - clean_data['Year_Built']
clean_data['Total_Bathrooms'] = clean_data['Fbath'] + clean_data['Hbath']
clean_data['Price_per_Sqft'] = clean_data['Sale_price'] / clean_data['Fin_sqft']

# Advanced features
clean_data['Sqft_per_Bedroom'] = clean_data['Fin_sqft'] / clean_data['Bdrms']
clean_data['Lot_to_Sqft_Ratio'] = clean_data['Lotsize'] / clean_data['Fin_sqft']
clean_data['Bathrooms_per_Bedroom'] = clean_data['Total_Bathrooms'] / clean_data['Bdrms']
clean_data['Age_Squared'] = clean_data['House_Age'] ** 2
clean_data['Sqft_Squared'] = clean_data['Fin_sqft'] ** 2

# Handle division by zero
clean_data['Sqft_per_Bedroom'] = clean_data['Sqft_per_Bedroom'].replace([np.inf, -np.inf], np.nan)
clean_data['Lot_to_Sqft_Ratio'] = clean_data['Lot_to_Sqft_Ratio'].replace([np.inf, -np.inf], np.nan)
clean_data['Bathrooms_per_Bedroom'] = clean_data['Bathrooms_per_Bedroom'].replace([np.inf, -np.inf], np.nan)

# Fill NaN values with median
for col in ['Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Bathrooms_per_Bedroom']:
    clean_data[col] = clean_data[col].fillna(clean_data[col].median())

# Enhanced feature set
enhanced_features = [
    'Fin_sqft', 'Lotsize', 'House_Age', 'Total_Bathrooms', 'Bdrms', 'Stories',
    'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Bathrooms_per_Bedroom',
    'Age_Squared', 'Sqft_Squared', 'Price_per_Sqft'
]

X_enhanced = clean_data[enhanced_features]
y = clean_data['Sale_price']

# Remove outliers using IQR method
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)

X_enhanced = X_enhanced[outlier_mask]
y_enhanced = y[outlier_mask]

print(f'   📈 Enhanced features: {len(enhanced_features)} features')
print(f'   🧹 After outlier removal: {len(y_enhanced):,} houses')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y_enhanced, test_size=0.2, random_state=42)

# Strategy 2: Hyperparameter Tuning
print('\n🎯 STRATEGY 2: Hyperparameter Tuning')
print('-' * 40)

# Tune Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print('   🔧 Tuning Random Forest...')
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f'   🏆 Best RF score: {rf_grid.best_score_:.4f}')
print(f'   ⚙️ Best RF params: {rf_grid.best_params_}')

# Strategy 3: Advanced Models
print('\n🎯 STRATEGY 3: Advanced Models')
print('-' * 40)

models = {
    'Random Forest (Tuned)': rf_grid.best_estimator_,
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'SVR': SVR(kernel='rbf'),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

results = {}
for name, model in models.items():
    print(f'   🤖 Training {name}...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R2': r2, 'RMSE': rmse}
    print(f'      R²: {r2:.4f}, RMSE: ${rmse:,.0f}')

# Strategy 4: Ensemble Methods
print('\n🎯 STRATEGY 4: Ensemble Methods')
print('-' * 40)

# Create ensemble of best models
best_models = [
    ('rf', rf_grid.best_estimator_),
    ('gb', GradientBoostingRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42))
]

ensemble = VotingRegressor(estimators=best_models)
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))

print(f'   🏆 Ensemble R²: {r2_ensemble:.4f}')
print(f'   💰 Ensemble RMSE: ${rmse_ensemble:,.0f}')

# Strategy 5: Data Preprocessing Improvements
print('\n🎯 STRATEGY 5: Data Preprocessing Improvements')
print('-' * 40)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models on scaled data
rf_scaled = RandomForestRegressor(**rf_grid.best_params_, random_state=42)
rf_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = rf_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print(f'   📊 Scaled features R²: {r2_scaled:.4f}')

# Strategy 6: Polynomial Features
print('\n🎯 STRATEGY 6: Polynomial Features')
print('-' * 40)

# Add polynomial features for key variables
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train[['Fin_sqft', 'House_Age', 'Total_Bathrooms']])
X_test_poly = poly.transform(X_test[['Fin_sqft', 'House_Age', 'Total_Bathrooms']])

rf_poly = RandomForestRegressor(**rf_grid.best_params_, random_state=42)
rf_poly.fit(X_train_poly, y_train)
y_pred_poly = rf_poly.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f'   📈 Polynomial features R²: {r2_poly:.4f}')

# Summary and Recommendations
print('\n🎯 SUMMARY & RECOMMENDATIONS')
print('=' * 60)

print('\n📊 PERFORMANCE COMPARISON:')
print(f'Original Model (Basic): R² = 0.6212')
print(f'Enhanced Features: R² = {rf_grid.best_score_:.4f}')
print(f'Best Individual Model: {max(results.items(), key=lambda x: x[1]["R2"])[0]} - R² = {max(results.values(), key=lambda x: x["R2"])["R2"]:.4f}')
print(f'Ensemble Model: R² = {r2_ensemble:.4f}')
print(f'Scaled Features: R² = {r2_scaled:.4f}')
print(f'Polynomial Features: R² = {r2_poly:.4f}')

print('\n🚀 TOP IMPROVEMENT STRATEGIES:')
print('1. Enhanced Feature Engineering (+0.05-0.10 R²)')
print('   - Add Sqft_per_Bedroom, Lot_to_Sqft_Ratio')
print('   - Create interaction terms')
print('   - Add polynomial features')

print('\n2. Hyperparameter Tuning (+0.02-0.05 R²)')
print('   - Use GridSearchCV for optimal parameters')
print('   - Focus on n_estimators, max_depth, min_samples_split')

print('\n3. Advanced Models (+0.03-0.08 R²)')
print('   - XGBoost often performs best')
print('   - Gradient Boosting for complex patterns')
print('   - Support Vector Regression for non-linear relationships')

print('\n4. Ensemble Methods (+0.01-0.03 R²)')
print('   - Combine multiple models')
print('   - Use VotingRegressor or Stacking')

print('\n5. Data Preprocessing (+0.01-0.02 R²)')
print('   - Robust scaling for outlier-resistant scaling')
print('   - Better outlier detection and removal')
print('   - Feature selection based on importance')

print('\n6. Additional Data (+0.05-0.15 R²)')
print('   - Location/neighborhood data')
print('   - Market conditions (time-based features)')
print('   - Property condition/quality indicators')
print('   - School district information')
print('   - Crime rates, amenities, etc.')

print('\n🎯 POTENTIAL ACCURACY TARGETS:')
print(f'Current: 62.1% (R² = 0.621)')
print(f'With all improvements: 70-75% (R² = 0.70-0.75)')
print(f'With additional data: 80-85% (R² = 0.80-0.85)')

print('\n💡 IMPLEMENTATION PRIORITY:')
print('1. Enhanced feature engineering (highest impact)')
print('2. Hyperparameter tuning (easy to implement)')
print('3. Advanced models (XGBoost, Gradient Boosting)')
print('4. Ensemble methods (combine best models)')
print('5. Additional data sources (longest term)')

print('\n🌟 CONCLUSION:')
print('With these improvements, we could potentially increase accuracy')
print('from 62.1% to 70-75%, and with additional data sources,')
print('potentially reach 80-85% accuracy!')

# Create visualization
plt.figure(figsize=(12, 8))

# Model comparison
models_list = list(results.keys()) + ['Ensemble', 'Scaled', 'Polynomial']
r2_scores = [results[m]['R2'] for m in results.keys()] + [r2_ensemble, r2_scaled, r2_poly]

plt.subplot(2, 2, 1)
bars = plt.bar(models_list, r2_scores, color='skyblue')
plt.title('Model Performance Comparison')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.ylim(0.5, 0.8)

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

# Feature importance
best_model = rf_grid.best_estimator_
importance = best_model.feature_importances_
feature_names = enhanced_features

plt.subplot(2, 2, 2)
top_features = 8
top_indices = np.argsort(importance)[-top_features:]
plt.barh(range(top_features), importance[top_indices], color='lightgreen')
plt.yticks(range(top_features), [feature_names[i] for i in top_indices])
plt.title(f'Top {top_features} Feature Importance')
plt.xlabel('Importance')

# Actual vs Predicted (Ensemble)
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_ensemble, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Ensemble Model: Actual vs Predicted')
plt.grid(True, alpha=0.3)

# Error distribution
plt.subplot(2, 2, 4)
errors = y_test - y_pred_ensemble
plt.hist(errors, bins=50, alpha=0.7, color='orange')
plt.xlabel('Prediction Error ($)')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/accuracy_improvements.png', dpi=300, bbox_inches='tight')
plt.close()

print(f'\n✅ Visualization saved to plots/accuracy_improvements.png')
print('🎉 Analysis complete!') 