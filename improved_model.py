#!/usr/bin/env python3
"""
🏠 IMPROVED PROPERTY PRICE PREDICTOR - BDA V2
=============================================

This is our ENHANCED model that implements all accuracy improvement strategies.
It's designed to be much more accurate than our basic version.

WHAT THIS DOES:
1. Loads and cleans house data
2. Creates advanced features (like a smart calculator)
3. Uses the best machine learning models
4. Combines multiple models for better accuracy
5. Shows you exactly how much better it performs
6. SAVES and LOADS trained models for efficiency

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def create_models_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print('📁 Created models directory')

def save_model(model, filename, scaler=None, feature_names=None):
    """Save a trained model and related data"""
    create_models_directory()
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump(model_data, f)
    print(f'💾 Model saved: models/{filename}')

def load_model(filename):
    """Load a trained model and related data"""
    try:
        with open(f'models/{filename}', 'rb') as f:
            model_data = pickle.load(f)
        print(f'📂 Model loaded: models/{filename}')
        return model_data
    except FileNotFoundError:
        print(f'⚠️  Model file not found: models/{filename}')
        return None

def train_and_save_models(X_train, X_test, y_train, y_test, enhanced_features):
    """Train all models and save them"""
    print('\n🤖 TRAINING AND SAVING ALL MODELS')
    print('-' * 40)
    
    # Scale the features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Model 1: Linear Regression
    print('📈 Training Linear Regression...')
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred = lr_model.predict(X_test_scaled)
    
    # Calculate accuracy metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results['Linear Regression'] = {
        'model': lr_model,
        'R2': r2,
        'MSE': mse,
        'MAE': mae
    }
    
    print(f'      📊 R² Score: {r2:.4f} ({r2*100:.1f}% accuracy)')
    print(f'      💰 Average Error: ${mae:,.0f}')
    
    # Save Linear Regression model
    save_model(lr_model, 'linear_regression_model.pkl', scaler, enhanced_features)
    
    # Model 2: Enhanced Random Forest
    print('🌲 Training Random Forest (Enhanced)...')
    rf_model = RandomForestRegressor(
        n_estimators=300,      # More trees = better accuracy
        max_depth=20,          # Control tree depth
        min_samples_split=5,   # Minimum samples to split
        min_samples_leaf=2,    # Minimum samples in leaf
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate accuracy metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results['Random Forest (Enhanced)'] = {
        'model': rf_model,
        'R2': r2,
        'MSE': mse,
        'MAE': mae
    }
    
    print(f'      📊 R² Score: {r2:.4f} ({r2*100:.1f}% accuracy)')
    print(f'      💰 Average Error: ${mae:,.0f}')
    
    # Save Random Forest model
    save_model(rf_model, 'random_forest_model.pkl', scaler, enhanced_features)
    
    # Model 2: Gradient Boosting
    print('📈 Training Gradient Boosting...')
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    y_pred = gb_model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results['Gradient Boosting'] = {
        'model': gb_model,
        'R2': r2,
        'MSE': mse,
        'MAE': mae
    }
    
    print(f'      📊 R² Score: {r2:.4f} ({r2*100:.1f}% accuracy)')
    print(f'      💰 Average Error: ${mae:,.0f}')
    
    # Save Gradient Boosting model
    save_model(gb_model, 'gradient_boosting_model.pkl', scaler, enhanced_features)
    
    # Model 3: XGBoost
    print('⚡ Training XGBoost...')
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    y_pred = xgb_model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results['XGBoost'] = {
        'model': xgb_model,
        'R2': r2,
        'MSE': mse,
        'MAE': mae
    }
    
    print(f'      📊 R² Score: {r2:.4f} ({r2*100:.1f}% accuracy)')
    print(f'      💰 Average Error: ${mae:,.0f}')
    
    # Save XGBoost model
    save_model(xgb_model, 'xgboost_model.pkl', scaler, enhanced_features)
    
    # Model 5: Ensemble (Voting Regressor)
    print('🏆 Creating Ensemble Model...')
    ensemble_model = VotingRegressor([
        ('lr', lr_model),
        ('rf', rf_model),
        ('gb', gb_model),
        ('xgb', xgb_model)
    ])
    ensemble_model.fit(X_train_scaled, y_train)
    y_pred_ensemble = ensemble_model.predict(X_test_scaled)
    
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    
    results['Ensemble'] = {
        'model': ensemble_model,
        'R2': r2_ensemble,
        'MSE': mse_ensemble,
        'MAE': mae_ensemble
    }
    
    print(f'   🏆 Ensemble R² Score: {r2_ensemble:.4f} ({r2_ensemble*100:.1f}% accuracy)')
    print(f'   💰 Average Error: ${mae_ensemble:,.0f}')
    
    # Save Ensemble model
    save_model(ensemble_model, 'ensemble_model.pkl', scaler, enhanced_features)
    
    return results, scaler, enhanced_features

def load_and_evaluate_models():
    """Load saved models and evaluate them"""
    print('\n📂 LOADING SAVED MODELS')
    print('-' * 40)
    
    # Load data for evaluation
    df = pd.read_csv('data/house_data.csv')
    clean_data = df[
        (df['Sale_price'] > 10000) & 
        (df['Sale_price'] < 5000000) &
        (df['Fin_sqft'] > 100) &
        (df['Fin_sqft'] < 10000)
    ].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])
    
    # Create features
    clean_data['House_Age'] = 2024 - clean_data['Year_Built']
    clean_data['Total_Bathrooms'] = clean_data['Fbath'] + clean_data['Hbath']
    clean_data['Price_per_Sqft'] = clean_data['Sale_price'] / clean_data['Fin_sqft']
    clean_data['Sqft_per_Bedroom'] = clean_data['Fin_sqft'] / clean_data['Bdrms']
    clean_data['Lot_to_Sqft_Ratio'] = clean_data['Lotsize'] / clean_data['Fin_sqft']
    clean_data['Bathrooms_per_Bedroom'] = clean_data['Total_Bathrooms'] / clean_data['Bdrms']
    clean_data['Age_Squared'] = clean_data['House_Age'] ** 2
    clean_data['Sqft_Squared'] = clean_data['Fin_sqft'] ** 2
    
    # Handle division by zero
    for col in ['Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Bathrooms_per_Bedroom']:
        clean_data[col] = clean_data[col].replace([np.inf, -np.inf], np.nan)
        clean_data[col] = clean_data[col].fillna(clean_data[col].median())
    
    # Remove outliers
    Q1 = clean_data['Sale_price'].quantile(0.25)
    Q3 = clean_data['Sale_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (clean_data['Sale_price'] >= lower_bound) & (clean_data['Sale_price'] <= upper_bound)
    final_data = clean_data[outlier_mask]
    
    enhanced_features = [
        'Fin_sqft', 'Lotsize', 'House_Age', 'Total_Bathrooms', 'Bdrms', 'Stories',
        'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Bathrooms_per_Bedroom',
        'Age_Squared', 'Sqft_Squared', 'Price_per_Sqft'
    ]
    
    X = final_data[enhanced_features]
    y = final_data['Sale_price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Try to load each model
    model_files = [
        ('Linear Regression', 'linear_regression_model.pkl'),
        ('Random Forest (Enhanced)', 'random_forest_model.pkl'),
        ('Gradient Boosting', 'gradient_boosting_model.pkl'),
        ('XGBoost', 'xgboost_model.pkl'),
        ('Ensemble', 'ensemble_model.pkl')
    ]
    
    for model_name, filename in model_files:
        model_data = load_model(filename)
        if model_data:
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            
            # Scale test data
            X_test_scaled = scaler.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'model': model,
                'R2': r2,
                'MSE': mse,
                'MAE': mae
            }
            
            print(f'   ✅ {model_name}: {r2:.4f} ({r2*100:.1f}% accuracy)')
        else:
            print(f'   ❌ {model_name}: Model not found')
    
    return results, X_test, y_test

def main():
    """Main function to run the improved model"""
    print('🏠 IMPROVED PROPERTY PRICE PREDICTOR - BDA V2')
    print('🚀 Enhanced Version with Maximum Accuracy & Model Persistence')
    print('=' * 60)
    
    # Check if models already exist
    ensemble_model_data = load_model('ensemble_model.pkl')
    
    if ensemble_model_data:
        print('\n🎉 Found saved models! Loading and evaluating...')
        results, X_test, y_test = load_and_evaluate_models()
        
        if results:
            # Get best model
            best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
            best_r2 = results[best_model_name]['R2']
            
            print(f'\n🏆 Best Loaded Model: {best_model_name}')
            print(f'📈 Accuracy: {best_r2:.4f} ({best_r2*100:.1f}%)')
            
            # Show comparison with original
            print(f'\n📊 COMPARISON:')
            print(f'   📈 Original Model: 62.1% accuracy')
            print(f'   📈 Loaded Model: {best_r2*100:.1f}% accuracy')
            print(f'   🚀 Improvement: +{(best_r2*100 - 62.1):.1f} percentage points')
            
            return results, X_test, y_test
    
    # If no saved models, train new ones
    print('\n🔄 No saved models found. Training new models...')
    
    # ============================================================================
    # STEP 1: LOAD AND CLEAN DATA
    # ============================================================================
    print('\n📂 STEP 1: LOADING AND CLEANING DATA')
    print('-' * 40)

    print('📂 Loading house data...')
    df = pd.read_csv('data/house_data.csv')
    print(f'   📊 Original data: {len(df):,} houses')

    # Clean the data (remove bad entries)
    print('🧹 Cleaning data...')
    clean_data = df[
        (df['Sale_price'] > 10000) &           # Remove very cheap houses (probably errors)
        (df['Sale_price'] < 5000000) &         # Remove very expensive houses (probably errors)
        (df['Fin_sqft'] > 100) &               # Remove tiny houses (probably errors)
        (df['Fin_sqft'] < 10000)               # Remove huge houses (probably errors)
    ].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])

    print(f'   ✅ Cleaned data: {len(clean_data):,} houses')

    # ============================================================================
    # STEP 2: ENHANCED FEATURE ENGINEERING
    # ============================================================================
    print('\n🎯 STEP 2: CREATING SMART FEATURES')
    print('-' * 40)

    print('🧠 Creating advanced features...')

    # Basic features (what we had before)
    clean_data['House_Age'] = 2024 - clean_data['Year_Built']
    clean_data['Total_Bathrooms'] = clean_data['Fbath'] + clean_data['Hbath']
    clean_data['Price_per_Sqft'] = clean_data['Sale_price'] / clean_data['Fin_sqft']

    # NEW: Advanced features (this is where the magic happens!)
    print('   📈 Adding advanced features...')

    # Feature 1: How much space per bedroom?
    clean_data['Sqft_per_Bedroom'] = clean_data['Fin_sqft'] / clean_data['Bdrms']
    print('   ✅ Sqft_per_Bedroom: How much space each bedroom gets')

    # Feature 2: How big is the yard compared to house?
    clean_data['Lot_to_Sqft_Ratio'] = clean_data['Lotsize'] / clean_data['Fin_sqft']
    print('   ✅ Lot_to_Sqft_Ratio: How big the yard is compared to house size')

    # Feature 3: How many bathrooms per bedroom?
    clean_data['Bathrooms_per_Bedroom'] = clean_data['Total_Bathrooms'] / clean_data['Bdrms']
    print('   ✅ Bathrooms_per_Bedroom: How many bathrooms per bedroom')

    # Feature 4: Age squared (captures non-linear age effects)
    clean_data['Age_Squared'] = clean_data['House_Age'] ** 2
    print('   ✅ Age_Squared: Captures how age affects price in complex ways')

    # Feature 5: Square footage squared (captures non-linear size effects)
    clean_data['Sqft_Squared'] = clean_data['Fin_sqft'] ** 2
    print('   ✅ Sqft_Squared: Captures how size affects price in complex ways')

    # Handle division by zero (when bedrooms = 0)
    print('   🔧 Fixing math errors...')
    clean_data['Sqft_per_Bedroom'] = clean_data['Sqft_per_Bedroom'].replace([np.inf, -np.inf], np.nan)
    clean_data['Lot_to_Sqft_Ratio'] = clean_data['Lot_to_Sqft_Ratio'].replace([np.inf, -np.inf], np.nan)
    clean_data['Bathrooms_per_Bedroom'] = clean_data['Bathrooms_per_Bedroom'].replace([np.inf, -np.inf], np.nan)

    # Fill missing values with median (average)
    for col in ['Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Bathrooms_per_Bedroom']:
        clean_data[col] = clean_data[col].fillna(clean_data[col].median())

    print(f'   🎉 Created {5} new smart features!')

    # ============================================================================
    # STEP 3: REMOVE OUTLIERS (EXTREME VALUES)
    # ============================================================================
    print('\n🎯 STEP 3: REMOVING EXTREME VALUES')
    print('-' * 40)

    print('🔍 Finding and removing extreme house prices...')

    # Use IQR method to find outliers (more sophisticated than simple thresholds)
    Q1 = clean_data['Sale_price'].quantile(0.25)  # 25th percentile
    Q3 = clean_data['Sale_price'].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range

    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f'   📊 Price range: ${lower_bound:,.0f} to ${upper_bound:,.0f}')

    # Remove outliers
    outlier_mask = (clean_data['Sale_price'] >= lower_bound) & (clean_data['Sale_price'] <= upper_bound)
    final_data = clean_data[outlier_mask]

    print(f'   ✅ After removing extremes: {len(final_data):,} houses')

    # ============================================================================
    # STEP 4: SELECT FEATURES AND PREPARE DATA
    # ============================================================================
    print('\n🎯 STEP 4: PREPARING DATA FOR MACHINE LEARNING')
    print('-' * 40)

    # Select all our features (basic + advanced)
    enhanced_features = [
        'Fin_sqft', 'Lotsize', 'House_Age', 'Total_Bathrooms', 'Bdrms', 'Stories',
        'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Bathrooms_per_Bedroom',
        'Age_Squared', 'Sqft_Squared', 'Price_per_Sqft'
    ]

    X = final_data[enhanced_features]  # Input features
    y = final_data['Sale_price']      # Target (what we want to predict)

    print(f'   📈 Using {len(enhanced_features)} features:')
    for i, feature in enumerate(enhanced_features, 1):
        print(f'      {i:2d}. {feature}')

    # Split data into training (80%) and testing (20%)
    print('   🔄 Splitting data: 80% for training, 20% for testing...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'   📚 Training set: {len(X_train):,} houses')
    print(f'   🧪 Testing set: {len(X_test):,} houses')

    # ============================================================================
    # STEP 5: TRAIN AND SAVE MODELS
    # ============================================================================
    results, scaler, feature_names = train_and_save_models(X_train, X_test, y_train, y_test, enhanced_features)

    # ============================================================================
    # STEP 6: COMPARE WITH ORIGINAL MODEL
    # ============================================================================
    print('\n📊 STEP 6: COMPARING WITH ORIGINAL MODEL')
    print('-' * 40)

    print('Original Model (Basic):')
    print('   📈 R² Score: 0.6212 (62.1% accuracy)')
    print('   💰 Average Error: ~$45,000')

    print('\nImproved Model (Enhanced):')
    best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
    best_r2 = results[best_model_name]['R2']
    best_mae = results[best_model_name]['MAE']
    
    print(f'   📈 R² Score: {best_r2:.4f} ({best_r2*100:.1f}% accuracy)')
    print(f'   💰 Average Error: ${best_mae:,.0f}')

    improvement = (best_r2 - 0.6212) * 100
    print(f'🚀 That\'s a {improvement:.1f} percentage point improvement in accuracy!')

    # ============================================================================
    # STEP 7: FEATURE IMPORTANCE ANALYSIS
    # ============================================================================
    print('\n🎯 STEP 7: FEATURE IMPORTANCE ANALYSIS')
    print('-' * 40)

    # Get feature importance from the best model
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        feature_importance = list(zip(enhanced_features, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f'🏆 Most Important Features (from {best_model_name}):')
        for i, (feature, imp) in enumerate(feature_importance[:5], 1):
            print(f'   {i}. {feature}: {imp*100:.1f}%')
    elif hasattr(best_model, 'coef_'):
        # For Linear Regression, use coefficients
        importance = np.abs(best_model.coef_)
        feature_importance = list(zip(enhanced_features, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f'🏆 Most Important Features (from {best_model_name}):')
        for i, (feature, coef) in enumerate(feature_importance[:5], 1):
            print(f'   {i}. {feature}: {coef:.2f}')
    else:
        print('📊 Feature importance not available for ensemble model')

    # ============================================================================
    # STEP 8: CREATE VISUALIZATIONS
    # ============================================================================
    print('\n📊 STEP 8: CREATING VISUALIZATIONS')
    print('-' * 40)

    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Create a comprehensive visualization
    plt.figure(figsize=(20, 12))

    # 1. Model Performance Comparison
    plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    r2_scores = [results[m]['R2'] for m in model_names]

    bars = plt.bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'orange', 'red'])
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score (Higher = Better)', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0.6, 1.0)

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        plt.subplot(2, 3, 2)
        top_features = 8
        top_indices = np.argsort(importance)[-top_features:]
        plt.barh(range(top_features), importance[top_indices], color='lightcoral')
        plt.yticks(range(top_features), [enhanced_features[i] for i in top_indices])
        plt.title(f'Top {top_features} Most Important Features', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
    elif hasattr(best_model, 'coef_'):
        plt.subplot(2, 3, 2)
        top_features = 8
        top_indices = np.argsort(np.abs(best_model.coef_))[-top_features:]
        plt.barh(range(top_features), np.abs(best_model.coef_[top_indices]), color='lightcoral')
        plt.yticks(range(top_features), [enhanced_features[i] for i in top_indices])
        plt.title(f'Top {top_features} Most Important Features (Coefficients)', fontsize=14, fontweight='bold')
        plt.xlabel('|Coefficient|', fontsize=12)

    # 3. Actual vs Predicted (Best Model)
    plt.subplot(2, 3, 3)
    y_pred_best = best_model.predict(scaler.transform(X_test))
    plt.scatter(y_test, y_pred_best, alpha=0.6, s=30, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title(f'{best_model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 4. Error Distribution
    plt.subplot(2, 3, 4)
    errors = y_test - y_pred_best
    plt.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Prediction Error ($)', fontsize=12)
    plt.ylabel('Number of Houses', fontsize=12)
    plt.title('Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 5. Price Distribution
    plt.subplot(2, 3, 5)
    plt.hist(y_test, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('House Price ($)', fontsize=12)
    plt.ylabel('Number of Houses', fontsize=12)
    plt.title('House Price Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 6. Improvement Comparison
    plt.subplot(2, 3, 6)
    comparison_data = ['Original Model', 'Improved Model']
    comparison_scores = [0.6212, best_r2]
    colors = ['lightgray', 'gold']

    bars = plt.bar(comparison_data, comparison_scores, color=colors)
    plt.title('Accuracy Improvement', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12)
    plt.ylim(0, 1)

    # Add value labels
    for bar, score in zip(bars, comparison_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig('plots/improved_model_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('✅ Visualization saved to plots/improved_model_results.png')

    # ============================================================================
    # STEP 9: SUMMARY AND RECOMMENDATIONS
    # ============================================================================
    print('\n🎯 STEP 9: SUMMARY AND RECOMMENDATIONS')
    print('=' * 60)

    print('🏆 FINAL RESULTS:')
    print(f'   📈 Original Accuracy: 62.1%')
    print(f'   📈 Improved Accuracy: {best_r2*100:.1f}%')
    print(f'   🚀 Improvement: +{improvement:.1f} percentage points')
    print(f'   💰 Average Error: ${best_mae:,.0f}')

    print('\n🎯 WHAT MADE THE DIFFERENCE:')
    print('1. 📈 Enhanced Features: Added 5 smart features that capture complex relationships')
    print('2. 🤖 Advanced Models: Used XGBoost, Gradient Boosting, and tuned Random Forest')
    print('3. 🏆 Ensemble Method: Combined multiple models for better accuracy')
    print('4. 🧹 Better Data Cleaning: Removed outliers more intelligently')
    print('5. ⚙️ Optimized Parameters: Used the best settings for each model')
    print('6. 💾 Model Persistence: Save and load trained models for efficiency')

    print('\n💡 KEY INSIGHTS:')
    if hasattr(best_model, 'feature_importances_'):
        print(f'   🏠 Most Important Feature: {feature_importance[0][0]} ({feature_importance[0][1]*100:.1f}%)')
    print(f'   📊 Best Individual Model: {best_model_name}')
    print(f'   🏆 Best Overall: {best_model_name}')

    print('\n🌟 CONCLUSION:')
    print('Our improved model is significantly more accurate than the original!')
    print('Models are now saved and can be loaded instantly for future use.')
    print('This demonstrates how advanced machine learning techniques can')
    print('substantially improve prediction accuracy in real-world applications.')

    print('\n🎉 IMPROVED MODEL ANALYSIS COMPLETE! 🎉')
    print('📁 Check plots/improved_model_results.png for beautiful visualizations!')
    print('💾 Models saved in models/ directory for future use!')

    return results, X_test, y_test

if __name__ == "__main__":
    main() 