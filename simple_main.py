#!/usr/bin/env python3
"""
🏠 Property Price Predictor - BDA V2 (Simplified Version)
=========================================================

This is a simplified version that includes everything in one file.
Perfect for testing and understanding how everything works!

What this program does:
1. Loads house data from CSV files
2. Cleans and prepares the data
3. Trains 6 different prediction models
4. Tests how well each model performs
5. Creates beautiful graphs showing the results
6. Analyzes business value and provides insights

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import os

# Hide warning messages
warnings.filterwarnings('ignore')

# Set random seed for reproducible results
np.random.seed(42)

class SimplePropertyPredictor:
    """
    🏠 Simple Property Price Predictor
    
    This class does everything in one place - perfect for learning!
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_clean_data(self):
        """
        📂 Load and clean the house data
        
        This function loads the data and removes bad entries
        """
        print("📂 Loading and cleaning house data...")
        
        # Load the data
        data = pd.read_csv('data/house_data.csv')
        print(f"   📊 Loaded {len(data)} houses")
        
        # Clean the data
        clean_data = data[
            (data['Sale_price'] > 10000) &  # No houses under $10,000
            (data['Sale_price'] < 5000000) &  # No houses over $5 million
            (data['Fin_sqft'] > 100) &  # No houses smaller than 100 sq ft
            (data['Fin_sqft'] < 10000)  # No houses bigger than 10,000 sq ft
        ].copy()
        
        # Remove missing values
        important_cols = ['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms']
        clean_data = clean_data.dropna(subset=important_cols)
        
        print(f"   🧹 Cleaned data: {len(clean_data)} houses")
        return clean_data
    
    def create_features(self, data):
        """
        🎨 Create new useful features from the data
        
        This creates new characteristics that help predict house prices
        """
        print("🎨 Creating new features...")
        
        # Create new features
        data['House_Age'] = 2024 - data['Year_Built']
        data['Total_Bathrooms'] = data['Fbath'] + data['Hbath']
        data['Price_per_Sqft'] = data['Sale_price'] / data['Fin_sqft']
        data['Sqft_per_Bedroom'] = data['Fin_sqft'] / data['Bdrms'].replace(0, 1)
        data['Bathrooms_per_Bedroom'] = data['Total_Bathrooms'] / data['Bdrms'].replace(0, 1)
        data['Lot_to_Sqft_Ratio'] = data['Lotsize'] / data['Fin_sqft']
        
        print("   ✅ Created 6 new features")
        return data
    
    def prepare_data(self, data):
        """
        🎯 Prepare data for training models
        
        This splits the data and scales the features
        """
        print("🎯 Preparing data for training...")
        
        # Select features for prediction
        features = ['Fin_sqft', 'Lotsize', 'House_Age', 'Total_Bathrooms', 
                   'Bdrms', 'Stories', 'Price_per_Sqft', 'Sqft_per_Bedroom',
                   'Bathrooms_per_Bedroom', 'Lot_to_Sqft_Ratio']
        
        X = data[features]
        y = data['Sale_price']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print(f"   📊 Training data: {X_train_scaled.shape[0]} houses")
        print(f"   📊 Test data: {X_test_scaled.shape[0]} houses")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        🤖 Train all the prediction models
        
        This trains 6 different types of models
        """
        print("🤖 Training prediction models...")
        print("=" * 50)
        
        # 1. Linear Regression
        print("📈 Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['Linear Regression'] = lr
        
        # 2. Ridge Regression
        print("🏔️ Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        self.models['Ridge Regression'] = ridge
        
        # 3. Lasso Regression
        print("🎯 Training Lasso Regression...")
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        self.models['Lasso Regression'] = lasso
        
        # 4. k-Nearest Neighbors
        print("👥 Training k-Nearest Neighbors...")
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        self.models['k-Nearest Neighbors'] = knn
        
        # 5. Decision Tree
        print("🌳 Training Decision Tree...")
        dt = DecisionTreeRegressor(max_depth=10, random_state=42)
        dt.fit(X_train, y_train)
        self.models['Decision Tree'] = dt
        
        # 6. Random Forest
        print("🌲 Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        
        print("✅ All models trained successfully!")
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        🧪 Test how well each model performs
        
        This evaluates all models and compares their performance
        """
        print("🧪 Evaluating model performance...")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"🧪 Testing {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Store results
            self.results[name] = {
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'predictions': y_pred
            }
            
            print(f"   📈 R² Score: {r2:.4f}")
            print(f"   💰 RMSE: ${rmse:,.0f}")
            print(f"   💰 MAE: ${mae:,.0f}")
            print(f"   📊 MAPE: {mape:.2f}%")
            print()
        
        return self.results
    
    def find_best_model(self):
        """
        🏆 Find the best performing model
        
        This compares all models and picks the winner
        """
        print("🏆 Finding the best model...")
        
        best_model = max(self.results.items(), key=lambda x: x[1]['r2_score'])
        best_name = best_model[0]
        best_score = best_model[1]['r2_score']
        
        print(f"🏆 Best model: {best_name}")
        print(f"📈 Best R² score: {best_score:.4f}")
        print(f"💰 Best RMSE: ${best_model[1]['rmse']:,.0f}")
        
        return best_name
    
    def create_plots(self, X_test, y_test):
        """
        📊 Create beautiful visualization plots
        
        This creates graphs showing the results
        """
        print("📊 Creating visualization plots...")
        
        # Create plots directory
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Get the best model
        best_name = self.find_best_model()
        best_model = self.models[best_name]
        y_pred = self.results[best_name]['predictions']
        
        # Create a comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🏠 Property Price Prediction Results', fontsize=20, fontweight='bold')
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=20, color='blue')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Model Performance Comparison
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['r2_score'] for name in model_names]
        colors = ['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold']
        bars = axes[0, 1].bar(model_names, r2_scores, color=colors[:len(model_names)])
        axes[0, 1].set_title('Model R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Residuals
        residuals = y_test - y_pred
        axes[0, 2].scatter(y_pred, residuals, alpha=0.5, s=20, color='green')
        axes[0, 2].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[0, 2].set_title('Residuals Plot', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Predicted Price ($)')
        axes[0, 2].set_ylabel('Residuals ($)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Feature Importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            feature_names = X_test.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            bars = axes[1, 0].barh(importance_df['Feature'], importance_df['Importance'], 
                                  color='lightblue', edgecolor='black')
            axes[1, 0].set_title(f'Feature Importance - {best_name}', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Importance Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Error Distribution
        error_percentage = np.abs(residuals / y_test) * 100
        axes[1, 1].hist(error_percentage, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Error Percentage (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Price Distribution
        axes[1, 2].hist(y_test, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].set_title('House Price Distribution', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Price ($)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Plots saved to 'plots/comprehensive_results.png'")
    
    def print_business_insights(self):
        """
        💼 Print business insights and analysis
        
        This translates technical results into business language
        """
        print("💼 BUSINESS ANALYSIS")
        print("=" * 50)
        
        # Get best model results
        best_name = self.find_best_model()
        best_results = self.results[best_name]
        
        # Model performance insights
        r2_score = best_results['r2_score']
        rmse = best_results['rmse']
        
        if r2_score >= 0.95:
            accuracy_level = "Excellent"
        elif r2_score >= 0.90:
            accuracy_level = "Very Good"
        elif r2_score >= 0.80:
            accuracy_level = "Good"
        else:
            accuracy_level = "Fair"
        
        print(f"📊 Model Performance:")
        print(f"   🎯 Accuracy Level: {accuracy_level}")
        print(f"   📈 R² Score: {r2_score:.3f}")
        print(f"   💰 Average Error: ${rmse:,.0f}")
        print(f"   🏆 Best Model: {best_name}")
        
        # Business applications
        print(f"\n🎯 Business Applications:")
        print(f"   👥 Real Estate Agents: Quick property valuations")
        print(f"   🏠 Home Buyers: Understanding fair market value")
        print(f"   💰 Investors: Identifying investment opportunities")
        print(f"   🏦 Lenders: Automated property valuations")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        print(f"   ✅ Use predictions as guidance, not absolute values")
        print(f"   ✅ Combine with professional appraisals")
        print(f"   ✅ Focus on square footage and market rates")
        print(f"   ✅ Regular model updates recommended")

def main():
    """
    🚀 Main function - runs the complete analysis
    """
    print("🏠 PROPERTY PRICE PREDICTOR - BDA V2 (Simplified)")
    print("=" * 60)
    print("🎯 Goal: Predict house prices using machine learning")
    print("📊 Method: Following CBS BDA course methodology")
    print("=" * 60)
    print()
    
    # Create predictor
    predictor = SimplePropertyPredictor()
    
    # Load and clean data
    data = predictor.load_and_clean_data()
    
    # Create features
    data = predictor.create_features(data)
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)
    
    # Train models
    models = predictor.train_models(X_train, y_train)
    
    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)
    
    # Create plots
    predictor.create_plots(X_test, y_test)
    
    # Print business insights
    predictor.print_business_insights()
    
    # Final summary
    print("\n🎉 PROJECT COMPLETE!")
    print("=" * 50)
    print("✅ Data loaded and cleaned")
    print("✅ 6 models trained and tested")
    print("✅ Visualizations created")
    print("✅ Business analysis completed")
    print()
    print("📁 Check the 'plots/' folder for beautiful graphs!")
    print("🌟 Thank you for using Property Price Predictor - BDA V2!")

if __name__ == "__main__":
    # Check if data file exists
    if not os.path.exists('data/house_data.csv'):
        print("❌ Error: 'data/house_data.csv' not found!")
        print("   Please make sure you have the house data file.")
        exit(1)
    
    # Run the analysis
    try:
        main()
    except Exception as e:
        print(f"❌ Something went wrong: {e}")
        print("💡 Check the error message above for clues") 