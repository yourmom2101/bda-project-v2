"""
🤖 Model Trainer Module
======================

This module trains different prediction models to guess house prices.
Think of it like teaching different students to solve the same math problem!

What it does:
1. Creates different types of prediction models (algorithms)
2. Trains each model on house data
3. Tests how well each model performs
4. Compares all models to find the best one
5. Saves the best model for future use

The models we try:
- Linear Regression (simple straight-line predictions)
- Ridge Regression (linear with extra rules)
- Lasso Regression (linear that picks important features)
- k-Nearest Neighbors (looks at similar houses)
- Decision Tree (makes decisions like "if big then expensive")
- Random Forest (many decision trees working together)

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import os

class ModelTrainer:
    """
    🎓 Model Trainer Class
    
    This class teaches different prediction models how to guess house prices.
    It's like having a school for computers!
    """
    
    def __init__(self):
        """
        🚀 Initialize the model trainer
        
        This is like setting up a classroom before the students arrive
        """
        self.models = {}  # Dictionary to store all our trained models
        self.results = {}  # Dictionary to store how well each model performs
        
    def train_linear_regression(self, X_train, y_train):
        """
        📈 Train a Linear Regression model
        
        This is the simplest type of prediction model.
        It tries to draw a straight line through the data points.
        Like saying "bigger houses cost more" in a simple way.
        
        Args:
            X_train: Features of houses (size, bedrooms, etc.)
            y_train: Actual prices of houses
            
        Returns:
            LinearRegression: The trained model
        """
        print("📈 Training Linear Regression model...")
        
        # Create a linear regression model
        model = LinearRegression()
        
        # Train the model (teach it the patterns)
        model.fit(X_train, y_train)
        
        print("✅ Linear Regression trained!")
        print(f"   📊 Intercept: ${model.intercept_:,.0f}")
        print(f"   📊 Coefficients: {len(model.coef_)} features")
        
        return model
    
    def train_ridge_regression(self, X_train, y_train):
        """
        🏔️ Train a Ridge Regression model
        
        This is like Linear Regression but with extra rules to prevent overfitting.
        Overfitting is when the model memorizes the training data too well
        and can't predict new houses accurately.
        
        Args:
            X_train: Features of houses
            y_train: Actual prices of houses
            
        Returns:
            Ridge: The trained model
        """
        print("🏔️ Training Ridge Regression model...")
        
        # Create a ridge regression model with regularization
        model = Ridge(alpha=1.0)  # alpha controls how strict the rules are
        
        # Train the model
        model.fit(X_train, y_train)
        
        print("✅ Ridge Regression trained!")
        print(f"   📊 Alpha (regularization): 1.0")
        
        return model
    
    def train_lasso_regression(self, X_train, y_train):
        """
        🎯 Train a Lasso Regression model
        
        This is like Ridge Regression but it can completely ignore
        unimportant features (set their coefficients to zero).
        It's good for feature selection.
        
        Args:
            X_train: Features of houses
            y_train: Actual prices of houses
            
        Returns:
            Lasso: The trained model
        """
        print("🎯 Training Lasso Regression model...")
        
        # Create a lasso regression model
        model = Lasso(alpha=0.1)  # alpha controls feature selection
        
        # Train the model
        model.fit(X_train, y_train)
        
        print("✅ Lasso Regression trained!")
        print(f"   📊 Alpha (regularization): 0.1")
        
        return model
    
    def train_knn_regression(self, X_train, y_train):
        """
        👥 Train a k-Nearest Neighbors model
        
        This model looks at the most similar houses to make predictions.
        Like saying "this house is similar to these 5 houses, so it should cost about the same."
        
        Args:
            X_train: Features of houses
            y_train: Actual prices of houses
            
        Returns:
            KNeighborsRegressor: The trained model
        """
        print("👥 Training k-Nearest Neighbors model...")
        
        # Try different values of k (number of neighbors to look at)
        k_values = [3, 5, 7, 10]
        best_k = 5
        best_score = -1
        
        # Find the best k value
        for k in k_values:
            model = KNeighborsRegressor(n_neighbors=k)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            avg_score = scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
        
        # Train the model with the best k
        model = KNeighborsRegressor(n_neighbors=best_k)
        model.fit(X_train, y_train)
        
        print("✅ k-Nearest Neighbors trained!")
        print(f"   📊 Best k value: {best_k}")
        print(f"   📊 Cross-validation score: {best_score:.3f}")
        
        return model
    
    def train_decision_tree(self, X_train, y_train):
        """
        🌳 Train a Decision Tree model
        
        This model makes decisions like a flowchart:
        "If the house is bigger than 2000 sq ft, then check if it has more than 3 bedrooms..."
        It's very easy to understand but can be complex.
        
        Args:
            X_train: Features of houses
            y_train: Actual prices of houses
            
        Returns:
            DecisionTreeRegressor: The trained model
        """
        print("🌳 Training Decision Tree model...")
        
        # Create a decision tree with reasonable complexity
        model = DecisionTreeRegressor(
            max_depth=10,        # Don't make it too complex
            min_samples_split=5, # Need at least 5 samples to split
            random_state=42      # For reproducible results
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        print("✅ Decision Tree trained!")
        print(f"   📊 Max depth: 10")
        print(f"   📊 Min samples to split: 5")
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """
        🌲 Train a Random Forest model
        
        This is like having many decision trees vote on the price.
        Each tree might make different decisions, but together they're very accurate.
        It's usually the best performing model!
        
        Args:
            X_train: Features of houses
            y_train: Actual prices of houses
            
        Returns:
            RandomForestRegressor: The trained model
        """
        print("🌲 Training Random Forest model...")
        
        # Create a random forest with many trees
        model = RandomForestRegressor(
            n_estimators=100,    # Use 100 decision trees
            max_depth=10,        # Don't make trees too complex
            min_samples_split=5, # Need at least 5 samples to split
            random_state=42      # For reproducible results
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        print("✅ Random Forest trained!")
        print(f"   📊 Number of trees: 100")
        print(f"   📊 Max depth per tree: 10")
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """
        🎓 Train all prediction models
        
        This function trains all 6 different types of models.
        It's like teaching a whole class of students!
        
        Args:
            X_train: Features of houses for training
            y_train: Actual prices of houses for training
            
        Returns:
            dict: Dictionary with all trained models
        """
        print("🎓 Training all prediction models...")
        print("=" * 50)
        
        # Train each type of model
        self.models['Linear Regression'] = self.train_linear_regression(X_train, y_train)
        print()
        
        self.models['Ridge Regression'] = self.train_ridge_regression(X_train, y_train)
        print()
        
        self.models['Lasso Regression'] = self.train_lasso_regression(X_train, y_train)
        print()
        
        self.models['k-Nearest Neighbors'] = self.train_knn_regression(X_train, y_train)
        print()
        
        self.models['Decision Tree'] = self.train_decision_tree(X_train, y_train)
        print()
        
        self.models['Random Forest'] = self.train_random_forest(X_train, y_train)
        print()
        
        print("🎉 All models trained successfully!")
        print(f"📊 Total models trained: {len(self.models)}")
        
        return self.models
    
    def evaluate_models(self, models, X_test, y_test):
        """
        🧪 Test how well each model performs
        
        This function tests all models on new data they haven't seen before.
        It's like giving all students the same final exam!
        
        Args:
            models: Dictionary of trained models
            X_test: Features of houses for testing
            y_test: Actual prices of houses for testing
            
        Returns:
            dict: Dictionary with performance results for each model
        """
        print("🧪 Evaluating all models...")
        print("=" * 50)
        
        results = {}
        
        for name, model in models.items():
            print(f"🧪 Testing {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate percentage error
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Store results
            results[name] = {
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
        
        self.results = results
        return results
    
    def get_best_model(self, results):
        """
        🏆 Find the best performing model
        
        This function compares all models and picks the winner!
        We use R² score as the main criterion (higher is better).
        
        Args:
            results: Dictionary with performance results
            
        Returns:
            str: Name of the best model
        """
        print("🏆 Finding the best model...")
        
        # Find the model with the highest R² score
        best_model = max(results.items(), key=lambda x: x[1]['r2_score'])
        best_name = best_model[0]
        best_score = best_model[1]['r2_score']
        
        print(f"🏆 Best model: {best_name}")
        print(f"📈 Best R² score: {best_score:.4f}")
        print(f"💰 Best RMSE: ${best_model[1]['rmse']:,.0f}")
        
        return best_name
    
    def get_feature_importance(self, model, feature_names):
        """
        📊 Get feature importance from tree-based models
        
        This tells us which features (like bedrooms, bathrooms) are most important
        for predicting house prices.
        
        Args:
            model: A trained model (works best with tree-based models)
            feature_names: Names of the features
            
        Returns:
            pandas.DataFrame: Feature importance scores
        """
        # Check if the model has feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importance = np.abs(model.coef_)
        else:
            return None
        
        # Create a DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model, model_name):
        """
        💾 Save the best model for future use
        
        This saves the model so we can use it later without retraining.
        It's like saving your homework so you can look at it later!
        
        Args:
            model: The trained model to save
            model_name: Name of the model
        """
        print(f"💾 Saving {model_name}...")
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save the model
        filename = f"models/{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, filename)
        
        print(f"✅ Model saved to: {filename}")
    
    def load_model(self, model_name):
        """
        📂 Load a saved model
        
        This loads a previously saved model so we can use it for predictions.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            The loaded model
        """
        filename = f"models/{model_name.lower().replace(' ', '_')}.joblib"
        
        if os.path.exists(filename):
            model = joblib.load(filename)
            print(f"✅ Model loaded from: {filename}")
            return model
        else:
            print(f"❌ Model file not found: {filename}")
            return None
    
    def print_model_comparison(self, results):
        """
        📊 Print a nice comparison of all models
        
        This creates a table showing how all models compare to each other.
        
        Args:
            results: Dictionary with performance results
        """
        print("📊 Model Performance Comparison")
        print("=" * 50)
        
        # Create a comparison table
        comparison_data = []
        for name, metrics in results.items():
            comparison_data.append({
                'Model': name,
                'R² Score': f"{metrics['r2_score']:.4f}",
                'RMSE': f"${metrics['rmse']:,.0f}",
                'MAE': f"${metrics['mae']:,.0f}",
                'MAPE': f"{metrics['mape']:.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print()
        
        # Find and highlight the best model
        best_model = max(results.items(), key=lambda x: x[1]['r2_score'])
        print(f"🏆 Best Model: {best_model[0]} (R² = {best_model[1]['r2_score']:.4f})") 