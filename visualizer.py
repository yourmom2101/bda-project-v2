"""
📊 Visualizer Module
====================

This module creates beautiful graphs and charts to help understand the data and results.
Think of it like hiring a professional photographer to take pictures of your work!

What it creates:
1. Data exploration plots (understanding the house data)
2. Model performance plots (how well each model predicts)
3. Feature importance plots (what affects house prices most)
4. Prediction accuracy plots (actual vs predicted prices)
5. Error analysis plots (where models make mistakes)

All plots are saved as high-quality images that you can use in reports or presentations.

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class Visualizer:
    """
    🎨 Visualizer Class
    
    This class creates beautiful and informative graphs.
    It's like having a professional data artist!
    """
    
    def __init__(self):
        """
        🚀 Initialize the visualizer
        
        This sets up the plotting style and creates the plots folder
        """
        # Set the style for all plots (makes them look professional)
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Set figure size and DPI for high-quality images
        self.fig_size = (12, 8)
        self.dpi = 300
        
    def create_data_exploration_plots(self, X, y):
        """
        📊 Create plots to explore and understand the house data
        
        These plots help us understand:
        - What the house prices look like (distribution)
        - How different features relate to price
        - What the data patterns are
        
        Args:
            X: Features of houses
            y: Actual prices of houses
        """
        print("📊 Creating data exploration plots...")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🏠 House Data Exploration', fontsize=20, fontweight='bold')
        
        # Plot 1: Price Distribution (how much do houses cost?)
        print("   📈 Creating price distribution plot...")
        axes[0, 0].hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('House Price Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Number of Houses')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add some statistics
        mean_price = y.mean()
        median_price = y.median()
        axes[0, 0].axvline(mean_price, color='red', linestyle='--', label=f'Mean: ${mean_price:,.0f}')
        axes[0, 0].axvline(median_price, color='green', linestyle='--', label=f'Median: ${median_price:,.0f}')
        axes[0, 0].legend()
        
        # Plot 2: Price vs Square Feet (bigger houses cost more?)
        print("   📏 Creating price vs square feet plot...")
        if 'Fin_sqft' in X.columns:
            axes[0, 1].scatter(X['Fin_sqft'], y, alpha=0.5, s=20, color='green')
            axes[0, 1].set_title('Price vs Square Feet', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Square Feet')
            axes[0, 1].set_ylabel('Price ($)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Price vs Bedrooms (more bedrooms = more expensive?)
        print("   🛏️  Creating price vs bedrooms plot...")
        if 'Bdrms' in X.columns:
            bedroom_data = [y[X['Bdrms']==i] for i in sorted(X['Bdrms'].unique())]
            axes[0, 2].boxplot(bedroom_data, labels=sorted(X['Bdrms'].unique()))
            axes[0, 2].set_title('Price by Number of Bedrooms', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Number of Bedrooms')
            axes[0, 2].set_ylabel('Price ($)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Price vs House Age (older houses cheaper?)
        print("   🏠 Creating price vs house age plot...")
        if 'House_Age' in X.columns:
            axes[1, 0].scatter(X['House_Age'], y, alpha=0.5, s=20, color='orange')
            axes[1, 0].set_title('Price vs House Age', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('House Age (Years)')
            axes[1, 0].set_ylabel('Price ($)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Price vs Bathrooms (more bathrooms = more expensive?)
        print("   🚿 Creating price vs bathrooms plot...")
        if 'Total_Bathrooms' in X.columns:
            bathroom_data = [y[X['Total_Bathrooms']==i] for i in sorted(X['Total_Bathrooms'].unique())]
            axes[1, 1].boxplot(bathroom_data, labels=sorted(X['Total_Bathrooms'].unique()))
            axes[1, 1].set_title('Price by Number of Bathrooms', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Number of Bathrooms')
            axes[1, 1].set_ylabel('Price ($)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Correlation Heatmap (which features are related?)
        print("   🔗 Creating correlation heatmap...")
        # Combine features and target for correlation analysis
        correlation_data = X.copy()
        correlation_data['Price'] = y
        corr_matrix = correlation_data.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 2], fmt='.2f')
        axes[1, 2].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('plots/data_exploration.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("✅ Data exploration plots saved to 'plots/data_exploration.png'")
    
    def create_model_performance_plots(self, results):
        """
        📊 Create plots comparing how well different models perform
        
        These plots show:
        - Which model is the best
        - How accurate each model is
        - How the models compare to each other
        
        Args:
            results: Dictionary with performance results for each model
        """
        print("📊 Creating model performance plots...")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🤖 Model Performance Comparison', fontsize=20, fontweight='bold')
        
        # Plot 1: R² Score Comparison (higher is better)
        print("   📈 Creating R² score comparison...")
        model_names = list(results.keys())
        r2_scores = [results[name]['r2_score'] for name in model_names]
        
        bars = axes[0, 0].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold'])
        axes[0, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: RMSE Comparison (lower is better)
        print("   💰 Creating RMSE comparison...")
        rmse_scores = [results[name]['rmse'] for name in model_names]
        
        bars = axes[0, 1].bar(model_names, rmse_scores, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold'])
        axes[0, 1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, rmse_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1000,
                           f'${score:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: MAPE Comparison (lower is better)
        print("   📊 Creating MAPE comparison...")
        mape_scores = [results[name]['mape'] for name in model_names]
        
        bars = axes[1, 0].bar(model_names, mape_scores, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold'])
        axes[1, 0].set_title('MAPE Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, mape_scores):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Model Ranking (overall performance)
        print("   🏆 Creating model ranking...")
        # Create a simple ranking based on R² score
        ranking_data = pd.DataFrame({
            'Model': model_names,
            'R² Score': r2_scores,
            'RMSE': rmse_scores,
            'MAPE': mape_scores
        }).sort_values('R² Score', ascending=True)
        
        bars = axes[1, 1].barh(ranking_data['Model'], ranking_data['R² Score'], 
                              color=['gold', 'lightcoral', 'pink', 'orange', 'lightgreen', 'skyblue'])
        axes[1, 1].set_title('Model Ranking (by R² Score)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('R² Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, ranking_data['R² Score']):
            width = bar.get_width()
            axes[1, 1].text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                           f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('plots/model_performance.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print("✅ Model performance plots saved to 'plots/model_performance.png'")
    
    def create_prediction_plots(self, y_test, y_pred, model_name):
        """
        📈 Create plots showing how well the best model predicts
        
        These plots show:
        - Actual vs Predicted prices
        - How accurate the predictions are
        - Where the model makes mistakes
        
        Args:
            y_test: Actual house prices
            y_pred: Predicted house prices
            model_name: Name of the model being plotted
        """
        print(f"📈 Creating prediction plots for {model_name}...")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'📊 {model_name} - Prediction Analysis', fontsize=20, fontweight='bold')
        
        # Plot 1: Actual vs Predicted (scatter plot)
        print("   📊 Creating actual vs predicted plot...")
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=20, color='blue')
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[0, 0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Residuals (errors)
        print("   📊 Creating residuals plot...")
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[0, 1].set_title('Residuals Plot', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Predicted Price ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals Distribution
        print("   📊 Creating residuals distribution...")
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Residuals ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add mean line
        mean_residual = residuals.mean()
        axes[1, 0].axvline(mean_residual, color='red', linestyle='--', 
                          label=f'Mean: ${mean_residual:,.0f}')
        axes[1, 0].legend()
        
        # Plot 4: Error Percentage
        print("   📊 Creating error percentage plot...")
        error_percentage = np.abs(residuals / y_test) * 100
        axes[1, 1].hist(error_percentage, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Error Percentage (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add mean line
        mean_error = error_percentage.mean()
        axes[1, 1].axvline(mean_error, color='blue', linestyle='--', 
                          label=f'Mean: {mean_error:.1f}%')
        axes[1, 1].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        filename = f'plots/{model_name.lower().replace(" ", "_")}_predictions.png'
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Prediction plots saved to '{filename}'")
    
    def create_feature_importance_plot(self, model, feature_names, model_name):
        """
        📊 Create a plot showing which features are most important
        
        This plot shows which characteristics (like bedrooms, bathrooms)
        have the biggest impact on house prices.
        
        Args:
            model: The trained model
            feature_names: Names of the features
            model_name: Name of the model
        """
        print(f"📊 Creating feature importance plot for {model_name}...")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            print(f"   ⚠️  {model_name} doesn't support feature importance")
            return
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                       color='lightblue', edgecolor='black')
        
        plt.title(f'Feature Importance - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, importance in zip(bars, importance_df['Importance']):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                    f'{importance:.3f}', ha='left', va='center', fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        filename = f'plots/{model_name.lower().replace(" ", "_")}_feature_importance.png'
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Feature importance plot saved to '{filename}'")
    
    def create_all_plots(self, X_test, y_test, y_pred, best_model, results):
        """
        🎨 Create all visualization plots
        
        This is the main function that creates all the plots we need.
        It's like creating a complete photo album of our analysis!
        
        Args:
            X_test: Test features
            y_test: Actual test prices
            y_pred: Predicted test prices
            best_model: The best performing model
            results: Performance results for all models
        """
        print("🎨 Creating all visualization plots...")
        print("=" * 50)
        
        # Get the name of the best model
        best_model_name = max(results.items(), key=lambda x: x[1]['r2_score'])[0]
        
        # Create all the different types of plots
        self.create_data_exploration_plots(X_test, y_test)
        self.create_model_performance_plots(results)
        self.create_prediction_plots(y_test, y_pred, best_model_name)
        
        # Create feature importance plot for tree-based models
        if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
            feature_names = X_test.columns.tolist()
            self.create_feature_importance_plot(best_model, feature_names, best_model_name)
        
        print("🎉 All plots created successfully!")
        print("📁 Check the 'plots/' folder to see all the graphs")
        
        # Print summary of what was created
        print("\n📊 Plots Created:")
        print("   📈 Data exploration plots")
        print("   🤖 Model performance comparison")
        print("   📊 Prediction accuracy analysis")
        print("   📊 Feature importance analysis") 