#!/usr/bin/env python3
"""
Model Performance Visualizations
Essential visualizations for BDA project evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def create_model_performance_visualizations():
    """Create comprehensive model performance visualizations"""
    
    print("📂 Loading house data and running improved model...")
    
    # Import and run the improved model to get predictions
    from improved_model import main
    results, X_test, y_test = main()
    
    # Get the best model
    best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
    best_model = results[best_model_name]['model']
    
    print(f"🏆 Using best model: {best_model_name}")
    
    # Load the scaler and feature names
    import pickle
    with open('models/ensemble_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Make predictions
    X_test_scaled = scaler.transform(X_test)
    y_pred = best_model.predict(X_test_scaled)
    
    # Create comprehensive model performance dashboard
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Actual vs Predicted Scatter Plot
    plt.subplot(3, 4, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=30, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title(f'{best_model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add R² score text
    r2_score = results[best_model_name]['R2']
    plt.text(0.05, 0.95, f'R² = {r2_score:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals Plot
    plt.subplot(3, 4, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=30, color='green')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Price ($)', fontsize=12)
    plt.ylabel('Residuals ($)', fontsize=12)
    plt.title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals Distribution
    plt.subplot(3, 4, 3)
    plt.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Residuals ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add mean and std text
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    plt.text(0.05, 0.95, f'Mean: ${mean_residual:,.0f}\nStd: ${std_residual:,.0f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Q-Q Plot for Residuals
    plt.subplot(3, 4, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. Feature Importance (if available)
    plt.subplot(3, 4, 5)
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Plot top 10 features
        top_features = feature_importance[:10]
        features, importances = zip(*top_features)
        
        plt.barh(range(len(features)), importances, color='orange', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor ensemble model', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
    
    # 6. Model Performance Comparison
    plt.subplot(3, 4, 6)
    model_names = list(results.keys())
    r2_scores = [results[m]['R2'] for m in model_names]
    
    bars = plt.bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'orange', 'red'], alpha=0.8)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0.6, 1.0)
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Error Distribution
    plt.subplot(3, 4, 7)
    plt.hist(residuals, bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Prediction Error ($)', fontsize=12)
    plt.ylabel('Number of Houses', fontsize=12)
    plt.title('Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 8. Price Distribution Comparison
    plt.subplot(3, 4, 8)
    plt.hist(y_test, bins=50, alpha=0.7, color='blue', label='Actual', edgecolor='black')
    plt.hist(y_pred, bins=50, alpha=0.7, color='orange', label='Predicted', edgecolor='black')
    plt.xlabel('House Price ($)', fontsize=12)
    plt.ylabel('Number of Houses', fontsize=12)
    plt.title('Price Distribution Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Prediction Accuracy by Price Range
    plt.subplot(3, 4, 9)
    # Create price bins
    price_bins = pd.cut(y_test, bins=10)
    accuracy_by_bin = []
    bin_labels = []
    
    for bin_name, group in price_bins.groupby(price_bins):
        if len(group) > 0:
            # Use integer indices instead of original indices
            group_indices = np.arange(len(y_test))[price_bins == bin_name]
            actual_prices = y_test.iloc[group_indices] if hasattr(y_test, 'iloc') else y_test[group_indices]
            predicted_prices = y_pred[group_indices]
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            accuracy_by_bin.append(100 - mape)
            bin_labels.append(f'{bin_name.left:,.0f}-\n{bin_name.right:,.0f}')
    
    if accuracy_by_bin:  # Only plot if we have data
        plt.bar(range(len(accuracy_by_bin)), accuracy_by_bin, color='green', alpha=0.8)
        plt.xlabel('Price Range ($)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Prediction Accuracy by Price Range', fontsize=14, fontweight='bold')
        plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Accuracy by price range\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Prediction Accuracy by Price Range', fontsize=14, fontweight='bold')
    
    # 10. Model Metrics Summary
    plt.subplot(3, 4, 10)
    ax = plt.gca()
    ax.axis('off')
    
    # Calculate metrics
    mae = results[best_model_name]['MAE']
    mse = results[best_model_name]['MSE']
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics_text = f"""
    📊 MODEL PERFORMANCE METRICS
    
    🏆 Best Model: {best_model_name}
    
    📈 Accuracy Metrics:
    • R² Score: {r2_score:.3f} ({r2_score*100:.1f}%)
    • MAE: ${mae:,.0f}
    • RMSE: ${rmse:,.0f}
    • MAPE: {mape:.1f}%
    
    📊 Residuals Analysis:
    • Mean Residual: ${mean_residual:,.0f}
    • Std Residual: ${std_residual:,.0f}
    • Min Residual: ${np.min(residuals):,.0f}
    • Max Residual: ${np.max(residuals):,.0f}
    
    🎯 Model Quality:
    {'✅ Excellent' if r2_score > 0.8 else '⚠️ Good' if r2_score > 0.6 else '🚨 Needs Improvement'}
    """
    
    plt.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9))
    
    # 11. Prediction vs Actual by Property Type
    plt.subplot(3, 4, 11)
    # Load original data to get property types
    df = pd.read_csv('data/house_data.csv')
    clean_data = df[
        (df['Sale_price'] > 10000) & 
        (df['Sale_price'] < 5000000) &
        (df['Fin_sqft'] > 100) &
        (df['Fin_sqft'] < 10000)
    ].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])
    
    # Get property types for test set (use simple approach)
    try:
        test_prop_types = clean_data['PropType'].value_counts()
        plt.pie(test_prop_types.values, labels=test_prop_types.index, autopct='%1.1f%%', startangle=90)
        plt.title('Property Type Distribution', fontsize=14, fontweight='bold')
    except:
        plt.text(0.5, 0.5, 'Property type\ndistribution\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Property Type Distribution', fontsize=14, fontweight='bold')
    
    # 12. Model Improvement Comparison
    plt.subplot(3, 4, 12)
    comparison_data = ['Original Model', 'Improved Model']
    comparison_scores = [0.6212, r2_score]  # Original vs improved
    colors = ['lightgray', 'gold']
    
    bars = plt.bar(comparison_data, comparison_scores, color=colors, alpha=0.8)
    plt.title('Model Improvement', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, score in zip(bars, comparison_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('model_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Model performance dashboard saved as: model_performance_dashboard.png")
    
    # Create additional specialized model analysis
    create_specialized_model_analysis(y_test, y_pred, residuals, feature_names, best_model)
    
    return results

def create_specialized_model_analysis(y_test, y_pred, residuals, feature_names, best_model):
    """Create specialized model analysis visualizations"""
    
    # 1. Detailed Residuals Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Residuals vs Actual
    axes[0, 0].scatter(y_test, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Actual Price ($)')
    axes[0, 0].set_ylabel('Residuals ($)')
    axes[0, 0].set_title('Residuals vs Actual Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted with trend line
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    # Add trend line
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(y_pred, p(y_pred), "r-", alpha=0.8)
    axes[0, 1].set_xlabel('Predicted Price ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residuals vs Predicted with Trend Line')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals distribution with normal curve
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True, color='blue', edgecolor='black')
    # Add normal distribution curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Distribution')
    axes[1, 0].set_xlabel('Residuals ($)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residuals Distribution vs Normal')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Residuals vs Normal Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_residuals_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Model residuals analysis saved as: model_residuals_analysis.png")

if __name__ == "__main__":
    create_model_performance_visualizations() 