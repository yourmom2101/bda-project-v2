#!/usr/bin/env python3
"""
Cross-Validation Analysis and Visualization
==========================================

This script creates comprehensive cross-validation analysis and visualizations
for the BDA V2 project final report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for cross-validation analysis"""
    print("Loading and preparing data...")
    
    # Load cleaned data
    df = pd.read_csv('data/house_data_cleaned.csv')
    
    # Prepare features
    enhanced_features = [
        'Fin_sqft', 'Lotsize', 'House_Age', 'Total_Bathrooms', 'Bdrms', 'Stories',
        'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Bathrooms_per_Bedroom',
        'Age_Squared', 'Sqft_Squared', 'Price_per_Sqft'
    ]
    
    X = df[enhanced_features]
    y = df['Sale_price']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, enhanced_features

def perform_cross_validation_analysis():
    """Perform comprehensive cross-validation analysis"""
    print("Performing cross-validation analysis...")
    
    X_scaled, y, features = load_and_prepare_data()
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }
    
    # Perform cross-validation
    cv_results = {}
    for name, model in models.items():
        print(f"Running cross-validation for {name}...")
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        cv_results[name] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'min': cv_scores.min(),
            'max': cv_scores.max()
        }
        print(f"  {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return cv_results, models

def create_cross_validation_visualizations(cv_results):
    """Create comprehensive cross-validation visualizations"""
    print("Creating cross-validation visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Validation Analysis - Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Cross-validation scores box plot
    ax1 = axes[0, 0]
    cv_data = []
    model_names = []
    for name, results in cv_results.items():
        cv_data.extend(results['scores'])
        model_names.extend([name] * len(results['scores']))
    
    cv_df = pd.DataFrame({'Model': model_names, 'CV Score': cv_data})
    sns.boxplot(data=cv_df, x='Model', y='CV Score', ax=ax1, palette='viridis')
    ax1.set_title('Cross-Validation Score Distribution', fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Mean CV scores with error bars
    ax2 = axes[0, 1]
    models = list(cv_results.keys())
    means = [cv_results[model]['mean'] for model in models]
    stds = [cv_results[model]['std'] for model in models]
    
    bars = ax2.bar(models, means, yerr=stds, capsize=5, color='skyblue', alpha=0.7)
    ax2.set_title('Mean Cross-Validation Scores with Standard Deviation', fontweight='bold')
    ax2.set_ylabel('Mean R² Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. CV scores heatmap
    ax3 = axes[1, 0]
    cv_matrix = np.array([cv_results[model]['scores'] for model in models])
    im = ax3.imshow(cv_matrix, cmap='RdYlGn', aspect='auto')
    ax3.set_xticks(range(5))
    ax3.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels(models)
    ax3.set_title('Cross-Validation Scores Heatmap', fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(5):
            text = ax3.text(j, i, f'{cv_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # 4. Model comparison summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for model in models:
        results = cv_results[model]
        summary_data.append([
            model,
            f"{results['mean']:.4f}",
            f"±{results['std']:.4f}",
            f"{results['min']:.4f}",
            f"{results['max']:.4f}"
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Model', 'Mean CV', 'Std Dev', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(models) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title('Cross-Validation Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('plots/cross_validation_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Cross-validation visualization saved to plots/cross_validation_analysis.png")
    
    return fig

def create_model_robustness_analysis(cv_results):
    """Create model robustness analysis"""
    print("Creating model robustness analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Robustness Analysis', fontsize=16, fontweight='bold')
    
    # 1. Coefficient of variation (lower is better)
    ax1.set_title('Model Stability (Coefficient of Variation)', fontweight='bold')
    models = list(cv_results.keys())
    cv_coefficients = []
    
    for model in models:
        results = cv_results[model]
        # Coefficient of variation = std/mean (lower is more stable)
        cv_coef = results['std'] / abs(results['mean']) if results['mean'] != 0 else float('inf')
        cv_coefficients.append(cv_coef)
    
    bars = ax1.bar(models, cv_coefficients, color=['red' if x > 0.1 else 'green' for x in cv_coefficients], alpha=0.7)
    ax1.set_ylabel('Coefficient of Variation (Std/Mean)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, cv_coef in zip(bars, cv_coefficients):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{cv_coef:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Performance range analysis
    ax2.set_title('Performance Range Analysis', fontweight='bold')
    ranges = []
    for model in models:
        results = cv_results[model]
        ranges.append(results['max'] - results['min'])
    
    bars = ax2.bar(models, ranges, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Performance Range (Max - Min)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, range_val in zip(bars, ranges):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{range_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/model_robustness_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Model robustness analysis saved to plots/model_robustness_analysis.png")
    
    return fig

def generate_cv_report(cv_results):
    """Generate a comprehensive cross-validation report"""
    print("Generating cross-validation report...")
    
    report = []
    report.append("CROSS-VALIDATION ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Model performance summary
    report.append("MODEL PERFORMANCE SUMMARY:")
    report.append("-" * 30)
    
    for model_name, results in cv_results.items():
        report.append(f"{model_name}:")
        report.append(f"  Mean CV Score: {results['mean']:.4f}")
        report.append(f"  Standard Deviation: ±{results['std']:.4f}")
        report.append(f"  Min Score: {results['min']:.4f}")
        report.append(f"  Max Score: {results['max']:.4f}")
        report.append(f"  Coefficient of Variation: {results['std']/abs(results['mean']):.4f}")
        report.append("")
    
    # Best performing model
    best_model = max(cv_results.items(), key=lambda x: x[1]['mean'])
    report.append(f"BEST PERFORMING MODEL: {best_model[0]}")
    report.append(f"Mean CV Score: {best_model[1]['mean']:.4f}")
    report.append("")
    
    # Model stability analysis
    report.append("MODEL STABILITY ANALYSIS:")
    report.append("-" * 30)
    
    for model_name, results in cv_results.items():
        stability = "STABLE" if results['std'] < 0.01 else "MODERATE" if results['std'] < 0.05 else "UNSTABLE"
        report.append(f"{model_name}: {stability} (Std: ±{results['std']:.4f})")
    
    # Save report
    with open('cross_validation_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("✅ Cross-validation report saved to cross_validation_report.txt")
    return report

def main():
    """Main function to run cross-validation analysis"""
    print("🏠 CROSS-VALIDATION ANALYSIS - BDA V2")
    print("=" * 50)
    
    # Perform cross-validation analysis
    cv_results, models = perform_cross_validation_analysis()
    
    # Create visualizations
    create_cross_validation_visualizations(cv_results)
    create_model_robustness_analysis(cv_results)
    
    # Generate report
    generate_cv_report(cv_results)
    
    print("\n🎉 CROSS-VALIDATION ANALYSIS COMPLETE!")
    print("📊 Visualizations saved to plots/ directory")
    print("📄 Report saved to cross_validation_report.txt")

if __name__ == "__main__":
    main() 