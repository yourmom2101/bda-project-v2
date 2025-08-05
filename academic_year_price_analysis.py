#!/usr/bin/env python3
"""
Year Built vs Sale Price Analysis
Academic analysis of building age impact on property values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean the dataset for analysis"""
    df = pd.read_csv('data/house_data.csv')
    
    # Remove extreme outliers and invalid data
    df_clean = df[
        (df['Sale_price'].between(10000, 2000000)) &
        (df['Fin_sqft'].between(200, 8000)) &
        (df['Year_Built'].between(1900, 2020))
    ].copy()
    
    # Remove missing values in critical columns
    df_clean = df_clean.dropna(subset=['Sale_price', 'Year_Built', 'Fin_sqft'])
    
    return df_clean

def perform_statistical_analysis(df):
    """Perform correlation and linear regression analysis"""
    # Correlation analysis
    correlation, p_value = pearsonr(df['Year_Built'], df['Sale_price'])
    
    # Linear regression
    X = df[['Year_Built']]
    y = df['Sale_price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    return correlation, p_value, slope, intercept, r_squared

def create_visualizations(df, correlation, p_value, slope, intercept, r_squared):
    """Create comprehensive visualizations"""
    
    # Create decade averages for trend analysis
    df['Decade'] = (df['Year_Built'] // 10) * 10
    decade_avg = df.groupby('Decade')['Sale_price'].mean().reset_index()
    
    # Main analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Scatter plot with regression line
    axes[0, 0].scatter(df['Year_Built'], df['Sale_price'], alpha=0.5, s=10)
    x_range = np.array([df['Year_Built'].min(), df['Year_Built'].max()])
    y_pred = intercept + slope * x_range
    axes[0, 0].plot(x_range, y_pred, 'r-', linewidth=2, 
                    label=f'R² = {r_squared:.3f}')
    axes[0, 0].set_xlabel('Year Built')
    axes[0, 0].set_ylabel('Sale Price ($)')
    axes[0, 0].set_title(f'Price vs Year Built\nCorrelation: {correlation:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Average price by decade
    axes[0, 1].plot(decade_avg['Decade'], decade_avg['Sale_price'], 'o-', linewidth=2)
    axes[0, 1].set_xlabel('Decade Built')
    axes[0, 1].set_ylabel('Average Price ($)')
    axes[0, 1].set_title('Average Price by Decade')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Price distribution by era
    df['Era'] = pd.cut(df['Year_Built'], 
                      bins=[1900, 1950, 1980, 2010, 2020],
                      labels=['Pre-1950', '1950-1979', '1980-2009', '2010+'])
    df.boxplot(column='Sale_price', by='Era', ax=axes[0, 2])
    axes[0, 2].set_title('Price Distribution by Era')
    axes[0, 2].set_xlabel('Building Era')
    axes[0, 2].set_ylabel('Sale Price ($)')
    
    # 4. Year distribution
    axes[1, 0].hist(df['Year_Built'], bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('Year Built')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Building Years')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Price distribution
    axes[1, 1].hist(df['Sale_price'], bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Sale Price ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Sale Prices')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Residuals plot
    predictions = intercept + slope * df['Year_Built']
    residuals = df['Sale_price'] - predictions
    axes[1, 2].scatter(df['Year_Built'], residuals, alpha=0.5, s=10)
    axes[1, 2].axhline(y=0, color='r', linestyle='--')
    axes[1, 2].set_xlabel('Year Built')
    axes[1, 2].set_ylabel('Residuals')
    axes[1, 2].set_title('Residuals Plot')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('year_price_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Dataset Size', f"{len(df):,} properties"],
        ['Year Range', f"{df['Year_Built'].min():.0f} - {df['Year_Built'].max():.0f}"],
        ['Price Range', f"${df['Sale_price'].min():,.0f} - ${df['Sale_price'].max():,.0f}"],
        ['Mean Price', f"${df['Sale_price'].mean():,.0f}"],
        ['Correlation', f"{correlation:.3f}"],
        ['P-value', f"{p_value:.2e}"],
        ['R-squared', f"{r_squared:.3f}"],
        ['Slope', f"{slope:.0f} $/year"],
        ['Intercept', f"${intercept:,.0f}"]
    ]
    
    table = ax.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.title('Statistical Summary: Year Built vs Sale Price', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('year_price_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function"""
    print("Year Built vs Sale Price Analysis")
    print("=" * 40)
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"Dataset size: {len(df):,} properties")
    
    # Perform statistical analysis
    print("Performing statistical analysis...")
    correlation, p_value, slope, intercept, r_squared = perform_statistical_analysis(df)
    
    # Print key results
    print(f"\nResults:")
    print(f"Correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.2e}")
    print(f"R-squared: {r_squared:.3f}")
    print(f"Slope: {slope:.0f} $/year")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df, correlation, p_value, slope, intercept, r_squared)
    print("Analysis complete.")

if __name__ == "__main__":
    main() 