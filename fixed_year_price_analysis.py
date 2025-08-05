#!/usr/bin/env python3
"""
Fixed Year vs Price Analysis with Connectivity and Linear Regression

Analyzes how building year impacts property prices,
incorporating linear regression and improved structure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression # Explicitly importing Linear Regression
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration and Style ---
plt.style.use('default')
sns.set_palette("Set2")

# --- Helper Functions (Defined First) ---

def calculate_feature_statistics(df):
    """Calculates basic statistics for numerical features."""
    stats_df = df.describe().transpose()
    stats_df['NaN Count'] = df.isnull().sum()
    stats_df['NaN %'] = (df.isnull().sum() / len(df)) * 100
    stats_df['Completeness %'] = 100 - stats_df['NaN %']
    return stats_df

def create_main_connectivity_chart(df, correlation, p_value, slope, intercept, r_squared, decade_avg):
    """
    Creates the main 6-panel comprehensive visualization for Year vs Price.
    Includes scatter plot with linear regression, decade averages, and distributions.
    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 24))
    fig.suptitle('Year Built vs. Sale Price: Comprehensive Connectivity Analysis', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout(pad=5.0)

    # 1. Main Scatter Plot with Linear Regression
    ax1 = axes[0, 0]
    sns.scatterplot(x='Year_Built', y='Sale_price', data=df, ax=ax1, alpha=0.6, s=20, color='skyblue')
    
    # Plot Linear Regression Line
    x_vals = np.array([df['Year_Built'].min(), df['Year_Built'].max()])
    y_vals = intercept + slope * x_vals
    ax1.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=2, label=f'Linear Regression (R²={r_squared:.2f})')
    
    ax1.set_title(f'1. Sale Price vs. Year Built (Correlation: {correlation:.2f}, p={p_value:.3f})', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year Built', fontsize=14)
    ax1.set_ylabel('Sale Price ($)', fontsize=14)
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)

    # 2. Average Price by Decade (Connected Line Chart)
    ax2 = axes[0, 1]
    sns.lineplot(x='Decade', y='Average_Price', data=decade_avg, marker='o', ax=ax2, color='purple', linewidth=3)
    ax2.fill_between(decade_avg['Decade'], decade_avg['Average_Price'], color='purple', alpha=0.1)
    for i, row in decade_avg.iterrows():
        ax2.text(row['Decade'], row['Average_Price'], f'${row["Average_Price"]:,.0f}', ha='center', va='bottom', fontsize=10)
    ax2.set_title('2. Average Sale Price by Decade', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Building Decade', fontsize=14)
    ax2.set_ylabel('Average Sale Price ($)', fontsize=14)
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 3. Price Distribution by Era (Box Plots)
    ax3 = axes[1, 0]
    df['Era'] = pd.cut(df['Year_Built'],
                       bins=[1900, 1940, 1970, 2000, 2024],
                       labels=['Pre-1940', '1940-1969', '1970-1999', '2000+'],
                       right=False)
    sns.boxplot(x='Era', y='Sale_price', data=df, ax=ax3, palette='viridis')
    ax3.set_title('3. Sale Price Distribution by Building Era', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Building Era', fontsize=14)
    ax3.set_ylabel('Sale Price ($)', fontsize=14)
    ax3.ticklabel_format(style='plain', axis='y')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 4. Year Built Distribution (Histogram)
    ax4 = axes[1, 1]
    sns.histplot(df['Year_Built'], kde=True, ax=ax4, color='teal', bins=30)
    ax4.set_title('4. Distribution of Year Built', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Year Built', fontsize=14)
    ax4.set_ylabel('Number of Properties', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.7)

    # 5. Price Distribution (Histogram)
    ax5 = axes[2, 0]
    sns.histplot(df['Sale_price'], kde=True, ax=ax5, color='orange', bins=30)
    ax5.set_title('5. Distribution of Sale Price', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Sale Price ($)', fontsize=14)
    ax5.set_ylabel('Number of Properties', fontsize=14)
    ax5.ticklabel_format(style='plain', axis='x')
    ax5.grid(True, linestyle='--', alpha=0.7)

    # 6. Residuals Plot (from Linear Regression)
    ax6 = axes[2, 1]
    predictions = intercept + slope * df['Year_Built']
    residuals = df['Sale_price'] - predictions
    sns.scatterplot(x=df['Year_Built'], y=residuals, ax=ax6, alpha=0.6, color='darkred')
    ax6.axhline(0, color='gray', linestyle='--', linewidth=2)
    ax6.set_title('6. Residuals Plot (Year Built vs. Sale Price)', fontsize=16, fontweight='bold')
    ax6.set_xlabel('Year Built', fontsize=14)
    ax6.set_ylabel('Residuals (Actual - Predicted Price)', fontsize=14)
    ax6.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('year_price_connectivity_analysis.png', bbox_inches='tight', dpi=300)
    print("✅ 'year_price_connectivity_analysis.png' created successfully.")
    plt.close(fig) # Close the figure to free memory

def create_detailed_year_analysis(df, correlation, p_value, slope, intercept, r_squared, decade_avg):
    """
    Creates a detailed 4-panel visualization for Year vs Price analysis.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 16))
    fig.suptitle('Detailed Year Built vs. Sale Price Analysis', fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout(pad=4.0)

    # 1. Scatter Plot with Regression Line and R-squared
    ax1 = axes[0, 0]
    sns.scatterplot(x='Year_Built', y='Sale_price', data=df, ax=ax1, alpha=0.7, s=30, color='blue')
    
    # Plot Linear Regression Line
    x_vals = np.array([df['Year_Built'].min(), df['Year_Built'].max()])
    y_vals = intercept + slope * x_vals
    ax1.plot(x_vals, y_vals, color='red', linestyle='-', linewidth=2, label=f'Linear Regression (R²={r_squared:.2f})')
    
    ax1.set_title('1. Sale Price vs. Year Built with Regression', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Year Built', fontsize=13)
    ax1.set_ylabel('Sale Price ($)', fontsize=13)
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)

    # 2. Average Price by Decade Bar Chart
    ax2 = axes[0, 1]
    sns.barplot(x='Decade', y='Average_Price', data=decade_avg, ax=ax2, palette='coolwarm')
    for index, row in decade_avg.iterrows():
        ax2.text(index, row['Average_Price'], f'${row["Average_Price"]:,.0f}', color='black', ha="center", va='bottom', fontsize=10)
    ax2.set_title('2. Average Sale Price by Decade (Bar Chart)', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Building Decade', fontsize=13)
    ax2.set_ylabel('Average Sale Price ($)', fontsize=13)
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # 3. Price Distribution by Year (Violin Plot for select years)
    ax3 = axes[1, 0]
    # Select a few representative years for violin plot to avoid clutter
    sample_years = df['Year_Built'].value_counts().nlargest(10).index.sort_values()
    sns.violinplot(x='Year_Built', y='Sale_price', data=df[df['Year_Built'].isin(sample_years)], ax=ax3, palette='pastel')
    ax3.set_title('3. Sale Price Distribution for Key Years', fontsize=15, fontweight='bold')
    ax3.set_xlabel('Year Built', fontsize=13)
    ax3.set_ylabel('Sale Price ($)', fontsize=13)
    ax3.ticklabel_format(style='plain', axis='y')
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

    # 4. Correlation Heatmap (Year_Built and Sale_price)
    ax4 = axes[1, 1]
    correlation_matrix = df[['Year_Built', 'Sale_price']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=.5, ax=ax4)
    ax4.set_title('4. Correlation Heatmap (Year Built vs. Sale Price)', fontsize=15, fontweight='bold')
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=12, rotation=0)

    plt.savefig('year_price_detailed_analysis.png', bbox_inches='tight', dpi=300)
    print("✅ 'year_price_detailed_analysis.png' created successfully.")
    plt.close(fig) # Close the figure to free memory

# --- Main Analysis Function ---

def analyze_year_price_relationship():
    """
    Main function to analyze the relationship between building year and property prices.
    Performs data loading, cleaning, linear regression, and generates visualizations.
    """
    print("📂 Loading house data...")
    df = pd.read_csv('data/house_data.csv')
    print(f"   Initial dataset shape: {df.shape}")

    # --- Data Cleaning (Emphasized and Improved) ---
    print("\n🧹 Cleaning data for analysis...")
    initial_records = len(df)
    
    # Filter out extreme outliers in Sale_price, Fin_sqft, and Year_Built
    clean_data = df[
        (df['Sale_price'] > 10000) &
        (df['Sale_price'] < 5000000) &
        (df['Fin_sqft'] > 100) &
        (df['Fin_sqft'] < 10000) &
        (df['Year_Built'] > 1900) & # Assuming properties built after 1900 are relevant
        (df['Year_Built'] <= 2024) # Up to current year
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Drop rows with NaN values in critical columns for this analysis
    critical_columns = ['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms']
    clean_data.dropna(subset=critical_columns, inplace=True)
    
    records_after_cleaning = len(clean_data)
    print(f"   Records after cleaning: {records_after_cleaning:,} (Removed {initial_records - records_after_cleaning:,} outliers/NaNs)")
    print("   Data cleaning complete.")

    # --- Feature Engineering (Decade) ---
    clean_data['Decade'] = (clean_data['Year_Built'] // 10) * 10
    decade_avg = clean_data.groupby('Decade')['Sale_price'].mean().reset_index()
    decade_avg.rename(columns={'Sale_price': 'Average_Price'}, inplace=True)

    # --- Statistical Analysis ---
    print("\n📈 Performing statistical analysis...")
    # Pearson Correlation
    correlation, p_value = pearsonr(clean_data['Year_Built'], clean_data['Sale_price'])
    print(f"   Pearson Correlation (Year_Built vs. Sale_price): {correlation:.2f} (p-value: {p_value:.3f})")

    # Linear Regression
    X = clean_data[['Year_Built']] # Independent variable (must be 2D)
    y = clean_data['Sale_price']   # Dependent variable

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y) # R-squared value

    print(f"   Linear Regression Model:")
    print(f"     Slope (Coefficient for Year_Built): {slope:,.2f}")
    print(f"     Intercept: {intercept:,.2f}")
    print(f"     R-squared: {r_squared:.2f}")
    print("   Statistical analysis complete.")

    # --- Visualization Generation ---
    print("\n🎨 Generating visualizations...")
    create_main_connectivity_chart(clean_data, correlation, p_value, slope, intercept, r_squared, decade_avg)
    create_detailed_year_analysis(clean_data, correlation, p_value, slope, intercept, r_squared, decade_avg)
    print("   Visualizations generated.")

    return clean_data, correlation, p_value, slope, intercept, r_squared

# --- Script Execution ---
if __name__ == "__main__":
    analyze_year_price_relationship() 