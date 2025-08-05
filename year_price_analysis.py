#!/usr/bin/env python3
"""
Year vs Price Analysis with Connectivity
Analyzes how building year impacts property prices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def analyze_year_price_relationship():
    """Analyze the relationship between building year and property prices"""
    
    print("📂 Loading house data...")
    df = pd.read_csv('data/house_data.csv')
    
    # Clean data for analysis
    clean_data = df[
        (df['Sale_price'] > 10000) & 
        (df['Sale_price'] < 5000000) &
        (df['Fin_sqft'] > 100) &
        (df['Fin_sqft'] < 10000) &
        (df['Year_Built'] > 1900) &  # Remove unrealistic years
        (df['Year_Built'] <= 2024)
    ].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])
    
    print(f"📊 Working with {len(clean_data):,} clean records")
    
    # Calculate correlation
    correlation, p_value = pearsonr(clean_data['Year_Built'], clean_data['Sale_price'])
    
    # Create comprehensive year-price analysis
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Main Scatter Plot with Trend Line
    plt.subplot(2, 3, 1)
    plt.scatter(clean_data['Year_Built'], clean_data['Sale_price'], alpha=0.6, s=20, color='blue')
    
    # Add trend line
    z = np.polyfit(clean_data['Year_Built'], clean_data['Sale_price'], 1)
    p = np.poly1d(z)
    plt.plot(clean_data['Year_Built'], p(clean_data['Year_Built']), "r-", linewidth=3, alpha=0.8)
    
    # Add correlation info
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}\nP-value: {p_value:.2e}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Year Built', fontsize=12, fontweight='bold')
    plt.ylabel('Sale Price ($)', fontsize=12, fontweight='bold')
    plt.title('Property Price vs Building Year\nwith Trend Line', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Average Price by Decade
    plt.subplot(2, 3, 2)
    clean_data['Decade'] = (clean_data['Year_Built'] // 10) * 10
    decade_avg = clean_data.groupby('Decade')['Sale_price'].mean()
    
    plt.plot(decade_avg.index, decade_avg.values, marker='o', linewidth=3, markersize=8, color='green')
    plt.fill_between(decade_avg.index, decade_avg.values, alpha=0.3, color='green')
    
    plt.xlabel('Decade Built', fontsize=12, fontweight='bold')
    plt.ylabel('Average Price ($)', fontsize=12, fontweight='bold')
    plt.title('Average Price by Decade', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for x, y in zip(decade_avg.index, decade_avg.values):
        plt.text(x, y + max(decade_avg.values) * 0.02, f'${y:,.0f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Price Distribution by Era
    plt.subplot(2, 3, 3)
    # Define eras
    def get_era(year):
        if year < 1950:
            return 'Pre-1950'
        elif year < 1970:
            return '1950-1969'
        elif year < 1990:
            return '1970-1989'
        elif year < 2010:
            return '1990-2009'
        else:
            return '2010+'
    
    clean_data['Era'] = clean_data['Year_Built'].apply(get_era)
    era_order = ['Pre-1950', '1950-1969', '1970-1989', '1990-2009', '2010+']
    
    # Create box plot
    era_data = [clean_data[clean_data['Era'] == era]['Sale_price'] for era in era_order]
    bp = plt.boxplot(era_data, labels=era_order, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'orange', 'red', 'purple']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Building Era', fontsize=12, fontweight='bold')
    plt.ylabel('Sale Price ($)', fontsize=12, fontweight='bold')
    plt.title('Price Distribution by Building Era', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Price per Square Foot by Year
    plt.subplot(2, 3, 4)
    clean_data['Price_per_Sqft'] = clean_data['Sale_price'] / clean_data['Fin_sqft']
    
    # Remove outliers for better visualization
    price_per_sqft_clean = clean_data[clean_data['Price_per_Sqft'] < 500]
    
    plt.scatter(price_per_sqft_clean['Year_Built'], price_per_sqft_clean['Price_per_Sqft'], 
               alpha=0.6, s=20, color='purple')
    
    # Add trend line for price per sqft
    z_pps = np.polyfit(price_per_sqft_clean['Year_Built'], price_per_sqft_clean['Price_per_Sqft'], 1)
    p_pps = np.poly1d(z_pps)
    plt.plot(price_per_sqft_clean['Year_Built'], p_pps(price_per_sqft_clean['Year_Built']), 
             "r-", linewidth=3, alpha=0.8)
    
    plt.xlabel('Year Built', fontsize=12, fontweight='bold')
    plt.ylabel('Price per Square Foot ($)', fontsize=12, fontweight='bold')
    plt.title('Price per Square Foot vs Building Year', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. Year Built Distribution
    plt.subplot(2, 3, 5)
    plt.hist(clean_data['Year_Built'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Year Built', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Properties', fontsize=12, fontweight='bold')
    plt.title('Distribution of Building Years', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    mean_year = clean_data['Year_Built'].mean()
    plt.axvline(mean_year, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_year:.0f}')
    plt.legend()
    
    # 6. Connectivity Analysis - Price Trends Over Time
    plt.subplot(2, 3, 6)
    
    # Create year bins and calculate statistics
    year_bins = pd.cut(clean_data['Year_Built'], bins=20)
    year_stats = clean_data.groupby(year_bins)['Sale_price'].agg(['mean', 'std', 'count']).dropna()
    
    # Plot mean with confidence intervals
    years = [interval.mid for interval in year_stats.index]
    means = year_stats['mean']
    stds = year_stats['std']
    
    plt.plot(years, means, marker='o', linewidth=3, markersize=6, color='blue', label='Mean Price')
    plt.fill_between(years, means - stds, means + stds, alpha=0.3, color='blue', label='±1 Std Dev')
    
    # Add trend line
    z_trend = np.polyfit(years, means, 1)
    p_trend = np.poly1d(z_trend)
    plt.plot(years, p_trend(years), "r--", linewidth=2, alpha=0.8, label='Trend Line')
    
    plt.xlabel('Year Built', fontsize=12, fontweight='bold')
    plt.ylabel('Average Price ($)', fontsize=12, fontweight='bold')
    plt.title('Price Trends with Confidence Intervals', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('year_price_connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Year-Price connectivity analysis saved as: year_price_connectivity_analysis.png")
    
    # Create additional detailed analysis
    create_detailed_year_analysis(clean_data, correlation, p_value, decade_avg)
    
    return clean_data, correlation, p_value

def create_detailed_year_analysis(df, correlation, p_value, decade_avg):
    """Create detailed year-price analysis with statistical insights"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Statistical Summary
    axes[0, 0].axis('off')
    
    # Calculate additional statistics
    year_range = df['Year_Built'].max() - df['Year_Built'].min()
    avg_price_old = df[df['Year_Built'] < 1950]['Sale_price'].mean()
    avg_price_new = df[df['Year_Built'] >= 2000]['Sale_price'].mean()
    price_difference = avg_price_new - avg_price_old
    price_ratio = avg_price_new / avg_price_old
    
    summary_text = f"""
    🏠 YEAR-PRICE CONNECTIVITY ANALYSIS
    
    📊 Statistical Summary:
    • Correlation Coefficient: {correlation:.3f}
    • P-value: {p_value:.2e}
    • Significance: {'✅ Significant' if p_value < 0.05 else '❌ Not Significant'}
    
    📈 Price Impact Analysis:
    • Year Range: {df['Year_Built'].min():.0f} - {df['Year_Built'].max():.0f} ({year_range:.0f} years)
    • Average Price (Pre-1950): ${avg_price_old:,.0f}
    • Average Price (2000+): ${avg_price_new:,.0f}
    • Price Difference: ${price_difference:,.0f}
    • Price Ratio: {price_ratio:.1f}x
    
    🎯 Key Insights:
    • {'Strong positive correlation' if correlation > 0.5 else 'Moderate correlation' if correlation > 0.3 else 'Weak correlation'}
    • Newer buildings are {'significantly more expensive' if price_ratio > 1.5 else 'somewhat more expensive' if price_ratio > 1.2 else 'similarly priced'}
    • {'Building year has strong impact' if abs(correlation) > 0.5 else 'Building year has moderate impact' if abs(correlation) > 0.3 else 'Building year has weak impact'} on price
    """
    
    axes[0, 0].text(0.05, 0.95, summary_text, transform=axes[0, 0].transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9))
    
    # 2. Price Growth Rate Analysis
    axes[0, 1].scatter(df['Year_Built'], df['Sale_price'], alpha=0.5, s=15, color='blue')
    
    # Add multiple trend lines for different periods
    recent_data = df[df['Year_Built'] >= 1980]
    older_data = df[df['Year_Built'] < 1980]
    
    if len(recent_data) > 10:
        z_recent = np.polyfit(recent_data['Year_Built'], recent_data['Sale_price'], 1)
        p_recent = np.poly1d(z_recent)
        axes[0, 1].plot(recent_data['Year_Built'], p_recent(recent_data['Year_Built']), 
                       "r-", linewidth=3, alpha=0.8, label='Recent (1980+)')
    
    if len(older_data) > 10:
        z_older = np.polyfit(older_data['Year_Built'], older_data['Sale_price'], 1)
        p_older = np.poly1d(z_older)
        axes[0, 1].plot(older_data['Year_Built'], p_older(older_data['Year_Built']), 
                       "g-", linewidth=3, alpha=0.8, label='Older (<1980)')
    
    axes[0, 1].set_xlabel('Year Built', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Sale Price ($)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Price Growth Rate by Period', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Price Premium by Decade
    axes[1, 0].bar(range(len(decade_avg)), decade_avg.values, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Decade', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average Price ($)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Price Premium by Decade', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(decade_avg)))
    axes[1, 0].set_xticklabels([f'{int(x)}s' for x in decade_avg.index], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(decade_avg.values):
        axes[1, 0].text(i, v + max(decade_avg.values) * 0.01, f'${v:,.0f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Connectivity Heatmap
    axes[1, 1].axis('off')
    
    # Create connectivity matrix
    year_bins = pd.cut(df['Year_Built'], bins=10)
    price_bins = pd.qcut(df['Sale_price'], q=10)
    
    connectivity_matrix = pd.crosstab(year_bins, price_bins, normalize='index')
    
    # Create heatmap
    im = axes[1, 1].imshow(connectivity_matrix.values, cmap='YlOrRd', aspect='auto')
    
    # Add labels
    axes[1, 1].set_xticks(range(len(connectivity_matrix.columns)))
    axes[1, 1].set_yticks(range(len(connectivity_matrix.index)))
    axes[1, 1].set_xticklabels([f'Q{i+1}' for i in range(len(connectivity_matrix.columns))], rotation=45)
    axes[1, 1].set_yticklabels([f'{interval.left:.0f}s' for interval in connectivity_matrix.index])
    
    axes[1, 1].set_xlabel('Price Quartiles', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Decade Built', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Year-Price Connectivity Heatmap', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('Connectivity Strength', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('year_price_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Detailed year-price analysis saved as: year_price_detailed_analysis.png")

if __name__ == "__main__":
    analyze_year_price_relationship() 