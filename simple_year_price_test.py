#!/usr/bin/env python3
"""
Simple Year-Price Test
Quick test to verify visualization generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def test_year_price_visualization():
    """Simple test of year-price visualization"""
    
    print("📂 Loading house data...")
    df = pd.read_csv('data/house_data.csv')
    
    # Clean data
    clean_data = df[
        (df['Sale_price'] > 10000) & 
        (df['Sale_price'] < 5000000) &
        (df['Fin_sqft'] > 100) &
        (df['Fin_sqft'] < 10000) &
        (df['Year_Built'] > 1900) &
        (df['Year_Built'] <= 2024)
    ].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])
    
    print(f"📊 Working with {len(clean_data):,} clean records")
    
    # Calculate correlation
    correlation, p_value = pearsonr(clean_data['Year_Built'], clean_data['Sale_price'])
    print(f"📈 Correlation: {correlation:.3f}, P-value: {p_value:.2e}")
    
    # Create simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Simple scatter plot
    ax1.scatter(clean_data['Year_Built'], clean_data['Sale_price'], alpha=0.6, s=20, color='blue')
    
    # Add trend line
    z = np.polyfit(clean_data['Year_Built'], clean_data['Sale_price'], 1)
    p = np.poly1d(z)
    ax1.plot(clean_data['Year_Built'], p(clean_data['Year_Built']), "r-", linewidth=3, alpha=0.8)
    
    ax1.set_xlabel('Year Built', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sale Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Property Price vs Building Year\nCorrelation: {correlation:.3f}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Average price by decade
    clean_data['Decade'] = (clean_data['Year_Built'] // 10) * 10
    decade_avg = clean_data.groupby('Decade')['Sale_price'].mean()
    
    ax2.plot(decade_avg.index, decade_avg.values, marker='o', linewidth=3, markersize=8, color='green')
    ax2.fill_between(decade_avg.index, decade_avg.values, alpha=0.3, color='green')
    
    ax2.set_xlabel('Decade Built', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Price ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Price by Decade', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(decade_avg.index, decade_avg.values):
        ax2.text(x, y + max(decade_avg.values) * 0.02, f'${y:,.0f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('year_price_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Simple year-price test saved as: year_price_test.png")
    
    # Print some statistics
    print(f"\n📊 STATISTICS:")
    print(f"   • Year range: {clean_data['Year_Built'].min():.0f} - {clean_data['Year_Built'].max():.0f}")
    print(f"   • Price range: ${clean_data['Sale_price'].min():,.0f} - ${clean_data['Sale_price'].max():,.0f}")
    print(f"   • Average price: ${clean_data['Sale_price'].mean():,.0f}")
    print(f"   • Correlation: {correlation:.3f} ({'Strong' if abs(correlation) > 0.5 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'} relationship)")
    print(f"   • P-value: {p_value:.2e} ({'Significant' if p_value < 0.05 else 'Not significant'})")
    
    return clean_data, correlation, p_value

if __name__ == "__main__":
    test_year_price_visualization() 