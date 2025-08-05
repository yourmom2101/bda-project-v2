#!/usr/bin/env python3
"""
Exploratory Data Analysis Visualizations
Comprehensive EDA following BDA course requirements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def create_eda_visualizations():
    """Create comprehensive EDA visualizations for BDA project"""
    
    print("📂 Loading house data...")
    df = pd.read_csv('data/house_data.csv')
    
    # Clean data for analysis
    clean_data = df[
        (df['Sale_price'] > 10000) & 
        (df['Sale_price'] < 5000000) &
        (df['Fin_sqft'] > 100) &
        (df['Fin_sqft'] < 10000)
    ].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])
    
    print(f"📊 Working with {len(clean_data):,} clean records")
    
    # Create comprehensive EDA dashboard
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Target Variable Distribution (Sale Price)
    plt.subplot(3, 4, 1)
    plt.hist(clean_data['Sale_price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Sale Price ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sale Prices')
    plt.grid(True, alpha=0.3)
    
    # 2. Log-transformed Sale Price (for better visualization)
    plt.subplot(3, 4, 2)
    plt.hist(np.log10(clean_data['Sale_price']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Log10(Sale Price)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Log-Transformed Sale Prices')
    plt.grid(True, alpha=0.3)
    
    # 3. Square Footage Distribution
    plt.subplot(3, 4, 3)
    plt.hist(clean_data['Fin_sqft'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Finished Square Footage')
    plt.ylabel('Frequency')
    plt.title('Distribution of Square Footage')
    plt.grid(True, alpha=0.3)
    
    # 4. Year Built Distribution
    plt.subplot(3, 4, 4)
    plt.hist(clean_data['Year_Built'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Year Built')
    plt.ylabel('Frequency')
    plt.title('Distribution of Year Built')
    plt.grid(True, alpha=0.3)
    
    # 5. Price vs Square Footage Scatter
    plt.subplot(3, 4, 5)
    plt.scatter(clean_data['Fin_sqft'], clean_data['Sale_price'], alpha=0.5, s=20)
    plt.xlabel('Square Footage')
    plt.ylabel('Sale Price ($)')
    plt.title('Price vs Square Footage')
    plt.grid(True, alpha=0.3)
    
    # 6. Price vs Year Built Scatter
    plt.subplot(3, 4, 6)
    plt.scatter(clean_data['Year_Built'], clean_data['Sale_price'], alpha=0.5, s=20)
    plt.xlabel('Year Built')
    plt.ylabel('Sale Price ($)')
    plt.title('Price vs Year Built')
    plt.grid(True, alpha=0.3)
    
    # 7. Price vs Bedrooms Box Plot
    plt.subplot(3, 4, 7)
    bedroom_data = clean_data[clean_data['Bdrms'] <= 8]  # Remove outliers
    sns.boxplot(x='Bdrms', y='Sale_price', data=bedroom_data)
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Sale Price ($)')
    plt.title('Price Distribution by Bedrooms')
    
    # 8. Price vs Stories Box Plot
    plt.subplot(3, 4, 8)
    story_data = clean_data[clean_data['Stories'] <= 4]  # Remove outliers
    sns.boxplot(x='Stories', y='Sale_price', data=story_data)
    plt.xlabel('Number of Stories')
    plt.ylabel('Sale Price ($)')
    plt.title('Price Distribution by Stories')
    
    # 9. Correlation Heatmap
    plt.subplot(3, 4, 9)
    numeric_cols = ['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms', 'Fbath', 'Hbath', 'Lotsize', 'Stories']
    correlation_matrix = clean_data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Matrix')
    
    # 10. Price Trends Over Time
    plt.subplot(3, 4, 10)
    clean_data['Sale_Year'] = pd.to_datetime(clean_data['Sale_date']).dt.year
    yearly_avg = clean_data.groupby('Sale_Year')['Sale_price'].mean()
    plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Sale Year')
    plt.ylabel('Average Price ($)')
    plt.title('Average Price Trends Over Time')
    plt.grid(True, alpha=0.3)
    
    # 11. Price per Square Foot Distribution
    plt.subplot(3, 4, 11)
    clean_data['Price_per_Sqft'] = clean_data['Sale_price'] / clean_data['Fin_sqft']
    price_per_sqft = clean_data[clean_data['Price_per_Sqft'] < 500]  # Remove outliers
    plt.hist(price_per_sqft['Price_per_Sqft'], bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Price per Square Foot ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Price per Square Foot')
    plt.grid(True, alpha=0.3)
    
    # 12. Property Type Distribution
    plt.subplot(3, 4, 12)
    prop_type_counts = clean_data['PropType'].value_counts()
    plt.pie(prop_type_counts.values, labels=prop_type_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution by Property Type')
    
    plt.tight_layout()
    plt.savefig('eda_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ EDA comprehensive analysis saved as: eda_comprehensive_analysis.png")
    
    # Create additional specialized visualizations
    create_specialized_visualizations(clean_data)
    
    return clean_data

def create_specialized_visualizations(df):
    """Create specialized visualizations for deeper insights"""
    
    # 1. Feature Relationships Matrix
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Price vs Square Footage with trend line
    axes[0, 0].scatter(df['Fin_sqft'], df['Sale_price'], alpha=0.5, s=20)
    z = np.polyfit(df['Fin_sqft'], df['Sale_price'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['Fin_sqft'], p(df['Fin_sqft']), "r--", alpha=0.8)
    axes[0, 0].set_xlabel('Square Footage')
    axes[0, 0].set_ylabel('Sale Price ($)')
    axes[0, 0].set_title('Price vs Square Footage with Trend Line')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Price vs Year Built with trend line
    axes[0, 1].scatter(df['Year_Built'], df['Sale_price'], alpha=0.5, s=20)
    z = np.polyfit(df['Year_Built'], df['Sale_price'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['Year_Built'], p(df['Year_Built']), "r--", alpha=0.8)
    axes[0, 1].set_xlabel('Year Built')
    axes[0, 1].set_ylabel('Sale Price ($)')
    axes[0, 1].set_title('Price vs Year Built with Trend Line')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Price vs Bedrooms
    bedroom_data = df[df['Bdrms'] <= 8]
    axes[1, 0].scatter(bedroom_data['Bdrms'], bedroom_data['Sale_price'], alpha=0.5, s=20)
    z = np.polyfit(bedroom_data['Bdrms'], bedroom_data['Sale_price'], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(bedroom_data['Bdrms'], p(bedroom_data['Bdrms']), "r--", alpha=0.8)
    axes[1, 0].set_xlabel('Number of Bedrooms')
    axes[1, 0].set_ylabel('Sale Price ($)')
    axes[1, 0].set_title('Price vs Bedrooms with Trend Line')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Price vs Total Bathrooms
    df['Total_Bathrooms'] = df['Fbath'] + df['Hbath']
    axes[1, 1].scatter(df['Total_Bathrooms'], df['Sale_price'], alpha=0.5, s=20)
    z = np.polyfit(df['Total_Bathrooms'], df['Sale_price'], 1)
    p = np.poly1d(z)
    axes[1, 1].plot(df['Total_Bathrooms'], p(df['Total_Bathrooms']), "r--", alpha=0.8)
    axes[1, 1].set_xlabel('Total Bathrooms')
    axes[1, 1].set_ylabel('Sale Price ($)')
    axes[1, 1].set_title('Price vs Total Bathrooms with Trend Line')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical Summary Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Summary statistics table
    axes[0, 0].axis('off')
    summary_stats = df[['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms']].describe()
    table = axes[0, 0].table(cellText=summary_stats.values,
                            rowLabels=summary_stats.index,
                            colLabels=summary_stats.columns,
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[0, 0].set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Outlier detection - Box plots
    axes[0, 1].boxplot([df['Sale_price'], df['Fin_sqft'], df['Year_Built'], df['Bdrms']], 
                       labels=['Price', 'SqFt', 'Year', 'Beds'])
    axes[0, 1].set_title('Outlier Detection - Box Plots')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Price distribution by property type
    prop_type_price = df.groupby('PropType')['Sale_price'].mean().sort_values(ascending=True)
    axes[1, 0].barh(prop_type_price.index, prop_type_price.values, color=['skyblue', 'lightgreen'])
    axes[1, 0].set_xlabel('Average Price ($)')
    axes[1, 0].set_title('Average Price by Property Type')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Price distribution by district
    district_price = df.groupby('District')['Sale_price'].mean().sort_values(ascending=True)
    axes[1, 1].bar(range(len(district_price)), district_price.values, color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('District')
    axes[1, 1].set_ylabel('Average Price ($)')
    axes[1, 1].set_title('Average Price by District')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_statistical_insights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ EDA specialized visualizations saved as:")
    print("   - eda_feature_relationships.png")
    print("   - eda_statistical_insights.png")

if __name__ == "__main__":
    create_eda_visualizations() 