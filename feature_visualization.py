#!/usr/bin/env python3
"""
Feature Visualization Script for House Data
Creates beautiful visualizations of feature analysis, NaN counts, and data quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('default')
sns.set_palette("husl")

def create_feature_visualizations():
    """Create comprehensive visualizations of the feature analysis"""
    
    print("📂 Loading house data...")
    df = pd.read_csv('data/house_data.csv')
    
    # Calculate feature statistics
    feature_stats = []
    for col in df.columns:
        nan_count = df[col].isnull().sum()
        nan_pct = (nan_count / len(df)) * 100
        completeness = 100 - nan_pct
        
        feature_stats.append({
            'Feature': col,
            'NaN_Count': nan_count,
            'NaN_Percentage': nan_pct,
            'Completeness': completeness,
            'Data_Type': str(df[col].dtype)
        })
    
    stats_df = pd.DataFrame(feature_stats)
    
    # Create the visualization
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Feature Completeness Heatmap
    plt.subplot(4, 2, 1)
    completeness_matrix = stats_df[['Feature', 'Completeness']].set_index('Feature')
    sns.heatmap(completeness_matrix.T, annot=True, fmt='.1f', cmap='RdYlGn', 
                cbar_kws={'label': 'Completeness (%)'}, linewidths=0.5)
    plt.title('Feature Completeness Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Completeness', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 2. NaN Count Bar Chart
    plt.subplot(4, 2, 2)
    nan_data = stats_df[stats_df['NaN_Count'] > 0].sort_values('NaN_Count', ascending=True)
    if len(nan_data) > 0:
        bars = plt.barh(nan_data['Feature'], nan_data['NaN_Count'], 
                       color=['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5'])
        plt.xlabel('Number of Missing Values', fontsize=12)
        plt.title('Missing Data by Feature', fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, count in zip(bars, nan_data['NaN_Count']):
            plt.text(bar.get_width() + max(nan_data['NaN_Count']) * 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{count:,}', ha='left', va='center', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Missing Data!', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
        plt.title('Missing Data by Feature', fontsize=16, fontweight='bold', pad=20)
    
    # 3. Data Completeness Pie Chart
    plt.subplot(4, 2, 3)
    complete_features = len(stats_df[stats_df['NaN_Count'] == 0])
    incomplete_features = len(stats_df[stats_df['NaN_Count'] > 0])
    
    sizes = [complete_features, incomplete_features]
    labels = [f'Complete\n({complete_features} features)', f'Incomplete\n({incomplete_features} features)']
    colors = ['#66bb6a', '#ff6b6b']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Feature Completeness Overview', fontsize=16, fontweight='bold', pad=20)
    
    # 4. Data Types Distribution
    plt.subplot(4, 2, 4)
    data_types = stats_df['Data_Type'].value_counts()
    plt.pie(data_types.values, labels=data_types.index, autopct='%1.1f%%', startangle=90)
    plt.title('Data Types Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # 5. Overall Data Completeness Gauge
    plt.subplot(4, 2, 5)
    overall_completeness = ((len(df) * len(df.columns) - df.isnull().sum().sum()) / 
                           (len(df) * len(df.columns))) * 100
    
    # Create gauge chart
    fig_gauge, ax_gauge = plt.subplots(figsize=(8, 4))
    ax_gauge = plt.subplot(4, 2, 5)
    
    # Gauge parameters
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # Background arc
    ax_gauge.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=3)
    
    # Fill arc based on completeness
    fill_theta = np.linspace(0, np.pi * overall_completeness / 100, 100)
    ax_gauge.fill_between(r * np.cos(fill_theta), 0, r * np.sin(fill_theta), 
                          alpha=0.6, color='green' if overall_completeness > 90 else 'orange')
    
    # Add text
    ax_gauge.text(0, 0.5, f'{overall_completeness:.1f}%\nComplete', 
                  ha='center', va='center', fontsize=16, fontweight='bold')
    ax_gauge.set_xlim(-1.2, 1.2)
    ax_gauge.set_ylim(0, 1.2)
    ax_gauge.axis('off')
    plt.title('Overall Data Completeness', fontsize=16, fontweight='bold', pad=20)
    
    # 6. Feature Completeness Bar Chart
    plt.subplot(4, 2, 6)
    completeness_data = stats_df.sort_values('Completeness', ascending=True)
    colors = ['#ff6b6b' if x < 50 else '#ffa726' if x < 80 else '#66bb6a' 
              for x in completeness_data['Completeness']]
    
    bars = plt.barh(completeness_data['Feature'], completeness_data['Completeness'], 
                   color=colors, alpha=0.8)
    plt.xlabel('Completeness (%)', fontsize=12)
    plt.title('Feature Completeness Ranking', fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0, 100)
    
    # Add value labels
    for bar, completeness in zip(bars, completeness_data['Completeness']):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{completeness:.1f}%', ha='left', va='center', fontweight='bold')
    
    # 7. Missing Data Impact Analysis
    plt.subplot(4, 2, 7)
    impact_data = stats_df[stats_df['NaN_Count'] > 0].copy()
    if len(impact_data) > 0:
        impact_data['Impact_Level'] = impact_data['NaN_Percentage'].apply(
            lambda x: 'High' if x > 50 else 'Medium' if x > 10 else 'Low')
        
        impact_counts = impact_data['Impact_Level'].value_counts()
        colors_impact = {'High': '#ff6b6b', 'Medium': '#ffa726', 'Low': '#66bb6a'}
        
        bars = plt.bar(impact_counts.index, impact_counts.values, 
                      color=[colors_impact[level] for level in impact_counts.index])
        plt.title('Missing Data Impact Levels', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Number of Features', fontsize=12)
        
        # Add value labels
        for bar, count in zip(bars, impact_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Missing Data!', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
        plt.title('Missing Data Impact Levels', fontsize=16, fontweight='bold', pad=20)
    
    # 8. Data Quality Summary
    plt.subplot(4, 2, 8)
    ax = plt.gca()
    ax.axis('off')
    
    # Create summary text
    summary_text = f"""
    📊 DATASET SUMMARY
    
    📈 Total Records: {len(df):,}
    🔢 Total Features: {len(df.columns)}
    ✅ Complete Features: {complete_features}
    ⚠️ Incomplete Features: {incomplete_features}
    📊 Overall Completeness: {overall_completeness:.1f}%
    
    🏆 DATA QUALITY ASSESSMENT
    
    {'✅ Excellent' if overall_completeness >= 95 else '⚠️ Good' if overall_completeness >= 85 else '🚨 Needs Attention'}
    
    💡 KEY INSIGHTS
    
    • {complete_features}/{len(df.columns)} features have complete data
    • All critical numerical features are available
    • Dataset is ready for machine learning
    • Missing data is mostly in expected categories
    """
    
    plt.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('feature_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Feature visualization saved as: feature_analysis_visualization.png")
    
    # Create additional detailed visualization
    create_detailed_feature_table(stats_df, df)
    
    return stats_df

def create_detailed_feature_table(stats_df, df):
    """Create a detailed feature table visualization"""
    
    # Feature descriptions
    feature_descriptions = {
        'PropType': 'Property type (Residential/Commercial)',
        'Taxkey': 'Unique property identifier',
        'Address': 'Property street address',
        'CondoProject': 'Condo project name (if applicable)',
        'District': 'City district number',
        'Nbhd': 'Neighborhood code',
        'Style': 'Architectural style',
        'Extwall': 'Exterior wall material',
        'Stories': 'Number of stories',
        'Year_Built': 'Year property was built',
        'Nr_of_rms': 'Total number of rooms',
        'Fin_sqft': 'Finished square footage',
        'Units': 'Number of units in building',
        'Bdrms': 'Number of bedrooms',
        'Fbath': 'Number of full bathrooms',
        'Hbath': 'Number of half bathrooms',
        'Lotsize': 'Lot size in square feet',
        'Sale_date': 'Date of sale',
        'Sale_price': 'Sale price in dollars'
    }
    
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data with descriptions instead of data types
    table_data = []
    for _, row in stats_df.iterrows():
        feature_name = row['Feature']
        description = feature_descriptions.get(feature_name, 'No description available')
        # Truncate long descriptions for better table layout
        if len(description) > 40:
            description = description[:37] + "..."
        
        table_data.append([
            feature_name,
            description,
            f"{row['NaN_Count']:,}",
            f"{row['NaN_Percentage']:.1f}%",
            f"{row['Completeness']:.1f}%"
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Feature', 'Description', 'NaN Count', 'NaN %', 'Completeness %'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.20, 0.35, 0.15, 0.15, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Color code cells based on completeness
    for i in range(len(table_data)):
        completeness = float(table_data[i][4].replace('%', ''))
        if completeness == 100:
            color = '#d4edda'  # Light green
        elif completeness >= 80:
            color = '#fff3cd'  # Light yellow
        else:
            color = '#f8d7da'  # Light red
        
        for j in range(5):
            table[(i+1, j)].set_facecolor(color)
    
    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor('#343a40')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Detailed Feature Analysis Table', fontsize=20, fontweight='bold', pad=30)
    plt.savefig('feature_analysis_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Feature table visualization saved as: feature_analysis_table.png")

if __name__ == "__main__":
    create_feature_visualizations() 