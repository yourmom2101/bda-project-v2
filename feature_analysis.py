#!/usr/bin/env python3
"""
Feature Analysis Script for House Data
Generates a comprehensive table of features with descriptions and NaN counts
"""

import pandas as pd
import numpy as np

def analyze_features():
    """Analyze the house data features and create a comprehensive table"""
    
    # Load the data
    print("📂 Loading house data...")
    df = pd.read_csv('data/house_data.csv')
    
    print(f"📊 Dataset Overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
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
    
    # Calculate statistics for each feature
    feature_stats = []
    
    for col in df.columns:
        nan_count = df[col].isnull().sum()
        nan_pct = (nan_count / len(df)) * 100
        
        # Get data type
        dtype = str(df[col].dtype)
        
        # Get unique values count for categorical data
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            unique_pct = (unique_count / len(df)) * 100
            unique_info = f"{unique_count} ({unique_pct:.1f}%)"
        else:
            unique_info = "N/A"
        
        # Get range for numerical data
        if df[col].dtype in ['int64', 'float64']:
            if nan_count < len(df):
                min_val = df[col].min()
                max_val = df[col].max()
                range_info = f"{min_val:,} - {max_val:,}"
            else:
                range_info = "All NaN"
        else:
            range_info = "N/A"
        
        feature_stats.append({
            'Feature': col,
            'Data_Type': dtype,
            'NaN_Count': nan_count,
            'NaN_Percentage': nan_pct,
            'Unique_Values': unique_info,
            'Range': range_info,
            'Description': feature_descriptions.get(col, 'No description available')
        })
    
    # Create DataFrame for better formatting
    stats_df = pd.DataFrame(feature_stats)
    
    # Print the comprehensive table
    print("\n" + "="*120)
    print("🏠 COMPREHENSIVE FEATURE ANALYSIS TABLE")
    print("="*120)
    
    print(f"{'Feature':<15} {'Data Type':<12} {'NaN Count':<10} {'NaN %':<8} {'Unique Values':<15} {'Range':<20} {'Description'}")
    print("-" * 120)
    
    for _, row in stats_df.iterrows():
        print(f"{row['Feature']:<15} {row['Data_Type']:<12} {row['NaN_Count']:<10} {row['NaN_Percentage']:<7.1f}% {row['Unique_Values']:<15} {row['Range']:<20} {row['Description']}")
    
    # Summary statistics
    print("\n" + "="*120)
    print("📊 SUMMARY STATISTICS")
    print("="*120)
    
    total_features = len(df.columns)
    features_with_nan = sum(df.isnull().sum() > 0)
    features_without_nan = total_features - features_with_nan
    total_nan_values = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    data_completeness = ((total_cells - total_nan_values) / total_cells) * 100
    
    print(f"📈 Total Features: {total_features}")
    print(f"✅ Features without NaN: {features_without_nan}")
    print(f"⚠️  Features with NaN: {features_with_nan}")
    print(f"🔢 Total NaN values: {total_nan_values:,}")
    print(f"📊 Overall data completeness: {data_completeness:.1f}%")
    
    # Most problematic features
    print(f"\n🚨 FEATURES WITH MOST MISSING DATA:")
    problematic_features = stats_df[stats_df['NaN_Count'] > 0].sort_values('NaN_Percentage', ascending=False)
    
    if len(problematic_features) > 0:
        for _, row in problematic_features.iterrows():
            print(f"   • {row['Feature']}: {row['NaN_Count']:,} NaN values ({row['NaN_Percentage']:.1f}%)")
    else:
        print("   ✅ No missing data found!")
    
    # Data quality insights
    print(f"\n💡 DATA QUALITY INSIGHTS:")
    print(f"   • {features_without_nan}/{total_features} features have complete data")
    print(f"   • {data_completeness:.1f}% of all data cells contain valid values")
    
    if data_completeness >= 95:
        print(f"   ✅ Excellent data quality!")
    elif data_completeness >= 85:
        print(f"   ⚠️  Good data quality with some missing values")
    elif data_completeness >= 70:
        print(f"   ⚠️  Moderate data quality - missing values need attention")
    else:
        print(f"   🚨 Poor data quality - significant missing data")
    
    # Save the analysis to a file
    with open('feature_analysis_report.txt', 'w') as f:
        f.write("HOUSE DATA FEATURE ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {len(df):,} records, {len(df.columns)} features\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("FEATURE TABLE:\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Feature':<15} {'Data Type':<12} {'NaN Count':<10} {'NaN %':<8} {'Description'}\n")
        f.write("-"*120 + "\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"{row['Feature']:<15} {row['Data_Type']:<12} {row['NaN_Count']:<10} {row['NaN_Percentage']:<7.1f}% {row['Description']}\n")
        
        f.write(f"\nSUMMARY:\n")
        f.write(f"Total Features: {total_features}\n")
        f.write(f"Features with NaN: {features_with_nan}\n")
        f.write(f"Total NaN values: {total_nan_values:,}\n")
        f.write(f"Data completeness: {data_completeness:.1f}%\n")
    
    print(f"\n💾 Analysis report saved to: feature_analysis_report.txt")
    
    return stats_df

if __name__ == "__main__":
    analyze_features() 