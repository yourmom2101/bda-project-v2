#!/usr/bin/env python3
"""
Create Cleaned House Data
=========================

This script creates a cleaned version of the house_data.csv file
and saves it as a separate file for analysis and documentation.
"""

import pandas as pd
import numpy as np

def create_cleaned_data():
    """Load original data, clean it, and save cleaned version"""
    
    print("Loading original house data...")
    df = pd.read_csv('data/house_data.csv')
    print(f"Original data shape: {df.shape}")
    print(f"Original data size: {df.shape[0]:,} houses")
    
    # Apply the same cleaning logic as in improved_model.py
    print("\nCleaning data...")
    df_clean = df[
        (df['Sale_price'].between(10000, 2000000)) &  # Realistic price range
        (df['Fin_sqft'].between(200, 8000)) &         # Realistic square footage
        (df['Year_Built'].between(1900, 2020))        # Realistic year range
    ].copy()
    
    # Remove rows with missing critical values
    df_clean = df_clean.dropna(subset=['Sale_price', 'Year_Built', 'Fin_sqft'])
    
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Cleaned data size: {df_clean.shape[0]:,} houses")
    print(f"Removed {df.shape[0] - df_clean.shape[0]:,} houses ({((df.shape[0] - df_clean.shape[0]) / df.shape[0] * 100):.1f}%)")
    
    # Create enhanced features (same as in improved_model.py)
    print("\nCreating enhanced features...")
    df_clean['House_Age'] = 2024 - df_clean['Year_Built']
    df_clean['Total_Bathrooms'] = df_clean['Fbath'] + df_clean['Hbath']
    df_clean['Price_per_Sqft'] = df_clean['Sale_price'] / df_clean['Fin_sqft']
    df_clean['Sqft_per_Bedroom'] = df_clean['Fin_sqft'] / df_clean['Bdrms']
    df_clean['Lot_to_Sqft_Ratio'] = df_clean['Lotsize'] / df_clean['Fin_sqft']
    df_clean['Bathrooms_per_Bedroom'] = df_clean['Total_Bathrooms'] / df_clean['Bdrms']
    df_clean['Age_Squared'] = df_clean['House_Age'] ** 2
    df_clean['Sqft_Squared'] = df_clean['Fin_sqft'] ** 2
    
    # Fix any infinite values from division
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=['Price_per_Sqft', 'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Bathrooms_per_Bedroom'])
    
    print(f"Final cleaned data shape: {df_clean.shape}")
    print(f"Final cleaned data size: {df_clean.shape[0]:,} houses")
    
    # Save cleaned data
    print("\nSaving cleaned data...")
    df_clean.to_csv('data/house_data_cleaned.csv', index=False)
    print("✅ Cleaned data saved to: data/house_data_cleaned.csv")
    
    # Print summary statistics
    print("\n📊 CLEANED DATA SUMMARY:")
    print("=" * 40)
    print(f"Total houses: {df_clean.shape[0]:,}")
    print(f"Total features: {df_clean.shape[1]}")
    print(f"Price range: ${df_clean['Sale_price'].min():,.0f} - ${df_clean['Sale_price'].max():,.0f}")
    print(f"Square footage range: {df_clean['Fin_sqft'].min():,.0f} - {df_clean['Fin_sqft'].max():,.0f} sqft")
    print(f"Year built range: {df_clean['Year_Built'].min()} - {df_clean['Year_Built'].max()}")
    print(f"Average price: ${df_clean['Sale_price'].mean():,.0f}")
    print(f"Average square footage: {df_clean['Fin_sqft'].mean():,.0f} sqft")
    
    return df_clean

if __name__ == "__main__":
    create_cleaned_data() 