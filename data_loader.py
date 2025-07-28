"""
📂 Data Loader Module
=====================

This module is responsible for loading, cleaning, and preparing house data.
Think of it like a data janitor - it makes sure all the data is clean and ready to use!

What it does:
1. Loads house data from CSV files
2. Cleans the data (removes bad entries, fixes problems)
3. Creates new useful features (like calculating house age)
4. Splits data into training and testing sets
5. Makes sure everything is ready for the prediction models

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class DataLoader:
    """
    🧹 Data Loader Class
    
    This class handles all the data preparation work.
    It's like having a professional data cleaner!
    """
    
    def __init__(self):
        """
        🚀 Initialize the data loader
        
        This is like setting up your workspace before starting work
        """
        self.data_path = 'data/house_data.csv'  # Where the house data is stored
        self.scaler = StandardScaler()          # Tool for scaling numbers
        
    def load_data(self):
        """
        📂 Load the house data from the CSV file
        
        This is like opening a book to read the information inside
        
        Returns:
            pandas.DataFrame: The house data (like a spreadsheet)
        """
        print("📂 Loading house data...")
        
        # Check if the data file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Data file not found: {self.data_path}")
        
        # Read the CSV file (like opening a spreadsheet)
        data = pd.read_csv(self.data_path)
        
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Total houses: {len(data)}")
        print(f"   🏠 Features: {len(data.columns)}")
        print(f"   📋 Columns: {list(data.columns)}")
        
        return data
    
    def clean_data(self, data):
        """
        🧹 Clean the house data
        
        This removes bad data and fixes problems, like:
        - Houses with impossible prices (like $0 or $1 billion)
        - Houses with missing important information
        - Houses with impossible sizes (like 0 square feet)
        
        Args:
            data (pandas.DataFrame): The raw house data
            
        Returns:
            pandas.DataFrame: Clean house data ready for analysis
        """
        print("🧹 Cleaning the data...")
        
        # Make a copy so we don't change the original data
        clean_data = data.copy()
        
        # Step 1: Remove houses with impossible prices
        print("   🏷️  Removing houses with impossible prices...")
        original_count = len(clean_data)
        
        # Remove houses with prices that are too low or too high
        clean_data = clean_data[
            (clean_data['Sale_price'] > 10000) &  # No house costs less than $10,000
            (clean_data['Sale_price'] < 5000000)  # No house costs more than $5 million
        ]
        
        removed_count = original_count - len(clean_data)
        print(f"      Removed {removed_count} houses with impossible prices")
        
        # Step 2: Remove houses with impossible sizes
        print("   📏 Removing houses with impossible sizes...")
        original_count = len(clean_data)
        
        clean_data = clean_data[
            (clean_data['Fin_sqft'] > 100) &    # No house smaller than 100 sq ft
            (clean_data['Fin_sqft'] < 10000)    # No house bigger than 10,000 sq ft
        ]
        
        removed_count = original_count - len(clean_data)
        print(f"      Removed {removed_count} houses with impossible sizes")
        
        # Step 3: Remove houses with missing important information
        print("   ❓ Removing houses with missing information...")
        original_count = len(clean_data)
        
        # Remove houses where important information is missing
        important_columns = ['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms']
        clean_data = clean_data.dropna(subset=important_columns)
        
        removed_count = original_count - len(clean_data)
        print(f"      Removed {removed_count} houses with missing information")
        
        # Step 4: Fix impossible values
        print("   🔧 Fixing impossible values...")
        
        # Make sure bedrooms is reasonable (between 0 and 10)
        clean_data['Bdrms'] = clean_data['Bdrms'].clip(0, 10)
        
        # Make sure bathrooms is reasonable (between 0 and 8)
        clean_data['Fbath'] = clean_data['Fbath'].clip(0, 8)
        clean_data['Hbath'] = clean_data['Hbath'].clip(0, 8)
        
        # Make sure year built is reasonable (between 1800 and 2024)
        clean_data['Year_Built'] = clean_data['Year_Built'].clip(1800, 2024)
        
        print(f"✅ Data cleaning complete!")
        print(f"   📊 Houses after cleaning: {len(clean_data)}")
        print(f"   🧹 Removed {len(data) - len(clean_data)} bad entries")
        
        return clean_data
    
    def create_features(self, data):
        """
        🎨 Create new useful features from existing data
        
        This is like creating new ingredients by combining existing ones.
        For example, if you have flour and eggs, you can make dough!
        
        Args:
            data (pandas.DataFrame): The clean house data
            
        Returns:
            pandas.DataFrame: Data with new features added
        """
        print("🎨 Creating new features...")
        
        # Make a copy so we don't change the original data
        feature_data = data.copy()
        
        # Feature 1: House Age (how old is the house?)
        print("   🏠 Creating house age feature...")
        feature_data['House_Age'] = 2024 - feature_data['Year_Built']
        
        # Feature 2: Total Bathrooms (full + half bathrooms)
        print("   🚿 Creating total bathrooms feature...")
        feature_data['Total_Bathrooms'] = feature_data['Fbath'] + feature_data['Hbath']
        
        # Feature 3: Price per Square Foot (how expensive is each square foot?)
        print("   💰 Creating price per square foot feature...")
        feature_data['Price_per_Sqft'] = feature_data['Sale_price'] / feature_data['Fin_sqft']
        
        # Feature 4: Square Feet per Bedroom (how much space per bedroom?)
        print("   🛏️  Creating square feet per bedroom feature...")
        feature_data['Sqft_per_Bedroom'] = feature_data['Fin_sqft'] / feature_data['Bdrms'].replace(0, 1)
        
        # Feature 5: Bathrooms per Bedroom (how many bathrooms per bedroom?)
        print("   🚿 Creating bathrooms per bedroom feature...")
        feature_data['Bathrooms_per_Bedroom'] = feature_data['Total_Bathrooms'] / feature_data['Bdrms'].replace(0, 1)
        
        # Feature 6: Lot Size per Square Foot (how big is the yard compared to house?)
        print("   🌳 Creating lot size ratio feature...")
        feature_data['Lot_to_Sqft_Ratio'] = feature_data['Lotsize'] / feature_data['Fin_sqft']
        
        print("✅ Feature creation complete!")
        print(f"   🎨 Created 6 new features")
        print(f"   📊 Total features now: {len(feature_data.columns)}")
        
        return feature_data
    
    def select_features(self, data):
        """
        🎯 Select the best features for prediction
        
        Not all features are equally useful for predicting house prices.
        This function picks the most important ones.
        
        Args:
            data (pandas.DataFrame): Data with all features
            
        Returns:
            pandas.DataFrame: Data with only the best features
        """
        print("🎯 Selecting the best features...")
        
        # List of features we want to use for prediction
        # These are the characteristics that most affect house prices
        selected_features = [
            'Fin_sqft',              # Size of the house
            'Lotsize',               # Size of the yard
            'House_Age',             # How old the house is
            'Total_Bathrooms',       # Number of bathrooms
            'Bdrms',                 # Number of bedrooms
            'Stories',               # Number of floors
            'Price_per_Sqft',        # Price per square foot
            'Sqft_per_Bedroom',      # Space per bedroom
            'Bathrooms_per_Bedroom', # Bathrooms per bedroom
            'Lot_to_Sqft_Ratio'      # Yard size compared to house size
        ]
        
        # Make sure all selected features exist in the data
        available_features = [col for col in selected_features if col in data.columns]
        
        # Select only the features we want
        feature_data = data[available_features].copy()
        
        print(f"✅ Feature selection complete!")
        print(f"   🎯 Selected {len(available_features)} features")
        print(f"   📋 Features: {available_features}")
        
        return feature_data
    
    def prepare_data(self, data):
        """
        🎯 Prepare the data for training prediction models
        
        This is the main function that does everything:
        1. Creates new features
        2. Selects the best features
        3. Splits data into training and testing sets
        4. Scales the data (makes all numbers similar size)
        
        Args:
            data (pandas.DataFrame): Clean house data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) - Data ready for training
        """
        print("🎯 Preparing data for training...")
        
        # Step 1: Create new features
        feature_data = self.create_features(data)
        
        # Step 2: Select the best features
        X = self.select_features(feature_data)
        
        # Step 3: Get the target variable (what we want to predict)
        y = data['Sale_price']
        
        # Step 4: Split data into training and testing sets
        print("   📊 Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,           # Use 20% for testing, 80% for training
            random_state=42          # Makes sure we get the same split every time
        )
        
        # Step 5: Scale the features (make all numbers similar size)
        print("   ⚖️  Scaling the features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame with column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print("✅ Data preparation complete!")
        print(f"   📊 Training data: {X_train_scaled.shape[0]} houses")
        print(f"   📊 Testing data: {X_test_scaled.shape[0]} houses")
        print(f"   🏠 Features: {X_train_scaled.shape[1]} characteristics")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_names(self):
        """
        📋 Get the names of the features we're using
        
        Returns:
            list: Names of the features used for prediction
        """
        return [
            'Fin_sqft',              # Size of the house
            'Lotsize',               # Size of the yard
            'House_Age',             # How old the house is
            'Total_Bathrooms',       # Number of bathrooms
            'Bdrms',                 # Number of bedrooms
            'Stories',               # Number of floors
            'Price_per_Sqft',        # Price per square foot
            'Sqft_per_Bedroom',      # Space per bedroom
            'Bathrooms_per_Bedroom', # Bathrooms per bedroom
            'Lot_to_Sqft_Ratio'      # Yard size compared to house size
        ] 