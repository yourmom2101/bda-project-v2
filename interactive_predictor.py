#!/usr/bin/env python3
"""
🏠 Interactive House Price Predictor
====================================

This script lets you try out the house price prediction model with your own examples!
Just like the guide promised - input house details and get price predictions.

Example:
- Input: "A house with 3 bedrooms, 2 bathrooms, 2000 square feet, built in 1990"
- Output: "This house should cost around $350,000"

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class InteractivePredictor:
    """
    🎯 Interactive House Price Predictor
    
    This class lets you input house details and get price predictions!
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.scaler = None
        self.feature_names = [
            'Fin_sqft', 'Lotsize', 'House_Age', 'Total_Bathrooms', 
            'Bdrms', 'Stories', 'Price_per_Sqft', 'Sqft_per_Bedroom',
            'Bathrooms_per_Bedroom', 'Lot_to_Sqft_Ratio'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Try to load the saved model
            if os.path.exists('models/best_model.pkl'):
                with open('models/best_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ Loaded trained model successfully!")
            else:
                print("⚠️  No saved model found. Training a new one...")
                self.train_new_model()
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("Training a new model...")
            self.train_new_model()
    
    def train_new_model(self):
        """Train a new model if no saved model exists"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            # Load and prepare data
            print("📂 Loading house data...")
            data = pd.read_csv('data/house_data.csv')
            
            # Clean data
            clean_data = data[
                (data['Sale_price'] > 10000) & 
                (data['Sale_price'] < 5000000) &
                (data['Fin_sqft'] > 100) &
                (data['Fin_sqft'] < 10000)
            ].dropna(subset=['Sale_price', 'Fin_sqft', 'Year_Built', 'Bdrms'])
            
            # Create features
            clean_data['House_Age'] = 2024 - clean_data['Year_Built']
            clean_data['Total_Bathrooms'] = clean_data['Fbath'] + clean_data['Hbath']
            clean_data['Price_per_Sqft'] = clean_data['Sale_price'] / clean_data['Fin_sqft']
            clean_data['Sqft_per_Bedroom'] = clean_data['Fin_sqft'] / clean_data['Bdrms'].replace(0, 1)
            clean_data['Bathrooms_per_Bedroom'] = clean_data['Total_Bathrooms'] / clean_data['Bdrms'].replace(0, 1)
            clean_data['Lot_to_Sqft_Ratio'] = clean_data['Lotsize'] / clean_data['Fin_sqft']
            
            # Select features
            X = clean_data[self.feature_names]
            y = clean_data['Sale_price']
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            print("✅ Model trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            print("Please make sure you have the house_data.csv file in the data/ folder.")
    
    def predict_price(self, bedrooms, bathrooms, square_feet, year_built, lot_size=None, stories=1):
        """
        🏠 Predict house price based on input features
        
        Args:
            bedrooms (int): Number of bedrooms
            bathrooms (float): Number of bathrooms (can be 2.5 for 2 full + 1 half)
            square_feet (int): Square footage of the house
            year_built (int): Year the house was built
            lot_size (int): Lot size in square feet (optional)
            stories (int): Number of stories (optional)
        
        Returns:
            float: Predicted house price
        """
        try:
            # Calculate derived features
            house_age = 2024 - year_built
            
            # Handle lot size (use average if not provided)
            if lot_size is None:
                lot_size = square_feet * 3  # Assume lot is 3x house size
            
            # Calculate ratios
            price_per_sqft = 200  # Default estimate, will be refined by model
            sqft_per_bedroom = square_feet / max(bedrooms, 1)
            bathrooms_per_bedroom = bathrooms / max(bedrooms, 1)
            lot_to_sqft_ratio = lot_size / square_feet
            
            # Create feature array
            features = np.array([[
                square_feet, lot_size, house_age, bathrooms, bedrooms, stories,
                price_per_sqft, sqft_per_bedroom, bathrooms_per_bedroom, lot_to_sqft_ratio
            ]])
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            predicted_price = self.model.predict(features_scaled)[0]
            
            return max(predicted_price, 10000)  # Ensure minimum reasonable price
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def format_price(self, price):
        """Format price nicely"""
        if price >= 1000000:
            return f"${price/1000000:.1f}M"
        elif price >= 1000:
            return f"${price/1000:.0f}K"
        else:
            return f"${price:,.0f}"
    
    def interactive_mode(self):
        """🎮 Interactive mode for user input"""
        print("\n" + "="*60)
        print("🏠 INTERACTIVE HOUSE PRICE PREDICTOR")
        print("="*60)
        print("🎯 Enter house details and get price predictions!")
        print("💡 Example: 3 bedrooms, 2.5 bathrooms, 2000 sq ft, built in 1990")
        print("="*60)
        
        while True:
            print("\n📝 Enter house details (or 'quit' to exit):")
            print("-" * 40)
            
            try:
                # Get user input
                bedrooms = input("Number of bedrooms: ").strip()
                if bedrooms.lower() == 'quit':
                    break
                bedrooms = int(bedrooms)
                
                bathrooms = input("Number of bathrooms (e.g., 2.5 for 2 full + 1 half): ").strip()
                if bathrooms.lower() == 'quit':
                    break
                bathrooms = float(bathrooms)
                
                square_feet = input("Square footage: ").strip()
                if square_feet.lower() == 'quit':
                    break
                square_feet = int(square_feet)
                
                year_built = input("Year built: ").strip()
                if year_built.lower() == 'quit':
                    break
                year_built = int(year_built)
                
                lot_size = input("Lot size in sq ft (press Enter for default): ").strip()
                if lot_size.lower() == 'quit':
                    break
                lot_size = int(lot_size) if lot_size else None
                
                stories = input("Number of stories (press Enter for 1): ").strip()
                if stories.lower() == 'quit':
                    break
                stories = int(stories) if stories else 1
                
                # Make prediction
                print("\n🔮 Making prediction...")
                predicted_price = self.predict_price(bedrooms, bathrooms, square_feet, year_built, lot_size, stories)
                
                if predicted_price:
                    print("\n" + "🎉" * 20)
                    print("🏠 PRICE PREDICTION")
                    print("🎉" * 20)
                    print(f"📊 House Details:")
                    print(f"   🛏️  Bedrooms: {bedrooms}")
                    print(f"   🚿 Bathrooms: {bathrooms}")
                    print(f"   📏 Square Feet: {square_feet:,}")
                    print(f"   🏗️  Year Built: {year_built}")
                    print(f"   🌳 Lot Size: {lot_size:,} sq ft" if lot_size else f"   🌳 Lot Size: {square_feet * 3:,} sq ft (estimated)")
                    print(f"   🏢 Stories: {stories}")
                    print()
                    print(f"💰 Predicted Price: {self.format_price(predicted_price)}")
                    print("🎉" * 20)
                else:
                    print("❌ Could not make prediction. Please check your input.")
                
            except ValueError:
                print("❌ Please enter valid numbers.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def example_predictions(self):
        """📋 Show some example predictions"""
        print("\n📋 EXAMPLE PREDICTIONS")
        print("=" * 40)
        
        examples = [
            (3, 2.5, 2000, 1990, "Family home"),
            (2, 1, 1200, 1985, "Starter home"),
            (4, 3, 3000, 2005, "Luxury home"),
            (1, 1, 800, 1970, "Small home"),
            (5, 4, 4000, 2010, "Large family home")
        ]
        
        for bedrooms, bathrooms, sqft, year, description in examples:
            price = self.predict_price(bedrooms, bathrooms, sqft, year)
            if price:
                print(f"🏠 {description}: {bedrooms}BR, {bathrooms}BA, {sqft:,}sqft, {year}")
                print(f"   💰 Predicted: {self.format_price(price)}")
                print()

def main():
    """🚀 Main function"""
    print("🏠 Loading Interactive House Price Predictor...")
    
    # Create predictor
    predictor = InteractivePredictor()
    
    # Show example predictions
    predictor.example_predictions()
    
    # Start interactive mode
    predictor.interactive_mode()

if __name__ == "__main__":
    main() 