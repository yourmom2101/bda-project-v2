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
            'Bathrooms_per_Bedroom', 'Lot_to_Sqft_Ratio', 'Age_Squared', 'Sqft_Squared'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Try to load the saved ensemble model first
            if os.path.exists('models/ensemble_model.pkl'):
                with open('models/ensemble_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                print("✅ Loaded ensemble model successfully!")
                return
            
            # Try to load random forest model as backup
            elif os.path.exists('models/random_forest_model.pkl'):
                with open('models/random_forest_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                print("✅ Loaded random forest model successfully!")
                return
                
            else:
                print("⚠️  No saved models found. Please run improved_model.py first to train and save models.")
                print("   This will create the models/ directory with trained models.")
                return
                
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("Please run improved_model.py first to train and save models.")
    
    def predict_price(self, bedrooms, bathrooms, square_feet, year_built, lot_size=None, stories=1):
        """
        Predict house price based on input features
        
        Args:
            bedrooms (int): Number of bedrooms
            bathrooms (float): Number of bathrooms (can be 1.5, 2.5, etc.)
            square_feet (int): Square footage of the house
            year_built (int): Year the house was built
            lot_size (int, optional): Lot size in square feet
            stories (int): Number of stories
            
        Returns:
            float: Predicted house price
        """
        if self.model is None:
            return None
            
        try:
            # Calculate derived features
            house_age = 2024 - year_built
            sqft_per_bedroom = square_feet / max(bedrooms, 1)  # Avoid division by zero
            bathrooms_per_bedroom = bathrooms / max(bedrooms, 1)
            
            # Use lot_size if provided, otherwise estimate based on square footage
            if lot_size is None:
                lot_size = square_feet * 1.5  # Typical lot is 1.5x house size
            
            lot_to_sqft_ratio = lot_size / square_feet
            age_squared = house_age ** 2
            sqft_squared = square_feet ** 2
            
            # Estimate price per square foot (this is a rough estimate)
            # In real applications, this would come from market data
            estimated_price_per_sqft = 150  # $150 per sq ft as baseline
            
            # Create feature array
            features = np.array([
                square_feet,           # Fin_sqft
                lot_size,              # Lotsize
                house_age,             # House_Age
                bathrooms,             # Total_Bathrooms
                bedrooms,              # Bdrms
                stories,               # Stories
                estimated_price_per_sqft,  # Price_per_Sqft
                sqft_per_bedroom,      # Sqft_per_Bedroom
                bathrooms_per_bedroom, # Bathrooms_per_Bedroom
                lot_to_sqft_ratio,     # Lot_to_Sqft_Ratio
                age_squared,           # Age_Squared
                sqft_squared           # Sqft_Squared
            ]).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            predicted_price = self.model.predict(features_scaled)[0]
            
            return max(predicted_price, 10000)  # Ensure minimum price of $10,000
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def format_price(self, price):
        """Format price nicely for display"""
        if price >= 1000000:
            return f"${price/1000000:.1f}M"
        elif price >= 1000:
            return f"${price/1000:.0f}K"
        else:
            return f"${price:,.0f}"
    
    def interactive_mode(self):
        """Run interactive mode for user input"""
        print("\n🏠 INTERACTIVE HOUSE PRICE PREDICTOR")
        print("=" * 50)
        print("Enter house details to get a price prediction!")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                # Get user input
                print("Enter house details:")
                bedrooms = input("   Number of bedrooms: ").strip()
                
                if bedrooms.lower() == 'quit':
                    print("👋 Thanks for using the House Price Predictor!")
                    break
                
                bedrooms = int(bedrooms)
                bathrooms = float(input("   Number of bathrooms: ").strip())
                square_feet = int(input("   Square footage: ").strip())
                year_built = int(input("   Year built: ").strip())
                
                # Optional inputs
                lot_size_input = input("   Lot size (sq ft, optional - press Enter to skip): ").strip()
                lot_size = int(lot_size_input) if lot_size_input else None
                
                stories_input = input("   Number of stories (optional - press Enter for 1): ").strip()
                stories = int(stories_input) if stories_input else 1
                
                # Make prediction
                predicted_price = self.predict_price(
                    bedrooms, bathrooms, square_feet, year_built, lot_size, stories
                )
                
                if predicted_price:
                    print(f"\n🏠 PRICE PREDICTION:")
                    print(f"   📊 Predicted Price: {self.format_price(predicted_price)}")
                    print(f"   📈 Price per sq ft: ${predicted_price/square_feet:.0f}")
                    print(f"   🏠 House Details: {bedrooms}BR, {bathrooms}BA, {square_feet:,} sq ft, built {year_built}")
                else:
                    print("❌ Could not make prediction. Please check your inputs.")
                
                print("\n" + "-" * 50 + "\n")
                
            except ValueError:
                print("❌ Please enter valid numbers for all fields.")
                print("\n" + "-" * 50 + "\n")
            except KeyboardInterrupt:
                print("\n👋 Thanks for using the House Price Predictor!")
                break
    
    def example_predictions(self):
        """Show some example predictions"""
        print("\n📊 EXAMPLE PREDICTIONS")
        print("=" * 40)
        
        examples = [
            (3, 2, 2000, 1990, None, 1),      # Typical family home
            (2, 1.5, 1200, 2000, None, 1),    # Small starter home
            (4, 3, 3000, 1985, None, 2),      # Large family home
            (1, 1, 800, 2010, None, 1),       # Small condo
            (5, 4, 4000, 1975, None, 2),      # Luxury home
        ]
        
        for i, (bedrooms, bathrooms, square_feet, year_built, lot_size, stories) in enumerate(examples, 1):
            predicted_price = self.predict_price(bedrooms, bathrooms, square_feet, year_built, lot_size, stories)
            
            if predicted_price:
                print(f"{i}. {bedrooms}BR, {bathrooms}BA, {square_feet:,} sq ft, built {year_built}")
                print(f"   → Predicted Price: {self.format_price(predicted_price)}")
                print(f"   → Price per sq ft: ${predicted_price/square_feet:.0f}")
                print()

def main():
    """Main function"""
    predictor = InteractivePredictor()
    
    if predictor.model is None:
        print("\n❌ No trained model available.")
        print("Please run: python3 improved_model.py")
        print("This will train and save the models for use with the interactive predictor.")
        return
    
    # Show example predictions
    predictor.example_predictions()
    
    # Run interactive mode
    predictor.interactive_mode()

if __name__ == "__main__":
    main() 