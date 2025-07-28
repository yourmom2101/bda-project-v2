#!/usr/bin/env python3
"""
🏠 Property Price Predictor - BDA V2
====================================

This is the main program that predicts house prices using smart computer algorithms.
It's like having a real estate expert in your computer!

What this program does:
1. Loads data about thousands of house sales
2. Teaches the computer to recognize patterns in house prices
3. Tests different prediction methods to find the best one
4. Creates beautiful graphs showing the results
5. Saves the best model for future use

Author: CBS BDA Student
Date: December 2024
"""

# Import the tools we need (like getting tools from a toolbox)
import pandas as pd                    # For working with data (like Excel)
import numpy as np                     # For doing math really fast
import matplotlib.pyplot as plt        # For making graphs
import seaborn as sns                  # For making graphs look pretty
import warnings                        # For hiding annoying warning messages
from pathlib import Path              # For working with files and folders
import os                             # For checking if files exist

# Import our own custom modules (like using special tools we made)
from data_loader import DataLoader    # Loads and cleans the house data
from model_trainer import ModelTrainer  # Trains the prediction models
from visualizer import Visualizer     # Creates beautiful graphs
from business_analyzer import BusinessAnalyzer  # Analyzes business value

# Hide warning messages so the output looks cleaner
warnings.filterwarnings('ignore')

# Set a random seed so we get the same results every time (like setting a seed for a plant)
np.random.seed(42)

def main():
    """
    🚀 Main function - this is where everything starts!
    
    Think of this like a recipe:
    1. Get the ingredients (data)
    2. Prepare the ingredients (clean data)
    3. Cook the food (train models)
    4. Taste the food (test models)
    5. Serve the food (show results)
    """
    
    print("🏠 PROPERTY PRICE PREDICTOR - BDA V2")
    print("=" * 50)
    print("🎯 Goal: Predict house prices using smart computer algorithms")
    print("📊 Method: Following CBS BDA course methodology")
    print("=" * 50)
    print()
    
    # Step 1: Load and prepare the data
    print("📂 STEP 1: Loading and preparing the data...")
    print("-" * 40)
    
    # Create a data loader (like hiring someone to get your ingredients)
    data_loader = DataLoader()
    
    # Load the house data (like getting ingredients from the store)
    house_data = data_loader.load_data()
    
    # Clean the data (like washing vegetables before cooking)
    clean_data = data_loader.clean_data(house_data)
    
    # Prepare the data for training (like chopping vegetables)
    X_train, X_test, y_train, y_test = data_loader.prepare_data(clean_data)
    
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Training data: {X_train.shape[0]} houses")
    print(f"   📊 Test data: {X_test.shape[0]} houses")
    print(f"   🏠 Features: {X_train.shape[1]} characteristics per house")
    print()
    
    # Step 2: Train the prediction models
    print("🤖 STEP 2: Training the prediction models...")
    print("-" * 40)
    
    # Create a model trainer (like hiring a chef)
    model_trainer = ModelTrainer()
    
    # Train all the different models (like trying different cooking methods)
    models = model_trainer.train_all_models(X_train, y_train)
    
    print("✅ All models trained successfully!")
    print()
    
    # Step 3: Test and evaluate the models
    print("🧪 STEP 3: Testing and evaluating the models...")
    print("-" * 40)
    
    # Test all models on the test data (like tasting all the dishes)
    results = model_trainer.evaluate_models(models, X_test, y_test)
    
    # Find the best model (like picking the tastiest dish)
    best_model_name = model_trainer.get_best_model(results)
    best_model = models[best_model_name]
    
    print(f"🏆 Best model: {best_model_name}")
    print(f"📈 Accuracy: {results[best_model_name]['r2_score']:.1%}")
    print(f"💰 Average error: ${results[best_model_name]['rmse']:,.0f}")
    print()
    
    # Step 4: Create beautiful visualizations
    print("📊 STEP 4: Creating beautiful visualizations...")
    print("-" * 40)
    
    # Create a visualizer (like hiring a photographer)
    visualizer = Visualizer()
    
    # Make predictions with the best model
    y_pred = best_model.predict(X_test)
    
    # Create all the graphs and charts
    visualizer.create_all_plots(X_test, y_test, y_pred, best_model, results)
    
    print("✅ All visualizations created successfully!")
    print("📁 Check the 'plots/' folder to see the graphs")
    print()
    
    # Step 5: Analyze business value
    print("💼 STEP 5: Analyzing business value...")
    print("-" * 40)
    
    # Create a business analyzer (like hiring a business consultant)
    business_analyzer = BusinessAnalyzer()
    
    # Analyze the business value of our predictions
    business_insights = business_analyzer.analyze_business_value(
        X_test, y_test, y_pred, best_model, results
    )
    
    # Print the business insights
    business_analyzer.print_insights(business_insights)
    print()
    
    # Step 6: Save the best model for future use
    print("💾 STEP 6: Saving the best model...")
    print("-" * 40)
    
    # Save the best model so we can use it later (like freezing leftovers)
    model_trainer.save_model(best_model, best_model_name)
    
    print("✅ Best model saved successfully!")
    print("📁 Check the 'models/' folder for the saved model")
    print()
    
    # Final summary
    print("🎉 PROJECT COMPLETE!")
    print("=" * 50)
    print("✅ Data loaded and cleaned")
    print("✅ Models trained and tested")
    print("✅ Visualizations created")
    print("✅ Business analysis completed")
    print("✅ Best model saved")
    print()
    print("📊 Results Summary:")
    print(f"   🏆 Best Model: {best_model_name}")
    print(f"   📈 Accuracy: {results[best_model_name]['r2_score']:.1%}")
    print(f"   💰 Average Error: ${results[best_model_name]['rmse']:,.0f}")
    print(f"   🏠 Houses Analyzed: {len(y_test)}")
    print()
    print("🎯 Next Steps:")
    print("   1. Look at the graphs in the 'plots/' folder")
    print("   2. Read the business insights above")
    print("   3. Use the saved model to predict new house prices")
    print("   4. Share your findings with others!")
    print()
    print("🌟 Thank you for using Property Price Predictor - BDA V2!")
    print("   Made with ❤️ for learning and helping people!")

def check_requirements():
    """
    🔍 Check if all required files and folders exist
    
    This is like checking if you have all the ingredients before cooking
    """
    print("🔍 Checking if everything is ready...")
    
    # Check if data folder exists
    if not os.path.exists('data'):
        print("❌ Error: 'data' folder not found!")
        print("   Please make sure you have the house data file.")
        return False
    
    # Check if data file exists
    if not os.path.exists('data/house_data.csv'):
        print("❌ Error: 'data/house_data.csv' not found!")
        print("   Please make sure you have the house data file.")
        return False
    
    # Create folders if they don't exist
    folders = ['models', 'plots', 'docs']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 Created '{folder}' folder")
    
    print("✅ Everything looks good!")
    return True

if __name__ == "__main__":
    """
    🚀 This is where the program starts when you run it
    
    It's like the "ON" button for the whole program
    """
    
    # Check if everything is ready
    if not check_requirements():
        print("❌ Please fix the issues above and try again.")
        exit(1)
    
    # Run the main program
    try:
        main()
    except Exception as e:
        print(f"❌ Something went wrong: {e}")
        print("💡 Try checking the error message above for clues")
        print("📞 If you need help, check the README.md file") 