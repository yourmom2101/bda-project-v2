#!/usr/bin/env python3
"""
🧪 Test Model Persistence
=========================

This script tests the model saving and loading functionality.
It demonstrates how models are saved and can be loaded instantly.

Author: CBS BDA Student
Date: December 2024
"""

import os
import time
import pickle
from improved_model import main as train_models, load_model

def test_model_persistence():
    """Test the model saving and loading functionality"""
    print("🧪 TESTING MODEL PERSISTENCE")
    print("=" * 50)
    
    # Check if models already exist
    if os.path.exists('models/ensemble_model.pkl'):
        print("✅ Found existing models!")
        
        # Test loading time
        print("\n⏱️  Testing model loading speed...")
        start_time = time.time()
        
        model_data = load_model('ensemble_model.pkl')
        
        load_time = time.time() - start_time
        print(f"📂 Model loaded in {load_time:.3f} seconds")
        
        if model_data:
            print("✅ Model loaded successfully!")
            print(f"   🏠 Model type: {type(model_data['model']).__name__}")
            print(f"   📊 Features: {len(model_data['feature_names'])}")
            print(f"   ⚙️  Scaler: {type(model_data['scaler']).__name__}")
        else:
            print("❌ Failed to load model")
            
    else:
        print("🔄 No saved models found. Training new models...")
        print("⏱️  This will take a few minutes for the first run...")
        
        start_time = time.time()
        train_models()
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Training completed in {training_time:.1f} seconds")
        print("💾 Models saved for future use!")
    
    # Test loading all model types
    print("\n📂 TESTING ALL MODEL TYPES:")
    print("-" * 30)
    
    model_files = [
        'ensemble_model.pkl',
        'random_forest_model.pkl', 
        'gradient_boosting_model.pkl',
        'xgboost_model.pkl'
    ]
    
    for filename in model_files:
        if os.path.exists(f'models/{filename}'):
            model_data = load_model(filename)
            if model_data:
                print(f"✅ {filename}: Loaded successfully")
            else:
                print(f"❌ {filename}: Failed to load")
        else:
            print(f"⚠️  {filename}: Not found")
    
    # Show file sizes
    print("\n📊 MODEL FILE SIZES:")
    print("-" * 25)
    
    total_size = 0
    for filename in model_files:
        filepath = f'models/{filename}'
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
            total_size += size
            print(f"   {filename}: {size:.2f} MB")
    
    print(f"   Total: {total_size:.2f} MB")
    
    print("\n🎉 MODEL PERSISTENCE TEST COMPLETE!")
    print("💡 Now you can use the interactive predictor instantly!")

if __name__ == "__main__":
    test_model_persistence() 