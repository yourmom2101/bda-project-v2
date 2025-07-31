"""
🏠 Property Price Predictor - BDA V2
====================================

Source code package for the Property Price Predictor project.

This package contains all the core modules for:
- Data loading and cleaning
- Model training and evaluation
- Visualization and plotting
- Business analysis and insights

Author: CBS BDA Student
Date: December 2024
"""

__version__ = "2.0.0"
__author__ = "CBS BDA Student"
__email__ = "student@cbs.dk"

# Import main modules for easy access
from .data_loader import DataLoader
from .model_trainer import ModelTrainer
from .visualizer import Visualizer
from .business_analyzer import BusinessAnalyzer

__all__ = [
    'DataLoader',
    'ModelTrainer', 
    'Visualizer',
    'BusinessAnalyzer'
] 