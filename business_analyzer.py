"""
💼 Business Analyzer Module
===========================

This module analyzes the business value and practical applications of our house price predictions.
Think of it like hiring a business consultant to tell you how to use your results!

What it analyzes:
1. Business value and ROI of the predictions
2. Market insights and trends
3. Risk assessment and limitations
4. Practical applications for different stakeholders
5. Recommendations for implementation

This helps translate technical results into business value that anyone can understand.

Author: CBS BDA Student
Date: December 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime

class BusinessAnalyzer:
    """
    💼 Business Analyzer Class
    
    This class analyzes the business value of our house price predictions.
    It's like having a business consultant explain what our results mean!
    """
    
    def __init__(self):
        """
        🚀 Initialize the business analyzer
        
        This sets up the analyzer with business metrics and insights
        """
        self.insights = {}
        self.recommendations = []
        self.risks = []
        
    def analyze_business_value(self, X_test, y_test, y_pred, best_model, results):
        """
        💰 Analyze the business value of our predictions
        
        This function calculates the business impact and value of our model.
        It's like calculating the ROI of our investment in this project!
        
        Args:
            X_test: Test features
            y_test: Actual house prices
            y_pred: Predicted house prices
            best_model: The best performing model
            results: Performance results for all models
            
        Returns:
            dict: Business insights and analysis
        """
        print("💼 Analyzing business value...")
        
        # Get the best model name and performance
        best_model_name = max(results.items(), key=lambda x: x[1]['r2_score'])[0]
        best_performance = results[best_model_name]
        
        # Calculate key business metrics
        self.insights = {
            'model_performance': self._analyze_model_performance(best_performance),
            'market_insights': self._analyze_market_insights(X_test, y_test),
            'feature_insights': self._analyze_feature_insights(best_model, X_test),
            'risk_assessment': self._assess_risks(y_test, y_pred),
            'business_applications': self._identify_business_applications(),
            'roi_analysis': self._calculate_roi(y_test, y_pred)
        }
        
        return self.insights
    
    def _analyze_model_performance(self, performance):
        """
        📊 Analyze the performance of our best model
        
        This translates technical metrics into business language.
        
        Args:
            performance: Performance metrics of the best model
            
        Returns:
            dict: Business interpretation of model performance
        """
        r2_score = performance['r2_score']
        rmse = performance['rmse']
        mape = performance['mape']
        
        # Interpret R² score in business terms
        if r2_score >= 0.95:
            accuracy_level = "Excellent"
            business_impact = "Very high confidence in predictions"
        elif r2_score >= 0.90:
            accuracy_level = "Very Good"
            business_impact = "High confidence in predictions"
        elif r2_score >= 0.80:
            accuracy_level = "Good"
            business_impact = "Moderate confidence in predictions"
        else:
            accuracy_level = "Fair"
            business_impact = "Limited confidence in predictions"
        
        # Interpret RMSE in business terms
        avg_price = 250000  # Assume average house price
        error_percentage = (rmse / avg_price) * 100
        
        if error_percentage <= 5:
            error_level = "Very Low"
            error_impact = "Predictions are very accurate"
        elif error_percentage <= 10:
            error_level = "Low"
            error_impact = "Predictions are reasonably accurate"
        elif error_percentage <= 20:
            error_level = "Moderate"
            error_impact = "Predictions have moderate accuracy"
        else:
            error_level = "High"
            error_impact = "Predictions have limited accuracy"
        
        return {
            'accuracy_level': accuracy_level,
            'business_impact': business_impact,
            'error_level': error_level,
            'error_impact': error_impact,
            'r2_score': r2_score,
            'rmse': rmse,
            'mape': mape,
            'error_percentage': error_percentage
        }
    
    def _analyze_market_insights(self, X_test, y_test):
        """
        📈 Analyze market insights from the data
        
        This extracts business insights about the housing market.
        
        Args:
            X_test: Test features
            y_test: Actual house prices
            
        Returns:
            dict: Market insights and trends
        """
        # Calculate market statistics
        avg_price = y_test.mean()
        median_price = y_test.median()
        price_range = y_test.max() - y_test.min()
        
        # Analyze price distribution
        price_quartiles = y_test.quantile([0.25, 0.5, 0.75])
        affordable_homes = (y_test <= price_quartiles[0.25]).sum()
        luxury_homes = (y_test >= price_quartiles[0.75]).sum()
        
        # Market segmentation
        total_homes = len(y_test)
        affordable_percentage = (affordable_homes / total_homes) * 100
        luxury_percentage = (luxury_homes / total_homes) * 100
        
        # Feature analysis for market insights
        market_insights = {}
        
        if 'Fin_sqft' in X_test.columns:
            # Price per square foot analysis
            price_per_sqft = y_test / X_test['Fin_sqft']
            avg_price_per_sqft = price_per_sqft.mean()
            market_insights['avg_price_per_sqft'] = avg_price_per_sqft
            market_insights['price_per_sqft_insight'] = f"Average price per square foot: ${avg_price_per_sqft:.0f}"
        
        if 'House_Age' in X_test.columns:
            # Age analysis
            avg_age = X_test['House_Age'].mean()
            market_insights['avg_house_age'] = avg_age
            market_insights['age_insight'] = f"Average house age: {avg_age:.0f} years"
        
        return {
            'avg_price': avg_price,
            'median_price': median_price,
            'price_range': price_range,
            'affordable_percentage': affordable_percentage,
            'luxury_percentage': luxury_percentage,
            'market_segments': {
                'affordable': affordable_homes,
                'mid_range': total_homes - affordable_homes - luxury_homes,
                'luxury': luxury_homes
            },
            'feature_insights': market_insights
        }
    
    def _analyze_feature_insights(self, model, X_test):
        """
        🎯 Analyze which features are most important for business
        
        This helps understand what drives house prices from a business perspective.
        
        Args:
            model: The best performing model
            X_test: Test features
            
        Returns:
            dict: Feature importance insights
        """
        feature_names = X_test.columns.tolist()
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return {'message': 'Feature importance not available for this model type'}
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Get top 3 most important features
        top_features = importance_df.head(3)
        
        # Business interpretation of feature importance
        feature_insights = []
        for _, row in top_features.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            if 'Fin_sqft' in feature:
                insight = f"Square footage is the most important factor ({importance:.1%} impact)"
            elif 'Price_per_Sqft' in feature:
                insight = f"Market rate per square foot is crucial ({importance:.1%} impact)"
            elif 'Bdrms' in feature:
                insight = f"Number of bedrooms significantly affects price ({importance:.1%} impact)"
            elif 'Bathrooms' in feature:
                insight = f"Number of bathrooms is important ({importance:.1%} impact)"
            elif 'Age' in feature:
                insight = f"House age affects pricing ({importance:.1%} impact)"
            else:
                insight = f"{feature} is important ({importance:.1%} impact)"
            
            feature_insights.append(insight)
        
        return {
            'top_features': top_features.to_dict('records'),
            'feature_insights': feature_insights,
            'business_implications': [
                "Focus on accurate square footage measurements",
                "Consider market rates when pricing",
                "Bedroom and bathroom count are key selling points"
            ]
        }
    
    def _assess_risks(self, y_test, y_pred):
        """
        ⚠️ Assess risks and limitations of the model
        
        This identifies potential problems and limitations for business use.
        
        Args:
            y_test: Actual house prices
            y_pred: Predicted house prices
            
        Returns:
            dict: Risk assessment and limitations
        """
        # Calculate prediction errors
        errors = np.abs(y_test - y_pred)
        error_percentages = (errors / y_test) * 100
        
        # Identify high-error predictions
        high_error_threshold = 20  # 20% error
        high_error_count = (error_percentages > high_error_threshold).sum()
        high_error_percentage = (high_error_count / len(y_test)) * 100
        
        # Identify extreme errors
        extreme_error_threshold = 50  # 50% error
        extreme_error_count = (error_percentages > extreme_error_threshold).sum()
        extreme_error_percentage = (extreme_error_count / len(y_test)) * 100
        
        risks = [
            f"{high_error_percentage:.1f}% of predictions have errors > 20%",
            f"{extreme_error_percentage:.1f}% of predictions have errors > 50%",
            "Model may not capture unique property features",
            "Market conditions can change over time",
            "Location-specific factors not fully captured"
        ]
        
        limitations = [
            "Cannot predict future market changes",
            "Does not account for property condition",
            "Limited to historical data patterns",
            "May not work well for unique properties",
            "Requires regular model updates"
        ]
        
        return {
            'high_error_percentage': high_error_percentage,
            'extreme_error_percentage': extreme_error_percentage,
            'risks': risks,
            'limitations': limitations,
            'recommendations': [
                "Use predictions as guidance, not absolute values",
                "Combine with professional appraisals",
                "Regularly update the model with new data",
                "Consider market-specific adjustments"
            ]
        }
    
    def _identify_business_applications(self):
        """
        🎯 Identify practical business applications
        
        This lists how different stakeholders can use the model.
        
        Returns:
            dict: Business applications for different users
        """
        applications = {
            'real_estate_agents': [
                "Quick property valuations for clients",
                "Setting competitive listing prices",
                "Market analysis for different neighborhoods",
                "Identifying underpriced properties"
            ],
            'home_buyers': [
                "Understanding fair market value",
                "Negotiating purchase prices",
                "Comparing properties across neighborhoods",
                "Budget planning and affordability assessment"
            ],
            'investors': [
                "Identifying investment opportunities",
                "Portfolio valuation and analysis",
                "Market trend analysis",
                "Risk assessment for property investments"
            ],
            'lenders': [
                "Automated property valuations",
                "Risk assessment for mortgage applications",
                "Portfolio management and monitoring",
                "Market analysis for lending decisions"
            ],
            'appraisers': [
                "Supporting appraisal reports",
                "Market trend analysis",
                "Quality control for valuations",
                "Efficiency improvements in appraisal process"
            ]
        }
        
        return applications
    
    def _calculate_roi(self, y_test, y_pred):
        """
        💰 Calculate potential ROI and business value
        
        This estimates the financial benefits of using the model.
        
        Args:
            y_test: Actual house prices
            y_pred: Predicted house prices
            
        Returns:
            dict: ROI analysis and business value
        """
        # Calculate average prediction error in dollars
        avg_error = np.mean(np.abs(y_test - y_pred))
        
        # Estimate business value based on error reduction
        # Assume traditional methods have 15% error vs our model's error
        traditional_error_rate = 0.15
        our_error_rate = np.mean(np.abs(y_test - y_pred) / y_test)
        error_reduction = traditional_error_rate - our_error_rate
        
        # Calculate potential savings
        avg_house_price = y_test.mean()
        potential_savings_per_house = avg_house_price * error_reduction
        
        # Estimate market size (example: 1000 houses per month)
        monthly_houses = 1000
        annual_houses = monthly_houses * 12
        annual_potential_savings = potential_savings_per_house * annual_houses
        
        # Development and maintenance costs (estimated)
        development_cost = 50000  # One-time development cost
        annual_maintenance = 10000  # Annual maintenance and updates
        
        # ROI calculation
        first_year_roi = (annual_potential_savings - development_cost - annual_maintenance) / (development_cost + annual_maintenance)
        
        return {
            'avg_error_dollars': avg_error,
            'error_reduction': error_reduction,
            'potential_savings_per_house': potential_savings_per_house,
            'annual_potential_savings': annual_potential_savings,
            'development_cost': development_cost,
            'annual_maintenance': annual_maintenance,
            'first_year_roi': first_year_roi,
            'payback_period_months': (development_cost + annual_maintenance) / (annual_potential_savings / 12)
        }
    
    def print_insights(self, insights):
        """
        📊 Print business insights in a clear, readable format
        
        This presents all the business analysis in an easy-to-understand way.
        
        Args:
            insights: Dictionary containing all business insights
        """
        print("💼 BUSINESS ANALYSIS RESULTS")
        print("=" * 60)
        
        # Model Performance
        print("\n📊 MODEL PERFORMANCE")
        print("-" * 30)
        perf = insights['model_performance']
        print(f"🎯 Accuracy Level: {perf['accuracy_level']}")
        print(f"📈 Business Impact: {perf['business_impact']}")
        print(f"⚠️  Error Level: {perf['error_level']}")
        print(f"💰 Error Impact: {perf['error_impact']}")
        print(f"📊 R² Score: {perf['r2_score']:.3f}")
        print(f"💰 Average Error: ${perf['rmse']:,.0f}")
        
        # Market Insights
        print("\n📈 MARKET INSIGHTS")
        print("-" * 30)
        market = insights['market_insights']
        print(f"🏠 Average House Price: ${market['avg_price']:,.0f}")
        print(f"📊 Median House Price: ${market['median_price']:,.0f}")
        print(f"💰 Price Range: ${market['price_range']:,.0f}")
        print(f"🏘️  Affordable Homes (<25th percentile): {market['affordable_percentage']:.1f}%")
        print(f"🏰 Luxury Homes (>75th percentile): {market['luxury_percentage']:.1f}%")
        
        # Feature Insights
        print("\n🎯 KEY DRIVERS OF HOUSE PRICES")
        print("-" * 30)
        features = insights['feature_insights']
        for insight in features['feature_insights']:
            print(f"• {insight}")
        
        # Risk Assessment
        print("\n⚠️  RISKS AND LIMITATIONS")
        print("-" * 30)
        risks = insights['risk_assessment']
        for risk in risks['risks'][:3]:  # Show top 3 risks
            print(f"• {risk}")
        
        # Business Applications
        print("\n🎯 BUSINESS APPLICATIONS")
        print("-" * 30)
        apps = insights['business_applications']
        print("👥 Real Estate Agents:")
        for app in apps['real_estate_agents'][:2]:
            print(f"   • {app}")
        
        print("🏠 Home Buyers:")
        for app in apps['home_buyers'][:2]:
            print(f"   • {app}")
        
        # ROI Analysis
        print("\n💰 BUSINESS VALUE & ROI")
        print("-" * 30)
        roi = insights['roi_analysis']
        print(f"💵 Potential Savings per House: ${roi['potential_savings_per_house']:,.0f}")
        print(f"📈 Annual Potential Savings: ${roi['annual_potential_savings']:,.0f}")
        print(f"🎯 First Year ROI: {roi['first_year_roi']:.1%}")
        print(f"⏰ Payback Period: {roi['payback_period_months']:.1f} months")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 30)
        print("✅ Use predictions as guidance, not absolute values")
        print("✅ Combine with professional appraisals")
        print("✅ Regularly update the model with new data")
        print("✅ Consider market-specific adjustments")
        print("✅ Focus on accurate square footage measurements")
        
        print("\n🎉 Business analysis complete!")
        print("📊 This model provides significant business value for house price prediction!") 