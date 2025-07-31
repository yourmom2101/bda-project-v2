# 🚀 Getting Started Guide - Property Price Predictor BDA V2

## Welcome! 👋

This guide will help you get started with the Property Price Predictor project. Even if you've never used Python or machine learning before, you'll be able to run this project and understand what it does!

## 📋 What You'll Learn

By the end of this guide, you'll be able to:
- ✅ Set up the project on your computer
- ✅ Run the house price prediction model
- ✅ Understand what the results mean
- ✅ Use the model for your own predictions
- ✅ Create beautiful visualizations

## 🛠️ What You Need

### Required Software
1. **Python 3.8 or newer** - The programming language we use
2. **Git** - To download the project (optional)
3. **A text editor** - Like VS Code, Notepad++, or even regular Notepad

### Computer Skills
- Basic computer skills (opening folders, running programs)
- Ability to use the command line/terminal (we'll show you how!)

## 📥 Step 1: Download the Project

### Option A: Using Git (Recommended)
```bash
# Open your terminal/command prompt and run:
git clone https://github.com/yourmom2101/bda-v2-project.git

# Go into the project folder
cd bda-v2-project
```

### Option B: Download as ZIP
1. Go to the project page on GitHub
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to a folder on your computer
5. Open the folder in your terminal/command prompt

## 🐍 Step 2: Install Python

### Check if Python is already installed
```bash
python --version
# or
python3 --version
```

If you see a version number (like "Python 3.9.0"), you're good to go!

### If Python is not installed:
1. **Windows**: Download from [python.org](https://www.python.org/downloads/)
2. **Mac**: Download from [python.org](https://www.python.org/downloads/) or use Homebrew
3. **Linux**: Use your package manager (e.g., `sudo apt install python3`)

## 📦 Step 3: Install Required Packages

The project needs some special tools (called "packages") to work. Here's how to install them:

```bash
# Make sure you're in the project folder
cd bda-v2-project

# Install all required packages
pip install -r requirements.txt
```

**What this does:**
- Installs pandas (for working with data)
- Installs scikit-learn (for machine learning)
- Installs matplotlib and seaborn (for creating graphs)
- Installs other necessary tools

**If you get an error:**
- Try: `pip3 install -r requirements.txt`
- Or: `python -m pip install -r requirements.txt`

## 🏠 Step 4: Understanding the Data

The project uses real house sales data to learn patterns. Here's what the data contains:

### House Characteristics
- **Square Feet** - How big the house is
- **Bedrooms** - Number of bedrooms
- **Bathrooms** - Number of bathrooms
- **Year Built** - When the house was built
- **Lot Size** - Size of the yard
- **Stories** - Number of floors

### What We're Predicting
- **Sale Price** - How much the house sold for

## 🚀 Step 5: Run the Project

Now the fun part! Let's run the project:

```bash
# Make sure you're in the project folder
python main.py
```

**What happens:**
1. The program loads the house data
2. It cleans and prepares the data
3. It trains 6 different prediction models
4. It tests how well each model performs
5. It creates beautiful graphs and charts
6. It saves the best model for future use

## 📊 Step 6: Understanding the Results

When the program finishes, you'll see:

### Model Performance
- **R² Score**: How accurate the predictions are (0-1, higher is better)
- **RMSE**: Average prediction error in dollars (lower is better)
- **MAPE**: Average percentage error (lower is better)

### Best Model
The program will tell you which model performed best (usually Random Forest).

### Generated Files
Check the `plots/` folder for beautiful graphs:
- `data_exploration.png` - Understanding the house data
- `model_performance.png` - Comparing different models
- `random_forest_predictions.png` - How well the best model predicts
- `random_forest_feature_importance.png` - What affects house prices most

## 🎯 Step 7: Making Your Own Predictions

Want to predict the price of a specific house? Here's how:

### Option A: Use the Saved Model
```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('models/random_forest.joblib')

# Create house data (example)
house_data = pd.DataFrame({
    'Fin_sqft': [2000],           # 2000 square feet
    'Lotsize': [5000],            # 5000 sq ft lot
    'House_Age': [15],            # 15 years old
    'Total_Bathrooms': [2.5],     # 2.5 bathrooms
    'Bdrms': [3],                 # 3 bedrooms
    'Stories': [2],               # 2 stories
    'Price_per_Sqft': [150],      # $150 per sq ft
    'Sqft_per_Bedroom': [667],    # 667 sq ft per bedroom
    'Bathrooms_per_Bedroom': [0.83], # 0.83 bathrooms per bedroom
    'Lot_to_Sqft_Ratio': [2.5]    # Lot is 2.5x house size
})

# Make prediction
predicted_price = model.predict(house_data)[0]
print(f"Predicted price: ${predicted_price:,.0f}")
```

### Option B: Use the Interactive Script
We can create a simple script that asks you questions about a house and predicts its price.

## 🔧 Troubleshooting

### Common Problems and Solutions

**Problem**: "ModuleNotFoundError: No module named 'pandas'"
**Solution**: Run `pip install -r requirements.txt` again

**Problem**: "FileNotFoundError: data/house_data.csv"
**Solution**: Make sure you're in the correct folder and the data file exists

**Problem**: "Permission denied" when installing packages
**Solution**: Try `pip install --user -r requirements.txt`

**Problem**: Python command not found
**Solution**: Make sure Python is installed and in your PATH

### Getting Help
If you're stuck:
1. Check the error message carefully
2. Make sure you followed all steps
3. Try running the commands again
4. Ask for help in the project discussions

## 📚 Next Steps

Once you've successfully run the project:

1. **Explore the Code**: Look at the different Python files to understand how it works
2. **Try Different Data**: Use your own house data if you have it
3. **Modify the Models**: Change parameters to see how it affects performance
4. **Create Visualizations**: Modify the plotting code to create custom graphs
5. **Learn More**: Read about machine learning and data science

## 🎓 Understanding the Methodology

This project follows the **CRISP-DM** methodology taught in business data analytics:

1. **Business Understanding** - What problem are we solving?
2. **Data Understanding** - What data do we have?
3. **Data Preparation** - Clean and prepare the data
4. **Modeling** - Train different prediction models
5. **Evaluation** - Test how well the models work
6. **Deployment** - Use the best model for predictions

## 🏆 Congratulations!

You've successfully:
- ✅ Set up a machine learning project
- ✅ Trained multiple prediction models
- ✅ Analyzed house price data
- ✅ Created professional visualizations
- ✅ Understood the business value

You're now ready to use machine learning for real-world problems!

## 📞 Need More Help?

- Check the main README.md file
- Look at the code comments (they're very detailed!)
- Ask questions in the project discussions
- Read about machine learning online

**Happy predicting! 🏠📊** 