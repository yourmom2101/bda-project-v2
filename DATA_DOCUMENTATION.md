# Data Documentation

## 📊 Data Files

### Original Data
- **File**: `data/house_data.csv`
- **Size**: 3.8MB
- **Records**: 34,523 houses
- **Features**: 19 original features
- **Status**: Raw, uncleaned dataset with outliers and missing values

### Cleaned Data
- **File**: `data/house_data_cleaned.csv` (generated locally)
- **Size**: 5.0MB
- **Records**: 28,104 houses
- **Features**: 27 features (19 original + 8 enhanced)
- **Status**: Cleaned and enhanced dataset ready for analysis

## 🔧 How to Generate Cleaned Data

### Option 1: Run the Cleaning Script
```bash
python3 create_cleaned_data.py
```

This will:
1. Load the original `house_data.csv`
2. Apply cleaning filters
3. Create enhanced features
4. Save as `house_data_cleaned.csv`

### Option 2: Use the Model Script
```bash
python3 improved_model.py
```

This will automatically clean the data during model training.

## 📈 Data Cleaning Process

### Filters Applied:
1. **Price Range**: $10,000 - $2,000,000
2. **Square Footage**: 200 - 8,000 sqft
3. **Year Built**: 1900 - 2020
4. **Missing Values**: Removed rows with missing critical data

### Enhanced Features Created:
1. `House_Age` = 2024 - Year_Built
2. `Total_Bathrooms` = Fbath + Hbath
3. `Price_per_Sqft` = Sale_price / Fin_sqft
4. `Sqft_per_Bedroom` = Fin_sqft / Bdrms
5. `Lot_to_Sqft_Ratio` = Lotsize / Fin_sqft
6. `Bathrooms_per_Bedroom` = Total_Bathrooms / Bdrms
7. `Age_Squared` = House_Age²
8. `Sqft_Squared` = Fin_sqft²

## 📊 Data Summary

### Original Data:
- **Total houses**: 34,523
- **Data loss**: 4,033 houses (11.7%)
- **Final cleaned**: 28,104 houses

### Cleaned Data Statistics:
- **Price range**: $10,000 - $2,000,000
- **Square footage range**: 256 - 7,664 sqft
- **Year built range**: 1900 - 2018
- **Average price**: $154,593
- **Average square footage**: 1,450 sqft

## 🚀 Quick Start

1. **Clone the repository**
2. **Run cleaning script**: `python3 create_cleaned_data.py`
3. **Use cleaned data**: `data/house_data_cleaned.csv`

## 📝 Note

The `data/` directory is in `.gitignore` to avoid tracking large files. The cleaned data file is generated locally when needed. 