import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Setup - Create a folder for images
if not os.path.exists('report_plots'):
    os.makedirs('report_plots')
    print("Created folder: 'report_plots' for your images.")

# 2. Load Data & Model
print("Loading data...")
df = pd.read_csv('dataset.csv')

try:
    # Try loading the best model first, then the standard one
    model = joblib.load('best_model.jb')
    print("Loaded 'best_model.jb'")
except:
    try:
        model = joblib.load('xgb_model.jb')
        print("Loaded 'xgb_model.jb'")
    except:
        print("⚠️ Model not found. Please run train_model.py first!")
        exit()

# 3. Prepare Data (Match the training columns)
# Ensure we use the exact features the model expects
features = ['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF',
           'FullBath', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces',
           'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', 'OpenPorchSF', 'LotArea',
           'CentralAir']

# Handle Preprocessing for Analysis
X = df[features].copy()
y = df['SalePrice']

# Basic cleaning for the analysis script to run smoothly
X['LotFrontage'] = X['LotFrontage'].fillna(X['LotFrontage'].mean())
X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
# Handle CentralAir (Map Yes/No to 1/0 if needed, or keep for Pipeline)
# Note: If using the 'best_model.jb' pipeline, it handles encoding automatically.
# If using 'xgb_model.jb', we might need to map manually. 
if X['CentralAir'].dtype == 'object' and not hasattr(model, 'named_steps'):
     X['CentralAir'] = X['CentralAir'].map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0})

# Split Data to calculate Test Metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# PART A: VISUALS
# ==========================================

# A1. Missing Values Plot
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Map (Yellow = Missing)')
plt.savefig('report_plots/1_missing_values.png')
plt.close()
print("✅ Generated Missing Values Plot")

# A2. Correlation Heatmap (Top 10 Features)
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
# Filter for top correlated features to SalePrice
cols = corr.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
            yticklabels=cols.values, xticklabels=cols.values, cmap='coolwarm')
plt.title('Correlation Heatmap (Top 10 Features)')
plt.savefig('report_plots/2_heatmap.png')
plt.close()
print("✅ Generated Heatmap")

# A3. Feature Skewness (SalePrice)
plt.figure(figsize=(8, 5))
sns.histplot(df['SalePrice'], kde=True, color='purple')
plt.title(f"Target Distribution (Skewness: {df['SalePrice'].skew():.2f})")
plt.savefig('report_plots/3_skewness.png')
plt.close()
print("✅ Generated Skewness Plot")

# A4. Trend Plot (Price vs Year Built)
plt.figure(figsize=(10, 5))
trend_data = df.groupby('YearBuilt')['SalePrice'].mean()
trend_data.plot(color='green')
plt.title('Trend: Average House Price by Year Built')
plt.ylabel('Average Price ($)')
plt.grid(True, alpha=0.3)
plt.savefig('report_plots/4_trend_plot.png')
plt.close()
print("✅ Generated Trend Plot")

# ==========================================
# PART B: MODEL METRICS & COMPARISON
# ==========================================

# Generate Predictions
y_pred = model.predict(X_test)

# B1. Metrics Calculation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"MSE (Mean Squared Error): {mse:,.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")
print("F1 Score: N/A (F1 is for Classification/Categories, this is Regression/Numbers)")
print("-" * 30)

# Save Metrics to text file
with open('report_plots/metrics.txt', 'w') as f:
    f.write(f"MSE: {mse:,.2f}\nRMSE: {rmse:,.2f}\nR2 Score: {r2:.4f}\n")

# B2. Comparison Plot (Actual vs Predicted)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Diagonal line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Comparison: Actual vs Predicted Prices')
plt.savefig('report_plots/5_comparison_plot.png')
plt.close()
print("✅ Generated Comparison Plot")

print("\nDone! Check the 'report_plots' folder for your images.")