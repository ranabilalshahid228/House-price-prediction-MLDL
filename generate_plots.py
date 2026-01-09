import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. SETUP
if not os.path.exists('report_images'):
    os.makedirs('report_images')

# Academic Style
plt.style.use('seaborn-v0_8-whitegrid')

# 2. LOAD RESOURCES
try:
    df = pd.read_csv("dataset.csv")
    # Clean data same as training to avoid errors
    df['TotalBsmtSF'] = pd.to_numeric(df['TotalBsmtSF'], errors='coerce').fillna(0)
    df['GrLivArea'] = pd.to_numeric(df['GrLivArea'], errors='coerce').fillna(0)
    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)
    
    # Load Model
    model = joblib.load("final_model.jb")
    print("✅ Data and Model Loaded.")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure dataset.csv and final_model.jb are in this folder.")
    exit()

# 3. PREPARE TEST DATA
features = ['OverallQual', 'TotalSF', 'GarageArea', 'YearBuilt', 'FullBath', 'Neighborhood']
target = 'SalePrice'

X = df[features]
y = df[target]

# Split (Same seed as training to see validation results)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate Predictions
y_pred = model.predict(X_test)

# --- PLOT 6: ACTUAL vs PREDICTED (The "Accuracy" Plot) ---
print("[1/3] Generating Actual vs Predicted...")
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='#2ca02c', edgecolor='k', s=40)
# Perfect Line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')

plt.title(f"Figure 6: Actual vs. Predicted Prices (R² = {r2_score(y_test, y_pred):.2f})", fontsize=14)
plt.xlabel("Actual Sale Price ($)", fontsize=12)
plt.ylabel("Predicted Sale Price ($)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('report_images/6_actual_vs_predicted.png', dpi=300)
plt.close()

# --- PLOT 7: RESIDUAL PLOT (The "Bias" Check) ---
print("[2/3] Generating Residual Plot...")
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='#1f77b4', edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2) # Zero line
plt.title("Figure 7: Residual Plot (Errors vs. Predictions)", fontsize=14)
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residual Error ($)")
plt.tight_layout()
plt.savefig('report_images/7_residual_plot.png', dpi=300)
plt.close()

# --- PLOT 8: ERROR DISTRIBUTION (The "Normality" Check) ---
print("[3/3] Generating Error Histogram...")
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='#d62728', bins=30)
plt.axvline(x=0, color='k', linestyle='--', lw=2)
plt.title("Figure 8: Distribution of Prediction Errors", fontsize=14)
plt.xlabel("Prediction Error ($)")
plt.tight_layout()
plt.savefig('report_images/8_error_distribution.png', dpi=300)
plt.close()

print("\n✅ DONE! Evaluation plots created in 'report_images'.")