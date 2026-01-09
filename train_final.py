import pandas as pd
import joblib
import os
import sys
import json
import shutil
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
DATA_FILE = "dataset.csv"
MODEL_FILE = "final_model.jb"      # New Name to avoid conflicts
METRICS_FILE = "final_metrics.json"

def train_model():
    print("="*40)
    print("      TRAINING FINAL MODEL (v4.0)      ")
    print("="*40)

    # 1. CLEANUP OLD FILES
    if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
    
    # 2. LOAD DATA
    try:
        if not os.path.exists(DATA_FILE): raise FileNotFoundError(f"'{DATA_FILE}' missing.")
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ DATA ERROR: {e}")
        sys.exit(1)

    print("[1/4] Preparing Data...")
    
    # FIX: Convert text 'NA' to Numbers and fill with 0
    df['TotalBsmtSF'] = pd.to_numeric(df['TotalBsmtSF'], errors='coerce').fillna(0)
    df['GrLivArea'] = pd.to_numeric(df['GrLivArea'], errors='coerce').fillna(0)
    
    # FEATURE ENGINEERING: Total Square Footage
    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    
    # REMOVE OUTLIERS (Improves Accuracy)
    df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)

    # DEFINING THE EXACT FEATURES
    # We use OneHotEncoder for Neighborhood because it is the most stable method
    features = ['OverallQual', 'TotalSF', 'GarageArea', 'YearBuilt', 'FullBath', 'Neighborhood']
    target = 'SalePrice'

    X = df[features]
    y = df[target]

    # 3. BUILD ROBUST PIPELINE
    numeric_features = ['OverallQual', 'TotalSF', 'GarageArea', 'YearBuilt', 'FullBath']
    categorical_features = ['Neighborhood']

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        # SAFE MODE: handle_unknown='ignore' prevents crashes if a new location is selected
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    # Model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=1, random_state=42))
    ])

    # 4. TRAIN
    print("[2/4] Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # 5. EVALUATE
    print("[3/4] Verifying Accuracy...")
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"   >> ACCURACY (R²): {r2:.4f}")

    # 6. SAVE
    print(f"[4/4] Saving to '{MODEL_FILE}'...")
    with open(METRICS_FILE, 'w') as f: json.dump({"r2_score": r2}, f)
    joblib.dump(pipeline, MODEL_FILE)
    print("\n✅ SUCCESS! Ready to run App.")

if __name__ == "__main__":
    train_model()