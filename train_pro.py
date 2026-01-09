import pandas as pd
import joblib
import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

# --- SAFE IMPORT FOR ENCODER ---
# This fixes the "ImportError" if your Scikit-Learn is old
try:
    from sklearn.preprocessing import TargetEncoder
    print("[INFO] Using Advanced TargetEncoder")
    categorical_encoder = TargetEncoder(target_type='continuous')
except ImportError:
    print("[INFO] TargetEncoder not found. Falling back to OneHotEncoder (Safe Mode)")
    from sklearn.preprocessing import OneHotEncoder
    categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# --- CONFIGURATION ---
DATA_FILE = "dataset.csv"
MODEL_FILE = "lite_model.jb"
METRICS_FILE = "model_metrics.json"
TARGET = 'SalePrice'

def train_model():
    print("="*40)
    print("      TRAINING PRO MODEL (FIXED)      ")
    print("="*40)

    # 1. LOAD DATA
    try:
        if not os.path.exists(DATA_FILE): raise FileNotFoundError(f"'{DATA_FILE}' not found.")
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ DATA LOAD ERROR: {e}")
        sys.exit(1)

    # 2. FIX DIRTY DATA (The likely cause of your error)
    print("[1/4] Cleaning & Preparing Data...")
    
    # Force columns to be numbers (turns "NA" or text into NaN)
    df['TotalBsmtSF'] = pd.to_numeric(df['TotalBsmtSF'], errors='coerce')
    df['GrLivArea'] = pd.to_numeric(df['GrLivArea'], errors='coerce')
    
    # Fill missing values with 0 so the math doesn't break
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
    df['GrLivArea'] = df['GrLivArea'].fillna(0)
    
    # Now safe to add
    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']

    # Remove huge outliers that confuse the model
    df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)

    # 3. DEFINE FEATURES
    # We use these exact columns
    features = ['OverallQual', 'TotalSF', 'GarageArea', 'YearBuilt', 'FullBath', 'Neighborhood']
    
    # Check if all columns exist
    missing = [col for col in features if col not in df.columns]
    if missing:
        print(f"❌ CRITICAL ERROR: Dataset is missing columns: {missing}")
        sys.exit(1)

    X = df[features]
    y = df[TARGET]

    # 4. BUILD PIPELINE
    numeric_features = ['OverallQual', 'TotalSF', 'GarageArea', 'YearBuilt', 'FullBath']
    categorical_features = ['Neighborhood']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('cat', categorical_encoder, categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42, n_jobs=1)) # n_jobs=1 fixes some Windows crashes
    ])

    # 5. HYPERPARAMETER TUNING
    print("[2/4] Optimizing Model (Grid Search)...")
    
    # Simplified grid to run faster and with less errors
    param_grid = {
        'model__n_estimators': [500, 1000],
        'model__learning_rate': [0.01, 0.05],
        'model__max_depth': [3, 5]
    }
    
    try:
        search = RandomizedSearchCV(pipeline, param_grid, n_iter=5, cv=3, scoring='r2', verbose=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
    except Exception as e:
        print(f"❌ TRAINING ERROR: {e}")
        sys.exit(1)

    # 6. EVALUATE & SAVE
    print("[3/4] Validating Accuracy...")
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"   >> FINAL ACCURACY (R²): {r2:.4f}")

    print(f"[4/4] Saving System Files...")
    with open(METRICS_FILE, 'w') as f: json.dump({"r2_score": r2}, f)
    joblib.dump(best_model, MODEL_FILE)
    print("\n✅ SUCCESS! Model Fixed and Ready.")

if __name__ == "__main__":
    train_model()