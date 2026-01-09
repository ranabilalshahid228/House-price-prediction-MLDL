import pandas as pd
import joblib
import os
import sys
import json
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
DATA_FILE = "dataset.csv"
MODEL_FILE = "lite_model.jb"
METRICS_FILE = "model_metrics.json"

# We added 'Neighborhood' to the features list
FEATURES = ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'FullBath', 'Neighborhood']
TARGET = 'SalePrice'

def train_model():
    print("="*40)
    print("      TRAINING IMPROVED MODEL (v2.0)      ")
    print("="*40)

    # 1. LOAD DATA
    try:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"'{DATA_FILE}' not found.")
        
        df = pd.read_csv(DATA_FILE)
        
        # Verify columns
        missing = [c for c in FEATURES if c not in df.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")

        X = df[FEATURES]
        y = df[TARGET]

    except Exception as e:
        print(f"❌ DATA ERROR: {e}")
        sys.exit(1)

    # 2. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. BUILD SMART PIPELINE
    # We treat 'Neighborhood' differently than numbers
    numeric_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'FullBath']
    categorical_features = ['Neighborhood']

    # Preprocessor: Handles Numbers and Text separately
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            
            # TargetEncoder is better than OneHot for High Cardinality (Many Neighborhoods)
            ('cat', TargetEncoder(target_type='continuous'), categorical_features)
        ])

    # Model: Tuned XGBoost for higher accuracy
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        ))
    ])

    # 4. TRAIN
    print("[1/3] Training XGBoost with Location Data...")
    model.fit(X_train, y_train)

    # 5. EVALUATE
    print("[2/3] Validating Accuracy...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"   >> New R² Score: {r2:.4f} (Should be > 0.86)")

    # 6. SAVE
    print(f"[3/3] Saving System Files...")
    with open(METRICS_FILE, 'w') as f:
        json.dump({"r2_score": r2}, f)
    joblib.dump(model, MODEL_FILE)
    print("\n✅ DONE! Location feature is now active.")

if __name__ == "__main__":
    train_model()