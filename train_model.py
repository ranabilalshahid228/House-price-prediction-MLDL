import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load Data
print("Loading dataset...")
try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("ERROR: dataset.csv not found. Please make sure it is in the same folder.")
    exit()

# 2. Define Features
features = ['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF',
           'FullBath', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces',
           'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', 'OpenPorchSF', 'LotArea',
           'CentralAir']

X = df[features]
y = df['SalePrice']

# 3. Build the Pipeline (Enterprise Grade)
numeric_features = [f for f in features if f != 'CentralAir']
categorical_features = ['CentralAir']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Define Stacking Model
estimators = [
    ('xgb', XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, objective='reg:squarederror')),
    ('gbr', GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6))
]

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', StackingRegressor(estimators=estimators, final_estimator=Ridge()))])

# 5. Train & Save
print("Training Stacking Model... (This usually takes 1-2 minutes)")
clf.fit(X, y)

joblib.dump(clf, 'best_model.jb')
print("✅ Success! Model saved as 'best_model.jb'")