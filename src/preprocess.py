from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import category_encoders as ce
import pandas as pd
import numpy as np
import pickle
import os

class DateTimeTransformer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        X_out = pd.DataFrame()
        for col in X.columns:
            dt = pd.to_datetime(X[col])
            X_out[f"{col}_year"] = dt.dt.year
            X_out[f"{col}_month"] = dt.dt.month
            X_out[f"{col}_day"] = dt.dt.day
            X_out[f"{col}_hour"] = dt.dt.hour
            X_out[f"{col}_dayofweek"] = dt.dt.dayofweek
        return X_out

def build_preprocessor(num_cols, low_card_cols, high_card_cols, dt_cols):
    """
    Create a scikit-learn preprocessing pipeline for numerical, categorical, and datetime data.
    """
    transformers = []
    
    if num_cols:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_transformer, num_cols))
    
    if low_card_cols:
        low_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat_low', low_card_transformer, low_card_cols))
        
    if high_card_cols:
        high_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target', ce.TargetEncoder()) # Requires target y at fit time if used directly, 
                                          # but for simplicity we'll use BinaryEncoder or Ordinal
        ])
        # Using BinaryEncoder as it doesn't need y for fit and handles high card better than OHE
        high_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('binary', ce.BinaryEncoder())
        ])
        transformers.append(('cat_high', high_card_transformer, high_card_cols))
        
    if dt_cols:
        dt_transformer = Pipeline(steps=[
            ('datetime', DateTimeTransformer())
        ])
        transformers.append(('dt', dt_transformer, dt_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor

def save_preprocessor(preprocessor, path="outputs/artifacts/preprocessor.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    return path
