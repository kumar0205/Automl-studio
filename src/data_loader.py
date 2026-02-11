import pandas as pd
import os

def load_data(path, sample_size=10000):
    """
    Load data from a CSV file with optional sampling.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    # Read only first 'sample_size' rows for efficiency if dataset is large
    df = pd.read_csv(path, nrows=sample_size)
    
    # Assume last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    return X, y, df.columns.tolist()
