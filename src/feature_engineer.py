import pandas as pd
import numpy as np

def detect_task(y):
    """
    Automatically detect if the task is classification or regression.
    """
    unique_count = len(np.unique(y))
    is_numeric = np.issubdtype(y.dtype, np.number)
    
    if is_numeric and unique_count > 10:
        return "regression"
    else:
        return "classification"

def get_column_types(df):
    """
    Detect numerical, categorical, and datetime columns.
    Identifies high-cardinality categorical columns.
    """
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Try to detect datetime columns
    dt_cols = []
    for col in df.select_dtypes(include=['object', 'datetime']).columns:
        if df[col].dtype == 'datetime64[ns]':
            dt_cols.append(col)
        else:
            try:
                pd.to_datetime(df[col], errors='raise')
                dt_cols.append(col)
            except:
                pass
                
    remaining_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]
    cat_cols = df[remaining_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Identify high-cardinality columns (heuristic: > 20 unique values)
    high_card_cols = [col for col in cat_cols if df[col].nunique() > 20]
    low_card_cols = [col for col in cat_cols if col not in high_card_cols]
    
    return num_cols, low_card_cols, high_card_cols, dt_cols
