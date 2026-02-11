import pytest
import pandas as pd
import numpy as np
from src.preprocess import build_preprocessor, DateTimeTransformer
from src.feature_engineer import get_column_types

def test_feature_detection():
    data = {
        'num': [1, 2, 3],
        'low_cat': ['a', 'b', 'a'],
        'high_cat': [f'cat_{i}' for i in range(3)], # 3 unique is low card here
        'date': ['2023-01-01', '2023-01-02', '2023-01-03']
    }
    df = pd.DataFrame(data)
    num, low, high, dt = get_column_types(df)
    
    assert 'num' in num
    assert 'low_cat' in low
    assert 'date' in dt
    assert len(high) == 0

def test_datetime_transformer():
    df = pd.DataFrame({'date': ['2023-01-01 12:00:00']})
    transformer = DateTimeTransformer()
    transformed = transformer.transform(df)
    
    assert transformed['date_year'][0] == 2023
    assert transformed['date_month'][0] == 1
    assert transformed['date_day'][0] == 1
    assert transformed['date_hour'][0] == 12

def test_preprocessor_pipeline():
    data = {
        'num': [1, 2, 3],
        'cat': ['a', 'b', 'a'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03']
    }
    df = pd.DataFrame(data)
    num, low, high, dt = get_column_types(df)
    preprocessor = build_preprocessor(num, low, high, dt)
    
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 3
    # num(1) + cat_low(2 for OHE) + dt(5) = 8 features
    assert X_transformed.shape[1] == 8
