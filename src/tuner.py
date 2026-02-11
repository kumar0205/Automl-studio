from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, X, y, task):
    """
    Apply light GridSearch for better model performance.
    Only tunes if model has known grid.
    """
    param_grid = {}
    
    model_type = type(model).__name__
    
    if model_type == "RandomForestClassifier" or model_type == "RandomForestRegressor":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }
    elif model_type == "GradientBoostingClassifier" or model_type == "GradientBoostingRegressor":
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100]
        }
    
    if not param_grid:
        return model
        
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2' if task == 'regression' else 'accuracy')
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_
