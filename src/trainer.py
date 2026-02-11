from sklearn.model_selection import train_test_split
from src.model_registry import get_models
from src.tuner import tune_hyperparameters

def train_best_model(X, y, task, tune=True):
    """
    Split data, iterate through models, optionally tune, and select the best one.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = get_models(task)
    
    best_model = None
    best_score = -1e9 # Use a very small number for regression comparison
    best_name = None
    
    # We use R2 for regression and Accuracy for classification for simple ranking
    score_name = "accuracy" if task == "classification" else "r2"
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Initial score
        score = model.score(X_test, y_test)
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
            
    # Optional: Tune the winner
    if tune and best_model:
        best_model = tune_hyperparameters(best_model, X_train, y_train, task)
        # Re-evaluate after tuning
        best_score = best_model.score(X_test, y_test)
            
    return best_model, best_score, best_name, X_test, y_test
