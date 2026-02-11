from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor

def get_models(task):
    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
    else:
        return {
            "Ridge Regression": Ridge(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
        }
