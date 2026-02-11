import json
import os
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

def evaluate_model(y_true, y_pred, task):
    """
    Calculate metrics based on task type.
    """
    metrics = {}
    if task == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_score"] = float(f1_score(y_true, y_pred, average='weighted'))
    else:
        metrics["r2_score"] = float(r2_score(y_true, y_pred))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    
    return metrics

def save_metrics(metrics, model_name, path="outputs/artifacts/metrics.json"):
    """
    Save metrics and model metadata to JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "model_name": model_name,
        "metrics": metrics
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=4)
    return path
