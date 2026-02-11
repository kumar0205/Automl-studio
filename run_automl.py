import argparse
import os
from src.data_loader import load_data
from src.preprocess import build_preprocessor, save_preprocessor
from src.feature_engineer import detect_task, get_column_types
from src.trainer import train_best_model
from src.evaluator import evaluate_model, save_metrics
from src.utils import save_model

def main():
    parser = argparse.ArgumentParser(description="AutoML Studio CLI")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--task", choices=["classification","regression"], help="Explicitly specify task (optional)")
    parser.add_argument("--sample", type=int, default=10000, help="Max rows to read")
    args = parser.parse_args()

    # 1. Load Data
    print(f"📂 Loading dataset: {args.dataset}...")
    X, y, all_cols = load_data(args.dataset, sample_size=args.sample)
    
    # 2. Detect Task
    task = args.task if args.task else detect_task(y)
    print(f"🎯 Task detected: {task.upper()}")
    
    # 3. Preprocessing
    num_cols, low_card_cols, high_card_cols, dt_cols = get_column_types(X)
    print(f"🔍 Features: {len(num_cols)} numerical, {len(low_card_cols)} low-card categorical, {len(high_card_cols)} high-card categorical, {len(dt_cols)} datetime")
    
    preprocessor = build_preprocessor(num_cols, low_card_cols, high_card_cols, dt_cols)
    X_processed = preprocessor.fit_transform(X)
    save_preprocessor(preprocessor)
    print("✅ Preprocessing pipeline saved to outputs/artifacts/preprocessor.pkl")
    
    # 4. Train
    print("🚀 Training models... (this might take a moment with GridSearch)")
    model, score, model_name, X_test, y_test = train_best_model(X_processed, y, task)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, task)
    save_metrics(metrics, model_name)
    print(f"📊 Best model: {model_name} | Score: {score:.4f}")
    
    # 6. Save
    save_model(model, "outputs/models/best_model.pkl")
    print("💾 Model saved to outputs/models/best_model.pkl")
    
    # Output for UI consumption
    print(f"TASK_TYPE={task}")
    print(f"BEST_MODEL={model_name}")
    print(f"BEST_SCORE={score}")

if __name__ == "__main__":
    main()
