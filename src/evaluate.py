"""Evaluate the saved XGBoost model on the held-out, most recent 20% of the data."""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:  # running as a script: python src/evaluate.py
    from data_loader import load_price_data
    from train import chronological_split, prepare_data
except ModuleNotFoundError:  # imported from a notebook
    from src.data_loader import load_price_data
    from src.train import chronological_split, prepare_data


def evaluate(data_path, model_path, plot_path=None):
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact["features"]

    prices = load_price_data(data_path, artifact.get("ticker", "AAPL"))
    X, y = prepare_data(prices)
    _, X_test, y_train, y_test = chronological_split(X, y)

    pred = model.predict(X_test[feature_names])

    # Baselines to compare against
    majority_class = int(y_train.mean() >= 0.5)
    majority_acc = accuracy_score(y_test, [majority_class] * len(y_test))

    results = {
        "test_rows": len(y_test),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "baseline_accuracy": majority_acc,
        "confusion_matrix": confusion_matrix(y_test, pred),
    }

    if plot_path:
        importances = sorted(
            zip(feature_names, model.feature_importances_), key=lambda t: t[1]
        )
        names, values = zip(*importances)
        plt.figure(figsize=(8, 4))
        plt.barh(names, values)
        plt.title("XGBoost feature importance")
        plt.tight_layout()
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prices.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/xgb_stock_direction.joblib"))
    parser.add_argument("--plot", type=Path, default=Path("models/feature_importance.png"))
    args = parser.parse_args()

    r = evaluate(args.data, args.model, args.plot)
    print(f"Test rows         : {r['test_rows']}")
    print(f"Accuracy          : {r['accuracy']:.4f}")
    print(f"Precision (Up)    : {r['precision']:.4f}")
    print(f"Recall (Up)       : {r['recall']:.4f}")
    print(f"F1 (Up)           : {r['f1']:.4f}")
    print(f"Baseline accuracy : {r['baseline_accuracy']:.4f}  (always predict the majority class)")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(r["confusion_matrix"])
    print(f"Saved feature importance plot to {args.plot}")


if __name__ == "__main__":
    main()