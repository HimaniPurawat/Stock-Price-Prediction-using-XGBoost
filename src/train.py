"""Train an XGBoost classifier that predicts next-day stock direction."""

import argparse
from pathlib import Path

import joblib
from xgboost import XGBClassifier

try:  # running as a script: python src/train.py
    from data_loader import load_price_data
    from features import build_features
    from target import build_target
except ModuleNotFoundError:  # imported from a notebook: from src.train import ...
    from src.data_loader import load_price_data
    from src.features import build_features
    from src.target import build_target

TEST_FRACTION = 0.2


def prepare_data(prices):
    """Return the feature matrix X and the 0/1 target y, aligned and NaN-free."""
    features = build_features(prices)
    dataset = features.join(build_target(prices)).dropna()
    X = dataset[features.columns]
    y = dataset["target"].astype(int)
    return X, y


def chronological_split(X, y, test_fraction=TEST_FRACTION):
    """Earlier rows train, later rows test. Never shuffled, so no future leakage."""
    split = int(len(X) * (1 - test_fraction))
    if split == 0 or split == len(X):
        raise ValueError("Not enough observations for a chronological split.")
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train(data_path, model_path, ticker="AAPL"):
    prices = load_price_data(data_path, ticker)
    X, y = prepare_data(prices)
    if len(X) < 100:
        raise ValueError("At least 100 usable observations are required to train.")

    X_train, X_test, y_train, y_test = chronological_split(X, y)

    # Balance the classes (Up days are usually a bit more common than Down days).
    pos = y_train.sum()
    neg = len(y_train) - pos
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(X.columns), "ticker": ticker},
        output_path,
    )
    print(f"Train rows: {len(X_train)} ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"Test rows : {len(X_test)} ({X_test.index[0].date()} to {X_test.index[-1].date()})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prices.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/xgb_stock_direction.joblib"))
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()
    saved = train(args.data, args.model, args.ticker)
    print(f"Saved model to {saved}")


if __name__ == "__main__":
    main()