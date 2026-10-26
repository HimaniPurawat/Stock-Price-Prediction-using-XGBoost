"""Rank a universe of stocks with one pooled XGBoost model (educational, not financial advice).

For every stock and date the model predicts the probability that the stock's
forward return over `horizon` trading days beats the median stock on that date.
Stocks are then ranked by that probability and the top K are "recommended".
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from xgboost import XGBClassifier

try:  # running as a script: python src/recommend.py
    from data_loader import download_stock_data
    from features import build_features
except ModuleNotFoundError:  # imported from a notebook
    from src.data_loader import download_stock_data
    from src.features import build_features


# NSE stocks for a stock recommendation project
# Yahoo Finance ticker format

UNIVERSE = [
    # Banking and financial services
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "HDFCLIFE.NS",
    "SBILIFE.NS",
    "SHRIRAMFIN.NS",

    # Information technology
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    
    "PERSISTENT.NS",

    # Energy and utilities
    "RELIANCE.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "TATAPOWER.NS",
    "ADANIGREEN.NS",

    # Automobiles
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "M&M.NS",
    "BAJAJ-AUTO.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",

    # FMCG and consumer goods
    "ITC.NS",
    "HINDUNILVR.NS",
    "NESTLEIND.NS",
    "BRITANNIA.NS",
    "TATACONSUM.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "TRENT.NS",

    # Healthcare and pharmaceuticals
    "SUNPHARMA.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "APOLLOHOSP.NS",
    "DIVISLAB.NS",

    # Infrastructure, cement and industrials
    "LT.NS",
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "ADANIPORTS.NS",
    "BEL.NS",
    "JSWSTEEL.NS",
    "TATASTEEL.NS",
]



def load_universe(path="data/universe.csv", tickers=UNIVERSE, refresh=False):
    """Long-format prices (index Date; columns close, volume, ticker). Downloads if needed."""
    path = Path(path)
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    frames = []
    for ticker in tickers:
        try:
            df = download_stock_data(ticker)
        except Exception as exc:
            print(f"Skipping {ticker}: {exc}")
            continue
        df = df[["close", "volume"]].copy()
        df["ticker"] = ticker
        frames.append(df)
    if not frames:
        raise ValueError("No data could be downloaded.")

    long = pd.concat(frames)
    long.index.name = "Date"
    path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(path)
    return long


def build_panel(long, horizon=5):
    """One row per (date, ticker): features plus the forward return over `horizon` days."""
    parts = []
    for ticker, g in long.groupby("ticker"):
        g = g.sort_index()
        feats = build_features(g)
        feats["fwd_return"] = g["close"].shift(-horizon) / g["close"] - 1
        feats["ticker"] = ticker
        parts.append(feats)
    panel = pd.concat(parts).sort_index()
    panel.index.name = "Date"
    return panel


def add_target(panel):
    """1 if the stock beat the median stock of that date, else 0 (rows without a future are dropped)."""
    labeled = panel.dropna(subset=["fwd_return"]).copy()
    median = labeled.groupby(level=0)["fwd_return"].transform("median")
    labeled["target"] = (labeled["fwd_return"] > median).astype(int)
    return labeled


def make_model():
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, eval_metric="logloss", random_state=42,
    )


def backtest(labeled, feature_cols, horizon=5, top_k=5, test_fraction=0.2):
    """Train on early dates, then every `horizon` days pick the top K and see how they did."""
    dates = np.sort(labeled.index.unique())
    cut = int(len(dates) * (1 - test_fraction))
    train = labeled[labeled.index.isin(dates[: cut - horizon])]  # gap avoids label overlap
    test = labeled[labeled.index.isin(dates[cut:])].copy()

    model = make_model().fit(train[feature_cols], train["target"])
    test["score"] = model.predict_proba(test[feature_cols])[:, 1]

    rows = []
    for d in dates[cut:][::horizon]:  # non-overlapping holding periods
        day = test.loc[[d]]
        if len(day) < 2 * top_k:
            continue
        top = day.nlargest(top_k, "score")
        mom = day.nlargest(top_k, "return_20d")  # simple momentum baseline
        rows.append({
            "date": d,
            "model": top["fwd_return"].mean(),
            "momentum": mom["fwd_return"].mean(),
            "all_stocks": day["fwd_return"].mean(),
            "precision_at_k": top["target"].mean(),
            "ndcg_at_k": ndcg_score(
                [day["fwd_return"].rank().values], [day["score"].values], k=top_k
            ),
        })
    return pd.DataFrame(rows).set_index("date")


def recommend_now(labeled, panel, feature_cols, top_k=5):
    """Fit on all labeled history and rank the stocks on the most recent trading day."""
    model = make_model().fit(labeled[feature_cols], labeled["target"])
    latest_date = panel.index.max()
    latest = panel[panel.index == latest_date].copy()
    latest["prob_beats_peers"] = model.predict_proba(latest[feature_cols])[:, 1]
    wanted = ["ticker", "prob_beats_peers", "return_20d", "rsi_14", "volatility_20d"]
    cols = [c for c in wanted if c in latest.columns]  # rsi_14 only exists with the RSI features
    return latest_date, latest.sort_values("prob_beats_peers", ascending=False)[cols].head(top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/universe.csv"))
    parser.add_argument("--refresh", action="store_true", help="re-download all prices")
    parser.add_argument("--horizon", type=int, default=5, help="holding period in trading days")
    parser.add_argument("--top", type=int, default=5, help="number of stocks to recommend")
    args = parser.parse_args()

    long = load_universe(args.data, refresh=args.refresh)
    panel = build_panel(long, args.horizon)
    labeled = add_target(panel)
    feature_cols = [c for c in panel.columns if c not in ("fwd_return", "ticker")]

    print(f"{long['ticker'].nunique()} stocks, {labeled.index.nunique()} dates, horizon {args.horizon}d")
    res = backtest(labeled, feature_cols, args.horizon, args.top)

    print(f"\nBacktest on the latest 20% of dates ({len(res)} non-overlapping {args.horizon}-day periods)")
    print(f"Avg {args.horizon}d return, model top-{args.top}   : {res['model'].mean():.3%}")
    print(f"Avg {args.horizon}d return, momentum top-{args.top}: {res['momentum'].mean():.3%}")
    print(f"Avg {args.horizon}d return, all stocks    : {res['all_stocks'].mean():.3%}")
    print(f"Periods model beat all-stocks average    : {(res['model'] > res['all_stocks']).mean():.1%}")
    print(f"Precision@{args.top} (random = 50%)           : {res['precision_at_k'].mean():.1%}")
    print(f"NDCG@{args.top}                              : {res['ndcg_at_k'].mean():.3f}")

    date, recs = recommend_now(labeled, panel, feature_cols, args.top)
    print(f"\nTop {args.top} by model score on {date.date()} (educational only, not advice):")
    print(recs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()