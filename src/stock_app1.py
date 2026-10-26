"""Stock prediction project in ONE file: backend + Streamlit frontend.

Run from a folder that contains (or will receive) data/ and models/:
    pip install pandas numpy scikit-learn xgboost yfinance joblib streamlit
    streamlit run stock_app.py

Educational project, not financial advice.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, ndcg_score, precision_score, recall_score,
)
from xgboost import XGBClassifier

DATA = Path("data/prices.csv")
UNIVERSE_CSV = Path("data/nifty100.csv")
MODEL = Path("models/xgb_stock_direction.joblib")
TEST_FRACTION = 0.2

UNIVERSE = [  # fallback Nifty 100 list, used only if the NSE list cannot be fetched
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "AMBUJACEM",
    "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BAJAJHLDNG", "BANKBARODA", "BEL", "BHARTIARTL", "BOSCHLTD", "BPCL", "BRITANNIA", "CANBK",
    "CGPOWER", "CHOLAFIN", "CIPLA", "COALINDIA", "DABUR", "DIVISLAB", "DLF", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GAIL", "GODREJCP", "GRASIM", "HAL", "HAVELLS", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "HINDZINC", "HYUNDAI",
    "ICICIBANK", "ICICIGI", "ICICIPRULI", "INDIGO", "INFY", "IOC", "IRFC", "ITC",
    "JINDALSTEL", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LICI", "LODHA", "LT", "M&M", "MARICO",
    "MARUTI", "MAXHEALTH", "MAZDOCK", "MOTHERSON", "NAUKRI", "NESTLEIND", "NTPC", "ONGC",
    "PFC", "PIDILITIND", "PNB", "POLICYBZR", "POWERGRID", "RECLTD", "RELIANCE", "SBICARD",
    "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SUNPHARMA", "SWIGGY",
    "TATACONSUM", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TMPV", "TORNTPHARM",
    "TRENT", "TVSMOTOR", "ULTRACEMCO", "UNITDSPR", "VBL", "VEDL", "WIPRO", "ZYDUSLIFE",
]
UNIVERSE = [t + ".NS" for t in UNIVERSE]
NSE_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"


# =============================================================== BACKEND: data
def download_stock_data(ticker="AAPL"):
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values("Price")
    df.columns = df.columns.str.lower()
    return df.sort_index()


def load_price_data(path=DATA, ticker="AAPL"):
    """Load prices from CSV; download and save first if the file is missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        download_stock_data(ticker).to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df.columns = df.columns.str.lower()
    return df.sort_index()


def nifty100_tickers():
    """Official Nifty 100 symbols from NSE; falls back to the built-in list if offline."""
    try:
        df = pd.read_csv(NSE_LIST_URL, storage_options={"User-Agent": "Mozilla/5.0"})
        syms = [str(x).strip() + ".NS" for x in df["Symbol"]]
        if len(syms) >= 90:
            return syms
    except Exception as exc:
        print(f"Could not fetch NSE list ({exc}); using built-in list.")
    return UNIVERSE


def load_universe(path=UNIVERSE_CSV):
    """Long-format prices for the Nifty 100 (columns: close, volume, ticker)."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    frames = []
    for t in nifty100_tickers():
        try:
            df = download_stock_data(t)
            df = df[["close", "volume"]].copy()
        except Exception as exc:
            print(f"Skipping {t}: {exc}")
            continue
        df["ticker"] = t
        frames.append(df)
    if not frames:
        raise ValueError("No data could be downloaded.")
    long = pd.concat(frames)
    long.index.name = "Date"
    path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(path)
    return long


# ========================================================== BACKEND: features
def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def build_features(prices):
    """16 features from past prices and volume only (no look-ahead)."""
    close = prices["close"]
    f = pd.DataFrame(index=prices.index)

    f["daily_return"] = close.pct_change()
    f["return_5d"] = close.pct_change(5)
    f["return_20d"] = close.pct_change(20)
    for lag in (1, 2, 3):
        f[f"return_lag_{lag}"] = f["daily_return"].shift(lag)

    f["volatility_5d"] = f["daily_return"].rolling(5).std()
    f["volatility_20d"] = f["daily_return"].rolling(20).std()

    f["sma_ratio_5d"] = close / close.rolling(5).mean() - 1
    f["sma_ratio_20d"] = close / close.rolling(20).mean() - 1

    f["rsi_14"] = rsi(close, 14)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd"] = macd / close
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    mid, std = close.rolling(20).mean(), close.rolling(20).std()
    f["bollinger_pctb"] = (close - (mid - 2 * std)) / (4 * std)

    if "volume" in prices.columns:
        f["volume_change"] = prices["volume"].pct_change()
        f["volume_ratio_20d"] = prices["volume"] / prices["volume"].rolling(20).mean()

    return f.replace([np.inf, -np.inf], np.nan).dropna()


def build_target(prices, horizon=5):
    """1 if the close is higher after `horizon` rows, else 0 (NaN where no future)."""
    future = prices["close"].shift(-horizon)
    return (future > prices["close"]).astype("float").where(future.notna()).rename("target")


# ============================================== BACKEND: single-stock model
def prepare_data(prices):
    features = build_features(prices)
    dataset = features.join(build_target(prices)).dropna()
    return dataset[features.columns], dataset["target"].astype(int)


def chronological_split(X, y, test_fraction=TEST_FRACTION):
    split = int(len(X) * (1 - test_fraction))
    if split == 0 or split == len(X):
        raise ValueError("Not enough observations for a chronological split.")
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train(data_path=DATA, model_path=MODEL, ticker="AAPL"):
    prices = load_price_data(data_path, ticker)
    X, y = prepare_data(prices)
    if len(X) < 100:
        raise ValueError("At least 100 usable observations are required to train.")
    X_train, _, y_train, _ = chronological_split(X, y)

    pos = y_train.sum()
    neg = len(y_train) - pos
    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=neg / max(pos, 1),
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns), "ticker": ticker}, model_path)
    return model_path


# ================================================== BACKEND: Nifty 100 ranking
def build_panel(long, horizon=5):
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
    """1 if the stock beat that date's median stock, else 0."""
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
    dates = np.sort(labeled.index.unique())
    cut = int(len(dates) * (1 - test_fraction))
    train_df = labeled[labeled.index.isin(dates[: cut - horizon])]  # gap avoids label overlap
    test = labeled[labeled.index.isin(dates[cut:])].copy()

    model = make_model().fit(train_df[feature_cols], train_df["target"])
    test["score"] = model.predict_proba(test[feature_cols])[:, 1]

    rows = []
    for d in dates[cut:][::horizon]:  # non-overlapping holding periods
        day = test.loc[[d]]
        if len(day) < 2 * top_k:
            continue
        top = day.nlargest(top_k, "score")
        mom = day.nlargest(top_k, "return_20d")
        rows.append({
            "date": d,
            "model": top["fwd_return"].mean(),
            "momentum": mom["fwd_return"].mean(),
            "all_stocks": day["fwd_return"].mean(),
            "precision_at_k": top["target"].mean(),
            "ndcg_at_k": ndcg_score([day["fwd_return"].rank().values], [day["score"].values], k=top_k),
        })
    return pd.DataFrame(rows).set_index("date")


def recommend_now(labeled, panel, feature_cols, top_k=5):
    model = make_model().fit(labeled[feature_cols], labeled["target"])
    latest_date = panel.index.max()
    latest = panel[panel.index == latest_date].copy()
    latest["prob_beats_peers"] = model.predict_proba(latest[feature_cols])[:, 1]
    cols = ["ticker", "prob_beats_peers", "return_20d", "rsi_14", "volatility_20d"]
    return latest_date, latest.sort_values("prob_beats_peers", ascending=False)[cols].head(top_k)


# ================================================================== FRONTEND
st.set_page_config(page_title="Stock Direction and Ranking", page_icon="📈", layout="wide")
st.title("Stock price prediction with XGBoost")
st.caption("Educational project, not financial advice.")

tab_single, tab_rank, tab_about = st.tabs(["Single stock", "Rank Nifty 100 stocks", "About"])


@st.cache_data(show_spinner="Loading prices...")
def get_prices():
    return load_price_data(DATA)


@st.cache_resource(show_spinner="Training model...")
def get_model(retrain: bool):
    if retrain or not MODEL.exists():
        train(DATA, MODEL)
    return joblib.load(MODEL)


with tab_single:
    prices = get_prices()
    retrain = st.toggle("Retrain model from scratch", value=False)
    artifact = get_model(retrain)
    model, cols = artifact["model"], artifact["features"]
    ticker = artifact.get("ticker", "AAPL")

    X, y = prepare_data(prices)
    _, X_test, y_train, y_test = chronological_split(X, y)
    pred = model.predict(X_test[cols])
    base = accuracy_score(y_test, [int(y_train.mean() >= 0.5)] * len(y_test))
    acc = accuracy_score(y_test, pred)

    st.subheader(f"Will {ticker} close higher in 5 trading days?")
    latest = build_features(prices).iloc[[-1]]
    p_up = float(model.predict_proba(latest[cols])[0, 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest data date", str(latest.index[0].date()))
    c2.metric("Probability of up", f"{p_up:.1%}")
    c3.metric("Model says", "Up" if p_up >= 0.5 else "Down")

    st.markdown("**Test-set performance** (latest 20% of days, never seen in training)")
    m = st.columns(5)
    m[0].metric("Accuracy", f"{acc:.1%}", f"{acc - base:+.1%} vs always-up")
    m[1].metric("Precision (up)", f"{precision_score(y_test, pred, zero_division=0):.1%}")
    m[2].metric("Recall (up)", f"{recall_score(y_test, pred, zero_division=0):.1%}")
    m[3].metric("F1 (up)", f"{f1_score(y_test, pred, zero_division=0):.1%}")
    m[4].metric("Test days", len(y_test))

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = pd.DataFrame(
            confusion_matrix(y_test, pred),
            index=["Actually down", "Actually up"],
            columns=["Predicted down", "Predicted up"],
        )
        st.dataframe(cm, use_container_width=True)
        st.markdown("**Closing price**")
        st.line_chart(prices["close"])
    with right:
        st.markdown("**Feature importance**")
        imp = pd.Series(model.feature_importances_, index=cols).sort_values()
        st.bar_chart(imp, horizontal=True)

    with st.expander("Predicted vs actual on the test period"):
        out = pd.DataFrame({"actual": y_test, "predicted": pred}, index=y_test.index)
        out["correct"] = out["actual"] == out["predicted"]
        st.dataframe(out.tail(60), use_container_width=True)


@st.cache_data(show_spinner="Building features for all stocks...")
def get_panel(horizon: int):
    long = load_universe(UNIVERSE_CSV)
    panel = build_panel(long, horizon)
    labeled = add_target(panel)
    feats = [c for c in panel.columns if c not in ("fwd_return", "ticker")]
    return long["ticker"].nunique(), panel, labeled, feats


@st.cache_data(show_spinner="Running backtest...")
def get_backtest(horizon: int, top_k: int):
    _, _, labeled, feats = get_panel(horizon)
    return backtest(labeled, feats, horizon, top_k)


@st.cache_data(show_spinner="Ranking stocks...")
def get_recs(horizon: int, top_k: int):
    _, panel, labeled, feats = get_panel(horizon)
    return recommend_now(labeled, panel, feats, top_k)


with tab_rank:
    if st.button("Re-download Nifty 100 data and retrain"):
        UNIVERSE_CSV.unlink(missing_ok=True)
        st.cache_data.clear()
        st.rerun()
    st.caption("First run downloads about 100 stocks from Yahoo Finance (a few minutes), then trains the ranker.")
    s1, s2 = st.columns(2)
    horizon = s1.slider("Holding period (trading days)", 1, 20, 5)
    top_k = s2.slider("Stocks to pick", 3, 10, 5)

    n_stocks = get_panel(horizon)[0]
    res = get_backtest(horizon, top_k)
    date, recs = get_recs(horizon, top_k)

    st.subheader(f"Top {top_k} picks on {date.date()}")
    st.dataframe(
        recs.rename(columns={
            "ticker": "Stock", "prob_beats_peers": "Prob. beats peers",
            "return_20d": "20-day return", "rsi_14": "RSI", "volatility_20d": "20-day volatility",
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader(f"Backtest: {len(res)} non-overlapping {horizon}-day periods, {n_stocks} Nifty 100 stocks loaded")
    k = st.columns(5)
    k[0].metric("Model top picks", f"{res['model'].mean():.3%}", help="Average return per period")
    k[1].metric("Momentum top picks", f"{res['momentum'].mean():.3%}")
    k[2].metric("All stocks", f"{res['all_stocks'].mean():.3%}")
    k[3].metric("Beat all-stock avg", f"{(res['model'] > res['all_stocks']).mean():.0%}")
    k[4].metric("Precision / NDCG", f"{res['precision_at_k'].mean():.0%} / {res['ndcg_at_k'].mean():.2f}")

    st.markdown("**Growth of 1 unit, compounding each period**")
    st.line_chart((1 + res[["model", "momentum", "all_stocks"]]).cumprod())

    with st.expander("Period-by-period results"):
        st.dataframe(res, use_container_width=True)


with tab_about:
    st.markdown(
        """
**What this does**

- *Single stock:* an XGBoost classifier predicts whether the close is higher 5 trading days from now,
  using 16 features built only from past prices and volume.
- *Ranking:* one pooled model, trained on the Nifty 100 stocks, scores each stock on whether each will beat the median stock,
  and picks the top K and compares them with a 20-day momentum baseline.

**How the test is kept honest**

- Chronological 80/20 split, rows never shuffled.
- Every feature at a date uses only that date and earlier.
- The ranking model leaves a gap between train and test so labels do not overlap.

**Limits**

- Short test windows, one train/test split, no trading costs.
- Overall accuracy can sit below the always-up baseline, so read precision with care.
        """
    )
