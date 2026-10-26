"""Feature engineering for historical stock prices."""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing is an exponential average with alpha = 1/window
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build price-only features from a date-indexed frame with a close column."""
    if "close" not in prices.columns:
        raise ValueError("Prices must contain a 'close' column.")

    close = prices["close"]
    features = pd.DataFrame(index=prices.index)

    # Returns and lagged returns (only past data, so there is no leakage)
    features["daily_return"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["return_20d"] = close.pct_change(20)
    for lag in (1, 2, 3):
        features[f"return_lag_{lag}"] = features["daily_return"].shift(lag)

    # Volatility
    features["volatility_5d"] = features["daily_return"].rolling(5).std()
    features["volatility_20d"] = features["daily_return"].rolling(20).std()

    # Trend: distance from moving averages
    features["sma_ratio_5d"] = close / close.rolling(5).mean() - 1
    features["sma_ratio_20d"] = close / close.rolling(20).mean() - 1

    # RSI
    features["rsi_14"] = rsi(close, 14)

    # MACD (divided by price so it doesn't depend on the price level)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    features["macd"] = macd / close
    features["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    # Bollinger %B: where the price sits inside its 20-day band (0 = lower, 1 = upper)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    features["bollinger_pctb"] = (close - (mid - 2 * std)) / (4 * std)

    if "volume" in prices.columns:
        features["volume_change"] = prices["volume"].pct_change()
        features["volume_ratio_20d"] = prices["volume"] / prices["volume"].rolling(20).mean()

    return features.replace([np.inf, -np.inf], np.nan).dropna()