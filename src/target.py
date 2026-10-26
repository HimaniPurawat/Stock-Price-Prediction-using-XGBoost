"""Target construction for directional stock-price prediction."""

import pandas as pd


def build_target(prices: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Return 1 when close is higher after ``horizon`` rows, otherwise 0."""
    if "close" not in prices.columns:
        raise ValueError("Prices must contain a 'close' column.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    future_close = prices["close"].shift(-horizon)
    target = (future_close > prices["close"]).astype("float")
    return target.where(future_close.notna()).rename("target")
