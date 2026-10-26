from pathlib import Path

import pandas as pd
import yfinance as yf


def download_stock_data(ticker="AAPL"):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values("Price")

    df.columns = df.columns.str.lower()

    return df.sort_index()


def load_price_data(path="data/prices.csv", ticker="AAPL"):
    """Load prices from a CSV. If the file does not exist, download and save it first."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        download_stock_data(ticker).to_csv(path)

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df.columns = df.columns.str.lower()
    return df.sort_index()


if __name__ == "__main__":
    data = load_price_data("data/prices.csv", "AAPL")
    print(data.head())