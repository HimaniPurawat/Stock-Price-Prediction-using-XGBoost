# Stock Prediction

A small starter project for exploring historical stock prices and training a
baseline model to predict whether the closing price will rise on the next
trading day. This is an educational example, not financial advice.

## Project layout

- `data/` — input CSV files; add your own `prices.csv` here.
- `notebooks/exploration.ipynb` — a short exploratory analysis.
- `src/` — data loading, feature engineering, target construction, training,
  and evaluation.
- `models/` — saved model artifacts.

## Input data

Provide a CSV at `data/prices.csv` with a `Date` column and either a `Close` or
`Adj Close` column. An optional `Volume` column is used to build an additional
feature. Common OHLC columns are accepted but are not required.

## Setup

From this directory, create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python src/train.py
python src/evaluate.py
```

Both commands accept `--data` and `--model` options to use different input and
output paths. The split is chronological: the first 80% of usable observations
are used for training and the last 20% for evaluation. Features use only
observations available at each prediction date. Results on historical data do
not guarantee future performance.

## Explore

Launch Jupyter from this directory and open `notebooks/exploration.ipynb`:

```bash
jupyter notebook
```
