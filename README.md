# MacroAI Regime Engine

MacroAI is a Streamlit dashboard I built to explore a simple question: can macroeconomic regimes help explain how SPY, QQQ, and TLT behave over the next few weeks?

The app pulls macro data from FRED, ETF prices from Yahoo Finance, groups similar macro environments with K-means clustering, and then tests whether a Random Forest model can produce useful 20-day return signals.

This is a portfolio project, not a trading system. The goal is to make the data, assumptions, model behavior, and backtest results easy to inspect.

## Why I Built This

Markets do not behave the same way in every macro environment. High inflation, rising rates, widening credit spreads, or elevated volatility can all change which assets lead or lag.

I wanted to build a dashboard that connects those macro conditions to ETF returns in a transparent way. Instead of only showing a model prediction, the app also shows the regime history, model quality checks, backtest results, risk metrics, and robustness tests.

## Data Sources

- FRED macroeconomic data through `fredapi`
- ETF price data from Yahoo Finance through `yfinance`
- ETFs: SPY, QQQ, and TLT
- Macro inputs include Treasury yields, yield curves, Fed Funds, CPI, unemployment, VIX, and corporate bond spread data

## What The App Does

- Shows ETF price and return data
- Displays selected FRED macro indicators
- Builds macro features such as levels and trailing changes
- Groups macro environments using K-means clustering
- Calculates ETF forward returns by macro regime
- Trains Random Forest models for SPY, QQQ, and TLT
- Shows bullish probabilities and signal strength labels
- Compares model accuracy against a simple baseline
- Runs a simple 20-day strategy backtest
- Calculates risk metrics such as drawdown, Sharpe, Sortino, and Calmar
- Tests robustness across horizons, confidence thresholds, cluster counts, and time periods
- Generates a plain-English analyst summary without calling an external AI API

## Machine Learning Approach

The regime model uses K-means clustering. Before clustering, the macro features are standardized so that large-scale variables do not dominate the distance calculations.

The prediction model uses a separate `RandomForestClassifier` for each ETF. For each model, the target is:

```text
1 if the ETF's forward return is positive
0 otherwise
```

The model uses macro features and the current K-means cluster as inputs. Forward returns are used only as labels, never as features.

## Backtest Approach

The backtest uses a chronological split. The model trains on the earlier portion of the data and makes predictions on the later holdout period. This is meant to avoid training on information from the future.

The simple strategy works like this:

- On each test-period date, predict bullish probabilities for SPY, QQQ, and TLT.
- Choose the ETF with the highest bullish probability.
- Only take the trade if that probability is above the selected threshold.
- Otherwise, hold cash.
- Measure performance using completed forward returns.
- Compare the strategy against SPY, QQQ, TLT, and an equal-weight SPY/QQQ/TLT benchmark.

## Limitations

This project is intentionally simple and should be treated as research.

- The backtest does not include transaction costs, taxes, slippage, or liquidity constraints.
- Forward-return windows overlap, so the observations are not fully independent.
- FRED data can be revised over time.
- Random Forest probabilities are model estimates, not reliable forecasts.
- Regime labels are interpretive and should be reviewed manually.
- Good historical performance in this dashboard does not mean the strategy will work in the future.

## Running Locally

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Add a FRED API key using either `.env`:

```text
FRED_API_KEY=your_fred_api_key_here
```

Or Streamlit secrets at `.streamlit/secrets.toml`:

```toml
FRED_API_KEY = "your_fred_api_key_here"
```

Run the app:

```powershell
streamlit run app.py
```

## Disclaimer

MacroAI is for research and educational use only. It is not financial advice and is not a recommendation to buy or sell any security.
