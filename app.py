import os
from dotenv import load_dotenv
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from fredapi import Fred
load_dotenv()
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MacroAI Regime Engine", layout="wide")

st.title("MacroAI Regime Engine")

# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Settings")

def get_fred_api_key():
    # Prefer environment variables for local development; Streamlit secrets support cloud deployment.
    api_key = os.getenv("FRED_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets.get("FRED_API_KEY")
    except Exception:
        return None


fred_api_key = get_fred_api_key()

if not fred_api_key:
    fred_api_key = st.sidebar.text_input("Enter your FRED API key", type="password")

tickers = ["SPY", "QQQ", "TLT"]
selected_ticker = st.sidebar.selectbox("Choose an ETF", tickers)

period = st.sidebar.selectbox(
    "Choose ETF time period",
    ["1y", "2y", "5y", "10y", "max"],
    index=2
)

macro_options = {
    "2Y Treasury Yield": "DGS2",
    "10Y Treasury Yield": "DGS10",
    "20Y Treasury Yield": "DGS20",
    "10Y-2Y Yield Curve": "T10Y2Y",
    "20Y-2Y Yield Curve": "CUSTOM_20Y_2Y",
    "20Y-10Y Yield Curve": "CUSTOM_20Y_10Y",
    "Fed Funds Rate": "FEDFUNDS",
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "Unemployment Rate": "UNRATE",
    "VIX": "VIXCLS",
    "BAA Corporate Bond Spread": "BAA10Y"
}

selected_macro_name = st.sidebar.selectbox(
    "Choose macro indicator",
    list(macro_options.keys())
)

selected_macro_code = macro_options[selected_macro_name]

n_clusters = st.sidebar.slider(
    "Number of K-means clusters",
    min_value=2,
    max_value=10,
    value=5
)

# -----------------------------
# Load ETF data
# -----------------------------
@st.cache_data
def load_price_data(ticker, period):
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])

    data["Daily Return"] = data["Close"].pct_change()
    data["20D Forward Return"] = data["Close"].shift(-20) / data["Close"] - 1

    return data

# -----------------------------
# Load FRED data
# -----------------------------
@st.cache_data
def load_fred_data(api_key, series_code):
    fred = Fred(api_key=api_key)

    # Custom yield curves
    if series_code == "CUSTOM_20Y_2Y":
        dgs20 = fred.get_series("DGS20")
        dgs2 = fred.get_series("DGS2")
        series = dgs20 - dgs2
        value_name = "20Y - 2Y Spread"

    elif series_code == "CUSTOM_20Y_10Y":
        dgs20 = fred.get_series("DGS20")
        dgs10 = fred.get_series("DGS10")
        series = dgs20 - dgs10
        value_name = "20Y - 10Y Spread"

    else:
        series = fred.get_series(series_code)
        value_name = "Value"

    data = pd.DataFrame(series, columns=[value_name])
    data = data.reset_index()
    data.columns = ["Date", "Value"]
    data["Date"] = pd.to_datetime(data["Date"])

    data["1M Change"] = data["Value"].diff(21)
    data["3M Change"] = data["Value"].diff(63)

    return data

price_data = load_price_data(selected_ticker, period)

overview_tab, market_tab, macro_tab, kmeans_tab, returns_tab, rf_tab, backtest_tab, robustness_tab, ai_summary_tab = st.tabs(
    [
        "Overview",
        "Market Data",
        "Macro Indicators",
        "K-Means Regime Classifier",
        "ETF Returns by Regime",
        "Random Forest Prediction",
        "Backtest",
        "Robustness Testing",
        "AI Analyst Summary"
    ]
)

# -----------------------------
# ETF price dashboard
# -----------------------------
with market_tab:
    st.header("ETF Price Dashboard")
    st.write(
        "This section pulls ETF price history from yfinance and calculates daily returns plus the next "
        "20-trading-day forward return used later in the regime analysis."
    )

    price_metric_col, daily_metric_col = st.columns(2)
    latest_price_row = price_data.dropna(subset=["Close"]).iloc[-1]
    price_metric_col.metric(f"{selected_ticker} Latest Close", f"${latest_price_row['Close']:.2f}")
    daily_metric_col.metric(
        "Latest Daily Return",
        f"{latest_price_row['Daily Return']:.2%}" if pd.notna(latest_price_row["Daily Return"]) else "N/A"
    )

    fig_price = px.line(
        price_data,
        x="Date",
        y="Close",
        title=f"{selected_ticker} Adjusted Close Price"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    with st.expander(f"Recent {selected_ticker} Return Data", expanded=False):
        st.dataframe(
            price_data[["Date", "Close", "Daily Return", "20D Forward Return"]]
            .tail(30)
            .style.format(
                {
                    "Close": "${:.2f}",
                    "Daily Return": "{:.2%}",
                    "20D Forward Return": "{:.2%}"
                },
                na_rep="N/A"
            ),
            use_container_width=True
        )

# -----------------------------
# Macro section
# -----------------------------
@st.cache_data
def load_multiple_fred_series(api_key):
    fred = Fred(api_key=api_key)

    series_codes = {
        "Fed Funds": "FEDFUNDS",
        "2Y Yield": "DGS2",
        "10Y Yield": "DGS10",
        "20Y Yield": "DGS20",
        "10Y-2Y Curve": "T10Y2Y",
        "CPI": "CPIAUCSL",
        "Core CPI": "CPILFESL",
        "Unemployment": "UNRATE",
        "VIX": "VIXCLS",
        "BAA Spread": "BAA10Y"
    }

    macro_df = pd.DataFrame()

    for name, code in series_codes.items():
        series = fred.get_series(code)
        macro_df[name] = series

    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.sort_index()

    # Custom yield curves
    macro_df["20Y-10Y Curve"] = macro_df["20Y Yield"] - macro_df["10Y Yield"]
    macro_df["20Y-2Y Curve"] = macro_df["20Y Yield"] - macro_df["2Y Yield"]

    # Forward fill missing values
    macro_df = macro_df.ffill()

    # Feature engineering
    feature_df = pd.DataFrame(index=macro_df.index)

    for col in macro_df.columns:
        feature_df[f"{col} 1M Change"] = macro_df[col].diff(21)
        feature_df[f"{col} 3M Change"] = macro_df[col].diff(63)

    feature_df["10Y Yield Level"] = macro_df["10Y Yield"]
    feature_df["20Y Yield Level"] = macro_df["20Y Yield"]
    feature_df["VIX Level"] = macro_df["VIX"]
    feature_df["Unemployment Level"] = macro_df["Unemployment"]
    feature_df["10Y-2Y Curve Level"] = macro_df["10Y-2Y Curve"]

    feature_df = feature_df.dropna()

    return macro_df, feature_df


@st.cache_data
def load_multi_asset_prices(tickers, period="max"):
    price_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        price_df[ticker] = data["Close"]

    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    # Calculate 20-day forward returns
    forward_returns = price_df.shift(-20) / price_df - 1

    forward_returns.columns = [
        f"{ticker} 20D Forward Return" for ticker in tickers
    ]

    return price_df, forward_returns


def calculate_forward_returns(price_df, tickers, horizon):
    forward_returns = price_df.shift(-horizon) / price_df - 1
    forward_returns.columns = [
        f"{ticker} {horizon}D Forward Return" for ticker in tickers
    ]
    return forward_returns


def force_datetime_ns(data, date_column="Date"):
    data = data.copy()
    data[date_column] = (
        pd.to_datetime(data[date_column], errors="coerce")
        .dt.floor("D")
        .astype("datetime64[ns]")
    )
    return data


def run_kmeans_regime_model(feature_df, n_clusters):
    # Scale macro features first so large-number features do not dominate K-means.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    elbow_rows = []
    max_k = min(10, len(feature_df) - 1)
    for k in range(2, max_k + 1):
        elbow_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        elbow_model.fit(scaled_features)
        elbow_rows.append({"k": k, "Inertia": elbow_model.inertia_})

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    silhouette = None
    if 1 < n_clusters < len(feature_df) and len(set(clusters)) > 1:
        silhouette = silhouette_score(scaled_features, clusters)

    clustered_features = feature_df.copy()
    clustered_features["Cluster"] = clusters

    pca = PCA(n_components=2, random_state=42)
    pca_values = pca.fit_transform(scaled_features)
    pca_data = pd.DataFrame(
        {
            "Date": clustered_features.index,
            "PC1": pca_values[:, 0],
            "PC2": pca_values[:, 1],
            "Cluster": clusters.astype(str)
        }
    )

    return clustered_features, scaled_features, pd.DataFrame(elbow_rows), silhouette, pca_data


def prepare_regime_return_data(clustered_features, forward_returns):
    cluster_data = clustered_features[["Cluster"]].reset_index()
    cluster_data = cluster_data.rename(columns={cluster_data.columns[0]: "Date"})

    returns_data = forward_returns.reset_index()
    returns_data = returns_data.rename(columns={returns_data.columns[0]: "Date"})

    # Force both Date columns to the exact same dtype before merge_asof.
    returns_data = force_datetime_ns(returns_data, "Date")
    cluster_data = force_datetime_ns(cluster_data, "Date")

    returns_data = returns_data.dropna(subset=["Date"]).sort_values("Date")
    cluster_data = cluster_data.dropna(subset=["Date"]).sort_values("Date")

    regime_return_data = pd.merge_asof(
        returns_data,
        cluster_data,
        on="Date",
        direction="backward"
    )

    return returns_data, cluster_data, regime_return_data.dropna()


def calculate_return_tables(regime_return_data, return_columns):
    grouped = regime_return_data.groupby("Cluster")

    average_returns = grouped[return_columns].mean()
    median_returns = grouped[return_columns].median()
    win_rates = grouped[return_columns].agg(lambda values: (values > 0).mean())
    observation_counts = grouped.size().rename("Observations").to_frame()

    return average_returns, median_returns, win_rates, observation_counts


def make_current_cluster_asset_table(current_cluster, average_returns, median_returns, win_rates, observation_counts):
    rows = []

    if current_cluster not in average_returns.index:
        return pd.DataFrame()

    for return_column in average_returns.columns:
        asset = return_column.replace(" 20D Forward Return", "")
        rows.append(
            {
                "Asset": asset,
                "Average 20D Forward Return": average_returns.loc[current_cluster, return_column],
                "Median 20D Forward Return": median_returns.loc[current_cluster, return_column],
                "Win Rate": win_rates.loc[current_cluster, return_column],
                "Observations": observation_counts.loc[current_cluster, "Observations"]
            }
        )

    return pd.DataFrame(rows).sort_values("Average 20D Forward Return", ascending=False)


def label_signal_strength(probability):
    if probability >= 0.60:
        return "Strong Bullish"
    if probability >= 0.55:
        return "Moderate Bullish"
    if probability >= 0.45:
        return "Neutral / Weak Signal"
    if probability >= 0.40:
        return "Moderate Bearish"
    return "Strong Bearish"


def summarize_signal_strength(signal_label):
    if "Strong" in signal_label:
        return "strong"
    if "Moderate" in signal_label:
        return "moderate"
    return "weak"


def format_percent_value(value):
    return f"{value:.2%}" if pd.notna(value) else "N/A"


def format_pct(value):
    return format_percent_value(value)


def colored_badge(label, color):
    color_map = {
        "green": ("#166534", "#dcfce7", "#86efac"),
        "red": ("#991b1b", "#fee2e2", "#fecaca"),
        "yellow": ("#854d0e", "#fef9c3", "#fde68a"),
        "gray": ("#374151", "#f3f4f6", "#d1d5db")
    }
    text_color, background, border = color_map.get(color, color_map["gray"])
    return (
        f"<span style='display:inline-block;padding:0.35rem 0.6rem;border-radius:0.4rem;"
        f"font-weight:700;color:{text_color};background:{background};border:1px solid {border};'>"
        f"{label}</span>"
    )


def colored_metric(label, value, color):
    color_map = {
        "green": ("#166534", "#f0fdf4", "#bbf7d0"),
        "red": ("#991b1b", "#fef2f2", "#fecaca"),
        "yellow": ("#854d0e", "#fffbeb", "#fde68a"),
        "gray": ("#374151", "#f9fafb", "#d1d5db")
    }
    text_color, background, border = color_map.get(color, color_map["gray"])
    return (
        f"<div style='padding:0.85rem 0.95rem;border-radius:0.5rem;"
        f"border:1px solid {border};background:{background};min-height:5.75rem;'>"
        f"<div style='font-size:0.78rem;font-weight:650;color:#4b5563;margin-bottom:0.35rem;'>"
        f"{label}</div>"
        f"<div style='font-size:1.15rem;font-weight:800;color:{text_color};line-height:1.25;'>"
        f"{value}</div>"
        f"</div>"
    )


def action_badge(action):
    return colored_badge(action, get_badge_color_for_action(action))


def signal_color(signal_strength):
    return get_badge_color_for_signal(signal_strength)


def probability_color(probability):
    return "green" if pd.notna(probability) and probability > 0.51 else "red"


def return_color(value):
    if pd.isna(value):
        return "gray"
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return "yellow"


def get_badge_color_for_action(action_label):
    if any(word in action_label for word in ["BUY", "OVERWEIGHT"]):
        return "green"
    if any(word in action_label for word in ["SELL", "AVOID"]):
        return "red"
    return "yellow"


def get_badge_color_for_signal(signal_strength):
    if signal_strength in ["Strong Bullish", "Moderate Bullish"]:
        return "green"
    if signal_strength in ["Strong Bearish", "Moderate Bearish"]:
        return "red"
    return "yellow"


def get_action_from_probability(probability, signal_strength, model_edge=None, backtest_is_weak=False, etf=None):
    if pd.isna(probability):
        action_label = "HOLD / WAIT FOR STRONGER SIGNAL"
        action = "Hold"
        reason = "Random Forest probabilities are unavailable for the current setup."
    elif (pd.notna(model_edge) and model_edge < 0) or backtest_is_weak:
        action_label = "HOLD / WAIT FOR STRONGER SIGNAL"
        action = "Hold"
        reason = "The current signal is softened because model edge or backtest evidence is weak."
    elif probability >= 0.55 and signal_strength in ["Moderate Bullish", "Strong Bullish"]:
        action_label = f"BUY / OVERWEIGHT {etf}" if etf else "BUY / OVERWEIGHT"
        action = "Buy"
        reason = "The top ETF has a bullish probability above 55% with a bullish signal strength label."
    elif 0.51 <= probability < 0.55:
        action_label = f"HOLD / SLIGHT OVERWEIGHT {etf}" if etf else "HOLD / SLIGHT OVERWEIGHT"
        action = "Hold"
        reason = "The top probability is positive but not strong enough for a full overweight signal."
    elif 0.45 <= probability < 0.51:
        action_label = "HOLD / NEUTRAL"
        action = "Hold"
        reason = "Bullish probabilities are clustered near neutral."
    else:
        action_label = "SELL / AVOID RISK ASSETS"
        action = "Sell"
        reason = "The best bullish probability is below 45%, indicating a bearish model signal."

    return {
        "label": action_label,
        "action": action,
        "reason": reason,
        "color": get_badge_color_for_action(action_label)
    }


def get_profile_value(profile, column_name):
    return profile[column_name] if column_name in profile.index and pd.notna(profile[column_name]) else None


def assign_regime_label(profile):
    vix_level = get_profile_value(profile, "VIX Level")
    yield_10y_change = get_profile_value(profile, "10Y Yield 3M Change")
    yield_20y_change = get_profile_value(profile, "20Y Yield 3M Change")
    cpi_change = get_profile_value(profile, "CPI 3M Change")
    core_cpi_change = get_profile_value(profile, "Core CPI 3M Change")
    unemployment_change = get_profile_value(profile, "Unemployment 3M Change")
    baa_spread_change = get_profile_value(profile, "BAA Spread 3M Change")
    curve_level = get_profile_value(profile, "10Y-2Y Curve Level")

    high_vix = vix_level is not None and vix_level >= 25
    low_vix = vix_level is not None and vix_level <= 18
    rising_rates = any(
        value is not None and value >= 0.40
        for value in [yield_10y_change, yield_20y_change]
    )
    falling_rates = any(
        value is not None and value <= -0.40
        for value in [yield_10y_change, yield_20y_change]
    )
    rising_inflation = any(
        value is not None and value >= 1.00
        for value in [cpi_change, core_cpi_change]
    )
    rising_unemployment = unemployment_change is not None and unemployment_change >= 0.20
    widening_credit = baa_spread_change is not None and baa_spread_change >= 0.20
    inverted_curve = curve_level is not None and curve_level <= -0.50
    stable_labor = unemployment_change is not None and unemployment_change <= 0.10
    stable_credit = baa_spread_change is not None and baa_spread_change <= 0.10

    if high_vix or rising_unemployment or widening_credit:
        return "Recession Fear / Risk-Off Regime"
    if rising_rates and rising_inflation:
        return "Inflation Pressure / Rising-Rate Regime"
    if falling_rates and not rising_inflation:
        return "Falling-Rate / Bond-Friendly Regime"
    if low_vix and stable_labor and stable_credit and not rising_inflation:
        return "Risk-On / Stable Growth Regime"
    if inverted_curve or rising_rates or rising_inflation:
        return "Late-Cycle / Mixed Macro Regime"
    return "Mixed / Transitional Macro Regime"


def get_distinctive_cluster_features(cluster_profiles, current_cluster, limit=4):
    profile = cluster_profiles.loc[current_cluster]
    profile_std = cluster_profiles.std().mask(lambda values: values == 0)
    relative_profile = ((profile - cluster_profiles.mean()) / profile_std).dropna()

    if relative_profile.empty:
        return profile.abs().sort_values(ascending=False).head(limit)

    return relative_profile.reindex(
        relative_profile.abs().sort_values(ascending=False).head(limit).index
    )


def format_feature_drivers(feature_scores):
    if feature_scores.empty:
        return "no single macro feature stands out from the cluster profile"

    driver_phrases = []
    for feature_name, score in feature_scores.items():
        direction = "above" if score > 0 else "below"
        driver_phrases.append(f"{feature_name} is {direction} its cross-cluster average")
    return "; ".join(driver_phrases)


def add_percent_display_column(data, source_column, display_column):
    display_data = data.copy()
    display_data[display_column] = display_data[source_column] * 100
    return display_data


def calculate_performance_metrics(return_data, return_column, label, periods_per_year, trade_mask=None):
    returns = return_data[return_column].dropna()
    if returns.empty:
        return {
            "Asset": label,
            "Cumulative Return": None,
            "Annualized Return": None,
            "Volatility": None,
            "Sharpe Ratio": None,
            "Max Drawdown": None,
            "Sortino Ratio": None,
            "Calmar Ratio": None,
            "Number of Trades": 0,
            "Win Rate": None,
            "Average Winning Trade": None,
            "Average Losing Trade": None
        }

    wealth = (1 + returns).cumprod()
    cumulative_return = wealth.iloc[-1] - 1
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(returns)) - 1
    volatility = returns.std() * (periods_per_year ** 0.5)
    sharpe_ratio = (
        (returns.mean() / returns.std()) * (periods_per_year ** 0.5)
        if returns.std() and pd.notna(returns.std())
        else None
    )
    drawdown = wealth / wealth.cummax() - 1
    max_drawdown = drawdown.min()
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * (periods_per_year ** 0.5)
    sortino_ratio = (
        annualized_return / downside_volatility
        if downside_volatility and pd.notna(downside_volatility)
        else None
    )
    calmar_ratio = (
        annualized_return / abs(max_drawdown)
        if max_drawdown and pd.notna(max_drawdown) and max_drawdown < 0
        else None
    )

    if trade_mask is not None:
        trade_returns = return_data.loc[trade_mask, return_column].dropna()
    else:
        trade_returns = returns

    return {
        "Asset": label,
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Number of Trades": len(trade_returns),
        "Win Rate": (trade_returns > 0).mean() if len(trade_returns) else None,
        "Average Winning Trade": trade_returns[trade_returns > 0].mean() if len(trade_returns) else None,
        "Average Losing Trade": trade_returns[trade_returns < 0].mean() if len(trade_returns) else None
    }


def train_random_forest_models(feature_df, forward_returns, asset_tickers):
    feature_data = feature_df.reset_index()
    feature_data = feature_data.rename(columns={feature_data.columns[0]: "Date"})
    feature_data = force_datetime_ns(feature_data, "Date")

    returns_data = forward_returns.reset_index()
    returns_data = returns_data.rename(columns={returns_data.columns[0]: "Date"})
    returns_data = force_datetime_ns(returns_data, "Date")

    model_data = pd.merge_asof(
        returns_data.dropna(subset=["Date"]).sort_values("Date"),
        feature_data.dropna(subset=["Date"]).sort_values("Date"),
        on="Date",
        direction="backward"
    ).dropna()

    feature_columns = feature_df.columns.tolist()
    latest_features = feature_df.iloc[[-1]]
    metrics_rows = []
    importance_tables = {}
    confusion_tables = {}
    probability_rows = []

    for ticker in asset_tickers:
        return_column = f"{ticker} 20D Forward Return"
        target_column = f"{ticker} Positive 20D Return"

        asset_data = model_data[["Date", return_column] + feature_columns].dropna().copy()
        asset_data[target_column] = (asset_data[return_column] > 0).astype(int)

        if len(asset_data) < 80 or asset_data[target_column].nunique() < 2:
            metrics_rows.append(
                {
                    "Asset": ticker,
                    "Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                    "ROC-AUC": None,
                    "Baseline Accuracy": None,
                    "Model Edge": None,
                    "Train Rows": None,
                    "Test Rows": len(asset_data),
                    "Note": "Not enough labeled data or only one target class."
                }
            )
            continue

        split_index = int(len(asset_data) * 0.80)
        train_data = asset_data.iloc[:split_index]
        test_data = asset_data.iloc[split_index:]

        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]

        if y_train.nunique() < 2:
            metrics_rows.append(
                {
                    "Asset": ticker,
                    "Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                    "ROC-AUC": None,
                    "Baseline Accuracy": None,
                    "Model Edge": None,
                    "Train Rows": len(train_data),
                    "Test Rows": len(test_data),
                    "Note": "Training window has only one target class."
                }
            )
            continue

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            roc_auc = None
        else:
            roc_auc = "calculate"

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        if roc_auc == "calculate":
            roc_auc = roc_auc_score(y_test, probabilities)

        model_accuracy = accuracy_score(y_test, predictions)
        baseline_accuracy = y_test.value_counts(normalize=True).max()
        model_edge = model_accuracy - baseline_accuracy
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        confusion_tables[ticker] = pd.DataFrame(
            matrix,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"]
        )

        metrics_rows.append(
            {
                "Asset": ticker,
                "Accuracy": model_accuracy,
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc,
                "Baseline Accuracy": baseline_accuracy,
                "Model Edge": model_edge,
                "Train Rows": len(train_data),
                "Test Rows": len(test_data),
                "Note": "Chronological 80/20 split"
            }
        )

        full_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
        full_model.fit(asset_data[feature_columns], asset_data[target_column])

        latest_probability = full_model.predict_proba(latest_features)[0, 1]
        probability_rows.append(
            {
                "Asset": ticker,
                "Latest Bullish Probability": latest_probability
            }
        )

        importance_tables[ticker] = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": full_model.feature_importances_
            }
        ).sort_values("Importance", ascending=False)

    return pd.DataFrame(metrics_rows), importance_tables, confusion_tables, pd.DataFrame(probability_rows)


def backtest_random_forest_strategy(feature_df, forward_returns, asset_tickers, probability_threshold=0.55):
    feature_data = feature_df.reset_index()
    feature_data = feature_data.rename(columns={feature_data.columns[0]: "Date"})
    feature_data = force_datetime_ns(feature_data, "Date")

    returns_data = forward_returns.reset_index()
    returns_data = returns_data.rename(columns={returns_data.columns[0]: "Date"})
    returns_data = force_datetime_ns(returns_data, "Date")

    feature_columns = feature_df.columns.tolist()
    return_columns = [f"{ticker} 20D Forward Return" for ticker in asset_tickers]
    model_data = pd.merge_asof(
        returns_data.dropna(subset=["Date"]).sort_values("Date"),
        feature_data.dropna(subset=["Date"]).sort_values("Date"),
        on="Date",
        direction="backward"
    ).dropna(subset=return_columns + feature_columns)

    if len(model_data) < 80:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    split_index = int(len(model_data) * 0.80)
    train_data = model_data.iloc[:split_index].copy()
    test_data = model_data.iloc[split_index:].copy()
    probability_data = pd.DataFrame({"Date": test_data["Date"].values})

    for ticker in asset_tickers:
        return_column = f"{ticker} 20D Forward Return"
        target_column = f"{ticker} Positive 20D Return"
        train_data[target_column] = (train_data[return_column] > 0).astype(int)

        if train_data[target_column].nunique() < 2:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(train_data[feature_columns], train_data[target_column])
        probability_data[f"{ticker} Bullish Probability"] = model.predict_proba(
            test_data[feature_columns]
        )[:, 1]

    backtest_rows = []
    for row_index, probability_row in probability_data.iterrows():
        probabilities = {
            ticker: probability_row[f"{ticker} Bullish Probability"] for ticker in asset_tickers
        }
        best_ticker = max(probabilities, key=probabilities.get)
        best_probability = probabilities[best_ticker]

        if best_probability > probability_threshold:
            selected_asset = best_ticker
            strategy_return = test_data.iloc[row_index][f"{best_ticker} 20D Forward Return"]
        else:
            selected_asset = "Cash"
            strategy_return = 0.0

        backtest_rows.append(
            {
                "Date": probability_row["Date"],
                "Selected Asset": selected_asset,
                "Highest Bullish Probability": best_probability,
                "Strategy 20D Return": strategy_return,
                "SPY 20D Forward Return": test_data.iloc[row_index]["SPY 20D Forward Return"],
                "QQQ 20D Forward Return": test_data.iloc[row_index]["QQQ 20D Forward Return"],
                "TLT 20D Forward Return": test_data.iloc[row_index]["TLT 20D Forward Return"]
            }
        )

    backtest_data = pd.DataFrame(backtest_rows)
    backtest_data["Equal Weight SPY/QQQ/TLT 20D Forward Return"] = backtest_data[
        ["SPY 20D Forward Return", "QQQ 20D Forward Return", "TLT 20D Forward Return"]
    ].mean(axis=1)
    trade_data = backtest_data[backtest_data["Selected Asset"] != "Cash"]
    strategy_average = backtest_data["Strategy 20D Return"].mean()
    trade_count = len(trade_data)

    summary_data = pd.DataFrame(
        [
            {"Metric": "Number of Trades", "Value": trade_count},
            {"Metric": "Average 20D Return", "Value": strategy_average},
            {
                "Metric": "Win Rate",
                "Value": (trade_data["Strategy 20D Return"] > 0).mean() if trade_count else None
            },
            {
                "Metric": "Best Trade",
                "Value": trade_data["Strategy 20D Return"].max() if trade_count else None
            },
            {
                "Metric": "Worst Trade",
                "Value": trade_data["Strategy 20D Return"].min() if trade_count else None
            }
        ]
    )

    comparison_data = pd.DataFrame(
        [
            {"Asset": "RF Strategy", "Average 20D Return": strategy_average},
            {"Asset": "SPY", "Average 20D Return": backtest_data["SPY 20D Forward Return"].mean()},
            {"Asset": "QQQ", "Average 20D Return": backtest_data["QQQ 20D Forward Return"].mean()},
            {"Asset": "TLT", "Average 20D Return": backtest_data["TLT 20D Forward Return"].mean()},
            {
                "Asset": "Equal Weight SPY/QQQ/TLT",
                "Average 20D Return": backtest_data[
                    ["SPY 20D Forward Return", "QQQ 20D Forward Return", "TLT 20D Forward Return"]
                ].mean(axis=1).mean()
            }
        ]
    )

    metric_columns = {
        "RF Strategy": "Strategy 20D Return",
        "SPY": "SPY 20D Forward Return",
        "QQQ": "QQQ 20D Forward Return",
        "TLT": "TLT 20D Forward Return",
        "Equal Weight SPY/QQQ/TLT": "Equal Weight SPY/QQQ/TLT 20D Forward Return"
    }
    periods_per_year = 252 / 20
    strategy_trade_mask = backtest_data["Selected Asset"] != "Cash"
    risk_rows = []
    curve_data = backtest_data[["Date"]].copy()
    drawdown_data = backtest_data[["Date"]].copy()
    rolling_data = backtest_data[["Date"]].copy()

    for label, return_column in metric_columns.items():
        trade_mask = strategy_trade_mask if label == "RF Strategy" else None
        risk_rows.append(
            calculate_performance_metrics(
                backtest_data,
                return_column,
                label,
                periods_per_year,
                trade_mask
            )
        )

        returns = backtest_data[return_column].fillna(0)
        wealth = (1 + returns).cumprod()
        curve_data[label] = wealth - 1
        drawdown_data[label] = wealth / wealth.cummax() - 1
        if len(returns) >= 252:
            rolling_data[label] = returns.rolling(252).apply(lambda values: (1 + values).prod() - 1)

    risk_metrics = pd.DataFrame(risk_rows)
    curve_chart_data = curve_data.melt(id_vars="Date", var_name="Asset", value_name="Cumulative Return")
    drawdown_chart_data = drawdown_data.melt(id_vars="Date", var_name="Asset", value_name="Drawdown")

    if len(backtest_data) >= 252:
        rolling_chart_data = rolling_data.melt(
            id_vars="Date",
            var_name="Asset",
            value_name="Rolling 12M Return"
        ).dropna()
    else:
        rolling_chart_data = pd.DataFrame()

    return (
        summary_data,
        comparison_data,
        risk_metrics,
        curve_chart_data,
        drawdown_chart_data,
        rolling_chart_data
    )


def run_probability_strategy_backtest(feature_df, forward_returns, asset_tickers, horizon, probability_threshold):
    feature_data = feature_df.reset_index()
    feature_data = feature_data.rename(columns={feature_data.columns[0]: "Date"})
    feature_data = force_datetime_ns(feature_data, "Date")

    returns_data = forward_returns.reset_index()
    returns_data = returns_data.rename(columns={returns_data.columns[0]: "Date"})
    returns_data = force_datetime_ns(returns_data, "Date")

    feature_columns = feature_df.columns.tolist()
    return_columns = [f"{ticker} {horizon}D Forward Return" for ticker in asset_tickers]
    model_data = pd.merge_asof(
        returns_data.dropna(subset=["Date"]).sort_values("Date"),
        feature_data.dropna(subset=["Date"]).sort_values("Date"),
        on="Date",
        direction="backward"
    ).dropna(subset=return_columns + feature_columns)

    if len(model_data) < 80:
        return pd.DataFrame()

    split_index = int(len(model_data) * 0.80)
    train_data = model_data.iloc[:split_index].copy()
    test_data = model_data.iloc[split_index:].copy()
    probability_data = pd.DataFrame({"Date": test_data["Date"].values})

    for ticker in asset_tickers:
        return_column = f"{ticker} {horizon}D Forward Return"
        target_column = f"{ticker} Positive {horizon}D Return"
        train_data[target_column] = (train_data[return_column] > 0).astype(int)

        if train_data[target_column].nunique() < 2:
            return pd.DataFrame()

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(train_data[feature_columns], train_data[target_column])
        probability_data[f"{ticker} Bullish Probability"] = model.predict_proba(
            test_data[feature_columns]
        )[:, 1]

    backtest_rows = []
    for row_index, probability_row in probability_data.iterrows():
        probabilities = {
            ticker: probability_row[f"{ticker} Bullish Probability"] for ticker in asset_tickers
        }
        best_ticker = max(probabilities, key=probabilities.get)
        best_probability = probabilities[best_ticker]

        if best_probability > probability_threshold:
            selected_asset = best_ticker
            strategy_return = test_data.iloc[row_index][f"{best_ticker} {horizon}D Forward Return"]
        else:
            selected_asset = "Cash"
            strategy_return = 0.0

        backtest_rows.append(
            {
                "Date": probability_row["Date"],
                "Selected Asset": selected_asset,
                "Highest Bullish Probability": best_probability,
                "Strategy Return": strategy_return
            }
        )

    return pd.DataFrame(backtest_rows)


def summarize_strategy_backtest(backtest_data):
    if backtest_data.empty:
        return {
            "Number of Trades": 0,
            "Average Return": None,
            "Win Rate": None,
            "Best Trade": None,
            "Worst Trade": None
        }

    trade_data = backtest_data[backtest_data["Selected Asset"] != "Cash"]
    trade_count = len(trade_data)
    return {
        "Number of Trades": trade_count,
        "Average Return": backtest_data["Strategy Return"].mean(),
        "Win Rate": (trade_data["Strategy Return"] > 0).mean() if trade_count else None,
        "Best Trade": trade_data["Strategy Return"].max() if trade_count else None,
        "Worst Trade": trade_data["Strategy Return"].min() if trade_count else None
    }


def assign_period_label(date_value):
    if date_value < pd.Timestamp("2020-01-01"):
        return "pre-2020"
    if date_value < pd.Timestamp("2022-01-01"):
        return "2020-2021"
    if date_value < pd.Timestamp("2023-01-01"):
        return "2022"
    return "2023-present"


if fred_api_key:
    macro_data = load_fred_data(fred_api_key, selected_macro_code)
    macro_all, macro_features = load_multiple_fred_series(fred_api_key)

    # Make a copy so Streamlit cache data is not modified directly.
    raw_macro_features = macro_features.copy()
    macro_features, scaled_features, elbow_data, silhouette, pca_data = run_kmeans_regime_model(
        raw_macro_features,
        n_clusters
    )
    current_cluster = int(macro_features["Cluster"].iloc[-1])
    cluster_profiles = macro_features.groupby("Cluster").mean()
    current_profile = cluster_profiles.loc[current_cluster].sort_values(ascending=False)
    current_regime_label = assign_regime_label(cluster_profiles.loc[current_cluster])
    current_cluster_drivers = get_distinctive_cluster_features(cluster_profiles, current_cluster)

    asset_tickers = ["SPY", "QQQ", "TLT"]
    return_columns = [
        "SPY 20D Forward Return",
        "QQQ 20D Forward Return",
        "TLT 20D Forward Return"
    ]

    asset_prices, asset_forward_returns = load_multi_asset_prices(asset_tickers, period="max")
    returns_data, cluster_data, regime_return_data = prepare_regime_return_data(
        macro_features,
        asset_forward_returns
    )
    average_returns, median_returns, win_rates, observation_counts = calculate_return_tables(
        regime_return_data,
        return_columns
    )
    current_cluster_asset_table = make_current_cluster_asset_table(
        current_cluster,
        average_returns,
        median_returns,
        win_rates,
        observation_counts
    )
    latest_backtest_date = regime_return_data["Date"].max()
    rf_metrics, rf_importances, rf_confusion_matrices, latest_probabilities = train_random_forest_models(
        macro_features,
        asset_forward_returns,
        asset_tickers
    )
    (
        rf_backtest_summary,
        rf_backtest_comparison,
        rf_backtest_risk_metrics,
        rf_backtest_curve,
        rf_backtest_drawdown,
        rf_backtest_rolling_returns
    ) = backtest_random_forest_strategy(
        macro_features,
        asset_forward_returns,
        asset_tickers
    )
    robustness_horizon_rows = []
    robustness_threshold_rows = []
    robustness_cluster_rows = []
    robustness_period_rows = []
    robustness_yearly_rows = []

    for horizon in [5, 20, 60]:
        horizon_returns = calculate_forward_returns(asset_prices, asset_tickers, horizon)
        horizon_backtest = run_probability_strategy_backtest(
            macro_features,
            horizon_returns,
            asset_tickers,
            horizon,
            0.55
        )
        horizon_summary = summarize_strategy_backtest(horizon_backtest)
        robustness_horizon_rows.append(
            {
                "Horizon": f"{horizon}D",
                "Average Return": horizon_summary["Average Return"],
                "Win Rate": horizon_summary["Win Rate"]
            }
        )

    for threshold in [0.50, 0.55, 0.60, 0.65]:
        threshold_backtest = run_probability_strategy_backtest(
            macro_features,
            asset_forward_returns,
            asset_tickers,
            20,
            threshold
        )
        threshold_summary = summarize_strategy_backtest(threshold_backtest)
        robustness_threshold_rows.append(
            {
                "Threshold": threshold,
                "Number of Trades": threshold_summary["Number of Trades"],
                "Average Return": threshold_summary["Average Return"],
                "Win Rate": threshold_summary["Win Rate"],
                "Best Trade": threshold_summary["Best Trade"],
                "Worst Trade": threshold_summary["Worst Trade"]
            }
        )

    for cluster_count in range(3, 9):
        cluster_test_features, _, _, _, _ = run_kmeans_regime_model(raw_macro_features, cluster_count)
        cluster_backtest = run_probability_strategy_backtest(
            cluster_test_features,
            asset_forward_returns,
            asset_tickers,
            20,
            0.55
        )
        cluster_summary = summarize_strategy_backtest(cluster_backtest)
        robustness_cluster_rows.append(
            {
                "Cluster Count": cluster_count,
                "Average Return": cluster_summary["Average Return"],
                "Win Rate": cluster_summary["Win Rate"]
            }
        )

    robustness_base_backtest = run_probability_strategy_backtest(
        macro_features,
        asset_forward_returns,
        asset_tickers,
        20,
        0.55
    )
    if not robustness_base_backtest.empty:
        robustness_base_backtest["Year"] = robustness_base_backtest["Date"].dt.year
        yearly_performance = robustness_base_backtest.groupby("Year")["Strategy Return"].mean().reset_index()
        robustness_yearly_rows = yearly_performance.rename(
            columns={"Strategy Return": "Average Return"}
        ).to_dict("records")

        robustness_base_backtest["Period"] = robustness_base_backtest["Date"].apply(assign_period_label)
        for period_name, period_data in robustness_base_backtest.groupby("Period"):
            period_summary = summarize_strategy_backtest(period_data)
            robustness_period_rows.append(
                {
                    "Period": period_name,
                    "Number of Trades": period_summary["Number of Trades"],
                    "Average Return": period_summary["Average Return"],
                    "Win Rate": period_summary["Win Rate"],
                    "Best Trade": period_summary["Best Trade"],
                    "Worst Trade": period_summary["Worst Trade"]
                }
            )

    robustness_horizon_table = pd.DataFrame(robustness_horizon_rows)
    robustness_threshold_table = pd.DataFrame(robustness_threshold_rows)
    robustness_cluster_table = pd.DataFrame(robustness_cluster_rows)
    robustness_period_table = pd.DataFrame(robustness_period_rows)
    robustness_yearly_table = pd.DataFrame(robustness_yearly_rows)

    rf_backtest_is_weak = True
    if not rf_backtest_summary.empty:
        backtest_summary_lookup = rf_backtest_summary.set_index("Metric")["Value"]
        backtest_average_return = backtest_summary_lookup.get("Average 20D Return")
        backtest_win_rate = backtest_summary_lookup.get("Win Rate")
        rf_backtest_is_weak = (
            (pd.notna(backtest_average_return) and backtest_average_return <= 0)
            or (pd.notna(backtest_win_rate) and backtest_win_rate < 0.50)
        )

    current_model_signal = None
    if not latest_probabilities.empty:
        model_signal_table = latest_probabilities.merge(
            rf_metrics[["Asset", "Model Edge"]],
            on="Asset",
            how="left"
        )
        model_signal_table["Signal Strength"] = model_signal_table[
            "Latest Bullish Probability"
        ].apply(label_signal_strength)
        current_model_signal = model_signal_table.sort_values(
            "Latest Bullish Probability",
            ascending=False
        ).iloc[0]
        current_model_action = get_action_from_probability(
            current_model_signal["Latest Bullish Probability"],
            current_model_signal["Signal Strength"],
            current_model_signal["Model Edge"],
            rf_backtest_is_weak,
            current_model_signal["Asset"]
        )
    else:
        current_model_action = get_action_from_probability(None, "Neutral / Weak Signal")

    with overview_tab:
        st.header("Overview")
        strategy_risk = None
        annualized_return = None
        if not rf_backtest_risk_metrics.empty:
            strategy_rows = rf_backtest_risk_metrics[
                rf_backtest_risk_metrics["Asset"] == "RF Strategy"
            ]
            if not strategy_rows.empty:
                strategy_risk = strategy_rows.iloc[0]
                annualized_return = strategy_risk["Annualized Return"]

        if current_model_signal is not None:
            overview_probability = current_model_signal["Latest Bullish Probability"]
            selected_etf = current_model_signal["Asset"]
            signal_strength = current_model_signal["Signal Strength"]
        else:
            overview_probability = None
            selected_etf = "N/A"
            signal_strength = "N/A"

        first_row = st.columns(3)
        first_row[0].markdown(
            colored_metric(
                "Current Macro Regime",
                f"Cluster {current_cluster} - {current_regime_label}",
                "gray"
            ),
            unsafe_allow_html=True
        )
        first_row[1].markdown(
            colored_metric(
                "Model Action",
                current_model_action["label"],
                current_model_action["color"]
            ),
            unsafe_allow_html=True
        )
        first_row[2].markdown(
            colored_metric("Selected ETF", selected_etf, "gray"),
            unsafe_allow_html=True
        )

        second_row = st.columns(3)
        second_row[0].markdown(
            colored_metric(
                "Bullish Probability",
                format_pct(overview_probability) if pd.notna(overview_probability) else "N/A",
                probability_color(overview_probability)
            ),
            unsafe_allow_html=True
        )
        second_row[1].markdown(
            colored_metric("Signal Strength", signal_strength, signal_color(signal_strength)),
            unsafe_allow_html=True
        )
        second_row[2].markdown(
            colored_metric(
                "Historical Annualized Return",
                format_pct(annualized_return) if pd.notna(annualized_return) else "N/A",
                return_color(annualized_return)
            ),
            unsafe_allow_html=True
        )

        if current_model_signal is None:
            st.info("Random Forest probabilities are unavailable for the current settings.")
        else:
            st.caption(
                "The model ranks SPY, QQQ, and TLT using macro features, the current regime cluster, "
                "and Random Forest 20-day forward-return probabilities."
            )

    # -----------------------------
    # Selected macro indicator chart
    # -----------------------------
    with macro_tab:
        st.header("Selected Macro Indicator Chart")
        st.write(
            "This tab shows the FRED macro series selected in the sidebar. The same FRED data source "
            "feeds the broader feature set used to classify market regimes."
        )

        fig_macro = px.line(
            macro_data,
            x="Date",
            y="Value",
            title=f"{selected_macro_name} ({selected_macro_code})"
        )
        st.plotly_chart(fig_macro, use_container_width=True)

        with st.expander(f"Recent {selected_macro_name} Data", expanded=False):
            st.dataframe(
                macro_data.tail(30).style.format(
                    {
                        "Value": "{:.2f}",
                        "1M Change": "{:.2f}",
                        "3M Change": "{:.2f}"
                    },
                    na_rep="N/A"
                ),
                use_container_width=True
            )

        with st.expander("Macro Feature Dataset Used for ML", expanded=False):
            st.write(
                "These engineered features include macro levels and trailing changes. They are descriptive "
                "inputs available at or before each observation date."
            )
            st.dataframe(raw_macro_features.tail(30), use_container_width=True)

    # -----------------------------
    # K-means regime classifier
    # -----------------------------
    with kmeans_tab:
        st.header("K-means Regime Classifier")
        st.write(
            "K-means groups similar macro environments into regimes. Features are standardized first, "
            "so yields, inflation, volatility, and spreads can be compared on a common scale."
        )

        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Current Macro Regime Cluster", current_cluster)
        if silhouette is not None:
            metric_col_2.metric("Silhouette Score", f"{silhouette:.3f}")
        else:
            metric_col_2.metric("Silhouette Score", "N/A")

        chart_col_1, chart_col_2 = st.columns(2)
        with chart_col_1:
            fig_elbow = px.line(
                elbow_data,
                x="k",
                y="Inertia",
                markers=True,
                title="Elbow Chart: K-means Inertia"
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.caption("Lower inertia means tighter clusters. The elbow helps judge whether more clusters add value.")

        with chart_col_2:
            fig_pca = px.scatter(
                pca_data,
                x="PC1",
                y="PC2",
                color="Cluster",
                hover_data=["Date"],
                title="PCA View of Macro Observations"
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption("PCA compresses the macro feature set into two dimensions so the clusters are easier to inspect.")

        cluster_plot_data = macro_features.reset_index()
        cluster_plot_data = cluster_plot_data.rename(columns={cluster_plot_data.columns[0]: "Date"})
        fig_cluster = px.line(
            cluster_plot_data,
            x="Date",
            y="Cluster",
            title="Macro Regime Cluster Over Time"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        with st.expander("Recent Regime History", expanded=False):
            st.dataframe(macro_features[["Cluster"]].tail(30), use_container_width=True)

        with st.expander("Cluster Profile Table", expanded=True):
            st.write("Average macro feature values by cluster. Use this to identify what each regime represents.")
            st.dataframe(cluster_profiles.style.format("{:.2f}"), use_container_width=True)

        with st.expander("Current Cluster Strongest Features", expanded=True):
            st.write(f"The current macro environment belongs to Cluster {current_cluster}.")
            profile_col_1, profile_col_2 = st.columns(2)
            with profile_col_1:
                st.write("Highest average features")
                st.dataframe(
                    current_profile.head(15).rename("Average Feature Value").to_frame().style.format("{:.2f}"),
                    use_container_width=True
                )
            with profile_col_2:
                st.write("Lowest average features")
                st.dataframe(
                    current_profile.tail(15).rename("Average Feature Value").to_frame().style.format("{:.2f}"),
                    use_container_width=True
                )

    # -----------------------------
    # ETF forward returns by regime
    # -----------------------------
    with returns_tab:
        st.header("ETF Forward Returns by Regime")
        st.write(
            "This section asks a practical allocation question: after similar macro regimes, how did SPY, "
            "QQQ, and TLT perform over the next 20 trading days?"
        )
        st.caption(
            "20-day forward returns are only available for dates at least 20 trading days in the past. "
            "The current regime uses the latest macro data, while the return statistics use historical "
            "completed 20-day periods."
        )

        if not current_cluster_asset_table.empty:
            best_row = current_cluster_asset_table.iloc[0]
            best_col, avg_col, win_col, date_col = st.columns(4)
            best_col.metric("Current Cluster Historical Best Asset", best_row["Asset"])
            avg_col.metric(
                "Average 20D Forward Return",
                f"{best_row['Average 20D Forward Return']:.2%}"
            )
            win_col.metric("Win Rate", f"{best_row['Win Rate']:.2%}")
            date_col.metric(
                "Latest Backtest Date",
                latest_backtest_date.strftime("%Y-%m-%d") if pd.notna(latest_backtest_date) else "Unavailable"
            )

        fig_returns = px.bar(
            average_returns.reset_index(),
            x="Cluster",
            y=return_columns,
            barmode="group",
            title="Average 20-Day Forward Returns by Macro Cluster"
        )
        st.plotly_chart(fig_returns, use_container_width=True)

        summary_tab, current_tab, raw_tab = st.tabs(["Summary Tables", "Current Cluster", "Merged Data"])
        with summary_tab:
            st.subheader("Average 20-Day Forward Return")
            st.dataframe(average_returns.style.format("{:.2%}"), use_container_width=True)

            st.subheader("Median 20-Day Forward Return")
            st.dataframe(median_returns.style.format("{:.2%}"), use_container_width=True)

            st.subheader("Win Rate")
            st.write("Win rate is the percentage of historical observations where the forward return was positive.")
            st.dataframe(win_rates.style.format("{:.2%}"), use_container_width=True)

            st.subheader("Observation Count")
            st.dataframe(observation_counts, use_container_width=True)

        with current_tab:
            st.write("Ranking of SPY, QQQ, and TLT inside the current macro cluster.")
            st.dataframe(
                current_cluster_asset_table.style.format(
                    {
                        "Average 20D Forward Return": "{:.2%}",
                        "Median 20D Forward Return": "{:.2%}",
                        "Win Rate": "{:.2%}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

        with raw_tab:
            st.write("Each row uses the latest macro cluster available at or before that ETF return date.")
            st.dataframe(
                regime_return_data.tail(50).style.format(
                    {
                        "SPY 20D Forward Return": "{:.2%}",
                        "QQQ 20D Forward Return": "{:.2%}",
                        "TLT 20D Forward Return": "{:.2%}"
                    },
                    na_rep="N/A"
                ),
                use_container_width=True
            )

    # -----------------------------
    # Random Forest prediction model
    # -----------------------------
    with rf_tab:
        st.header("Random Forest 20D Return Prediction")
        st.write(
            "The Random Forest models estimate whether each ETF's next 20-day return is positive. "
            "The test set is the most recent 20% of observations, which avoids random shuffling of time-series data."
        )

        if not latest_probabilities.empty:
            probability_cols = st.columns(len(latest_probabilities))
            for index, probability_row in latest_probabilities.reset_index(drop=True).iterrows():
                probability_cols[index].metric(
                    f"{probability_row['Asset']} Bullish Probability",
                    f"{probability_row['Latest Bullish Probability']:.2%}"
                )

        st.subheader("Model Test Metrics")
        st.dataframe(
            rf_metrics.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "ROC-AUC": "{:.3f}",
                    "Baseline Accuracy": "{:.2%}",
                    "Model Edge": "{:.2%}"
                },
                na_rep="N/A"
            ),
            use_container_width=True,
            hide_index=True
        )

        for _, metric_row in rf_metrics.iterrows():
            if pd.notna(metric_row["Model Edge"]) and metric_row["Model Edge"] <= 0:
                st.warning(
                    f"{metric_row['Asset']}: The model does not currently beat a simple baseline for this ETF."
                )

        with st.expander("Latest Bullish Probabilities", expanded=True):
            if latest_probabilities.empty:
                st.info("Latest probabilities are unavailable because the model did not have enough class variety.")
            else:
                signal_table = latest_probabilities.merge(
                    rf_metrics[["Asset", "Accuracy", "Model Edge"]],
                    on="Asset",
                    how="left"
                )
                signal_table["Signal Strength"] = signal_table["Latest Bullish Probability"].apply(
                    label_signal_strength
                )
                signal_table = signal_table.rename(
                    columns={
                        "Asset": "ETF",
                        "Latest Bullish Probability": "Bullish Probability",
                        "Accuracy": "Test Accuracy"
                    }
                )
                st.dataframe(
                    signal_table[
                        ["ETF", "Bullish Probability", "Signal Strength", "Test Accuracy", "Model Edge"]
                    ].style.format(
                        {
                            "Bullish Probability": "{:.2%}",
                            "Test Accuracy": "{:.2%}",
                            "Model Edge": "{:.2%}"
                        },
                        na_rep="N/A"
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                top_signal = signal_table.sort_values("Bullish Probability", ascending=False).iloc[0]
                top_etf = top_signal["ETF"]
                signal_level = summarize_signal_strength(top_signal["Signal Strength"])
                top_features = rf_importances.get(top_etf, pd.DataFrame()).head(3)

                st.subheader("Random Forest Interpretation")
                if top_features.empty:
                    feature_text = "Top feature importances are unavailable for this ETF."
                else:
                    feature_text = ", ".join(top_features["Feature"].tolist())

                st.write(
                    f"The current macro regime is Cluster {current_cluster}. "
                    f"The ETF with the highest bullish probability is {top_etf} at "
                    f"{top_signal['Bullish Probability']:.2%}. "
                    f"This is a {signal_level} signal based on the label "
                    f"'{top_signal['Signal Strength']}'. "
                    f"The top three model features for {top_etf} are: {feature_text}."
                )

        with st.expander("Confusion Matrices by ETF", expanded=True):
            st.write("Rows are actual outcomes and columns are model predictions on the chronological test set.")
            for ticker in asset_tickers:
                if ticker in rf_confusion_matrices:
                    st.write(f"{ticker} confusion matrix")
                    st.dataframe(rf_confusion_matrices[ticker], use_container_width=True)

        with st.expander("Feature Importance by ETF", expanded=True):
            st.write("Feature importance shows which macro inputs the fitted Random Forest used most.")
            for ticker in asset_tickers:
                if ticker in rf_importances:
                    st.write(f"{ticker} top macro features")
                    top_importance = rf_importances[ticker].head(10)
                    st.dataframe(
                        top_importance.style.format({"Importance": "{:.3f}"}),
                        use_container_width=True,
                        hide_index=True
                    )

    with backtest_tab:
        st.header("Simple Random Forest Backtest")
        st.write(
            "Each test-period date chooses the ETF with the highest predicted bullish probability. "
            "The strategy takes a position only when that probability is above 55%; otherwise it holds cash. "
            "Models are trained on the first 80% of chronological observations and tested on the later 20%."
        )

        if rf_backtest_summary.empty or rf_backtest_comparison.empty:
            st.info("Backtest is unavailable because the training window does not have enough class variety.")
        else:
            trade_count = int(rf_backtest_summary.loc[
                rf_backtest_summary["Metric"] == "Number of Trades",
                "Value"
            ].iloc[0])
            average_return = rf_backtest_summary.loc[
                rf_backtest_summary["Metric"] == "Average 20D Return",
                "Value"
            ].iloc[0]
            win_rate = rf_backtest_summary.loc[
                rf_backtest_summary["Metric"] == "Win Rate",
                "Value"
            ].iloc[0]
            best_trade = rf_backtest_summary.loc[
                rf_backtest_summary["Metric"] == "Best Trade",
                "Value"
            ].iloc[0]
            worst_trade = rf_backtest_summary.loc[
                rf_backtest_summary["Metric"] == "Worst Trade",
                "Value"
            ].iloc[0]

            backtest_cols = st.columns(5)
            backtest_cols[0].metric("Number of Trades", trade_count)
            backtest_cols[1].metric("Average 20D Return", f"{average_return:.2%}")
            backtest_cols[2].metric("Win Rate", f"{win_rate:.2%}" if pd.notna(win_rate) else "N/A")
            backtest_cols[3].metric("Best Trade", f"{best_trade:.2%}" if pd.notna(best_trade) else "N/A")
            backtest_cols[4].metric("Worst Trade", f"{worst_trade:.2%}" if pd.notna(worst_trade) else "N/A")

            fig_backtest = px.bar(
                rf_backtest_comparison,
                x="Asset",
                y="Average 20D Return",
                title="Random Forest Strategy vs ETF Average 20-Day Returns"
            )
            st.plotly_chart(fig_backtest, use_container_width=True)

            st.dataframe(
                rf_backtest_comparison.style.format({"Average 20D Return": "{:.2%}"}),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Risk and Performance Metrics")
            rf_backtest_risk_display = rf_backtest_risk_metrics.rename(
                columns={"Cumulative Return": "Historical Cumulative Return"}
            )
            hidden_unreliable_columns = [
                "Max Drawdown",
                "Calmar Ratio",
                "Average Winning Trade",
                "Average Losing Trade"
            ]
            rf_backtest_risk_display = rf_backtest_risk_display.drop(
                columns=[column for column in hidden_unreliable_columns if column in rf_backtest_risk_display.columns]
            )
            st.dataframe(
                rf_backtest_risk_display.style.format(
                    {
                        "Historical Cumulative Return": "{:.2%}",
                        "Annualized Return": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "Sharpe Ratio": "{:.2f}",
                        "Sortino Ratio": "{:.2f}",
                        "Number of Trades": "{:.0f}",
                        "Win Rate": "{:.2%}",
                    },
                    na_rep="N/A"
                ),
                use_container_width=True,
                hide_index=True
            )

            rf_backtest_curve_display = rf_backtest_curve.rename(
                columns={"Cumulative Return": "Historical Cumulative Return"}
            )
            fig_cumulative = px.line(
                rf_backtest_curve_display,
                x="Date",
                y="Historical Cumulative Return",
                color="Asset",
                title="Historical Cumulative Return Curve"
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)

            if rf_backtest_rolling_returns.empty:
                st.info("Rolling 12-month return needs at least 252 backtest observations.")
            else:
                fig_rolling = px.line(
                    rf_backtest_rolling_returns,
                    x="Date",
                    y="Rolling 12M Return",
                    color="Asset",
                    title="Rolling 12-Month Return"
                )
                st.plotly_chart(fig_rolling, use_container_width=True)

    with robustness_tab:
        st.header("Robustness Testing")
        st.write(
            "These tests rerun the strategy under different assumptions. Each test keeps the same rule: "
            "models train on the earlier chronological window and score only later observations."
        )

        st.subheader("Forward Return Horizon Sensitivity")
        if robustness_horizon_table.empty:
            st.info("Horizon sensitivity is unavailable because there is not enough completed return history.")
        else:
            st.dataframe(
                robustness_horizon_table.style.format(
                    {"Average Return": "{:.2%}", "Win Rate": "{:.2%}"},
                    na_rep="N/A"
                ),
                use_container_width=True,
                hide_index=True
            )
            horizon_chart_data = add_percent_display_column(
                robustness_horizon_table,
                "Average Return",
                "Average Return (%)"
            )
            fig_horizon_return = px.bar(
                horizon_chart_data,
                x="Horizon",
                y="Average Return (%)",
                title="Horizon vs Average Strategy Return",
                text=horizon_chart_data["Average Return (%)"].map(lambda value: f"{value:.2f}%")
            )
            fig_horizon_return.update_traces(textposition="outside")
            fig_horizon_return.update_yaxes(ticksuffix="%", tickformat=".2f")
            st.plotly_chart(fig_horizon_return, use_container_width=True)

        st.subheader("Confidence Threshold Sensitivity")
        if robustness_threshold_table.empty:
            st.info("Threshold sensitivity is unavailable because there is not enough completed return history.")
        else:
            st.dataframe(
                robustness_threshold_table.style.format(
                    {
                        "Threshold": "{:.0%}",
                        "Average Return": "{:.2%}",
                        "Win Rate": "{:.2%}",
                        "Best Trade": "{:.2%}",
                        "Worst Trade": "{:.2%}"
                    },
                    na_rep="N/A"
                ),
                use_container_width=True,
                hide_index=True
            )
            threshold_chart_data = robustness_threshold_table.copy()
            threshold_chart_data["Threshold Label"] = threshold_chart_data["Threshold"].map("{:.0%}".format)
            threshold_chart_data["Average Return (%)"] = threshold_chart_data["Average Return"] * 100
            chart_col_1, chart_col_2 = st.columns(2)
            with chart_col_1:
                fig_threshold_return = px.line(
                    threshold_chart_data,
                    x="Threshold Label",
                    y="Average Return (%)",
                    markers=True,
                    title="Threshold vs Average Strategy Return"
                )
                fig_threshold_return.update_traces(
                    text=threshold_chart_data["Average Return (%)"].map(lambda value: f"{value:.2f}%")
                )
                fig_threshold_return.update_yaxes(ticksuffix="%", tickformat=".2f")
                st.plotly_chart(fig_threshold_return, use_container_width=True)
            with chart_col_2:
                fig_threshold_win_rate = px.line(
                    threshold_chart_data,
                    x="Threshold Label",
                    y="Win Rate",
                    markers=True,
                    title="Threshold vs Win Rate"
                )
                fig_threshold_win_rate.update_yaxes(tickformat=".2%")
                st.plotly_chart(fig_threshold_win_rate, use_container_width=True)

        st.subheader("Cluster Count Sensitivity")
        if robustness_cluster_table.empty:
            st.info("Cluster sensitivity is unavailable because there is not enough completed return history.")
        else:
            st.dataframe(
                robustness_cluster_table.style.format(
                    {"Average Return": "{:.2%}", "Win Rate": "{:.2%}"},
                    na_rep="N/A"
                ),
                use_container_width=True,
                hide_index=True
            )

        st.subheader("Time-Period Robustness")
        if robustness_yearly_table.empty:
            st.info("Yearly robustness is unavailable because the holdout period has no completed trades.")
        else:
            st.dataframe(
                robustness_period_table.style.format(
                    {
                        "Average Return": "{:.2%}",
                        "Win Rate": "{:.2%}",
                        "Best Trade": "{:.2%}",
                        "Worst Trade": "{:.2%}"
                    },
                    na_rep="N/A"
                ),
                use_container_width=True,
                hide_index=True
            )
            st.write("Yearly strategy performance")
            st.dataframe(
                robustness_yearly_table.style.format({"Average Return": "{:.2%}"}),
                use_container_width=True,
                hide_index=True
            )
            yearly_chart_data = add_percent_display_column(
                robustness_yearly_table,
                "Average Return",
                "Average Return (%)"
            )
            fig_yearly_returns = px.bar(
                yearly_chart_data,
                x="Year",
                y="Average Return (%)",
                title="Yearly Strategy Returns",
                text=yearly_chart_data["Average Return (%)"].map(lambda value: f"{value:.2f}%")
            )
            fig_yearly_returns.update_traces(textposition="outside")
            fig_yearly_returns.update_yaxes(ticksuffix="%", tickformat=".2f")
            st.plotly_chart(fig_yearly_returns, use_container_width=True)

    # -----------------------------
    # AI analyst summary
    # -----------------------------
    with ai_summary_tab:
        st.header("AI Analyst Summary")

        if latest_probabilities.empty:
            st.info("AI Analyst Summary is unavailable because the Random Forest models did not produce probabilities.")
        else:
            analyst_signal_table = latest_probabilities.merge(
                rf_metrics[["Asset", "Accuracy", "Model Edge"]],
                on="Asset",
                how="left"
            )
            analyst_signal_table["Signal Strength"] = analyst_signal_table[
                "Latest Bullish Probability"
            ].apply(label_signal_strength)
            top_signal = analyst_signal_table.sort_values(
                "Latest Bullish Probability",
                ascending=False
            ).iloc[0]
            top_etf = top_signal["Asset"]
            signal_label = top_signal["Signal Strength"]
            signal_level = summarize_signal_strength(signal_label)
            ranked_signals = analyst_signal_table.sort_values(
                "Latest Bullish Probability",
                ascending=False
            ).reset_index(drop=True)
            second_probability = (
                ranked_signals.loc[1, "Latest Bullish Probability"]
                if len(ranked_signals) > 1
                else None
            )
            probability_spread = (
                top_signal["Latest Bullish Probability"] - second_probability
                if second_probability is not None
                else None
            )
            probabilities_are_close = probability_spread is not None and probability_spread < 0.05
            top_features = rf_importances.get(top_etf, pd.DataFrame()).head(3)
            feature_text = (
                ", ".join(top_features["Feature"].tolist())
                if not top_features.empty
                else "feature importances are unavailable"
            )
            cluster_driver_text = format_feature_drivers(current_cluster_drivers)

            if rf_backtest_summary.empty:
                backtest_edge_is_weak = True
            else:
                summary_lookup = rf_backtest_summary.set_index("Metric")["Value"]
                average_return = summary_lookup.get("Average 20D Return")
                win_rate = summary_lookup.get("Win Rate")
                backtest_edge_is_weak = (
                    pd.notna(average_return) and average_return <= 0
                ) or (
                    pd.notna(win_rate) and win_rate < 0.50
                )

            if rf_backtest_risk_metrics.empty:
                strategy_cumulative_return = None
            else:
                strategy_risk = rf_backtest_risk_metrics[
                    rf_backtest_risk_metrics["Asset"] == "RF Strategy"
                ].iloc[0]
                strategy_cumulative_return = strategy_risk["Cumulative Return"]

            weak_model_edge = pd.notna(top_signal["Model Edge"]) and top_signal["Model Edge"] <= 0.02
            weak_historical_edge = (
                backtest_edge_is_weak
                or (pd.notna(strategy_cumulative_return) and strategy_cumulative_return <= 0)
            )

            if not probabilities_are_close and signal_level in ["moderate", "strong"]:
                strategy_view = (
                    f"The dashboard currently favors {top_etf} for the 20D horizon because it has the highest "
                    f"bullish probability and the signal strength is {signal_level}."
                )
            else:
                strategy_view = (
                    "The dashboard favors a cautious equal-weight or cash-like approach because the ETF "
                    "probabilities are close together or the strongest signal is weak."
                )

            if weak_model_edge or weak_historical_edge:
                reliability_warning = (
                    " However, the backtest/model edge is weak, so the signal is not reliable enough to treat "
                    "as a standalone allocation rule."
                )
            else:
                reliability_warning = ""

            st.markdown(
                f"""
**Current Regime:** Cluster {current_cluster} - {current_regime_label}

**Why This Cluster:** The strongest current-cluster profile drivers are: {cluster_driver_text}.

**Model Signal:** The model currently favors {top_etf}. Its bullish probability is
{top_signal['Latest Bullish Probability']:.2%}, which is a {signal_level} signal using the
"{signal_label}" label.

**Key Features:** The top Random Forest features for {top_etf} are: {feature_text}.

**Interpretation:** {strategy_view}{reliability_warning}
"""
            )

        st.subheader("Model Action")
        if current_model_signal is None:
            st.info("Model action is unavailable because the Random Forest models did not produce probabilities.")
        else:
            action_probability = current_model_signal["Latest Bullish Probability"]
            action_signal_strength = current_model_signal["Signal Strength"]
            action_etf = current_model_signal["Asset"]

            st.markdown(action_badge(f"Model Action: {current_model_action['label']}"), unsafe_allow_html=True)
            action_cols = st.columns(5)
            action_cols[0].markdown(
                colored_metric("Selected ETF", action_etf, "gray"),
                unsafe_allow_html=True
            )
            action_cols[1].markdown(
                colored_metric("Action", current_model_action["action"].upper(), current_model_action["color"]),
                unsafe_allow_html=True
            )
            action_cols[2].markdown(
                colored_metric(
                    "Bullish Probability",
                    format_pct(action_probability),
                    probability_color(action_probability)
                ),
                unsafe_allow_html=True
            )
            action_cols[3].markdown(
                colored_metric("Signal Strength", action_signal_strength, signal_color(action_signal_strength)),
                unsafe_allow_html=True
            )
            action_cols[4].markdown(
                colored_metric("Macro Regime", f"Cluster {current_cluster}", "gray"),
                unsafe_allow_html=True
            )
            st.caption(f"Reason: {current_model_action['reason']}")

        interpretation_guide = pd.DataFrame({
            "Pattern You See": [
                "High VIX, rising unemployment, widening BAA spread",
                "Rising 2Y/10Y/20Y yields, rising CPI/Core CPI",
                "Falling yields, low VIX, falling inflation pressure",
                "Low VIX, stable yields, improving macro data",
                "Yield curve deeply negative, unemployment starting to rise"
            ],
            "Possible Regime Label": [
                "Recession Fear / Risk-Off Regime",
                "Inflation Pressure / Rising-Rate Regime",
                "Falling-Rate / Bond-Friendly Regime",
                "Risk-On / Stable Growth Regime",
                "Late-Cycle / Mixed Macro Regime"
            ],
            "Assets That May Matter": [
                "TLT, GLD, XLP, XLV",
                "XLE, DXY, short-duration bonds",
                "TLT, QQQ, growth stocks",
                "SPY, QQQ, IWM, cyclicals",
                "TLT, defensive sectors, cash"
            ]
        })

        st.subheader("How to Read the Dashboard")
        st.write(
            f"The current observation is assigned to Cluster {current_cluster}. Start with the cluster profile, "
            "then compare the historical ETF return ranking with the Random Forest probabilities."
        )

        if not current_cluster_asset_table.empty:
            st.write("Current cluster historical ETF ranking:")
            st.dataframe(
                current_cluster_asset_table.style.format(
                    {
                        "Average 20D Forward Return": "{:.2%}",
                        "Median 20D Forward Return": "{:.2%}",
                        "Win Rate": "{:.2%}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

        st.subheader("Manual Regime Label Guide")
        st.dataframe(interpretation_guide, use_container_width=True, hide_index=True)

        st.success(
            "Macro regime clustering, ETF return analysis, and Random Forest probability models are ready."
        )

    st.caption("This dashboard is a research tool, not financial advice.")
else:
    for unavailable_tab in [
        overview_tab,
        macro_tab,
        kmeans_tab,
        returns_tab,
        rf_tab,
        backtest_tab,
        robustness_tab,
        ai_summary_tab
    ]:
        with unavailable_tab:
            st.info("Enter a FRED API key in the sidebar to load macro data, regimes, return analysis, and ML models.")
    st.caption("This dashboard is a research tool, not financial advice.")
