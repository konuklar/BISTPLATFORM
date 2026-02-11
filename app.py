# ============================================================================
# BIST / Quantum BIST Platform Portfolio Analytics (Cloud-Safe, Truth-First)
# - Quantum BIST Platform only (yfinance)
# - No TensorFlow (Streamlit Cloud-safe)
# - Strong data parsing (MultiIndex-safe)
# - Strict validity checks (no silent shape mismatches)
# - Execute button gating (no crash-on-load)
# - Portfolio analytics + VaR/CVaR + Optimization (PyPortfolioOpt)
# ============================================================================
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="BIST Portfolio Terminal (Quantum BIST Platform)",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TICKERS = [
    "XU100.IS", "THYAO.IS", "ASELS.IS", "KCHOL.IS", "BIMAS.IS",
    "GARAN.IS", "AKBNK.IS", "SISE.IS", "TUPRS.IS", "SAHOL.IS"
]

RISK_FREE_RATES = {
    "TRY": 0.35,   # Default placeholder; adjust as needed for your committee-grade work
    "USD": 0.045,
    "EUR": 0.03,
}
DEFAULT_RFR = float(RISK_FREE_RATES.get("TRY", 0.0))
DEFAULT_RFR = float(np.clip(DEFAULT_RFR, 0.0, 0.50))


@dataclass
class AppState:
    run: bool = False
    last_fetch_ok: bool = False
    last_error: str = ""


def _init_state():
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
    if "ui_disable_css" not in st.session_state:
        st.session_state["ui_disable_css"] = True  # SAFE UI by default


_init_state()


def _apply_safe_css(disable: bool = True) -> None:
    """Minimal CSS to keep sidebar readable. No risky theme overrides."""
    if disable:
        return
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { min-width: 370px; max-width: 370px; }
        .stSidebar .stMarkdown, .stSidebar label, .stSidebar span { font-size: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _is_intraday(interval: str) -> bool:
    return interval in {"1h", "30m", "15m", "5m"}


def _periods_per_year(interval: str) -> int:
    """
    Conservative periods/year mapping.
    For daily: 252 trading days.
    For intraday, we approximate 7 trading hours/day for BIST (adjust if needed).
    """
    if interval == "1d":
        return 252
    if interval == "1h":
        return 252 * 7
    if interval == "30m":
        return 252 * 7 * 2
    if interval == "15m":
        return 252 * 7 * 4
    if interval == "5m":
        return 252 * 7 * 12
    return 252


# =========================
# DATA FETCHING (YFINANCE)
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_yahoo_data(
    tickers: List[str],
    start: datetime,
    end: datetime,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Quantum BIST Platform via yfinance.
    Returns a DataFrame of prices (Close or Adj Close) with columns = tickers.
    This function is MultiIndex-safe for yfinance output.
    """
    if not tickers:
        raise ValueError("No tickers selected.")

    # Yahoo has intraday history limits; keep it realistic.
    # We don't silently reshape; we fail with a meaningful error.
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if data is None or len(data) == 0:
        raise ValueError("Quantum BIST Platform returned empty data. Check tickers / date range / interval.")

    # Identify which field to use
    # If auto_adjust=True, 'Close' is effectively adjusted.
    preferred_field = "Close"
    fallback_field = "Adj Close"

    # Single ticker may return flat columns: Open High Low Close Volume Adj Close (maybe).
    if not isinstance(data.columns, pd.MultiIndex):
        # choose Close or Adj Close depending on availability
        if preferred_field in data.columns:
            px_series = data[preferred_field].copy()
        elif fallback_field in data.columns:
            px_series = data[fallback_field].copy()
        else:
            raise ValueError(f"Price field not found in Yahoo response columns: {list(data.columns)}")
        px_df = pd.DataFrame({tickers[0]: px_series})
        px_df.index = pd.to_datetime(px_df.index)
        return px_df.sort_index()

    # MultiIndex: could be (ticker, field) or (field, ticker)
    lvl0 = list(map(str, data.columns.get_level_values(0).unique()))
    lvl1 = list(map(str, data.columns.get_level_values(1).unique()))

    tickers_set = set(map(str, tickers))
    lvl0_has_tickers = any(x in tickers_set for x in lvl0)
    lvl1_has_tickers = any(x in tickers_set for x in lvl1)

    def _extract_price_from_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        missing = []
        for t in tickers:
            t = str(t)
            if lvl0_has_tickers:
                # columns = (ticker, field)
                if (t, preferred_field) in df.columns:
                    out[t] = df[(t, preferred_field)]
                elif (t, fallback_field) in df.columns:
                    out[t] = df[(t, fallback_field)]
                else:
                    missing.append(t)
            else:
                # columns = (field, ticker)
                if (preferred_field, t) in df.columns:
                    out[t] = df[(preferred_field, t)]
                elif (fallback_field, t) in df.columns:
                    out[t] = df[(fallback_field, t)]
                else:
                    missing.append(t)

        if missing:
            raise ValueError(f"Missing price data for tickers: {missing}. Try different interval/date range.")
        px = pd.DataFrame(out)
        px.index = pd.to_datetime(px.index)
        return px

    px_df = _extract_price_from_multiindex(data)
    px_df = px_df.sort_index()
    return px_df


def validate_prices(
    prices: pd.DataFrame,
    min_rows: int = 60,
    allow_fill: str = "intersection",
) -> pd.DataFrame:
    """
    Validates and cleans price data.
    allow_fill:
      - 'intersection': drop rows with any missing
      - 'ffill': forward fill
      - 'ffill_bfill': forward fill then backfill
    """
    if prices is None or prices.empty:
        raise ValueError("Prices dataframe is empty.")

    # Drop columns that are fully NA
    prices = prices.dropna(axis=1, how="all")
    if prices.shape[1] == 0:
        raise ValueError("All selected tickers have empty prices (all-NA).")

    # Ensure monotonic datetime index
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.sort_index()

    # Handle missing
    if allow_fill == "intersection":
        cleaned = prices.dropna(axis=0, how="any")
    elif allow_fill == "ffill":
        cleaned = prices.ffill()
        cleaned = cleaned.dropna(axis=0, how="any")
    elif allow_fill == "ffill_bfill":
        cleaned = prices.ffill().bfill()
        cleaned = cleaned.dropna(axis=0, how="any")
    else:
        raise ValueError(f"Unknown missing-data handling method: {allow_fill}")

    if cleaned.shape[0] < min_rows:
        raise ValueError(
            f"Not enough valid rows after cleaning: {cleaned.shape[0]} < {min_rows}. "
            "Try longer date range, different interval, or fewer tickers."
        )

    # Ensure positive prices
    if (cleaned <= 0).any().any():
        raise ValueError("Non-positive prices detected after cleaning. Check data integrity.")

    return cleaned


def compute_returns(prices: pd.DataFrame, return_type: str) -> pd.DataFrame:
    """
    Computes returns from clean prices.
    Ensures no shape mismatch by using the return index directly.
    """
    if return_type == "simple":
        rets = prices.pct_change()
    elif return_type == "log":
        rets = np.log(prices).diff()
    else:
        raise ValueError("return_type must be 'simple' or 'log'.")

    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if rets.empty:
        raise ValueError("Returns are empty after computation. Possibly too many missing values.")
    return rets


# =========================
# METRICS / RISK
# =========================
def annualize_return(returns: pd.Series, freq: int) -> float:
    return float((1 + returns.mean()) ** freq - 1) if np.isfinite(returns.mean()) else np.nan


def annualize_vol(returns: pd.Series, freq: int) -> float:
    return float(returns.std(ddof=1) * math.sqrt(freq)) if np.isfinite(returns.std(ddof=1)) else np.nan


def sharpe_ratio(returns: pd.Series, rfr_annual: float, freq: int) -> float:
    rfr_period = (1 + rfr_annual) ** (1 / freq) - 1
    excess = returns - rfr_period
    denom = excess.std(ddof=1)
    if denom is None or denom == 0 or not np.isfinite(denom):
        return np.nan
    return float((excess.mean() / denom) * math.sqrt(freq))


def sortino_ratio(returns: pd.Series, rfr_annual: float, freq: int) -> float:
    rfr_period = (1 + rfr_annual) ** (1 / freq) - 1
    excess = returns - rfr_period
    downside = excess[excess < 0]
    denom = downside.std(ddof=1)
    if denom is None or denom == 0 or not np.isfinite(denom):
        return np.nan
    return float((excess.mean() / denom) * math.sqrt(freq))


def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())


def historical_var_cvar(returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Historical VaR and CVaR (Expected Shortfall) at level alpha (e.g., 0.05 => 95% VaR).
    Returns are in return space (negative for losses).
    """
    x = returns.dropna().values
    if len(x) < 50:
        return np.nan, np.nan
    var = np.quantile(x, alpha)
    tail = x[x <= var]
    cvar = tail.mean() if len(tail) else np.nan
    return float(var), float(cvar)


def portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    pr = returns_df.values @ w
    return pd.Series(pr, index=returns_df.index, name="Portfolio")


def metrics_table(returns_df: pd.DataFrame, rfr_annual: float, freq: int, alpha: float) -> pd.DataFrame:
    rows = []
    for col in returns_df.columns:
        s = returns_df[col].dropna()
        cum = (1 + s).cumprod()
        r = annualize_return(s, freq)
        v = annualize_vol(s, freq)
        sh = sharpe_ratio(s, rfr_annual, freq)
        so = sortino_ratio(s, rfr_annual, freq)
        mdd = max_drawdown(cum)
        var, cvar = historical_var_cvar(s, alpha=alpha)
        rows.append([col, r, v, sh, so, mdd, var, cvar])

    df = pd.DataFrame(
        rows,
        columns=["Asset", "Ann.Return", "Ann.Vol", "Sharpe", "Sortino", "MaxDD", f"VaR({int((1-alpha)*100)}%)", f"CVaR({int((1-alpha)*100)}%)"],
    ).set_index("Asset")
    return df


# =========================
# OPTIMIZATION (PYPFOPT)
# =========================
def try_optimize_portfolio(
    prices: pd.DataFrame,
    method: str,
    rfr_annual: float,
    weight_bounds: Tuple[float, float],
) -> Tuple[Optional[pd.Series], str]:
    """
    Attempts PyPortfolioOpt optimization. Returns (weights_series, message).
    """
    try:
        from pypfopt import expected_returns, risk_models
        from pypfopt.efficient_frontier import EfficientFrontier
    except Exception as e:
        return None, f"PyPortfolioOpt not available or failed to import: {e}"

    try:
        mu = expected_returns.mean_historical_return(prices, frequency=252)  # we use 252 baseline for annual mu
        S = risk_models.sample_cov(prices, frequency=252)

        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

        if method == "Max Sharpe":
            ef.max_sharpe(risk_free_rate=rfr_annual)
        elif method == "Min Volatility":
            ef.min_volatility()
        elif method == "Max Quadratic Utility":
            ef.max_quadratic_utility(risk_aversion=1.0)
        else:
            # default
            ef.min_volatility()

        w = ef.clean_weights()
        ws = pd.Series(w).reindex(prices.columns).fillna(0.0)
        return ws, "Optimization successful."
    except Exception as e:
        return None, f"Optimization failed: {e}"


# =========================
# PLOTS
# =========================
def plot_cumulative(returns_df: pd.DataFrame, title: str) -> go.Figure:
    cum = (1 + returns_df).cumprod()
    fig = go.Figure()
    for col in cum.columns:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[col], mode="lines", name=str(col)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Growth of $1", height=420, legend_title_text="")
    return fig


def plot_corr(returns_df: pd.DataFrame, title: str) -> go.Figure:
    corr = returns_df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title=title,
    )
    fig.update_layout(height=520)
    return fig


def plot_weights(weights: pd.Series, title: str) -> go.Figure:
    w = weights[weights > 0].sort_values(ascending=False)
    fig = go.Figure([go.Bar(x=w.index.astype(str), y=w.values)])
    fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Weight", height=360)
    return fig


# =========================
# UI
# =========================
def sidebar_controls() -> Dict:
    st.sidebar.title("⚙️ Control Panel")

    st.sidebar.checkbox(
        "Disable custom UI tweaks (recommended if sidebar looks odd)",
        value=st.session_state["ui_disable_css"],
        key="ui_disable_css",
    )
    _apply_safe_css(disable=st.session_state["ui_disable_css"])

    st.sidebar.markdown("### 📌 Data Source")
    st.sidebar.info("Quantum BIST Platform (Quantum")

    st.sidebar.markdown("### 🧾 Universe")
    tickers = st.sidebar.multiselect(
        "Tickers",
        options=sorted(list(set(DEFAULT_TICKERS))),
        default=DEFAULT_TICKERS[:6],
        help="Use Quantum BIST Platform tickers (BIST usually ends with .IS)",
    )

    today = date.today()
    default_start = today - timedelta(days=365 * 2)

    st.sidebar.markdown("### 📅 Date Range")
    start_date = st.sidebar.date_input("Start", value=default_start)
    end_date = st.sidebar.date_input("End", value=today)

    st.sidebar.markdown("### ⏱ Interval & Returns")
    interval = st.sidebar.selectbox("Interval", options=["1d", "1h", "30m", "15m", "5m"], index=0)
    return_type = st.sidebar.selectbox("Return Type", options=["simple", "log"], index=0)
    auto_adjust = st.sidebar.checkbox("auto_adjust (recommended)", value=True)

    st.sidebar.markdown("### 🧹 Missing Data Handling")
    align_method = st.sidebar.selectbox(
        "Method",
        options=["intersection", "ffill", "ffill_bfill"],
        index=0,
        help="intersection = drop any row with missing values (truth-first).",
    )

    st.sidebar.markdown("### 💼 Risk Parameters")
    currency = st.sidebar.selectbox("Risk-free currency", options=["TRY", "USD", "EUR"], index=0)
    rfr_default = float(np.clip(float(RISK_FREE_RATES.get(currency, DEFAULT_RFR)), 0.0, 0.50))
    rfr_pct = st.sidebar.slider("Risk-free rate (annual, %)", min_value=0.0, max_value=50.0, value=rfr_default * 100.0, step=0.25)
    rfr_annual = rfr_pct / 100.0

    alpha = st.sidebar.slider("VaR confidence", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    var_alpha = 1 - alpha  # tail probability

    st.sidebar.markdown("### 🧠 Optimization")
    opt_enabled = st.sidebar.checkbox("Enable Optimization (PyPortfolioOpt)", value=True)
    opt_method = st.sidebar.selectbox("Optimization objective", options=["Max Sharpe", "Min Volatility", "Max Quadratic Utility"], index=0)
    long_only = st.sidebar.checkbox("Long-only", value=True)
    if long_only:
        lb, ub = 0.0, 1.0
    else:
        lb, ub = st.sidebar.slider("Weight lower bound", -1.0, 0.0, -0.2, 0.05), st.sidebar.slider("Weight upper bound", 0.0, 2.0, 1.0, 0.05)

    st.sidebar.markdown("---")
    colA, colB = st.sidebar.columns(2)
    with colA:
        execute = st.button("🚀 Execute", type="primary", use_container_width=True)
    with colB:
        reset = st.button("⟲ Reset", use_container_width=True)

    if reset:
        st.session_state["app_state"].run = False
        st.session_state["app_state"].last_fetch_ok = False
        st.session_state["app_state"].last_error = ""

    if execute:
        st.session_state["app_state"].run = True

    return {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "return_type": return_type,
        "auto_adjust": auto_adjust,
        "align_method": align_method,
        "rfr_annual": rfr_annual,
        "currency": currency,
        "var_alpha": var_alpha,
        "opt_enabled": opt_enabled,
        "opt_method": opt_method,
        "weight_bounds": (float(lb), float(ub)) if isinstance(lb, (int, float)) else (0.0, 1.0),
    }


def main():
    cfg = sidebar_controls()
    st.title("🏛️ BIST Portfolio Terminal — Quantum BIST Platform Edition")

    # Redundant Execute button on main page (visibility guarantee)
    st.markdown("### Run Control")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🚀 Execute Analysis", type="primary", use_container_width=True):
            st.session_state["app_state"].run = True
    with c2:
        if st.button("⟲ Reset", use_container_width=True):
            st.session_state["app_state"].run = False
            st.session_state["app_state"].last_fetch_ok = False
            st.session_state["app_state"].last_error = ""
    with c3:
        st.caption("Tip: If sidebar looks unreadable, keep 'Disable custom UI tweaks' ON.")

    if not st.session_state["app_state"].run:
        st.info("Select parameters in the left Control Panel, then click **🚀 Execute**.")
        return

    # Validate date range
    start_dt = datetime.combine(cfg["start_date"], datetime.min.time())
    end_dt = datetime.combine(cfg["end_date"], datetime.min.time()) + timedelta(days=1)

    if end_dt <= start_dt:
        st.error("Invalid date range: End must be after Start.")
        return

    if _is_intraday(cfg["interval"]):
        max_days = 58  # conservative
        if (end_dt - start_dt).days > max_days:
            st.warning(
                f"Intraday interval '{cfg['interval']}' has limited history on Yahoo. "
                f"Your range is {(end_dt - start_dt).days} days. Consider using 1d or shorten range."
            )

    if not cfg["tickers"]:
        st.error("Please select at least one ticker.")
        return

    # Fetch + validate + returns
    with st.spinner("Fetching Quantum BIST Platform data (yfinance)..."):
        try:
            raw_prices = fetch_yahoo_data(
                tickers=cfg["tickers"],
                start=start_dt,
                end=end_dt,
                interval=cfg["interval"],
                auto_adjust=cfg["auto_adjust"],
            )
            prices = validate_prices(raw_prices, min_rows=60, allow_fill=cfg["align_method"])
            rets = compute_returns(prices, return_type=cfg["return_type"])
            st.session_state["app_state"].last_fetch_ok = True
            st.session_state["app_state"].last_error = ""
        except Exception as e:
            st.session_state["app_state"].last_fetch_ok = False
            st.session_state["app_state"].last_error = str(e)
            st.error(f"Data loading failed: {e}")
            st.stop()

    # Display data summary
    st.markdown("## ✅ Data Quality & Summary")
    left, right = st.columns([1.2, 1])
    with left:
        st.write("**Prices (head)**")
        st.dataframe(prices.head(10), use_container_width=True)
    with right:
        st.write("**Validity checks**")
        st.metric("Rows (prices)", int(prices.shape[0]))
        st.metric("Rows (returns)", int(rets.shape[0]))
        st.metric("Assets", int(prices.shape[1]))
        st.metric("Interval", cfg["interval"])
        st.metric("Return Type", cfg["return_type"])

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Risk & VaR", "🔗 Correlations", "🧠 Optimization"])

    freq = _periods_per_year(cfg["interval"])

    with tab1:
        st.subheader("Cumulative Performance")
        fig = plot_cumulative(rets, "Assets: Cumulative Growth (based on returns)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Portfolio (User Weights)")
        st.caption("Set weights below (auto-normalized). This is *not* the optimized portfolio.")
        cols = st.columns(min(4, len(rets.columns)))
        w = []
        for i, asset in enumerate(rets.columns):
            with cols[i % len(cols)]:
                w.append(st.number_input(f"{asset} weight", min_value=0.0, max_value=1.0, value=round(1.0 / len(rets.columns), 4), step=0.01))
        w = np.array(w, dtype=float)
        if w.sum() <= 0:
            st.warning("Weights sum to zero. Please set weights > 0.")
        else:
            w = w / w.sum()
            port = portfolio_returns(rets, w)
            port_df = pd.concat([rets, port], axis=1)
            figp = plot_cumulative(port_df[[port.name]], "User Portfolio: Cumulative Growth")
            st.plotly_chart(figp, use_container_width=True)

    with tab2:
        st.subheader("Risk Metrics (Annualized + VaR/CVaR)")
        mt = metrics_table(rets, rfr_annual=cfg["rfr_annual"], freq=freq, alpha=cfg["var_alpha"])
        st.dataframe(mt.style.format("{:.4f}"), use_container_width=True)

        st.caption("VaR/CVaR are historical tail estimates from the selected return series (truth-first).")

    with tab3:
        st.subheader("Correlation Matrix")
        st.plotly_chart(plot_corr(rets, "Return Correlations"), use_container_width=True)

    with tab4:
        st.subheader("PyPortfolioOpt Optimization")
        if not cfg["opt_enabled"]:
            st.info("Enable optimization in the sidebar to run PyPortfolioOpt.")
        else:
            w_opt, msg = try_optimize_portfolio(
                prices=prices,
                method=cfg["opt_method"],
                rfr_annual=cfg["rfr_annual"],
                weight_bounds=cfg["weight_bounds"],
            )
            st.write(msg)
            if w_opt is None:
                st.warning("Optimization did not produce weights. Check dependencies / constraints.")
            else:
                st.write("**Optimized weights**")
                st.dataframe(w_opt.to_frame("weight").style.format("{:.4f}"), use_container_width=True)
                st.plotly_chart(plot_weights(w_opt, f"Optimized Weights — {cfg['opt_method']}"), use_container_width=True)

                # Optimized portfolio performance
                port_opt = portfolio_returns(rets, w_opt.values)
                fig_opt = plot_cumulative(pd.DataFrame({port_opt.name: port_opt}), "Optimized Portfolio: Cumulative Growth")
                st.plotly_chart(fig_opt, use_container_width=True)

    st.markdown("---")
    st.caption("If you still see a data error, send the exact ticker list + interval + date range so we can reproduce it precisely.")


if __name__ == "__main__":
    main()
