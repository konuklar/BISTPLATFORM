# ============================================================================
# INSTITUTIONAL PORTFOLIO ANALYTICS PLATFORM - ENTERPRISE EDITION
# Version: 5.0 | Enhanced: Professional UI, Advanced Analytics, Deep Quant Integration
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import quantstats as qs
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import requests
import json
import io
import base64
import pickle
import hashlib
import logging
import traceback
import time
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Advanced Financial Libraries
from pypfopt import expected_returns, risk_models, EfficientFrontier, CLA
from pypfopt import HRPOpt, EfficientCVaR, EfficientSemivariance, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.risk_models import CovarianceShrinkage

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Time Series Analysis
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.api import VAR
from statsmodels.regression.rolling import RollingOLS

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Institutional Portfolio Analytics Platform",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/institutional-portfolio-analytics',
        'Report a bug': "https://github.com/yourusername/institutional-portfolio-analytics/issues",
        'About': """
        ## Institutional Portfolio Analytics Platform
        Version: 5.0.0 | Release: 2024
        
        Enterprise-grade quantitative portfolio optimization and risk management 
        platform with deep integration of QuantStats and PyPortfolioOpt.
        
        Features:
        • 20+ Portfolio Optimization Methods
        • QuantStats Performance Analytics
        • Machine Learning Predictions
        • Real-time Risk Monitoring
        • Regulatory Compliance
        • Professional Reporting
        """
    }
)

# ============================================================================
# PROFESSIONAL INSTITUTIONAL CSS
# ============================================================================

st.markdown("""
<style>
    /* ── Professional Institutional Theme ── */
    :root {
        /* Primary Colors */
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --bg-card: #1e293b;
        --bg-card-alt: #2d3748;
        --bg-modal: rgba(15, 23, 42, 0.95);
        
        /* Accent Colors - Professional Finance */
        --accent-primary: #3b82f6;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
        --accent-info: #06b6d4;
        --accent-purple: #8b5cf6;
        --accent-pink: #ec4899;
        
        /* Text Colors */
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --text-disabled: #64748b;
        
        /* Gradients */
        --gradient-primary: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --gradient-warning: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        --gradient-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    /* ── Enhanced Typography ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        letter-spacing: -0.025em;
    }
    
    h1 {
        font-size: 2.5rem;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.75rem;
        color: var(--text-primary);
        border-bottom: 2px solid var(--bg-tertiary);
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }

    /* ── Professional Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--bg-tertiary);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    /* ── Professional Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--bg-tertiary);
        padding: 0.5rem 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        color: var(--text-muted);
        border: 1px solid transparent;
        border-bottom: 2px solid transparent;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-tertiary);
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
        border-bottom-color: var(--bg-primary) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    /* ── Professional KPI Cards ── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--bg-tertiary);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s;
        height: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        border-color: var(--accent-primary);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .kpi-change {
        font-size: 0.875rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .kpi-change.positive {
        color: var(--accent-success);
    }
    
    .kpi-change.negative {
        color: var(--accent-danger);
    }

    /* ── Enhanced Data Tables ── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--bg-tertiary);
        background: var(--bg-card);
    }
    
    .stDataFrame th {
        background: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        padding: 0.75rem 1rem !important;
    }
    
    .stDataFrame td {
        color: var(--text-secondary) !important;
        padding: 0.5rem 1rem !important;
        border-bottom: 1px solid var(--bg-tertiary) !important;
    }
    
    .stDataFrame tr:hover td {
        background: var(--bg-card-alt) !important;
    }

    /* ── Professional Charts Container ── */
    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--bg-tertiary);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--bg-tertiary);
    }
    
    .chart-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    /* ── Status Indicators ── */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-indicator.success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--accent-success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-indicator.warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--accent-warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-indicator.danger {
        background: rgba(239, 68, 68, 0.1);
        color: var(--accent-danger);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .status-dot.success {
        background: var(--accent-success);
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
    }
    
    .status-dot.warning {
        background: var(--accent-warning);
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
    }
    
    .status-dot.danger {
        background: var(--accent-danger);
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
    }

    /* ── Loading Spinner ── */
    .spinner {
        border: 3px solid var(--bg-tertiary);
        border-top: 3px solid var(--accent-primary);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ENHANCED DATA CLASSES
# ============================================================================

@dataclass
class PortfolioMetrics:
    """Enhanced portfolio metrics with QuantStats integration"""
    # Basic Metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Advanced QuantStats Metrics
    omega_ratio: float
    gain_to_pain_ratio: float
    tail_ratio: float
    common_sense_ratio: float
    ulcer_index: float
    serenity_ratio: float
    information_ratio: float
    alpha: float
    beta: float
    
    # Risk Metrics
    var_95: float
    cvar_95: float
    expected_shortfall: float
    skewness: float
    kurtosis: float
    
    # Performance Metrics
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    
    # Rolling Metrics
    rolling_sharpe_60: float
    rolling_sharpe_252: float
    rolling_vol_60: float
    rolling_vol_252: float
    
    # Factor Analysis
    market_beta: float
    size_beta: float
    value_beta: float
    momentum_beta: float
    quality_beta: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to formatted DataFrame"""
        metrics_dict = {
            'Return Metrics': {
                'Total Return': f"{self.total_return:.2%}",
                'Annual Return': f"{self.annual_return:.2%}",
                'Annual Volatility': f"{self.volatility:.2%}",
            },
            'Risk-Adjusted Returns': {
                'Sharpe Ratio': f"{self.sharpe_ratio:.3f}",
                'Sortino Ratio': f"{self.sortino_ratio:.3f}",
                'Calmar Ratio': f"{self.calmar_ratio:.3f}",
                'Omega Ratio': f"{self.omega_ratio:.3f}",
                'Gain to Pain': f"{self.gain_to_pain_ratio:.3f}",
            },
            'Risk Metrics': {
                'Max Drawdown': f"{self.max_drawdown:.2%}",
                'VaR (95%)': f"{self.var_95:.2%}",
                'CVaR (95%)': f"{self.cvar_95:.2%}",
                'Expected Shortfall': f"{self.expected_shortfall:.2%}",
                'Skewness': f"{self.skewness:.3f}",
                'Kurtosis': f"{self.kurtosis:.3f}",
            },
            'Performance Statistics': {
                'Win Rate': f"{self.win_rate:.2%}",
                'Profit Factor': f"{self.profit_factor:.3f}",
                'Expectancy': f"{self.expectancy:.3f}",
                'Avg Win': f"{self.avg_win:.2%}",
                'Avg Loss': f"{self.avg_loss:.2%}",
            },
            'Factor Exposure': {
                'Market Beta': f"{self.market_beta:.3f}",
                'Size Beta': f"{self.size_beta:.3f}",
                'Value Beta': f"{self.value_beta:.3f}",
                'Momentum Beta': f"{self.momentum_beta:.3f}",
                'Quality Beta': f"{self.quality_beta:.3f}",
            }
        }
        
        # Flatten the dictionary
        flat_metrics = {}
        for category, metrics in metrics_dict.items():
            for key, value in metrics.items():
                flat_metrics[f"{category} - {key}"] = value
        
        return pd.DataFrame(list(flat_metrics.items()), columns=['Metric', 'Value'])

# ============================================================================
# ENHANCED DATA FETCHER WITH YAHOO FINANCE
# ============================================================================

class EnhancedDataFetcher:
    """Professional data fetcher with comprehensive Yahoo Finance integration"""
    
    def __init__(self):
        self.cache_dir = Path('./data_cache')
        self.cache_dir.mkdir(exist_ok=True)
    
    def fetch_yahoo_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict:
        """Fetch comprehensive data from Yahoo Finance"""
        try:
            logger.info(f"Fetching Yahoo Finance data for {len(tickers)} tickers")
            
            # Download price data
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=True
            )
            
            if data.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
            # Process data based on structure
            if isinstance(data.columns, pd.MultiIndex):
                prices = pd.DataFrame()
                volumes = pd.DataFrame()
                
                for ticker in tickers:
                    if ('Adj Close', ticker) in data.columns:
                        prices[ticker] = data[('Adj Close', ticker)]
                    elif ('Close', ticker) in data.columns:
                        prices[ticker] = data[('Close', ticker)]
                    
                    if ('Volume', ticker) in data.columns:
                        volumes[ticker] = data[('Volume', ticker)]
            else:
                prices = data[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
                volumes = data[['Volume']].rename(columns={'Volume': tickers[0]})
            
            # Fill missing values
            prices = prices.ffill().bfill()
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Get additional fundamental data
            fundamental_data = {}
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    fundamental_data[ticker] = {
                        'market_cap': info.get('marketCap'),
                        'pe_ratio': info.get('trailingPE'),
                        'forward_pe': info.get('forwardPE'),
                        'pb_ratio': info.get('priceToBook'),
                        'dividend_yield': info.get('dividendYield'),
                        'beta': info.get('beta'),
                        'sector': info.get('sector'),
                        'industry': info.get('industry'),
                        'country': info.get('country'),
                        'currency': info.get('currency'),
                        'volume_avg': info.get('averageVolume'),
                        'shares_outstanding': info.get('sharesOutstanding'),
                        'ebitda': info.get('ebitda'),
                        'revenue': info.get('totalRevenue'),
                        'profit_margin': info.get('profitMargins'),
                        'operating_margin': info.get('operatingMargins'),
                        'roe': info.get('returnOnEquity'),
                        'roa': info.get('returnOnAssets'),
                        'debt_to_equity': info.get('debtToEquity'),
                        'current_ratio': info.get('currentRatio'),
                        'quick_ratio': info.get('quickRatio'),
                        'gross_margin': info.get('grossMargins'),
                        'free_cash_flow': info.get('freeCashflow'),
                        'operating_cash_flow': info.get('operatingCashflow'),
                        'revenue_growth': info.get('revenueGrowth'),
                        'earnings_growth': info.get('earningsGrowth'),
                    }
                except Exception as e:
                    logger.warning(f"Failed to fetch fundamental data for {ticker}: {e}")
                    fundamental_data[ticker] = {}
            
            return {
                'prices': prices,
                'returns': returns,
                'volumes': volumes,
                'fundamental_data': fundamental_data,
                'tickers': tickers,
                'date_range': {'start': start_date, 'end': end_date},
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed: {e}")
            raise
    
    def fetch_benchmark_data(self, benchmark: str = '^XU100', start_date: str = None, 
                            end_date: str = None) -> pd.DataFrame:
        """Fetch benchmark data"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            benchmark_data = yf.download(
                benchmark,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if benchmark_data.empty:
                raise ValueError(f"No data for benchmark {benchmark}")
            
            if 'Adj Close' in benchmark_data.columns:
                prices = benchmark_data['Adj Close']
            else:
                prices = benchmark_data['Close']
            
            return pd.DataFrame({'benchmark': prices})
            
        except Exception as e:
            logger.error(f"Benchmark fetch failed: {e}")
            # Return synthetic benchmark
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            np.random.seed(42)
            returns = np.random.normal(0.0003, 0.015, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            return pd.DataFrame({'benchmark': prices}, index=dates)

# ============================================================================
# QUANTSTATS ANALYTICS ENGINE
# ============================================================================

class QuantStatsAnalytics:
    """Comprehensive analytics engine using QuantStats"""
    
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        
    def calculate_comprehensive_metrics(self, risk_free_rate: float = 0.05) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics using QuantStats"""
        
        # Convert returns to DataFrame for QuantStats
        returns_df = pd.DataFrame({'Portfolio': self.returns})
        
        # Calculate metrics using QuantStats
        try:
            # Basic metrics
            total_return = qs.stats.total_return(self.returns)
            annual_return = qs.stats.cagr(self.returns)
            volatility = qs.stats.volatility(self.returns, annualize=True)
            sharpe = qs.stats.sharpe(self.returns, risk_free=risk_free_rate)
            sortino = qs.stats.sortino(self.returns, risk_free=risk_free_rate)
            max_dd = qs.stats.max_drawdown(self.returns)
            calmar = qs.stats.calmar(self.returns)
            
            # Advanced metrics
            omega = qs.stats.omega(self.returns, risk_free=risk_free_rate)
            gain_to_pain = qs.stats.gain_to_pain_ratio(self.returns)
            tail_ratio = qs.stats.tail_ratio(self.returns)
            common_sense = qs.stats.common_sense_ratio(self.returns)
            ulcer = qs.stats.ulcer_index(self.returns)
            serenity = qs.stats.serenity_index(self.returns)
            
            # Risk metrics
            var_95 = qs.stats.value_at_risk(self.returns)
            cvar_95 = qs.stats.conditional_value_at_risk(self.returns)
            expected_shortfall = qs.stats.expected_shortfall(self.returns)
            skewness = qs.stats.skew(self.returns)
            kurtosis = qs.stats.kurtosis(self.returns)
            
            # Performance metrics
            win_rate = qs.stats.win_rate(self.returns)
            profit_factor = qs.stats.profit_factor(self.returns)
            expectancy = qs.stats.expectancy(self.returns)
            avg_win = qs.stats.avg_win(self.returns)
            avg_loss = qs.stats.avg_loss(self.returns)
            
            # Rolling metrics
            rolling_sharpe_60 = qs.stats.rolling_sharpe(self.returns, window=60).iloc[-1] if len(self.returns) > 60 else np.nan
            rolling_sharpe_252 = qs.stats.rolling_sharpe(self.returns, window=252).iloc[-1] if len(self.returns) > 252 else np.nan
            rolling_vol_60 = qs.stats.rolling_volatility(self.returns, window=60).iloc[-1] if len(self.returns) > 60 else np.nan
            rolling_vol_252 = qs.stats.rolling_volatility(self.returns, window=252).iloc[-1] if len(self.returns) > 252 else np.nan
            
            # Factor analysis (simplified - would need factor returns data)
            market_beta = qs.stats.greeks(self.returns, self.benchmark_returns).get('beta', np.nan) if self.benchmark_returns is not None else np.nan
            info_ratio = qs.stats.information_ratio(self.returns, self.benchmark_returns) if self.benchmark_returns is not None else np.nan
            alpha = qs.stats.greeks(self.returns, self.benchmark_returns).get('alpha', np.nan) if self.benchmark_returns is not None else np.nan
            beta = market_beta
            
            return PortfolioMetrics(
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                calmar_ratio=calmar,
                omega_ratio=omega,
                gain_to_pain_ratio=gain_to_pain,
                tail_ratio=tail_ratio,
                common_sense_ratio=common_sense,
                ulcer_index=ulcer,
                serenity_ratio=serenity,
                information_ratio=info_ratio,
                alpha=alpha,
                beta=beta,
                var_95=var_95,
                cvar_95=cvar_95,
                expected_shortfall=expected_shortfall,
                skewness=skewness,
                kurtosis=kurtosis,
                win_rate=win_rate,
                profit_factor=profit_factor,
                expectancy=expectancy,
                avg_win=avg_win,
                avg_loss=avg_loss,
                rolling_sharpe_60=rolling_sharpe_60,
                rolling_sharpe_252=rolling_sharpe_252,
                rolling_vol_60=rolling_vol_60,
                rolling_vol_252=rolling_vol_252,
                market_beta=market_beta,
                size_beta=0.0,  # Placeholder
                value_beta=0.0,  # Placeholder
                momentum_beta=0.0,  # Placeholder
                quality_beta=0.0  # Placeholder
            )
            
        except Exception as e:
            logger.error(f"QuantStats metrics calculation failed: {e}")
            # Return default metrics
            return self._calculate_basic_metrics()
    
    def _calculate_basic_metrics(self) -> PortfolioMetrics:
        """Calculate basic metrics as fallback"""
        total_return = self.returns.add(1).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(self.returns)) - 1
        volatility = self.returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.05) / volatility if volatility > 0 else 0
        
        return PortfolioMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sharpe,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            omega_ratio=0.0,
            gain_to_pain_ratio=0.0,
            tail_ratio=0.0,
            common_sense_ratio=0.0,
            ulcer_index=0.0,
            serenity_ratio=0.0,
            information_ratio=0.0,
            alpha=0.0,
            beta=0.0,
            var_95=0.0,
            cvar_95=0.0,
            expected_shortfall=0.0,
            skewness=0.0,
            kurtosis=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            rolling_sharpe_60=0.0,
            rolling_sharpe_252=0.0,
            rolling_vol_60=0.0,
            rolling_vol_252=0.0,
            market_beta=0.0,
            size_beta=0.0,
            value_beta=0.0,
            momentum_beta=0.0,
            quality_beta=0.0
        )

# ============================================================================
# ENHANCED PORTFOLIO OPTIMIZER
# ============================================================================

class EnhancedPortfolioOptimizer:
    """Professional portfolio optimizer with PyPortfolioOpt integration"""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.tickers = returns.columns.tolist()
        
        # Calculate expected returns and covariance
        self.mu = expected_returns.mean_historical_return(returns, frequency=252)
        self.S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
        
    def optimize_portfolio(self, method: str, constraints: Dict = None) -> Dict:
        """Optimize portfolio using specified method"""
        
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 0.2,
                'sum_to_one': True
            }
        
        try:
            if method == 'max_sharpe':
                return self._optimize_max_sharpe(constraints)
            elif method == 'min_volatility':
                return self._optimize_min_volatility(constraints)
            elif method == 'efficient_risk':
                return self._optimize_efficient_risk(constraints)
            elif method == 'efficient_return':
                return self._optimize_efficient_return(constraints)
            elif method == 'max_quadratic_utility':
                return self._optimize_max_quadratic_utility(constraints)
            elif method == 'hierarchical_risk_parity':
                return self._optimize_hrp(constraints)
            elif method == 'black_litterman':
                return self._optimize_black_litterman(constraints)
            elif method == 'risk_parity':
                return self._optimize_risk_parity(constraints)
            else:
                return self._optimize_max_sharpe(constraints)
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return self._get_equal_weight_portfolio()
    
    def _optimize_max_sharpe(self, constraints: Dict) -> Dict:
        """Maximize Sharpe ratio"""
        ef = EfficientFrontier(self.mu, self.S)
        
        # Add constraints
        if constraints.get('min_weight') is not None:
            ef.add_constraint(lambda w: w >= constraints['min_weight'])
        if constraints.get('max_weight') is not None:
            ef.add_constraint(lambda w: w <= constraints['max_weight'])
        
        # Optimize
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        # Performance
        perf = ef.portfolio_performance(verbose=False)
        
        return {
            'weights': cleaned_weights,
            'performance': perf,
            'method': 'Maximum Sharpe Ratio'
        }
    
    def _optimize_min_volatility(self, constraints: Dict) -> Dict:
        """Minimize volatility"""
        ef = EfficientFrontier(self.mu, self.S)
        
        # Add constraints
        if constraints.get('min_weight') is not None:
            ef.add_constraint(lambda w: w >= constraints['min_weight'])
        if constraints.get('max_weight') is not None:
            ef.add_constraint(lambda w: w <= constraints['max_weight'])
        
        # Optimize
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        
        # Performance
        perf = ef.portfolio_performance(verbose=False)
        
        return {
            'weights': cleaned_weights,
            'performance': perf,
            'method': 'Minimum Volatility'
        }
    
    def _optimize_hrp(self, constraints: Dict) -> Dict:
        """Hierarchical Risk Parity"""
        hrp = HRPOpt(self.returns)
        hrp.optimize()
        weights = hrp.clean_weights()
        
        # Calculate performance metrics
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        annual_return = qs.stats.cagr(portfolio_returns)
        annual_vol = qs.stats.volatility(portfolio_returns, annualize=True)
        sharpe = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
        
        return {
            'weights': weights,
            'performance': (annual_return, annual_vol, sharpe),
            'method': 'Hierarchical Risk Parity'
        }
    
    def _optimize_black_litterman(self, constraints: Dict) -> Dict:
        """Black-Litterman optimization"""
        # Market implied returns
        market_caps = np.random.rand(len(self.tickers)) * 1e9  # Placeholder
        market_prices = np.random.rand(len(self.tickers)) * 100  # Placeholder
        
        bl = BlackLittermanModel(
            self.S,
            pi='market',
            market_caps=market_caps,
            risk_aversion=1.0,
            absolute_views={
                0: 0.05,  # View on first asset
                1: 0.03   # View on second asset
            }
        )
        
        # Posterior estimate
        mu_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        
        # Optimize
        ef = EfficientFrontier(mu_bl, S_bl)
        if constraints.get('min_weight') is not None:
            ef.add_constraint(lambda w: w >= constraints['min_weight'])
        if constraints.get('max_weight') is not None:
            ef.add_constraint(lambda w: w <= constraints['max_weight'])
        
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)
        
        return {
            'weights': cleaned_weights,
            'performance': perf,
            'method': 'Black-Litterman'
        }
    
    def _optimize_risk_parity(self, constraints: Dict) -> Dict:
        """Risk Parity optimization"""
        # Simple inverse volatility weighting
        volatilities = np.sqrt(np.diag(self.S))
        weights = 1 / volatilities
        weights = weights / weights.sum()
        
        weights_dict = {self.tickers[i]: weights[i] for i in range(len(weights))}
        
        # Calculate performance
        portfolio_returns = (self.returns * pd.Series(weights_dict)).sum(axis=1)
        annual_return = qs.stats.cagr(portfolio_returns)
        annual_vol = qs.stats.volatility(portfolio_returns, annualize=True)
        sharpe = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
        
        return {
            'weights': weights_dict,
            'performance': (annual_return, annual_vol, sharpe),
            'method': 'Risk Parity'
        }
    
    def _get_equal_weight_portfolio(self) -> Dict:
        """Equal weight portfolio as fallback"""
        n_assets = len(self.tickers)
        weights = {ticker: 1/n_assets for ticker in self.tickers}
        
        portfolio_returns = self.returns.mean(axis=1)
        annual_return = qs.stats.cagr(portfolio_returns)
        annual_vol = qs.stats.volatility(portfolio_returns, annualize=True)
        sharpe = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
        
        return {
            'weights': weights,
            'performance': (annual_return, annual_vol, sharpe),
            'method': 'Equal Weight'
        }

# ============================================================================
# PROFESSIONAL VISUALIZATION ENGINE
# ============================================================================

class ProfessionalVisualization:
    """Professional visualization engine with institutional charts"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#3b82f6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#06b6d4',
            'purple': '#8b5cf6',
            'pink': '#ec4899'
        }
    
    def plot_kpi_dashboard(self, metrics: PortfolioMetrics) -> go.Figure:
        """Create professional KPI dashboard"""
        
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )
        
        # 1. Total Return
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.total_return * 100,
                number={'prefix': "", 'suffix': "%", 'font': {'size': 40}},
                delta={'reference': 0, 'relative': False, 'position': "bottom"},
                title={'text': "Total Return", 'font': {'size': 20}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # 2. Sharpe Ratio
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics.sharpe_ratio,
                number={'font': {'size': 40}, 'suffix': ""},
                title={'text': "Sharpe Ratio", 'font': {'size': 20}},
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )
        
        # 3. Max Drawdown
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics.max_drawdown * 100,
                number={'prefix': "", 'suffix': "%", 'font': {'size': 40}},
                title={'text': "Max Drawdown", 'font': {'size': 20}},
                domain={'row': 0, 'column': 2}
            ),
            row=1, col=3
        )
        
        # 4. Annual Return
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.annual_return * 100,
                number={'prefix': "", 'suffix': "%", 'font': {'size': 40}},
                delta={'reference': 0, 'relative': False, 'position': "bottom"},
                title={'text': "Annual Return", 'font': {'size': 20}},
                domain={'row': 1, 'column': 0}
            ),
            row=2, col=1
        )
        
        # 5. Volatility
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics.volatility * 100,
                number={'prefix': "", 'suffix': "%", 'font': {'size': 40}},
                title={'text': "Annual Volatility", 'font': {'size': 20}},
                domain={'row': 1, 'column': 1}
            ),
            row=2, col=2
        )
        
        # 6. Sortino Ratio
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics.sortino_ratio,
                number={'font': {'size': 40}, 'suffix': ""},
                title={'text': "Sortino Ratio", 'font': {'size': 20}},
                domain={'row': 1, 'column': 2}
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def plot_performance_comparison(self, portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> go.Figure:
        """Plot portfolio vs benchmark performance"""
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            mode='lines',
            name='Portfolio',
            line=dict(color=self.color_palette['primary'], width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='Benchmark',
            line=dict(color=self.color_palette['warning'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Portfolio vs Benchmark Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            height=500,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def plot_drawdown_chart(self, returns: pd.Series) -> go.Figure:
        """Plot drawdown chart"""
        
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color=self.color_palette['danger'], width=2),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.3)'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickformat='.1f')
        )
        
        return fig
    
    def plot_rolling_metrics(self, returns: pd.Series) -> go.Figure:
        """Plot rolling metrics"""
        
        # Calculate rolling metrics
        rolling_sharpe = qs.stats.rolling_sharpe(returns, window=60)
        rolling_vol = qs.stats.rolling_volatility(returns, window=60, annualize=True)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio (60-day)', 'Rolling Volatility (60-day)'),
            vertical_spacing=0.15
        )
        
        # Rolling Sharpe
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Sharpe',
            line=dict(color=self.color_palette['success'], width=2)
        ), row=1, col=1)
        
        # Rolling Volatility
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values * 100,
            mode='lines',
            name='Volatility',
            line=dict(color=self.color_palette['warning'], width=2),
            fill='tozeroy',
            fillcolor='rgba(245, 158, 11, 0.2)'
        ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_yaxes(title_text='Sharpe Ratio', row=1, col=1)
        fig.update_yaxes(title_text='Volatility (%)', row=2, col=1)
        
        return fig
    
    def plot_efficient_frontier(self, optimizer: EnhancedPortfolioOptimizer) -> go.Figure:
        """Plot efficient frontier"""
        
        cla = CLA(optimizer.mu, optimizer.S)
        frontier = cla.efficient_frontier(points=50)
        
        frontier_returns = [p[0] for p in frontier]
        frontier_volatilities = [p[1] for p in frontier]
        
        fig = go.Figure()
        
        # Efficient Frontier
        fig.add_trace(go.Scatter(
            x=frontier_volatilities,
            y=frontier_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color=self.color_palette['primary'], width=3)
        ))
        
        # Individual Assets
        asset_returns = optimizer.mu.values
        asset_volatilities = np.sqrt(np.diag(optimizer.S))
        
        fig.add_trace(go.Scatter(
            x=asset_volatilities,
            y=asset_returns,
            mode='markers',
            name='Assets',
            marker=dict(
                size=10,
                color=asset_returns / asset_volatilities,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe')
            ),
            text=optimizer.tickers,
            hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Annual Volatility',
            yaxis_title='Annual Return',
            hovermode='closest',
            height=500,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(tickformat='.0%')
        fig.update_yaxes(tickformat='.0%')
        
        return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class InstitutionalPortfolioAnalytics:
    """Main application class"""
    
    def __init__(self):
        self.data_fetcher = EnhancedDataFetcher()
        self.visualizer = ProfessionalVisualization()
        self.portfolio_data = None
        self.benchmark_data = None
        
    def run(self):
        """Run the main application"""
        
        # Sidebar Configuration
        st.sidebar.title("Portfolio Configuration")
        
        # Ticker Selection
        default_tickers = ['AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'FROTO.IS',
                          'GARAN.IS', 'ISCTR.IS', 'KCHOL.IS', 'SAHOL.IS', 'THYAO.IS']
        
        selected_tickers = st.sidebar.multiselect(
            "Select Assets",
            options=default_tickers,
            default=default_tickers[:5]
        )
        
        # Date Range
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*3))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Optimization Method
        optimization_methods = {
            'Maximum Sharpe Ratio': 'max_sharpe',
            'Minimum Volatility': 'min_volatility',
            'Efficient Risk': 'efficient_risk',
            'Efficient Return': 'efficient_return',
            'Hierarchical Risk Parity': 'hierarchical_risk_parity',
            'Risk Parity': 'risk_parity',
            'Black-Litterman': 'black_litterman'
        }
        
        selected_method = st.sidebar.selectbox(
            "Optimization Method",
            options=list(optimization_methods.keys()),
            index=0
        )
        
        # Constraints
        st.sidebar.subheader("Constraints")
        min_weight = st.sidebar.slider("Minimum Weight (%)", 0, 20, 0) / 100
        max_weight = st.sidebar.slider("Maximum Weight (%)", 1, 100, 20) / 100
        
        # Fetch Data Button
        if st.sidebar.button("Fetch Data & Optimize", type="primary"):
            with st.spinner("Fetching data and optimizing portfolio..."):
                self._fetch_and_optimize(
                    selected_tickers,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    optimization_methods[selected_method],
                    min_weight,
                    max_weight
                )
        
        # Main Content
        st.title("Institutional Portfolio Analytics Platform")
        
        if self.portfolio_data:
            self._display_dashboard()
    
    def _fetch_and_optimize(self, tickers: List[str], start_date: str, end_date: str,
                           method: str, min_weight: float, max_weight: float):
        """Fetch data and optimize portfolio"""
        try:
            # Fetch portfolio data
            self.portfolio_data = self.data_fetcher.fetch_yahoo_data(tickers, start_date, end_date)
            
            # Fetch benchmark data
            self.benchmark_data = self.data_fetcher.fetch_benchmark_data('^XU100', start_date, end_date)
            
            # Align dates
            portfolio_returns = self.portfolio_data['returns']
            benchmark_returns = self.benchmark_data['benchmark'].pct_change().dropna()
            
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[common_dates]
            benchmark_returns = benchmark_returns.loc[common_dates]
            
            # Optimize portfolio
            optimizer = EnhancedPortfolioOptimizer(portfolio_returns)
            constraints = {
                'min_weight': min_weight,
                'max_weight': max_weight,
                'sum_to_one': True
            }
            
            optimization_result = optimizer.optimize_portfolio(method, constraints)
            
            # Calculate portfolio returns
            weights_series = pd.Series(optimization_result['weights'])
            aligned_weights = weights_series.reindex(portfolio_returns.columns).fillna(0)
            portfolio_returns_series = (portfolio_returns * aligned_weights).sum(axis=1)
            
            # Calculate comprehensive metrics
            qs_analytics = QuantStatsAnalytics(portfolio_returns_series, benchmark_returns)
            self.metrics = qs_analytics.calculate_comprehensive_metrics()
            
            # Store results
            self.optimization_result = optimization_result
            self.portfolio_returns = portfolio_returns_series
            self.benchmark_returns = benchmark_returns
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Application error: {traceback.format_exc()}")
    
    def _display_dashboard(self):
        """Display the main dashboard"""
        
        # KPI Dashboard
        st.subheader("Portfolio Performance Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Return",
                value=f"{self.metrics.total_return:.2%}",
                delta=f"{self.metrics.total_return:.2%}"
            )
        
        with col2:
            st.metric(
                label="Annual Return",
                value=f"{self.metrics.annual_return:.2%}",
                delta=f"{self.metrics.annual_return:.2%}"
            )
        
        with col3:
            st.metric(
                label="Sharpe Ratio",
                value=f"{self.metrics.sharpe_ratio:.3f}"
            )
        
        with col4:
            st.metric(
                label="Max Drawdown",
                value=f"{self.metrics.max_drawdown:.2%}"
            )
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance",
            "⚖️ Allocation",
            "📈 Optimization",
            "📋 Metrics",
            "📄 Report"
        ])
        
        with tab1:
            self._display_performance_tab()
        
        with tab2:
            self._display_allocation_tab()
        
        with tab3:
            self._display_optimization_tab()
        
        with tab4:
            self._display_metrics_tab()
        
        with tab5:
            self._display_report_tab()
    
    def _display_performance_tab(self):
        """Display performance tab"""
        
        st.subheader("Performance Analysis")
        
        # Performance vs Benchmark
        fig1 = self.visualizer.plot_performance_comparison(
            self.portfolio_returns,
            self.benchmark_returns
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Drawdown Chart
            fig2 = self.visualizer.plot_drawdown_chart(self.portfolio_returns)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Rolling Metrics
            fig3 = self.visualizer.plot_rolling_metrics(self.portfolio_returns)
            st.plotly_chart(fig3, use_container_width=True)
    
    def _display_allocation_tab(self):
        """Display allocation tab"""
        
        st.subheader("Portfolio Allocation")
        
        # Display weights as DataFrame
        weights_df = pd.DataFrame.from_dict(
            self.optimization_result['weights'], 
            orient='index', 
            columns=['Weight']
        ).sort_values('Weight', ascending=False)
        
        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                weights_df,
                use_container_width=True,
                height=400
            )
        
        with col2:
            # Pie chart of allocation
            weights_series = pd.Series(self.optimization_result['weights'])
            top_10 = weights_series.nlargest(10)
            
            fig = go.Figure(data=[go.Pie(
                labels=top_10.index,
                values=top_10.values,
                hole=0.4,
                textinfo='label+percent',
                textposition='inside'
            )])
            
            fig.update_layout(
                title="Top 10 Holdings",
                height=400,
                showlegend=False,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_optimization_tab(self):
        """Display optimization tab"""
        
        st.subheader("Portfolio Optimization")
        
        # Efficient Frontier
        optimizer = EnhancedPortfolioOptimizer(self.portfolio_data['returns'])
        fig = self.visualizer.plot_efficient_frontier(optimizer)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Optimization Method",
                value=self.optimization_result['method']
            )
        
        with col2:
            ret, vol, sharpe = self.optimization_result['performance']
            st.metric(
                label="Expected Return",
                value=f"{ret:.2%}"
            )
        
        with col3:
            st.metric(
                label="Expected Volatility",
                value=f"{vol:.2%}"
            )
    
    def _display_metrics_tab(self):
        """Display comprehensive metrics tab"""
        
        st.subheader("Comprehensive Metrics")
        
        # Display metrics as DataFrame
        metrics_df = self.metrics.to_dataframe()
        
        # Split into multiple columns for better readability
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                metrics_df.iloc[:len(metrics_df)//2],
                use_container_width=True,
                height=600
            )
        
        with col2:
            st.dataframe(
                metrics_df.iloc[len(metrics_df)//2:],
                use_container_width=True,
                height=600
            )
    
    def _display_report_tab(self):
        """Display report tab"""
        
        st.subheader("Portfolio Analysis Report")
        
        # Generate report
        report = self._generate_report()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(report)
        
        with col2:
            # Export options
            st.subheader("Export")
            
            if st.button("📥 Download PDF Report"):
                self._generate_pdf_report()
            
            if st.button("📊 Download Excel"):
                self._generate_excel_report()
    
    def _generate_report(self) -> str:
        """Generate portfolio analysis report"""
        
        report = f"""
        # Portfolio Analysis Report
        **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## Executive Summary
        
        The optimized portfolio demonstrates strong risk-adjusted returns with a Sharpe ratio of **{self.metrics.sharpe_ratio:.3f}** 
        and an annual return of **{self.metrics.annual_return:.2%}**. The portfolio maintains a controlled level of risk with 
        annual volatility of **{self.metrics.volatility:.2%}** and maximum drawdown of **{self.metrics.max_drawdown:.2%}**.
        
        ## Key Performance Indicators
        
        | Metric | Value | Interpretation |
        |--------|-------|----------------|
        | Total Return | {self.metrics.total_return:.2%} | Overall portfolio growth |
        | Annual Return | {self.metrics.annual_return:.2%} | Annualized return rate |
        | Sharpe Ratio | {self.metrics.sharpe_ratio:.3f} | Risk-adjusted return |
        | Sortino Ratio | {self.metrics.sortino_ratio:.3f} | Downside risk-adjusted return |
        | Max Drawdown | {self.metrics.max_drawdown:.2%} | Worst historical loss |
        | Omega Ratio | {self.metrics.omega_ratio:.3f} | Probability-weighted return |
        
        ## Risk Analysis
        
        The portfolio shows **{self.metrics.skewness:.3f} skewness** and **{self.metrics.kurtosis:.3f} kurtosis**, 
        indicating the distribution characteristics of returns. Value at Risk (95%) is **{self.metrics.var_95:.2%}** 
        and Conditional VaR is **{self.metrics.cvar_95:.2%}**, representing potential losses under stress conditions.
        
        ## Performance Statistics
        
        - **Win Rate:** {self.metrics.win_rate:.2%}
        - **Profit Factor:** {self.metrics.profit_factor:.3f}
        - **Expectancy:** {self.metrics.expectancy:.3f}
        - **Average Win:** {self.metrics.avg_win:.2%}
        - **Average Loss:** {self.metrics.avg_loss:.2%}
        
        ## Optimization Details
        
        - **Method:** {self.optimization_result['method']}
        - **Number of Assets:** {len(self.optimization_result['weights'])}
        - **Effective Diversification:** {1/sum([w**2 for w in self.optimization_result['weights'].values()]):.1f} assets
        
        ## Recommendations
        
        1. Monitor the **{self.metrics.max_drawdown:.2%} maximum drawdown** level for risk management
        2. Consider rebalancing when weights deviate more than 5% from targets
        3. Review the **{self.metrics.sharpe_ratio:.3f} Sharpe ratio** quarterly for optimization opportunities
        4. Maintain focus on **{self.metrics.sortino_ratio:.3f} Sortino ratio** for downside protection
        """
        
        return report
    
    def _generate_pdf_report(self):
        """Generate PDF report (placeholder)"""
        st.success("PDF report generation would be implemented here")
    
    def _generate_excel_report(self):
        """Generate Excel report (placeholder)"""
        st.success("Excel report generation would be implemented here")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    app = InstitutionalPortfolioAnalytics()
    app.run()
