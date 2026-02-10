# ============================================================================
# BIST ENTERPRISE QUANT PORTFOLIO OPTIMIZATION SUITE PRO MAX ULTRA
# Version: 9.0 | Features: Full PyPortfolioOpt + QuantStats Integration
# Institutional-Grade Analytics & Visualizations
# ============================================================================

import warnings
import sys
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import requests
import base64
import logging
import traceback
import time
import os
import json
import io
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# ─────────────────────────────────────────────────────────────────────────────
# CORE IMPORTS WITH ROBUST ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

# Force disable some warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('portfolio_analytics.log')
    ]
)
logger = logging.getLogger(__name__)

# Enhanced error handling for imports
def safe_import(module_name, import_func=None):
    """Safely import modules with detailed error reporting"""
    try:
        if import_func:
            return import_func()
        else:
            return __import__(module_name)
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing {module_name}: {e}")
        return None

# Try importing yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
    logger.info(f"yfinance version: {yf.__version__}")
except ImportError as e:
    st.error(f"yfinance import error: {e}")
    HAS_YFINANCE = False

# Try importing PyPortfolioOpt with fallbacks
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt import objective_functions
    from pypfopt.cla import CLA
    HAS_PYPFOPT = True
    logger.info("PyPortfolioOpt imported successfully")
except ImportError as e:
    logger.error(f"PyPortfolioOpt import error: {e}")
    HAS_PYPFOPT = False

# Try importing QuantStats
try:
    import quantstats as qs
    # Extend pandas for quantstats functionality
    try:
        qs.extend_pandas()
        HAS_QUANTSTATS = True
        logger.info("QuantStats imported successfully")
    except:
        HAS_QUANTSTATS = False
except ImportError:
    HAS_QUANTSTATS = False

# Try importing scikit-learn components
try:
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS FOR TYPE SAFETY
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationMethod(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    EFFICIENT_RISK = "efficient_risk"
    EFFICIENT_RETURN = "efficient_return"
    HRP = "hrp"
    MAX_QUADRATIC_UTILITY = "max_quadratic_utility"
    MAX_RETURN = "max_return"

class RiskModel(str, Enum):
    LEDOIT_WOLF = "ledoit_wolf"
    SAMPLE_COV = "sample_cov"
    SEMICOVARIANCE = "semicovariance"
    EXPONENTIAL_COV = "exp_cov"
    ORACLE = "oracle_approximating"

class ReturnModel(str, Enum):
    MEAN_HISTORICAL = "mean_historical"
    EMA_HISTORICAL = "ema_historical"
    CAPM = "capm_return"

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Quant Portfolio Lab Pro MAX ULTRA",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit/streamlit',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# BIST Portfolio Optimization Suite v9.0"
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL CSS THEME WITH GRADIENTS & ANIMATIONS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&family=Source+Sans+Pro:wght@400;600&display=swap');
    
    :root {
        --primary-dark: #0a1929;
        --secondary-dark: #1a2536;
        --accent-blue: #0066cc;
        --accent-green: #00cc88;
        --accent-red: #ff4d4d;
        --accent-purple: #9d4edd;
        --accent-orange: #ff6b35;
        --accent-cyan: #00f2fe;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --border-color: #2d3748;
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --gradient-4: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --gradient-5: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95%;
    }
    
    /* Professional Metrics with 3D Effect */
    .metric-card-3d {
        background: linear-gradient(145deg, var(--secondary-dark), var(--primary-dark));
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 8px 8px 16px rgba(0, 0, 0, 0.3),
                   -4px -4px 10px rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        transform-style: preserve-3d;
        perspective: 1000px;
    }
    
    .metric-card-3d::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: 0.6s;
    }
    
    .metric-card-3d:hover::before {
        left: 100%;
    }
    
    .metric-card-3d:hover {
        transform: translateY(-8px) rotateX(5deg) rotateY(-5deg);
        box-shadow: 12px 12px 24px rgba(0, 0, 0, 0.4),
                   -6px -6px 12px rgba(255, 255, 255, 0.1);
        border-color: var(--accent-blue);
    }
    
    /* Glass Effect Cards */
    .glass-card {
        background: rgba(26, 37, 54, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        background: rgba(26, 37, 54, 0.9);
        border-color: var(--accent-blue);
        box-shadow: 0 12px 48px rgba(0, 102, 204, 0.3);
    }
    
    /* Enhanced DataFrames */
    .dataframe-container {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        overflow: hidden;
        background: var(--secondary-dark);
    }
    
    /* Enhanced Tabs with 3D Effect */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(26, 37, 54, 0.9), rgba(10, 25, 41, 0.9));
        padding: 0.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 2rem;
        background: transparent;
        border-radius: 12px;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 102, 204, 0.1);
        border-color: var(--accent-blue);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-1) !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        border: none;
        transform: translateY(-2px);
    }
    
    /* 3D Buttons */
    .stButton > button {
        background: var(--gradient-3);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 30px rgba(79, 172, 254, 0.6);
    }
    
    /* Primary Action Button */
    .primary-button > button {
        background: var(--gradient-1) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Headers with Animated Gradient Text */
    .animated-gradient-text {
        background: linear-gradient(
            90deg,
            var(--accent-blue),
            var(--accent-green),
            var(--accent-purple),
            var(--accent-orange),
            var(--accent-blue)
        );
        background-size: 300% 300%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-animation 8s ease infinite;
    }
    
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    
    /* Enhanced Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-dark), rgba(10, 25, 41, 0.95));
        border-right: 1px solid var(--border-color);
        box-shadow: 8px 0 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Custom Scrollbar with Gradient */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-dark);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-1);
        border-radius: 5px;
        border: 2px solid var(--primary-dark);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gradient-2);
    }
    
    /* Notification Badges with Pulse Animation */
    .badge-pulse {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-left: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .badge-success {
        background: var(--gradient-4);
        color: white;
    }
    
    .badge-warning {
        background: var(--gradient-5);
        color: white;
    }
    
    .badge-danger {
        background: var(--gradient-2);
        color: white;
    }
    
    .badge-info {
        background: var(--gradient-3);
        color: white;
    }
    
    /* Progress Bars with Gradient */
    .stProgress > div > div > div {
        background: var(--gradient-1) !important;
        border-radius: 10px;
    }
    
    /* Tooltip Styling */
    .stTooltip {
        background: rgba(26, 37, 54, 0.95) !important;
        border: 1px solid var(--accent-blue) !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(26, 37, 54, 0.7) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2) !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(26, 37, 54, 0.7) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(26, 37, 54, 0.9) !important;
        border-color: var(--accent-blue) !important;
    }
    
    /* Divider Styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

BIST100_TICKERS = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EKGYO.IS',
    'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS', 'HALKB.IS', 'HEKTS.IS',
    'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'ODAS.IS',
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TAVHL.IS',
    'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TTKOM.IS',
    'TUPRS.IS', 'VAKBN.IS', 'VESTL.IS', 'YKBNK.IS', 'FENER.IS', 'GSRAY.IS',
    'MAVI.IS', 'SNKRN.IS', 'BFREN.IS', 'CCOLA.IS', 'ENJSA.IS', 'FMIZP.IS',
    'KOZAA.IS', 'MGROS.IS', 'OTKAR.IS', 'PETUN.IS', 'SODA.IS', 'TMSN.IS',
    'ULKER.IS', 'YATAS.IS', 'ZOREN.IS'
]

SECTOR_MAPPING = {
    'Banking': ['AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'HALKB.IS', 'YKBNK.IS', 'TSKB.IS', 'VAKBN.IS'],
    'Industry': ['ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EREGL.IS', 'GUBRF.IS', 'TKFEN.IS'],
    'Automotive': ['FROTO.IS', 'TOASO.IS', 'KCHOL.IS', 'OTKAR.IS'],
    'Technology': ['THYAO.IS', 'TCELL.IS', 'TTKOM.IS', 'VESTL.IS'],
    'Energy': ['PETKM.IS', 'TUPRS.IS', 'PETUN.IS'],
    'Holding': ['SAHOL.IS', 'KRDMD.IS', 'KOZAA.IS'],
    'Construction': ['EKGYO.IS', 'ODAS.IS', 'YATAS.IS'],
    'Textile': ['SASA.IS', 'MAVI.IS'],
    'Glass': ['SISE.IS', 'SODA.IS'],
    'Tourism': ['TAVHL.IS'],
    'Healthcare': ['HEKTS.IS'],
    'Food': ['PGSUS.IS', 'ULKER.IS', 'CCOLA.IS'],
    'Retail': ['BIMAS.IS', 'MGROS.IS'],
    'Sports': ['FENER.IS', 'GSRAY.IS'],
    'Electricity': ['ENJSA.IS'],
    'Finance': ['FMIZP.IS']
}

BENCHMARKS = {
    'BIST 100': 'XU100.IS',
    'BIST 30': 'XU030.IS', 
    'USD/TRY': 'TRY=X',
    'EUR/TRY': 'EURTRY=X',
    'Gold': 'GC=F',
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'BTC-USD': 'BTC-USD',
    'VIX': '^VIX',
    'Brent Oil': 'BZ=F'
}

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED DATA SOURCE WITH MULTIPLE FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketData:
    """Data class for market data"""
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    tickers: List[str]
    start_date: datetime
    end_date: datetime
    
    def to_dict(self):
        return {
            'close': self.close,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'returns': self.returns
        }

class AdvancedDataSource:
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.max_retries = 3
        self.retry_delay = 1
    
    def _generate_cache_key(self, tickers: List[str], start_date: datetime, 
                           end_date: datetime, interval: str) -> str:
        """Generate cache key for data"""
        key_data = {
            'tickers': sorted(tickers),
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'interval': interval
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @st.cache_data(ttl=3600, show_spinner="📊 Fetching market data...")
    def fetch_market_data(_self, tickers: List[str], start_date: datetime, 
                         end_date: datetime, interval: str = '1d') -> Optional[MarketData]:
        """Robust data fetching with retry logic and fallbacks"""
        
        if not HAS_YFINANCE:
            st.error("❌ yfinance is not installed")
            return None
        
        if not tickers:
            st.error("❌ No tickers provided")
            return None
        
        # Ensure tickers is a list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Remove duplicates
        tickers = list(set(tickers))
        
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_data_frames = []
        successful_tickers = []
        
        # Strategy 1: Try to fetch all at once
        for retry in range(self.max_retries):
            try:
                logger.info(f"Attempt {retry + 1}: Fetching all tickers together")
                data = yf.download(
                    tickers=tickers,
                    start=start_date,
                    end=end_date + timedelta(days=1),  # Include end date
                    interval=interval,
                    progress=False,
                    show_errors=False,
                    threads=True,
                    timeout=30
                )
                
                if data.empty:
                    logger.warning(f"Empty data returned on attempt {retry + 1}")
                    time.sleep(self.retry_delay)
                    continue
                
                # Process data based on structure
                if len(tickers) == 1:
                    # Single ticker
                    close_prices = data['Close'].to_frame(tickers[0])
                    open_prices = data['Open'].to_frame(tickers[0])
                    high_prices = data['High'].to_frame(tickers[0])
                    low_prices = data['Low'].to_frame(tickers[0])
                    volumes = data['Volume'].to_frame(tickers[0])
                else:
                    # Multiple tickers
                    close_prices = pd.DataFrame()
                    open_prices = pd.DataFrame()
                    high_prices = pd.DataFrame()
                    low_prices = pd.DataFrame()
                    volumes = pd.DataFrame()
                    
                    # Check if data has multi-level columns
                    if isinstance(data.columns, pd.MultiIndex):
                        for ticker in tickers:
                            if (ticker, 'Close') in data.columns:
                                close_prices[ticker] = data[(ticker, 'Close')]
                                open_prices[ticker] = data[(ticker, 'Open')]
                                high_prices[ticker] = data[(ticker, 'High')]
                                low_prices[ticker] = data[(ticker, 'Low')]
                                volumes[ticker] = data[(ticker, 'Volume')]
                                successful_tickers.append(ticker)
                    else:
                        # Fallback: Single column returned
                        close_prices = data[['Close']].copy()
                        close_prices.columns = [tickers[0]]
                        successful_tickers = [tickers[0]]
                
                if not close_prices.empty:
                    break
                    
            except Exception as e:
                logger.error(f"Batch download attempt {retry + 1} failed: {e}")
                time.sleep(self.retry_delay * (retry + 1))
        
        # Strategy 2: If batch failed, try individual tickers
        if close_prices.empty or len(successful_tickers) < len(tickers):
            logger.info("Falling back to individual ticker downloads")
            close_prices = pd.DataFrame()
            
            for ticker in tickers:
                if ticker in successful_tickers:
                    continue
                    
                for retry in range(self.max_retries):
                    try:
                        logger.info(f"Fetching individual ticker: {ticker}")
                        ticker_data = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date + timedelta(days=1),
                            progress=False,
                            show_errors=False,
                            timeout=20
                        )
                        
                        if not ticker_data.empty:
                            close_prices[ticker] = ticker_data['Close']
                            successful_tickers.append(ticker)
                            break
                    except Exception as e:
                        logger.error(f"Failed to download {ticker}: {e}")
                        time.sleep(self.retry_delay)
        
        # Check if we have any data
        if close_prices.empty:
            st.error(f"❌ Could not fetch data for any tickers: {tickers}")
            return None
        
        # Clean and prepare data
        close_prices = close_prices.ffill().bfill()
        close_prices = close_prices.dropna(axis=1, how='all')
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Create other price series (simplified for performance)
        open_prices = close_prices.copy()  # Approximate for simplicity
        high_prices = close_prices.copy()
        low_prices = close_prices.copy()
        volumes = pd.DataFrame(index=close_prices.index)
        
        for col in close_prices.columns:
            volumes[col] = 1000000  # Placeholder volume
        
        # Create MarketData object
        market_data = MarketData(
            close=close_prices,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            volume=volumes,
            returns=returns,
            tickers=successful_tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Successfully fetched data for {len(successful_tickers)} tickers")
        return market_data
    
    def fetch_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data with caching"""
        cache_key = f"fundamental_{ticker}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamental_data = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'website': info.get('website'),
                'long_business_summary': info.get('longBusinessSummary'),
                'employees': info.get('fullTimeEmployees')
            }
            
            # Clean None values
            fundamental_data = {k: v for k, v in fundamental_data.items() if v is not None}
            
            # Cache the data
            self.cache[cache_key] = fundamental_data
            
            return fundamental_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch fundamental data for {ticker}: {e}")
            return None

# ─────────────────────────────────────────────────────────────────────────────
# QUANTITATIVE PORTFOLIO OPTIMIZER WITH ADVANCED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class QuantitativePortfolioOptimizer:
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame):
        self.prices = prices
        self.returns = returns
        self.tickers = prices.columns.tolist()
        
        if not HAS_PYPFOPT:
            raise ImportError("PyPortfolioOpt is required for optimization")
        
        # Validate data
        self._validate_data()
        
        # Initialize optimization models
        self._initialize_models()
        
        # Performance tracking
        self.optimization_history = []
    
    def _validate_data(self):
        """Validate input data"""
        if self.prices.empty or self.returns.empty:
            raise ValueError("Prices or returns data is empty")
        
        if len(self.tickers) < 2:
            raise ValueError("At least 2 assets required for optimization")
        
        # Check for NaN values and handle
        self.prices = self.prices.ffill().bfill()
        self.returns = self.returns.ffill().bfill()
        
        # Remove any columns that are all NaN
        self.prices = self.prices.dropna(axis=1, how='all')
        self.returns = self.returns.dropna(axis=1, how='all')
        
        # Ensure tickers match
        self.tickers = list(set(self.prices.columns) & set(self.returns.columns))
        
        if len(self.tickers) < 2:
            raise ValueError("Insufficient valid tickers after cleaning")
    
    def _initialize_models(self):
        """Initialize expected returns and risk models"""
        try:
            # Expected return models
            self.mu_models = {
                ReturnModel.MEAN_HISTORICAL.value: expected_returns.mean_historical_return(self.prices),
                ReturnModel.EMA_HISTORICAL.value: expected_returns.ema_historical_return(self.prices, span=500),
                ReturnModel.CAPM.value: expected_returns.capm_return(self.prices),
            }
            
            # Risk models
            self.risk_models = {
                RiskModel.SAMPLE_COV.value: risk_models.sample_cov(self.returns),
                RiskModel.SEMICOVARIANCE.value: risk_models.semicovariance(self.returns),
                RiskModel.EXPONENTIAL_COV.value: risk_models.exp_cov(self.returns),
                RiskModel.LEDOIT_WOLF.value: risk_models.CovarianceShrinkage(self.prices).ledoit_wolf(),
                RiskModel.ORACLE.value: risk_models.CovarianceShrinkage(self.prices).oracle_approximating(),
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def optimize(self, method: str = OptimizationMethod.MAX_SHARPE.value, 
                risk_model: str = RiskModel.LEDOIT_WOLF.value,
                return_model: str = ReturnModel.MEAN_HISTORICAL.value,
                **kwargs) -> Tuple[Dict, Tuple]:
        """Main optimization method with comprehensive error handling"""
        
        try:
            # Get selected models
            if return_model not in self.mu_models:
                return_model = ReturnModel.MEAN_HISTORICAL.value
            
            if risk_model not in self.risk_models:
                risk_model = RiskModel.LEDOIT_WOLF.value
            
            mu = self.mu_models[return_model]
            S = self.risk_models[risk_model]
            
            # Risk-free rate
            risk_free_rate = kwargs.get('risk_free_rate', 0.0)
            
            # Store optimization parameters
            opt_params = {
                'method': method,
                'risk_model': risk_model,
                'return_model': return_model,
                'risk_free_rate': risk_free_rate,
                'timestamp': datetime.now()
            }
            
            # Perform optimization based on method
            if method == OptimizationMethod.MAX_SHARPE.value:
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == OptimizationMethod.MIN_VOLATILITY.value:
                ef = EfficientFrontier(mu, S)
                ef.min_volatility()
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == OptimizationMethod.MAX_QUADRATIC_UTILITY.value:
                ef = EfficientFrontier(mu, S)
                ef.max_quadratic_utility()
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == OptimizationMethod.EFFICIENT_RISK.value:
                ef = EfficientFrontier(mu, S)
                target_vol = kwargs.get('target_volatility', 0.15)
                ef.efficient_risk(target_volatility=target_vol)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == OptimizationMethod.EFFICIENT_RETURN.value:
                ef = EfficientFrontier(mu, S)
                target_return = kwargs.get('target_return', 0.20)
                ef.efficient_return(target_return=target_return)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == OptimizationMethod.HRP.value:
                hrp = HRPOpt(self.returns)
                hrp.optimize()
                weights = hrp.clean_weights()
                
                # Calculate performance metrics for HRP
                port_returns = (self.returns * pd.Series(weights)).sum(axis=1)
                annual_return = port_returns.mean() * 252
                annual_vol = port_returns.std() * np.sqrt(252)
                sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
                performance = (annual_return, annual_vol, sharpe)
                
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # Store in history
            opt_params.update({
                'weights': weights,
                'performance': performance
            })
            self.optimization_history.append(opt_params)
            
            return weights, performance
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            # Fallback to simple equal weighting
            st.warning(f"Optimization failed, using equal weighting: {str(e)}")
            
            equal_weights = {ticker: 1/len(self.tickers) for ticker in self.tickers}
            port_returns = (self.returns * pd.Series(equal_weights)).sum(axis=1)
            annual_return = port_returns.mean() * 252
            annual_vol = port_returns.std() * np.sqrt(252)
            sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            return equal_weights, (annual_return, annual_vol, sharpe)
    
    def generate_efficient_frontier(self, points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate efficient frontier"""
        mu = self.mu_models[ReturnModel.MEAN_HISTORICAL.value]
        S = self.risk_models[RiskModel.LEDOIT_WOLF.value]
        
        ef = EfficientFrontier(mu, S)
        mus, sigmas, weights = ef.efficient_frontier(points=points)
        
        return mus, sigmas, weights
    
    def calculate_discrete_allocation(self, weights: Dict, 
                                     total_portfolio_value: float = 1000000) -> Tuple[Dict, float]:
        """Calculate discrete share allocation"""
        try:
            latest_prices = get_latest_prices(self.prices)
            da = DiscreteAllocation(
                weights, 
                latest_prices, 
                total_portfolio_value=total_portfolio_value
            )
            
            # Try both allocation methods
            try:
                allocation, leftover = da.lp_portfolio()
            except:
                allocation, leftover = da.greedy_portfolio()
            
            return allocation, leftover
        except Exception as e:
            logger.error(f"Discrete allocation error: {e}")
            return {}, 0.0
    
    def monte_carlo_simulation(self, n_simulations: int = 10000, 
                              time_horizon: int = 252) -> Dict:
        """Run Monte Carlo simulation for portfolio returns"""
        try:
            # Get expected returns and covariance
            mu = self.mu_models[ReturnModel.MEAN_HISTORICAL.value].values
            S = self.risk_models[RiskModel.LEDOIT_WOLF.value].values
            
            # Cholesky decomposition for correlated random variables
            L = np.linalg.cholesky(S)
            
            # Generate simulations
            simulations = np.zeros((n_simulations, time_horizon))
            
            for i in range(n_simulations):
                # Generate correlated random returns
                z = np.random.normal(size=(time_horizon, len(mu)))
                correlated_returns = mu + np.dot(z, L.T)
                
                # Calculate portfolio path
                portfolio_returns = np.exp(np.cumsum(correlated_returns, axis=0))
                simulations[i] = portfolio_returns[:, 0]  # First asset as proxy
            
            # Calculate statistics
            final_values = simulations[:, -1]
            
            stats_dict = {
                'mean_final_value': np.mean(final_values),
                'median_final_value': np.median(final_values),
                'std_final_value': np.std(final_values),
                'var_95': np.percentile(final_values, 5),
                'var_99': np.percentile(final_values, 1),
                'cvar_95': final_values[final_values <= np.percentile(final_values, 5)].mean(),
                'simulations': simulations,
                'final_values': final_values
            }
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {e}")
            return {}

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED ANALYTICS ENGINE WITH MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedAnalyticsEngine:
    def __init__(self, portfolio_returns: pd.Series, 
                 benchmark_returns: pd.Series = None,
                 risk_free_rate: float = 0.0):
        
        # Ensure returns are pandas Series
        if isinstance(portfolio_returns, pd.DataFrame):
            self.portfolio_returns = portfolio_returns.iloc[:, 0]
        else:
            self.portfolio_returns = portfolio_returns
        
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Validate returns
        self._validate_returns()
        
        # Initialize QuantStats if available
        if HAS_QUANTSTATS:
            try:
                qs.extend_pandas()
            except:
                pass
    
    def _validate_returns(self):
        """Validate return series"""
        if self.portfolio_returns.empty:
            raise ValueError("Portfolio returns are empty")
        
        # Remove any NaN or inf values
        self.portfolio_returns = self.portfolio_returns.replace([np.inf, -np.inf], np.nan)
        self.portfolio_returns = self.portfolio_returns.dropna()
        
        if self.benchmark_returns is not None:
            self.benchmark_returns = self.benchmark_returns.replace([np.inf, -np.inf], np.nan)
            self.benchmark_returns = self.benchmark_returns.dropna()
    
    def calculate_comprehensive_metrics(self) -> Dict:
        """Calculate comprehensive performance and risk metrics"""
        metrics = {}
        
        # Basic return metrics
        metrics['Total Return'] = (1 + self.portfolio_returns).prod() - 1
        
        # CAGR
        total_days = (self.portfolio_returns.index[-1] - self.portfolio_returns.index[0]).days
        metrics['CAGR'] = ((1 + metrics['Total Return']) ** (365 / total_days)) - 1 if total_days > 0 else 0
        
        # Volatility metrics
        metrics['Annual Volatility'] = self.portfolio_returns.std() * np.sqrt(252)
        metrics['Annual Downside Deviation'] = self.portfolio_returns[self.portfolio_returns < 0].std() * np.sqrt(252) if len(self.portfolio_returns[self.portfolio_returns < 0]) > 0 else 0
        
        # Maximum Drawdown
        cum_returns = (1 + self.portfolio_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        metrics['Max Drawdown'] = drawdown.min()
        metrics['Avg Drawdown'] = drawdown.mean()
        
        # Risk-adjusted metrics (manual calculation if QuantStats not available)
        if metrics['Annual Volatility'] > 0:
            metrics['Sharpe Ratio'] = (metrics['CAGR'] - self.risk_free_rate) / metrics['Annual Volatility']
            
            if metrics['Annual Downside Deviation'] > 0:
                metrics['Sortino Ratio'] = (metrics['CAGR'] - self.risk_free_rate) / metrics['Annual Downside Deviation']
            else:
                metrics['Sortino Ratio'] = 0
        else:
            metrics['Sharpe Ratio'] = 0
            metrics['Sortino Ratio'] = 0
        
        # Calmar Ratio
        if abs(metrics['Max Drawdown']) > 0:
            metrics['Calmar Ratio'] = metrics['CAGR'] / abs(metrics['Max Drawdown'])
        else:
            metrics['Calmar Ratio'] = 0
        
        # Value at Risk
        metrics['VaR (95%)'] = np.percentile(self.portfolio_returns, 5)
        metrics['CVaR (95%)'] = self.portfolio_returns[self.portfolio_returns <= metrics['VaR (95%)']].mean()
        
        # Statistical metrics
        metrics['Skewness'] = stats.skew(self.portfolio_returns)
        metrics['Kurtosis'] = stats.kurtosis(self.portfolio_returns)
        metrics['Jarque-Bera Stat'] = stats.jarque_bera(self.portfolio_returns)[0]
        metrics['Jarque-Bera p-value'] = stats.jarque_bera(self.portfolio_returns)[1]
        
        # Win/Loss metrics
        positive_returns = self.portfolio_returns[self.portfolio_returns > 0]
        negative_returns = self.portfolio_returns[self.portfolio_returns < 0]
        
        metrics['Win Rate'] = len(positive_returns) / len(self.portfolio_returns) if len(self.portfolio_returns) > 0 else 0
        metrics['Avg Win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
        metrics['Avg Loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
        metrics['Profit Factor'] = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else float('inf')
        
        # Benchmark comparison metrics
        if self.benchmark_returns is not None:
            # Align dates
            common_idx = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
            if len(common_idx) > 0:
                port_aligned = self.portfolio_returns.loc[common_idx]
                bench_aligned = self.benchmark_returns.loc[common_idx]
                
                # Beta and Alpha
                covariance = np.cov(port_aligned, bench_aligned)[0, 1]
                bench_variance = np.var(bench_aligned)
                metrics['Beta'] = covariance / bench_variance if bench_variance > 0 else 0
                
                port_return = port_aligned.mean() * 252
                bench_return = bench_aligned.mean() * 252
                metrics['Alpha'] = port_return - (self.risk_free_rate + metrics['Beta'] * (bench_return - self.risk_free_rate))
                
                # Tracking Error and Information Ratio
                tracking_error = (port_aligned - bench_aligned).std() * np.sqrt(252)
                metrics['Tracking Error'] = tracking_error
                metrics['Information Ratio'] = metrics['Alpha'] / tracking_error if tracking_error > 0 else 0
        
        # Use QuantStats for additional metrics if available
        if HAS_QUANTSTATS:
            try:
                qs_metrics = qs.reports.metrics(
                    self.portfolio_returns,
                    self.benchmark_returns if self.benchmark_returns is not None else self.portfolio_returns,
                    rf=self.risk_free_rate,
                    display=False,
                    mode='full'
                )
                
                # Add QuantStats specific metrics
                if isinstance(qs_metrics, dict):
                    metrics.update({
                        'Omega Ratio': qs_metrics.get('Omega', 0),
                        'Tail Ratio': qs_metrics.get('Tail Ratio', 0),
                        'Common Sense Ratio': qs_metrics.get('Common Sense Ratio', 0),
                        'Daily Value at Risk': qs_metrics.get('Daily Value at Risk', 0),
                        'Expected Shortfall (cVaR)': qs_metrics.get('Expected Shortfall (cVaR)', 0),
                    })
            except:
                pass
        
        return metrics
    
    def create_interactive_tearsheet(self) -> go.Figure:
        """Create comprehensive interactive tearsheet"""
        
        # Calculate cumulative returns
        cum_port = (1 + self.portfolio_returns).cumprod()
        
        if self.benchmark_returns is not None:
            cum_bench = (1 + self.benchmark_returns).cumprod()
        
        # Create subplots with advanced layout
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Cumulative Returns', 'Daily Returns Distribution', 'Rolling Sharpe (6M)',
                'Drawdown Analysis', 'Monthly Returns Heatmap', 'Return QQ Plot',
                'Rolling Volatility (6M)', 'Underwater Plot', 'Return Autocorrelation',
                'Risk-Return Scatter', 'Monte Carlo Simulation', 'Performance vs Benchmark'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Cumulative Returns (Row 1, Col 1)
        fig.add_trace(
            go.Scatter(
                x=cum_port.index,
                y=cum_port.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='#00cc88', width=3),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Cumulative Return: %{y:.2%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        if self.benchmark_returns is not None:
            fig.add_trace(
                go.Scatter(
                    x=cum_bench.index,
                    y=cum_bench.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#0066cc', width=2, dash='dash'),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Cumulative Return: %{y:.2%}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Daily Returns Distribution (Row 1, Col 2)
        fig.add_trace(
            go.Histogram(
                x=self.portfolio_returns.values,
                nbinsx=50,
                name='Return Distribution',
                marker_color='#0066cc',
                opacity=0.7,
                hovertemplate='Return: %{x:.2%}<br>Frequency: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add normal distribution overlay
        x_range = np.linspace(self.portfolio_returns.min(), self.portfolio_returns.max(), 100)
        pdf = stats.norm.pdf(x_range, self.portfolio_returns.mean(), self.portfolio_returns.std())
        pdf = pdf * len(self.portfolio_returns) * (x_range[1] - x_range[0])
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=pdf,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='#ff6b35', width=2, dash='dash'),
                hovertemplate='Return: %{x:.2%}<br>Density: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe Ratio (Row 1, Col 3)
        rolling_window = 126  # 6 months
        rolling_sharpe = self.portfolio_returns.rolling(rolling_window).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='#9d4edd', width=2),
                fill='tozeroy',
                fillcolor='rgba(157, 78, 221, 0.2)',
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Sharpe Ratio: %{y:.2f}<extra></extra>'
            ),
            row=1, col=3
        )
        
        # 4. Drawdown Analysis (Row 2, Col 1)
        drawdown = (cum_returns - cum_returns.expanding().max()) / cum_returns.expanding().max()
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff4d4d', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 77, 77, 0.3)',
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 5. Monthly Returns Heatmap (Row 2, Col 2)
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month_name(),
            'Return': monthly_returns.values
        })
        
        monthly_pivot = monthly_df.pivot_table(
            index='Year', 
            columns='Month', 
            values='Return', 
            aggfunc='mean'
        )
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_pivot = monthly_pivot.reindex(columns=month_order)
        
        fig.add_trace(
            go.Heatmap(
                z=monthly_pivot.values,
                x=monthly_pivot.columns,
                y=monthly_pivot.index,
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(title="Return", len=0.2, y=0.7),
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Return QQ Plot (Row 2, Col 3)
        sorted_returns = np.sort(self.portfolio_returns)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sorted_returns))
        )
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_returns,
                mode='markers',
                name='QQ Plot',
                marker=dict(
                    size=5,
                    color='#4facfe',
                    line=dict(width=1, color='white')
                ),
                hovertemplate='Theoretical Quantile: %{x:.3f}<br>Sample Quantile: %{y:.3f}<extra></extra>'
            ),
            row=2, col=3
        )
        
        # Add 45-degree line
        min_val = min(theoretical_quantiles.min(), sorted_returns.min())
        max_val = max(theoretical_quantiles.max(), sorted_returns.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='45° Line',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1400,
            showlegend=True,
            template='plotly_dark',
            title_text="📊 Advanced Portfolio Analytics Dashboard",
            title_font_size=24,
            title_x=0.5,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".0%", row=1, col=1)
        
        fig.update_xaxes(title_text="Daily Return", tickformat=".1%", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Date", row=1, col=3)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=3)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Year", row=2, col=2)
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=3)
        fig.update_yaxes(title_text="Sample Quantiles", tickformat=".1%", row=2, col=3)
        
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION WITH ADVANCED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.optimized = False
        st.session_state.portfolio_data = None
        st.session_state.benchmark_data = None
        st.session_state.optimization_results = None
        st.session_state.analytics_results = None
    
    # Custom sidebar header with 3D effect
    st.sidebar.markdown("""
    <div style="
        text-align: center; 
        padding: 1.5rem; 
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
        border-radius: 20px; 
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    ">
        <h2 style="color: white; margin: 0; font-size: 1.8rem;">⚡ BIST Quant Pro ULTRA</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            v9.0 - Institutional Edition
        </p>
        <div style="display: flex; justify-content: center; gap: 0.5rem; margin-top: 1rem;">
            <span class="badge-pulse badge-success">AI-Powered</span>
            <span class="badge-pulse badge-info">Real-time</span>
            <span class="badge-pulse badge-warning">Advanced</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Advanced Configuration")
        
        # Date Selection with presets
        col1, col2 = st.columns(2)
        with col1:
            date_preset = st.selectbox(
                "Time Period",
                ["Custom", "1 Year", "3 Years", "5 Years", "Max"],
                help="Select predefined time period"
            )
        
        if date_preset == "Custom":
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    datetime.now() - timedelta(days=365 * 3),
                    help="Select start date for analysis"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    datetime.now(),
                    help="Select end date for analysis"
                )
        else:
            end_date = datetime.now()
            if date_preset == "1 Year":
                start_date = end_date - timedelta(days=365)
            elif date_preset == "3 Years":
                start_date = end_date - timedelta(days=365 * 3)
            elif date_preset == "5 Years":
                start_date = end_date - timedelta(days=365 * 5)
            else:  # Max
                start_date = datetime(2010, 1, 1)
            
            with col1:
                st.date_input("Start Date", start_date, disabled=True)
            with col2:
                st.date_input("End Date", end_date, disabled=True)
        
        # Asset Selection with advanced filtering
        st.markdown("### 📊 Asset Selection")
        
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            selected_sector = st.selectbox(
                "Sector Filter",
                ["All Sectors"] + list(SECTOR_MAPPING.keys()),
                help="Filter assets by sector"
            )
        
        with col_filter2:
            min_market_cap = st.selectbox(
                "Market Cap",
                ["Any", "Large Cap", "Mid Cap", "All"],
                help="Filter by market capitalization"
            )
        
        # Determine available tickers
        if selected_sector == "All Sectors":
            available_tickers = BIST100_TICKERS
        else:
            available_tickers = SECTOR_MAPPING.get(selected_sector, [])
        
        # Search box with advanced features
        search_query = st.text_input(
            "🔍 Search Ticker or Company",
            placeholder="Type to search...",
            help="Search by ticker symbol or company name"
        )
        
        if search_query:
            available_tickers = [
                t for t in available_tickers 
                if search_query.upper() in t or search_query.lower() in t.lower()
            ]
        
        # Multi-select with select all/deselect all
        col_select_all, col_deselect = st.columns(2)
        with col_select_all:
            if st.button("Select All", use_container_width=True, type="secondary"):
                st.session_state.selected_assets = available_tickers[:10]  # Limit to 10
        
        with col_deselect:
            if st.button("Deselect All", use_container_width=True, type="secondary"):
                st.session_state.selected_assets = []
        
        # Asset selection with limit
        max_assets = 15
        assets = st.multiselect(
            f"Select Assets (Max {max_assets})",
            available_tickers,
            default=['THYAO.IS', 'GARAN.IS', 'ASELS.IS', 'AKBNK.IS', 'FROTO.IS'],
            max_selections=max_assets,
            key="selected_assets",
            help=f"Select up to {max_assets} assets for portfolio optimization"
        )
        
        # Benchmark Selection with multiple benchmarks
        st.markdown("### 📈 Benchmark Selection")
        benchmark_symbol = st.selectbox(
            "Primary Benchmark",
            list(BENCHMARKS.keys()),
            index=0,
            help="Select primary benchmark for comparison"
        )
        
        # Additional benchmarks
        additional_benchmarks = st.multiselect(
            "Additional Benchmarks",
            [b for b in BENCHMARKS.keys() if b != benchmark_symbol],
            help="Select additional benchmarks for comparison"
        )
        
        # Advanced Optimization Parameters
        st.markdown("### ⚡ Advanced Optimization")
        
        optimization_method = st.selectbox(
            "Optimization Method",
            [
                OptimizationMethod.MAX_SHARPE.value,
                OptimizationMethod.MIN_VOLATILITY.value,
                OptimizationMethod.EFFICIENT_RISK.value,
                OptimizationMethod.EFFICIENT_RETURN.value,
                OptimizationMethod.HRP.value,
                OptimizationMethod.MAX_QUADRATIC_UTILITY.value
            ],
            help="Select portfolio optimization method"
        )
        
        col_risk, col_return = st.columns(2)
        with col_risk:
            risk_model = st.selectbox(
                "Risk Model",
                [
                    RiskModel.LEDOIT_WOLF.value,
                    RiskModel.SAMPLE_COV.value,
                    RiskModel.SEMICOVARIANCE.value,
                    RiskModel.EXPONENTIAL_COV.value,
                    RiskModel.ORACLE.value
                ],
                help="Select covariance estimation method"
            )
        
        with col_return:
            return_model = st.selectbox(
                "Return Model",
                [
                    ReturnModel.MEAN_HISTORICAL.value,
                    ReturnModel.EMA_HISTORICAL.value,
                    ReturnModel.CAPM.value
                ],
                help="Select expected return estimation method"
            )
        
        # Advanced Parameters
        with st.expander("🔬 Advanced Parameters", expanded=False):
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                0.0, 50.0, 30.0, 0.1,
                help="Annual risk-free rate in percentage"
            ) / 100
            
            if optimization_method == OptimizationMethod.EFFICIENT_RISK.value:
                target_volatility = st.slider(
                    "Target Volatility",
                    0.05, 0.50, 0.15, 0.01,
                    help="Target annual volatility for efficient risk optimization"
                )
            else:
                target_volatility = 0.15
            
            if optimization_method == OptimizationMethod.EFFICIENT_RETURN.value:
                target_return = st.slider(
                    "Target Return",
                    0.05, 1.0, 0.20, 0.01,
                    help="Target annual return for efficient return optimization"
                )
            else:
                target_return = 0.20
            
            # Advanced constraints
            enable_constraints = st.checkbox("Enable Portfolio Constraints", False)
            if enable_constraints:
                col_min, col_max = st.columns(2)
                with col_min:
                    min_weight = st.slider("Minimum Weight per Asset", 0.0, 0.3, 0.0, 0.01)
                with col_max:
                    max_weight = st.slider("Maximum Weight per Asset", 0.1, 1.0, 0.3, 0.01)
        
        # Reporting Options
        st.markdown("### 📊 Reporting Options")
        
        reporting_cols = st.columns(2)
        with reporting_cols[0]:
            generate_full_report = st.checkbox(
                "Full QuantStats Report",
                True,
                help="Generate comprehensive HTML report"
            )
            
            show_tearsheet = st.checkbox(
                "Interactive Tearsheet",
                True,
                help="Display interactive performance visualization"
            )
        
        with reporting_cols[1]:
            calculate_discrete = st.checkbox(
                "Discrete Allocation",
                False,
                help="Calculate actual share allocations"
            )
            
            monte_carlo_sim = st.checkbox(
                "Monte Carlo Simulation",
                False,
                help="Run Monte Carlo simulation"
            )
        
        if calculate_discrete:
            portfolio_value = st.number_input(
                "Portfolio Value (TRY)",
                10000, 10000000, 1000000, 10000,
                help="Total portfolio value for discrete allocation"
            )
        
        # Data Refresh Button
        st.markdown("---")
        col_refresh, col_reset = st.columns(2)
        with col_refresh:
            if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
                st.cache_data.clear()
                st.session_state.data_loaded = False
                st.rerun()
        
        with col_reset:
            if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main Dashboard Header with Animation
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem; 
        background: linear-gradient(135deg, rgba(10,25,41,0.95), rgba(26,37,54,0.95));
        border-radius: 25px; 
        margin-bottom: 2rem; 
        border: 1px solid rgba(45, 55, 72, 0.5);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0,102,204,0.1) 0%, transparent 70%);
            animation: float 20s infinite linear;
        "></div>
        
        <h1 class="animated-gradient-text" style="margin: 0; font-size: 3.5rem; position: relative;">
            📊 BIST Enterprise Portfolio Analytics Suite
        </h1>
        <p style="font-size: 1.3rem; color: #b0b0b0; margin-top: 1rem; position: relative;">
            Professional Portfolio Optimization & Risk Analytics Platform
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 2rem; position: relative;">
            <span class="badge-pulse badge-success">Real-time Analytics</span>
            <span class="badge-pulse badge-warning">Machine Learning</span>
            <span class="badge-pulse badge-danger">Risk Management</span>
            <span class="badge-pulse badge-info">AI Optimized</span>
        </div>
    </div>
    
    <style>
        @keyframes float {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check requirements
    if not HAS_YFINANCE:
        st.error("""
        ## ❌ Missing Required Packages
        
        Please install the following packages:
        ```bash
        pip install yfinance pypfopt quantstats streamlit plotly pandas numpy scipy
        ```
        
        For Streamlit Cloud, add these to your `requirements.txt` file.
        """)
        return
    
    if not HAS_PYPFOPT:
        st.warning("""
        ⚠️ **PyPortfolioOpt is not fully installed.** 
        Some optimization features may be limited. Please install:
        ```bash
        pip install pypfopt
        ```
        """)
    
    # Validate asset selection
    if len(assets) < 2:
        st.warning("""
        ⚠️ **Please select at least 2 assets for portfolio optimization.**
        
        - Use the asset selection panel in the sidebar
        - You can filter by sector or search for specific tickers
        - Select at least 2 different assets
        """)
        
        # Show quick selection options
        st.info("💡 **Quick Selection:** Try these popular BIST stocks:")
        col1, col2, col3, col4 = st.columns(4)
        
        popular_stocks = [
            ('THYAO.IS', 'Turkish Airlines'),
            ('GARAN.IS', 'Garanti Bank'),
            ('ASELS.IS', 'Aselsan'),
            ('AKBNK.IS', 'Akbank'),
            ('FROTO.IS', 'Ford Otosan'),
            ('SASA.IS', 'Sasa Polyester'),
            ('TUPRS.IS', 'Tüpraş'),
            ('YKBNK.IS', 'Yapı Kredi')
        ]
        
        for idx, (ticker, name) in enumerate(popular_stocks):
            col = [col1, col2, col3, col4][idx % 4]
            if col.button(f"{ticker}\n{name}", use_container_width=True):
                if 'selected_assets' in st.session_state:
                    st.session_state.selected_assets = list(set(st.session_state.selected_assets + [ticker]))
                else:
                    st.session_state.selected_assets = [ticker]
                st.rerun()
        
        return
    
    # Data Loading Section with Progress
    with st.spinner("🔄 Loading market data and performing analysis..."):
        # Initialize data source
        data_source = AdvancedDataSource()
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch portfolio data
        status_text.text("📥 Downloading portfolio data...")
        portfolio_data = data_source.fetch_market_data(
            assets,
            start_date,
            end_date
        )
        progress_bar.progress(30)
        
        if portfolio_data is None or portfolio_data.close.empty:
            st.error("""
            ❌ **Failed to load portfolio data!**
            
            **Troubleshooting:**
            1. Try selecting fewer stocks (2-5 stocks work best)
            2. Try a shorter date range (last 1 year)
            3. Check if Yahoo Finance is accessible
            4. Try these test tickers: ['THYAO.IS', 'GARAN.IS']
            """)
            return
        
        # Fetch benchmark data
        status_text.text("📊 Downloading benchmark data...")
        benchmark_data = data_source.fetch_market_data(
            [BENCHMARKS[benchmark_symbol]],
            start_date,
            end_date
        )
        progress_bar.progress(60)
        
        # Fetch additional benchmarks
        additional_benchmark_data = {}
        if additional_benchmarks:
            for bench in additional_benchmarks:
                bench_data = data_source.fetch_market_data(
                    [BENCHMARKS[bench]],
                    start_date,
                    end_date
                )
                if bench_data and not bench_data.close.empty:
                    additional_benchmark_data[bench] = bench_data.returns.iloc[:, 0]
        
        progress_bar.progress(80)
        
        # Store data in session state
        st.session_state.data_loaded = True
        st.session_state.portfolio_data = portfolio_data
        st.session_state.benchmark_data = benchmark_data
        
        # Initialize optimizer
        try:
            status_text.text("⚡ Optimizing portfolio...")
            optimizer = QuantitativePortfolioOptimizer(
                portfolio_data.close,
                portfolio_data.returns
            )
            
            # Optimize portfolio
            weights, performance = optimizer.optimize(
                method=optimization_method,
                risk_model=risk_model,
                return_model=return_model,
                target_volatility=target_volatility,
                target_return=target_return,
                risk_free_rate=risk_free_rate
            )
            
            # Calculate portfolio returns
            portfolio_returns = (portfolio_data.returns * pd.Series(weights)).sum(axis=1)
            
            # Get benchmark returns
            benchmark_returns = None
            if benchmark_data and not benchmark_data.returns.empty:
                benchmark_returns = benchmark_data.returns.iloc[:, 0]
            
            st.session_state.optimized = True
            st.session_state.optimization_results = {
                'weights': weights,
                'performance': performance,
                'portfolio_returns': portfolio_returns,
                'benchmark_returns': benchmark_returns
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            logger.error(f"Optimization error: {str(e)}")
            logger.error(traceback.format_exc())
            return
    
    # Performance Dashboard
    st.markdown("## 📈 Performance Dashboard")
    
    # Top Metrics in 3D cards
    metric_cols = st.columns(5)
    
    performance_metrics = st.session_state.optimization_results['performance']
    portfolio_returns = st.session_state.optimization_results['portfolio_returns']
    
    with metric_cols[0]:
        st.markdown('<div class="metric-card-3d">', unsafe_allow_html=True)
        st.metric(
            "Expected Return",
            f"{performance_metrics[0]:.2%}",
            delta=None,
            delta_color="normal",
            help="Annualized expected return"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown('<div class="metric-card-3d">', unsafe_allow_html=True)
        st.metric(
            "Expected Volatility",
            f"{performance_metrics[1]:.2%}",
            delta=None,
            help="Annualized portfolio volatility"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown('<div class="metric-card-3d">', unsafe_allow_html=True)
        st.metric(
            "Sharpe Ratio",
            f"{performance_metrics[2]:.2f}",
            delta=None,
            help="Risk-adjusted return (Sharpe Ratio)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[3]:
        var_95 = np.percentile(portfolio_returns, 5)
        st.markdown('<div class="metric-card-3d">', unsafe_allow_html=True)
        st.metric(
            "Daily VaR (95%)",
            f"{var_95:.2%}",
            delta=None,
            help="Value at Risk at 95% confidence"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[4]:
        if HAS_QUANTSTATS:
            max_dd = qs.stats.max_drawdown(portfolio_returns)
        else:
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()
        
        st.markdown('<div class="metric-card-3d">', unsafe_allow_html=True)
        st.metric(
            "Max Drawdown",
            f"{max_dd:.2%}",
            delta=None,
            help="Maximum historical drawdown"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Tabs with Advanced Features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Portfolio Overview",
        "📊 Optimization Analysis", 
        "⚠️ Risk Analytics",
        "📈 Performance Analytics",
        "🤖 AI Insights",
        "📑 Reports & Export"
    ])
    
    with tab1:
        # Portfolio Overview with Glass Effect
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("### 🎯 Optimal Allocation")
            
            # Convert weights to DataFrame
            weights = st.session_state.optimization_results['weights']
            weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
            weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            
            # Create interactive pie chart
            fig_pie = px.pie(
                weights_df,
                values='Weight',
                names=weights_df.index,
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.Viridis,
                labels={'Weight': 'Allocation %'},
                title="Portfolio Allocation",
                template="plotly_dark"
            )
            
            fig_pie.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=1.3,
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(255,255,255,0.2)',
                    borderwidth=1
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
                marker=dict(line=dict(color='#000000', width=2))
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Fundamental data for top holdings
            if len(weights_df) > 0:
                st.markdown("#### 📊 Top Holdings Fundamentals")
                top_holdings = weights_df.head(3).index.tolist()
                
                for ticker in top_holdings:
                    fund_data = data_source.fetch_fundamental_data(ticker)
                    if fund_data:
                        with st.expander(f"🔍 {ticker} - Fundamentals", expanded=False):
                            col_fund1, col_fund2 = st.columns(2)
                            with col_fund1:
                                if 'market_cap' in fund_data:
                                    st.metric("Market Cap", f"${fund_data['market_cap']/1e9:.1f}B")
                                if 'pe_ratio' in fund_data:
                                    st.metric("P/E Ratio", f"{fund_data['pe_ratio']:.1f}")
                            with col_fund2:
                                if 'dividend_yield' in fund_data:
                                    st.metric("Dividend Yield", f"{fund_data['dividend_yield']:.2%}")
                                if 'beta' in fund_data:
                                    st.metric("Beta", f"{fund_data['beta']:.2f}")
        
        with col_right:
            st.markdown("### 📈 Cumulative Performance")
            
            # Calculate cumulative returns
            cum_port = (1 + portfolio_returns).cumprod()
            
            fig_cum = go.Figure()
            
            fig_cum.add_trace(go.Scatter(
                x=cum_port.index,
                y=cum_port.values,
                name='Optimized Portfolio',
                line=dict(color='#00cc88', width=4, shape='spline'),
                hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(0, 204, 136, 0.1)'
            ))
            
            benchmark_returns = st.session_state.optimization_results.get('benchmark_returns')
            if benchmark_returns is not None:
                cum_bench = (1 + benchmark_returns).cumprod()
                fig_cum.add_trace(go.Scatter(
                    x=cum_bench.index,
                    y=cum_bench.values,
                    name=benchmark_symbol,
                    line=dict(color='#0066cc', width=3, dash='dash', shape='spline'),
                    hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>'
                ))
            
            # Add additional benchmarks
            for bench_name, bench_ret in additional_benchmark_data.items():
                cum_bench_add = (1 + bench_ret).cumprod()
                fig_cum.add_trace(go.Scatter(
                    x=cum_bench_add.index,
                    y=cum_bench_add.values,
                    name=bench_name,
                    line=dict(width=2, dash='dot'),
                    hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>',
                    visible='legendonly'
                ))
            
            fig_cum.update_layout(
                template="plotly_dark",
                height=500,
                hovermode='x unified',
                yaxis_title="Cumulative Return",
                xaxis_title="Date",
                yaxis_tickformat=".0%",
                title="Portfolio vs Benchmark Performance",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='rgba(255,255,255,0.2)',
                    borderwidth=1
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_cum, use_container_width=True)
            
            # Rolling metrics in expander
            with st.expander("📊 Rolling Metrics Analysis", expanded=False):
                rolling_window = st.slider("Rolling Window (days)", 30, 252, 126, 10)
                
                # Calculate rolling metrics
                rolling_sharpe = portfolio_returns.rolling(rolling_window).apply(
                    lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
                )
                
                rolling_vol = portfolio_returns.rolling(rolling_window).std() * np.sqrt(252)
                
                fig_rolling = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        f'Rolling Sharpe Ratio ({rolling_window} days)',
                        f'Rolling Volatility ({rolling_window} days)'
                    ),
                    vertical_spacing=0.15
                )
                
                fig_rolling.add_trace(
                    go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe.values,
                        name='Sharpe Ratio',
                        line=dict(color='#9d4edd', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(157, 78, 221, 0.2)',
                        hovertemplate='%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                fig_rolling.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values,
                        name='Volatility',
                        line=dict(color='#ff6b35', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(255, 107, 53, 0.2)',
                        hovertemplate='%{x|%Y-%m-%d}<br>Volatility: %{y:.2%}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig_rolling.update_layout(
                    height=500,
                    template="plotly_dark",
                    showlegend=False,
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                fig_rolling.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
                fig_rolling.update_yaxes(title_text="Volatility", tickformat=".0%", row=2, col=1)
                
                st.plotly_chart(fig_rolling, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Discrete allocation section
        if calculate_discrete:
            st.markdown("### 📦 Discrete Allocation")
            
            allocation, leftover = optimizer.calculate_discrete_allocation(
                weights, portfolio_value
            )
            
            if allocation:
                alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
                alloc_df['Price (TRY)'] = get_latest_prices(portfolio_data.close)[alloc_df.index]
                alloc_df['Value (TRY)'] = alloc_df['Shares'] * alloc_df['Price (TRY)']
                alloc_df['% of Portfolio'] = alloc_df['Value (TRY)'] / portfolio_value
                
                st.dataframe(
                    alloc_df.style.format({
                        'Shares': '{:,.0f}',
                        'Price (TRY)': '₺{:,.2f}',
                        'Value (TRY)': '₺{:,.2f}',
                        '% of Portfolio': '{:.2%}'
                    }).background_gradient(cmap='Greens', subset=['Value (TRY)']),
                    use_container_width=True,
                    height=300
                )
                
                col_alloc1, col_alloc2 = st.columns(2)
                with col_alloc1:
                    st.info(f"💰 **Total Invested:** ₺{alloc_df['Value (TRY)'].sum():,.2f}")
                with col_alloc2:
                    st.info(f"💵 **Remaining Cash:** ₺{leftover:,.2f}")
    
    with tab2:
        # Optimization Analysis
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## ⚡ Optimization Analysis")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            st.markdown("### Efficient Frontier")
            
            try:
                # Generate efficient frontier
                mus, sigmas, frontier_weights = optimizer.generate_efficient_frontier()
                
                fig_frontier = go.Figure()
                
                # Plot efficient frontier
                fig_frontier.add_trace(go.Scatter(
                    x=sigmas,
                    y=mus,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='white', width=4, shape='spline'),
                    fill='tonexty',
                    fillcolor='rgba(255, 255, 255, 0.1)',
                    hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
                
                # Plot optimal portfolio
                fig_frontier.add_trace(go.Scatter(
                    x=[performance_metrics[1]],
                    y=[performance_metrics[0]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=25,
                        symbol='star',
                        line=dict(color='white', width=3)
                    ),
                    name='Optimal Portfolio',
                    hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{text:.2f}<extra></extra>',
                    text=[performance_metrics[2]]
                ))
                
                # Plot individual assets
                individual_returns = []
                individual_risks = []
                
                for asset in portfolio_data.close.columns:
                    asset_returns = portfolio_data.returns[asset]
                    asset_return = asset_returns.mean() * 252
                    asset_risk = asset_returns.std() * np.sqrt(252)
                    
                    fig_frontier.add_trace(go.Scatter(
                        x=[asset_risk],
                        y=[asset_return],
                        mode='markers+text',
                        marker=dict(size=15, color='lightblue', opacity=0.7),
                        text=[asset],
                        textposition="top center",
                        name=asset,
                        showlegend=False,
                        hovertemplate='%{text}<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                    ))
                
                fig_frontier.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis_title="Annualized Volatility (Risk)",
                    yaxis_title="Annualized Return",
                    title="Efficient Frontier with Individual Assets",
                    hovermode='closest',
                    xaxis_tickformat=".0%",
                    yaxis_tickformat=".0%",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        bgcolor='rgba(0,0,0,0.5)',
                        bordercolor='rgba(255,255,255,0.2)',
                        borderwidth=1
                    )
                )
                
                st.plotly_chart(fig_frontier, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating efficient frontier: {str(e)}")
        
        with col_opt2:
            st.markdown("### Optimization Parameters")
            
            # Display optimization settings
            opt_params = {
                'Method': optimization_method.replace('_', ' ').title(),
                'Risk Model': risk_model.replace('_', ' ').title(),
                'Return Model': return_model.replace('_', ' ').title(),
                'Risk-Free Rate': f"{risk_free_rate:.2%}",
                'Target Volatility': f"{target_volatility:.2%}" if optimization_method == OptimizationMethod.EFFICIENT_RISK.value else 'N/A',
                'Target Return': f"{target_return:.2%}" if optimization_method == OptimizationMethod.EFFICIENT_RETURN.value else 'N/A',
                'Number of Assets': len(assets),
                'Date Range': f"{start_date} to {end_date}",
                'Data Points': len(portfolio_returns)
            }
            
            for key, value in opt_params.items():
                col_key, col_val = st.columns([1, 2])
                with col_key:
                    st.markdown(f"**{key}:**")
                with col_val:
                    st.markdown(f"{value}")
            
            # Weight distribution analysis
            st.markdown("#### Weight Distribution")
            
            weights_series = pd.Series(weights)
            
            fig_weights = px.histogram(
                x=weights_series.values,
                nbins=20,
                title="Weight Distribution",
                labels={'x': 'Weight', 'y': 'Count'},
                template="plotly_dark"
            )
            
            fig_weights.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_weights, use_container_width=True)
            
            # Concentration metrics
            st.markdown("#### Concentration Analysis")
            
            col_conc1, col_conc2, col_conc3 = st.columns(3)
            
            with col_conc1:
                # Herfindahl-Hirschman Index (HHI)
                hhi = np.sum(weights_series.values ** 2)
                st.metric("HHI Index", f"{hhi:.4f}")
            
            with col_conc2:
                # Gini coefficient
                sorted_weights = np.sort(weights_series.values)
                n = len(sorted_weights)
                cum_weights = np.cumsum(sorted_weights)
                gini = (n + 1 - 2 * np.sum(cum_weights) / cum_weights[-1]) / n
                st.metric("Gini Coefficient", f"{gini:.3f}")
            
            with col_conc3:
                # Top 3 concentration
                top3_concentration = weights_series.nlargest(3).sum()
                st.metric("Top 3 Concentration", f"{top3_concentration:.1%}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Risk Analytics
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## ⚠️ Advanced Risk Analytics")
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            st.markdown("### 📉 Value at Risk Analysis")
            
            # Calculate VaR at different confidence levels
            confidence_levels = [0.90, 0.95, 0.99]
            var_methods = ['Historical', 'Parametric (Normal)', 'Cornish-Fisher']
            
            var_data = []
            for cl in confidence_levels:
                # Historical VaR
                var_hist = np.percentile(portfolio_returns, (1 - cl) * 100)
                cvar_hist = portfolio_returns[portfolio_returns <= var_hist].mean()
                
                # Parametric VaR (Normal)
                var_param = portfolio_returns.mean() + stats.norm.ppf(1 - cl) * portfolio_returns.std()
                
                # Cornish-Fisher VaR
                z = stats.norm.ppf(1 - cl)
                s = stats.skew(portfolio_returns)
                k = stats.kurtosis(portfolio_returns)
                z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
                var_cf = portfolio_returns.mean() + z_cf * portfolio_returns.std()
                
                var_data.append({
                    'Confidence': f'{int(cl*100)}%',
                    'Historical': var_hist,
                    'Parametric': var_param,
                    'Cornish-Fisher': var_cf,
                    'CVaR': cvar_hist
                })
            
            var_df = pd.DataFrame(var_data)
            
            # Display VaR table
            st.dataframe(
                var_df.style.format({
                    'Historical': '{:.4f}',
                    'Parametric': '{:.4f}',
                    'Cornish-Fisher': '{:.4f}',
                    'CVaR': '{:.4f}'
                }).background_gradient(cmap='Reds_r', subset=['Historical', 'CVaR']),
                use_container_width=True,
                height=200
            )
            
            # VaR visualization
            fig_var = go.Figure()
            
            for idx, row in var_df.iterrows():
                fig_var.add_trace(go.Bar(
                    name=row['Confidence'],
                    x=['Historical', 'Parametric', 'Cornish-Fisher'],
                    y=[row['Historical'], row['Parametric'], row['Cornish-Fisher']],
                    text=[f'{row["Historical"]:.3%}', f'{row["Parametric"]:.3%}', f'{row["Cornish-Fisher"]:.3%}'],
                    textposition='auto',
                    hovertemplate='Method: %{x}<br>VaR: %{y:.3%}<extra></extra>'
                ))
            
            fig_var.update_layout(
                title='Value at Risk by Confidence Level',
                template='plotly_dark',
                height=400,
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col_risk2:
            st.markdown("### 📊 Drawdown Analysis")
            
            # Calculate drawdown series
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown_series = (cum_returns - running_max) / running_max
            
            fig_dd = go.Figure()
            
            fig_dd.add_trace(go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values,
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=3),
                name='Drawdown',
                hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>'
            ))
            
            # Find and mark significant drawdowns
            threshold = -0.05  # 5% drawdown threshold
            significant_dd = drawdown_series[drawdown_series < threshold]
            
            if not significant_dd.empty:
                for date, value in significant_dd.nsmallest(3).items():
                    fig_dd.add_annotation(
                        x=date,
                        y=value,
                        text=f"{value:.1%}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="yellow",
                        font=dict(color="yellow", size=12)
                    )
            
            fig_dd.update_layout(
                template="plotly_dark",
                height=400,
                title="Portfolio Drawdown Analysis",
                yaxis_title="Drawdown",
                yaxis_tickformat=".0%",
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Drawdown statistics
            if len(drawdown_series) > 0:
                dd_stats = {
                    'Max Drawdown': drawdown_series.min(),
                    'Average Drawdown': drawdown_series.mean(),
                    'Drawdown Days': len(drawdown_series[drawdown_series < 0]),
                    'Recovery Days (Avg)': 0,  # Would need more complex calculation
                    'Pain Index': abs(drawdown_series[drawdown_series < 0].mean())
                }
                
                dd_stats_df = pd.DataFrame.from_dict(dd_stats, orient='index', columns=['Value'])
                st.dataframe(
                    dd_stats_df.style.format('{:.4f}'),
                    use_container_width=True,
                    height=200
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        # Performance Analytics
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 📈 Advanced Performance Analytics")
        
        # Initialize analytics engine
        analytics_engine = AdvancedAnalyticsEngine(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate
        )
        
        # Calculate comprehensive metrics
        with st.spinner("Calculating advanced metrics..."):
            advanced_metrics = analytics_engine.calculate_comprehensive_metrics()
        
        # Display metrics in categorized sections
        st.markdown("### 📊 Performance Metrics")
        
        # Return Metrics
        with st.expander("📈 Return Metrics", expanded=True):
            return_cols = st.columns(4)
            return_metrics = {
                'Total Return': advanced_metrics.get('Total Return', 0),
                'CAGR': advanced_metrics.get('CAGR', 0),
                'Annual Return': performance_metrics[0],
                'Best Day': advanced_metrics.get('Avg Win', 0),
                'Worst Day': advanced_metrics.get('Avg Loss', 0),
                'Win Rate': advanced_metrics.get('Win Rate', 0),
                'Profit Factor': advanced_metrics.get('Profit Factor', 0)
            }
            
            for idx, (key, value) in enumerate(return_metrics.items()):
                with return_cols[idx % 4]:
                    if isinstance(value, (int, float)):
                        if 'Rate' in key or 'Factor' in key:
                            display_val = f"{value:.2f}"
                        elif 'Return' in key or 'Day' in key:
                            display_val = f"{value:.2%}"
                        else:
                            display_val = f"{value:.4f}"
                        st.metric(key, display_val)
        
        # Risk Metrics
        with st.expander("⚠️ Risk Metrics"):
            risk_cols = st.columns(4)
            risk_metrics = {
                'Annual Volatility': advanced_metrics.get('Annual Volatility', 0),
                'Downside Deviation': advanced_metrics.get('Annual Downside Deviation', 0),
                'Max Drawdown': advanced_metrics.get('Max Drawdown', 0),
                'VaR (95%)': advanced_metrics.get('VaR (95%)', 0),
                'CVaR (95%)': advanced_metrics.get('CVaR (95%)', 0),
                'Skewness': advanced_metrics.get('Skewness', 0),
                'Kurtosis': advanced_metrics.get('Kurtosis', 0)
            }
            
            for idx, (key, value) in enumerate(risk_metrics.items()):
                with risk_cols[idx % 4]:
                    if isinstance(value, (int, float)):
                        if 'Volatility' in key or 'Drawdown' in key or 'VaR' in key or 'CVaR' in key:
                            display_val = f"{value:.2%}"
                        else:
                            display_val = f"{value:.3f}"
                        st.metric(key, display_val)
        
        # Risk-Adjusted Metrics
        with st.expander("📐 Risk-Adjusted Metrics"):
            ratio_cols = st.columns(4)
            ratio_metrics = {
                'Sharpe Ratio': advanced_metrics.get('Sharpe Ratio', 0),
                'Sortino Ratio': advanced_metrics.get('Sortino Ratio', 0),
                'Calmar Ratio': advanced_metrics.get('Calmar Ratio', 0),
                'Omega Ratio': advanced_metrics.get('Omega Ratio', 0),
                'Information Ratio': advanced_metrics.get('Information Ratio', 0),
                'Alpha': advanced_metrics.get('Alpha', 0),
                'Beta': advanced_metrics.get('Beta', 0),
                'Tracking Error': advanced_metrics.get('Tracking Error', 0)
            }
            
            for idx, (key, value) in enumerate(ratio_metrics.items()):
                with ratio_cols[idx % 4]:
                    if isinstance(value, (int, float)):
                        if 'Alpha' in key or 'Tracking Error' in key:
                            display_val = f"{value:.2%}"
                        else:
                            display_val = f"{value:.3f}"
                        st.metric(key, display_val)
        
        # Generate tearsheet if requested
        if show_tearsheet:
            st.markdown("### 📊 Interactive Tearsheet")
            
            with st.spinner("Generating advanced tearsheet..."):
                try:
                    tearsheet_fig = analytics_engine.create_interactive_tearsheet()
                    st.plotly_chart(tearsheet_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to generate tearsheet: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        # AI Insights (Placeholder for advanced features)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 🤖 AI-Powered Insights")
        
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            st.markdown("### 🧠 Predictive Analytics")
            
            # Monte Carlo Simulation
            if monte_carlo_sim:
                with st.spinner("Running Monte Carlo simulation..."):
                    try:
                        mc_results = optimizer.monte_carlo_simulation(
                            n_simulations=5000,
                            time_horizon=252
                        )
                        
                        if mc_results:
                            st.markdown("#### Monte Carlo Simulation Results")
                            
                            # Display key statistics
                            col_mc1, col_mc2, col_mc3 = st.columns(3)
                            
                            with col_mc1:
                                st.metric(
                                    "Mean Final Value",
                                    f"{mc_results['mean_final_value']:.2f}"
                                )
                            
                            with col_mc2:
                                st.metric(
                                    "VaR (95%)",
                                    f"{mc_results['var_95']:.2f}"
                                )
                            
                            with col_mc3:
                                st.metric(
                                    "CVaR (95%)",
                                    f"{mc_results['cvar_95']:.2f}"
                                )
                            
                            # Plot distribution of final values
                            fig_mc = go.Figure()
                            
                            fig_mc.add_trace(go.Histogram(
                                x=mc_results['final_values'],
                                nbinsx=50,
                                name='Final Values',
                                marker_color='#4facfe',
                                opacity=0.7,
                                hovertemplate='Final Value: %{x:.2f}<br>Count: %{y}<extra></extra>'
                            ))
                            
                            fig_mc.update_layout(
                                title='Monte Carlo Simulation - Final Value Distribution',
                                template='plotly_dark',
                                height=400,
                                xaxis_title='Final Portfolio Value',
                                yaxis_title='Frequency',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig_mc, use_container_width=True)
                    except Exception as e:
                        st.error(f"Monte Carlo simulation failed: {str(e)}")
            else:
                st.info("Enable Monte Carlo Simulation in sidebar to see predictive analytics")
            
            # Portfolio Stress Testing
            st.markdown("#### 📉 Stress Testing")
            
            stress_scenarios = {
                'Market Crash (-20%)': -0.20,
                'Correction (-10%)': -0.10,
                'Volatility Spike (+50%)': 0.50,
                'Interest Rate Hike (+2%)': 0.02
            }
            
            for scenario, impact in stress_scenarios.items():
                with st.expander(f"📊 {scenario}", expanded=False):
                    col_stress1, col_stress2 = st.columns(2)
                    with col_stress1:
                        st.metric("Scenario Impact", f"{impact:.1%}")
                    with col_stress2:
                        # Simplified impact calculation
                        stressed_return = performance_metrics[0] + impact
                        stressed_sharpe = (stressed_return - risk_free_rate) / performance_metrics[1] if performance_metrics[1] > 0 else 0
                        st.metric("Estimated Sharpe", f"{stressed_sharpe:.2f}")
        
        with col_ai2:
            st.markdown("### 🔍 Smart Recommendations")
            
            # Portfolio optimization suggestions
            st.markdown("#### ⚡ Optimization Suggestions")
            
            suggestions = []
            
            # Check for high concentration
            weights_series = pd.Series(weights)
            top3_concentration = weights_series.nlargest(3).sum()
            
            if top3_concentration > 0.6:  # More than 60% in top 3
                suggestions.append(f"⚠️ High concentration in top 3 holdings ({top3_concentration:.1%}) - consider diversifying")
            
            # Check for low Sharpe ratio
            if performance_metrics[2] < 0.5:
                suggestions.append(f"📉 Low Sharpe ratio ({performance_metrics[2]:.2f}) - consider risk-adjusted optimization")
            
            # Check for high volatility
            if performance_metrics[1] > 0.25:
                suggestions.append(f"⚡ High volatility ({performance_metrics[1]:.1%}) - consider volatility targeting")
            
            # Check for negative skewness
            skewness = advanced_metrics.get('Skewness', 0)
            if skewness < -0.5:
                suggestions.append(f"📊 Negative return skew ({skewness:.2f}) - tail risk may be underestimated")
            
            if suggestions:
                for suggestion in suggestions:
                    st.warning(suggestion)
            else:
                st.success("✅ Portfolio appears well-optimized based on current metrics")
            
            # Sector allocation analysis
            st.markdown("#### 📈 Sector Allocation")
            
            sector_allocation = {}
            for ticker, weight in weights.items():
                for sector, tickers in SECTOR_MAPPING.items():
                    if ticker in tickers:
                        sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
                        break
            
            if sector_allocation:
                sector_df = pd.DataFrame.from_dict(sector_allocation, orient='index', columns=['Weight'])
                sector_df = sector_df.sort_values('Weight', ascending=False)
                
                fig_sector = px.pie(
                    sector_df,
                    values='Weight',
                    names=sector_df.index,
                    title="Portfolio Sector Allocation",
                    template="plotly_dark"
                )
                
                fig_sector.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_sector, use_container_width=True)
            
            # Rebalancing recommendations
            st.markdown("#### 🔄 Rebalancing Insights")
            
            col_rebal1, col_rebal2 = st.columns(2)
            
            with col_rebal1:
                # Calculate tracking error from target weights
                target_weights = np.array(list(weights.values()))
                current_weights = target_weights  # Simplified - in reality would compare to current holdings
                tracking_error = np.sqrt(np.sum((current_weights - target_weights) ** 2))
                
                if tracking_error > 0.1:
                    st.metric("Rebalancing Need", "High", delta=f"{tracking_error:.1%}")
                else:
                    st.metric("Rebalancing Need", "Low", delta=f"{tracking_error:.1%}")
            
            with col_rebal2:
                # Suggested rebalancing frequency
                st.metric("Suggested Frequency", "Quarterly")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        # Reports & Export
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 📑 Professional Reporting & Export")
        
        col_report1, col_report2 = st.columns(2)
        
        with col_report1:
            st.markdown("### 📊 Report Generation")
            
            # QuantStats Full Report
            if generate_full_report and HAS_QUANTSTATS:
                st.markdown("#### QuantStats Full Report")
                
                if st.button("📈 Generate Comprehensive Report", use_container_width=True, type="primary"):
                    with st.spinner("Generating comprehensive QuantStats report..."):
                        try:
                            # Initialize analytics engine
                            analytics_engine = AdvancedAnalyticsEngine(
                                portfolio_returns,
                                benchmark_returns,
                                risk_free_rate
                            )
                            
                            # Generate HTML report
                            buffer = io.StringIO()
                            
                            if benchmark_returns is not None:
                                qs.reports.html(
                                    portfolio_returns,
                                    benchmark_returns,
                                    rf=risk_free_rate,
                                    title='BIST Portfolio Analytics Report',
                                    output=buffer
                                )
                            else:
                                qs.reports.html(
                                    portfolio_returns,
                                    rf=risk_free_rate,
                                    title='Portfolio Analytics Report',
                                    output=buffer
                                )
                            
                            html_report = buffer.getvalue()
                            buffer.close()
                            
                            # Display and provide download
                            with st.expander("📋 Preview Report", expanded=True):
                                st.components.v1.html(html_report, height=600, scrolling=True)
                            
                            # Download button
                            b64 = base64.b64encode(html_report.encode()).decode()
                            href = f'''
                            <a href="data:text/html;base64,{b64}" 
                               download="bist_portfolio_report.html"
                               style="text-decoration: none;">
                               <button style="
                                   background: var(--gradient-1);
                                   color: white;
                                   padding: 1rem 2rem;
                                   border: none;
                                   border-radius: 10px;
                                   cursor: pointer;
                                   font-weight: bold;
                                   font-size: 1rem;
                                   width: 100%;
                                   margin-top: 1rem;
                               ">
                               📥 Download HTML Report
                               </button>
                            </a>
                            '''
                            st.markdown(href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Failed to generate QuantStats report: {str(e)}")
            else:
                st.info("Enable 'Full QuantStats Report' in sidebar to generate comprehensive reports")
            
            # Performance Summary Card
            st.markdown("#### 📋 Performance Summary")
            
            summary_metrics = {
                'Optimization Method': optimization_method.replace('_', ' ').title(),
                'Expected Return': f"{performance_metrics[0]:.2%}",
                'Expected Volatility': f"{performance_metrics[1]:.2%}",
                'Sharpe Ratio': f"{performance_metrics[2]:.2f}",
                'Max Drawdown': f"{advanced_metrics.get('Max Drawdown', 0):.2%}",
                'Win Rate': f"{advanced_metrics.get('Win Rate', 0):.1%}",
                'Number of Assets': len(assets),
                'Date Range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            }
            
            for key, value in summary_metrics.items():
                col_key, col_val = st.columns([2, 1])
                with col_key:
                    st.markdown(f"**{key}:**")
                with col_val:
                    st.markdown(f"{value}")
        
        with col_report2:
            st.markdown("### 📤 Data Export")
            
            # Export options
            export_options = st.multiselect(
                "Select Data to Export",
                ['Portfolio Weights', 'Returns Data', 'Performance Metrics', 
                 'Optimization Parameters', 'Risk Metrics', 'All Data'],
                default=['Portfolio Weights', 'Performance Metrics']
            )
            
            if st.button("📊 Export Selected Data", use_container_width=True):
                export_data = {}
                
                if 'Portfolio Weights' in export_options or 'All Data' in export_options:
                    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                    export_data['weights'] = weights_df
                
                if 'Returns Data' in export_options or 'All Data' in export_options:
                    returns_df = pd.DataFrame({
                        'Portfolio': portfolio_returns,
                        'Benchmark': benchmark_returns if benchmark_returns is not None else np.nan
                    })
                    export_data['returns'] = returns_df
                
                if 'Performance Metrics' in export_options or 'All Data' in export_options:
                    metrics_df = pd.DataFrame.from_dict(advanced_metrics, orient='index', columns=['Value'])
                    export_data['metrics'] = metrics_df
                
                if 'Optimization Parameters' in export_options or 'All Data' in export_options:
                    params = {
                        'optimization_method': optimization_method,
                        'risk_model': risk_model,
                        'return_model': return_model,
                        'risk_free_rate': risk_free_rate,
                        'target_volatility': target_volatility if optimization_method == OptimizationMethod.EFFICIENT_RISK.value else None,
                        'target_return': target_return if optimization_method == OptimizationMethod.EFFICIENT_RETURN.value else None,
                        'assets': assets,
                        'benchmark': benchmark_symbol,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    }
                    export_data['parameters'] = pd.DataFrame.from_dict(params, orient='index', columns=['Value'])
                
                # Create Excel file with multiple sheets
                if export_data:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        for sheet_name, df in export_data.items():
                            df.to_excel(writer, sheet_name=sheet_name)
                    
                    buffer.seek(0)
                    
                    # Download button
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                    href = f'''
                    <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" 
                       download="portfolio_analysis.xlsx"
                       style="text-decoration: none;">
                       <button style="
                           background: var(--gradient-3);
                           color: white;
                           padding: 1rem 2rem;
                           border: none;
                           border-radius: 10px;
                           cursor: pointer;
                           font-weight: bold;
                           font-size: 1rem;
                           width: 100%;
                           margin-top: 1rem;
                       ">
                       📥 Download Excel Report
                       </button>
                    </a>
                    '''
                    st.markdown(href, unsafe_allow_html=True)
            
            # Individual CSV exports
            st.markdown("#### 📁 Individual Data Exports")
            
            export_cols = st.columns(3)
            
            with export_cols[0]:
                if st.button("Export Weights CSV", use_container_width=True):
                    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                    csv = weights_df.to_csv()
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'''
                    <a href="data:file/csv;base64,{b64}" download="portfolio_weights.csv">
                    <button style="
                        background: rgba(79, 172, 254, 0.2);
                        color: white;
                        padding: 0.5rem;
                        border: 1px solid #4facfe;
                        border-radius: 8px;
                        cursor: pointer;
                        font-weight: bold;
                        width: 100%;
                    ">
                    📥 Weights CSV
                    </button>
                    </a>
                    '''
                    st.markdown(href, unsafe_allow_html=True)
            
            with export_cols[1]:
                if st.button("Export Returns CSV", use_container_width=True):
                    returns_df = pd.DataFrame({
                        'Portfolio': portfolio_returns,
                        'Benchmark': benchmark_returns if benchmark_returns is not None else np.nan
                    })
                    csv = returns_df.to_csv()
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'''
                    <a href="data:file/csv;base64,{b64}" download="returns_data.csv">
                    <button style="
                        background: rgba(79, 172, 254, 0.2);
                        color: white;
                        padding: 0.5rem;
                        border: 1px solid #4facfe;
                        border-radius: 8px;
                        cursor: pointer;
                        font-weight: bold;
                        width: 100%;
                    ">
                    📥 Returns CSV
                    </button>
                    </a>
                    '''
                    st.markdown(href, unsafe_allow_html=True)
            
            with export_cols[2]:
                if st.button("Export Metrics CSV", use_container_width=True):
                    metrics_df = pd.DataFrame.from_dict(advanced_metrics, orient='index', columns=['Value'])
                    csv = metrics_df.to_csv()
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'''
                    <a href="data:file/csv;base64,{b64}" download="performance_metrics.csv">
                    <button style="
                        background: rgba(79, 172, 254, 0.2);
                        color: white;
                        padding: 0.5rem;
                        border: 1px solid #4facfe;
                        border-radius: 8px;
                        cursor: pointer;
                        font-weight: bold;
                        width: 100%;
                    ">
                    📥 Metrics CSV
                    </button>
                    </a>
                    '''
                    st.markdown(href, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with advanced information
    st.markdown("---")
    
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("""
        ### 📚 Resources
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [PyPortfolioOpt Guide](https://pyportfolioopt.readthedocs.io)
        - [QuantStats Documentation](https://github.com/ranaroussi/quantstats)
        - [Yahoo Finance API](https://github.com/ranaroussi/yfinance)
        """)
    
    with col_footer2:
        st.markdown("""
        ### ⚠️ Disclaimer
        This tool is for educational and research purposes only. 
        Past performance does not guarantee future results. 
        Always conduct your own research and consult with financial 
        advisors before making investment decisions.
        """)
    
    with col_footer3:
        st.markdown("""
        ### 🔧 Technical Details
        - **Version:** 9.0 ULTRA
        - **Last Updated:** """ + datetime.now().strftime("%Y-%m-%d") + """
        - **Data Source:** Yahoo Finance
        - **Optimization:** PyPortfolioOpt
        - **Analytics:** QuantStats
        - **Visualization:** Plotly
        """)
    
    st.markdown("""
    <div style="text-align: center; color: #b0b0b0; font-size: 0.9rem; padding: 2rem;">
        <p>BIST Enterprise Portfolio Analytics Suite v9.0 ULTRA | Powered by Streamlit, PyPortfolioOpt & QuantStats</p>
        <p>📊 Institutional-Grade Portfolio Optimization & Risk Management Platform</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION ENTRY POINT WITH ADVANCED ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        # Check for required packages
        if not HAS_YFINANCE:
            st.error("""
            ## ❌ Missing Required Packages
            
            **Critical Error:** yfinance is not installed.
            
            Please install with:
            ```bash
            pip install yfinance pypfopt quantstats streamlit plotly pandas numpy scipy
            ```
            
            For Streamlit Cloud, ensure `requirements.txt` includes:
            ```
            yfinance>=0.2.28
            pypfopt>=1.5.5
            quantstats>=0.0.62
            streamlit>=1.28.0
            plotly>=5.17.0
            pandas>=2.0.0
            numpy>=1.24.0
            scipy>=1.11.0
            scikit-learn>=1.3.0
            ```
            """)
        else:
            # Run the main application
            main()
            
    except Exception as e:
        st.error(f"""
        ## 🚨 Application Error
        
        **Error Type:** {type(e).__name__}
        **Error Details:** {str(e)}
        """)
        
        # Show detailed error information in expander
        with st.expander("🔍 View Technical Details & Traceback"):
            st.code(traceback.format_exc(), language="python")
        
        # Show system information
        with st.expander("🖥️ System Information"):
            col_sys1, col_sys2 = st.columns(2)
            
            with col_sys1:
                st.write("**Python Version:**", sys.version.split()[0])
                st.write("**Platform:**", sys.platform)
            
            with col_sys2:
                try:
                    import pandas as pd
                    st.write("**Pandas Version:**", pd.__version__)
                except:
                    st.write("**Pandas:** Not installed")
                
                try:
                    import numpy as np
                    st.write("**NumPy Version:**", np.__version__)
                except:
                    st.write("**NumPy:** Not installed")
        
        # Troubleshooting guide
        st.markdown("""
        ## 🔧 Troubleshooting Guide
        
        1. **Check Internet Connection:**
           - The app needs to download data from Yahoo Finance
           - Test connectivity: https://finance.yahoo.com
        
        2. **Package Installation Issues:**
           ```bash
           # Try installing with specific versions
           pip install yfinance==0.2.28
           pip install pypfopt==1.5.5
           pip install quantstats==0.0.62
           ```
        
        3. **Data Fetching Issues:**
           - Try selecting fewer stocks (2-5)
           - Use a shorter date range (last 1-2 years)
           - Some tickers might not be available on Yahoo Finance
        
        4. **Memory/Performance Issues:**
           - Reduce the number of selected assets
           - Use a shorter date range
           - Close other applications using high memory
        
        5. **Streamlit Cloud Specific:**
           - Check the logs in Streamlit Cloud dashboard
           - Ensure `requirements.txt` is correctly formatted
           - The app may need a few minutes to build initially
        """)
        
        # Quick test buttons
        st.markdown("### 🧪 Quick Diagnostic Tests")
        
        col_test1, col_test2, col_test3 = st.columns(3)
        
        with col_test1:
            if st.button("Test yfinance", use_container_width=True):
                try:
                    import yfinance as yf
                    test_data = yf.download("THYAO.IS", period="1mo", progress=False)
                    if not test_data.empty:
                        st.success(f"✅ yfinance working! Shape: {test_data.shape}")
                    else:
                        st.error("❌ yfinance returned empty data")
                except Exception as e:
                    st.error(f"❌ yfinance test failed: {e}")
        
        with col_test2:
            if st.button("Test PyPortfolioOpt", use_container_width=True):
                try:
                    from pypfopt import expected_returns
                    st.success("✅ PyPortfolioOpt imported successfully!")
                except Exception as e:
                    st.error(f"❌ PyPortfolioOpt test failed: {e}")
        
        with col_test3:
            if st.button("Refresh Application", use_container_width=True, type="primary"):
                st.cache_data.clear()
                if 'initialized' in st.session_state:
                    del st.session_state.initialized
                st.rerun()
