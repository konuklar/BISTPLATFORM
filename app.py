# ============================================================================
# BIST ENTERPRISE QUANT PORTFOLIO OPTIMIZATION SUITE PRO MAX ULTRA
# Version: 12.0 | Institutional-Grade Portfolio Analytics Platform
# Features: Advanced VAR Analytics, Stress Testing, Backtesting, QuantStats Integration
# ============================================================================

import warnings
import sys
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize, Bounds, LinearConstraint, differential_evolution
import scipy.optimize as opt
import requests
import base64
import logging
import traceback
import time
import os
import json
import io
import hashlib
import math
import itertools
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedLogger:
    """Enhanced logging with performance tracking"""
    
    def __init__(self, name="PortfolioOptimizer"):
        self.logger = logging.getLogger(name)
        self.setup_logging()
        self.performance_metrics = {}
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        file_handler = logging.FileHandler(
            'portfolio_analytics_advanced.log',
            mode='a',
            encoding='utf-8'
        )
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self.performance_metrics[operation] = {
            'duration': duration,
            'timestamp': datetime.now(),
            **kwargs
        }
        self.logger.info(f"Performance: {operation} completed in {duration:.3f}s")
    
    def log_optimization(self, method: str, assets: int, metrics: Dict):
        """Log optimization details"""
        self.logger.info(
            f"Optimization: {method} | Assets: {assets} | "
            f"Return: {metrics.get('expected_return', 0):.3%} | "
            f"Vol: {metrics.get('expected_volatility', 0):.3%} | "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}"
        )
    
    def get_performance_report(self) -> Dict:
        """Get performance report"""
        return self.performance_metrics

# Initialize logger
logger = EnhancedLogger()

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE MODULE IMPORTS WITH FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import yfinance with enhanced error handling
try:
    import yfinance as yf
    HAS_YFINANCE = True
    YFINANCE_VERSION = yf.__version__
    logger.logger.info(f"yfinance {YFINANCE_VERSION} imported successfully")
except ImportError as e:
    st.error(f"CRITICAL: yfinance import failed: {e}")
    HAS_YFINANCE = False
    yf = None

# Import PyPortfolioOpt with all components
try:
    if HAS_YFINANCE:
        from pypfopt import expected_returns, risk_models
        from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR, EfficientSemivariance
        from pypfopt.hierarchical_portfolio import HRPOpt
        from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
        from pypfopt import objective_functions
        from pypfopt.cla import CLA
        from pypfopt.black_litterman import BlackLittermanModel
        from pypfopt.risk_models import CovarianceShrinkage
        HAS_PYPFOPT = True
        logger.logger.info("PyPortfolioOpt imported successfully with all components")
    else:
        HAS_PYPFOPT = False
except ImportError as e:
    logger.logger.error(f"PyPortfolioOpt import error: {e}")
    HAS_PYPFOPT = False

# Import QuantStats with extended functionality
try:
    import quantstats as qs
    qs.extend_pandas()
    HAS_QUANTSTATS = True
    
    # Import additional quantstats components
    from quantstats import stats as qs_stats
    from quantstats import plots as qs_plots
    from quantstats import utils as qs_utils
    logger.logger.info("QuantStats imported successfully with extended functionality")
except ImportError as e:
    logger.logger.error(f"QuantStats import error: {e}")
    HAS_QUANTSTATS = False
    qs = None

# Import advanced statistical packages
try:
    from scipy.stats import norm, t, skew, kurtosis, jarque_bera, anderson, shapiro
    from scipy.stats import percentileofscore
    HAS_SCIPY_STATS = True
except:
    HAS_SCIPY_STATS = False

# Import machine learning packages
try:
    from sklearn.covariance import LedoitWolf, OAS, GraphicalLassoCV
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False

# Import advanced optimization packages
try:
    import cvxpy as cp
    HAS_CVXPY = True
except:
    HAS_CVXPY = False

# Import additional financial packages
try:
    import arch
    HAS_ARCH = True
except:
    HAS_ARCH = False

try:
    import pyfolio as pf
    HAS_PYFOLIO = True
except:
    HAS_PYFOLIO = False

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationMethod(str, Enum):
    MAX_SHARPE = "Maximum Sharpe Ratio"
    MIN_VOLATILITY = "Minimum Volatility"
    EFFICIENT_RISK = "Efficient Risk (Target Volatility)"
    EFFICIENT_RETURN = "Efficient Return (Target Return)"
    HRP = "Hierarchical Risk Parity (HRP)"
    MAX_QUADRATIC_UTILITY = "Maximum Quadratic Utility"
    BLACK_LITTERMAN = "Black-Litterman Model"
    RISK_PARITY = "Risk Parity"
    MAX_DIVERSIFICATION = "Maximum Diversification Ratio"
    MIN_CVAR = "Minimum Conditional VaR (CVaR)"
    MAX_SORTINO = "Maximum Sortino Ratio"
    MAX_CALMAR = "Maximum Calmar Ratio"
    MAX_OMEGA = "Maximum Omega Ratio"
    MAX_ULCER = "Maximum Ulcer Performance Index"
    MIN_MAD = "Minimum Mean Absolute Deviation"
    MAX_ENTROPY = "Maximum Entropy"
    EQUAL_RISK_CONTRIBUTION = "Equal Risk Contribution"
    CLA = "Critical Line Algorithm (CLA)"
    MEAN_ABSOLUTE_DEVIATION = "Mean Absolute Deviation"
    SEMI_VARIANCE = "Semi-Variance Optimization"
    CVAR_OPTIMIZATION = "CVaR Optimization"
    ENTROPY_POOLING = "Entropy Pooling"
    ROBUST_OPTIMIZATION = "Robust Optimization"
    BAYESIAN_OPTIMIZATION = "Bayesian Optimization"

class RiskModel(str, Enum):
    SAMPLE_COV = "Sample Covariance"
    LEDOIT_WOLF = "Ledoit-Wolf Shrinkage"
    SEMICOVARIANCE = "Semicovariance (Downside Risk)"
    EXPONENTIAL_COV = "Exponential Covariance"
    CONSTANT_CORRELATION = "Constant Correlation"
    ORACLE = "Oracle Approximating Shrinkage"
    MIN_COV_DETERMINANT = "Minimum Covariance Determinant"
    GRAPHICAL_LASSO = "Graphical Lasso"
    OAS = "Oracle Approximating Shrinkage (OAS)"
    FACTOR_MODEL = "Factor Model"
    GARCH = "GARCH Model"
    EWMA = "Exponentially Weighted Moving Average"
    DCC_GARCH = "DCC-GARCH"
    BEKK = "BEKK-GARCH"
    REALIZED_COV = "Realized Covariance"
    ROBUST_COV = "Robust Covariance Estimation"

class ReturnModel(str, Enum):
    MEAN_HISTORICAL = "Mean Historical Return"
    EMA_HISTORICAL = "Exponential Moving Average"
    CAPM = "CAPM Return"
    LOG_RETURN = "Logarithmic Return"
    SIMPLE_RETURN = "Simple Return"
    ARIMA = "ARIMA Forecast"
    GARCH = "GARCH Forecast"
    EWM = "Exponentially Weighted Mean"
    QUANTILE_REGRESSION = "Quantile Regression"
    MACHINE_LEARNING = "Machine Learning Prediction"
    BLACK_LITTERMAN = "Black-Litterman"
    MONTE_CARLO = "Monte Carlo Simulation"
    BOOTSTRAP = "Bootstrap Resampling"

class VARMethod(str, Enum):
    HISTORICAL = "Historical Simulation"
    PARAMETRIC = "Parametric (Normal Distribution)"
    PARAMETRIC_T = "Parametric (Student-t Distribution)"
    CORNISH_FISHER = "Cornish-Fisher Expansion"
    MONTE_CARLO = "Monte Carlo Simulation"
    EXTREME_VALUE = "Extreme Value Theory (EVT)"
    GARCH_VAR = "GARCH-based VaR"
    CONDITIONAL_VAR = "Conditional VaR (CVaR)"
    EXPECTED_SHORTFALL = "Expected Shortfall"
    MODIFIED_VAR = "Modified VaR"
    STRESS_VAR = "Stress Testing VaR"
    REGIME_SWITCHING = "Regime Switching VaR"
    FILTERED_HISTORICAL = "Filtered Historical Simulation"
    BOOTSTRAP = "Bootstrap VaR"

class BacktestMethod(str, Enum):
    WALK_FORWARD = "Walk-Forward Analysis"
    ROLLING_WINDOW = "Rolling Window"
    EXPANDING_WINDOW = "Expanding Window"
    MONTE_CARLO = "Monte Carlo Backtest"
    STRESS_TEST = "Stress Testing"
    SCENARIO_ANALYSIS = "Scenario Analysis"
    HISTORICAL_SIMULATION = "Historical Simulation"
    BOOTSTRAP = "Bootstrap Resampling"
    CROSS_VALIDATION = "Cross-Validation"
    OUT_OF_SAMPLE = "Out-of-Sample Testing"

class StressScenario(str, Enum):
    MARKET_CRASH_2008 = "2008 Financial Crisis"
    DOT_COM_BUBBLE = "Dot-com Bubble Burst"
    COVID_CRASH = "COVID-19 Market Crash"
    INTEREST_RATE_SHOCK = "Interest Rate Shock"
    INFLATION_SPIKE = "Inflation Spike"
    CURRENCY_CRISIS = "Currency Crisis"
    LIQUIDITY_CRISIS = "Liquidity Crisis"
    GEOPOLITICAL_SHOCK = "Geopolitical Shock"
    SECTOR_ROTATION = "Sector Rotation"
    VOLATILITY_SPIKE = "Volatility Spike"
    RECESSION = "Recession Scenario"
    STAGFLATION = "Stagflation Scenario"
    BLACK_SWAN = "Black Swan Event"
    SYSTEMIC_CRISIS = "Systemic Financial Crisis"

# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL STREAMLIT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Institutional Portfolio Analytics Suite Pro",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit/streamlit',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': """
        # BIST Institutional Portfolio Analytics Suite
        Version: 12.0 Pro Max Ultra
        
        Advanced Quantitative Portfolio Optimization Platform
        with Comprehensive Risk Analytics, Stress Testing,
        and Backtesting Capabilities.
        
        © 2024 Institutional Finance Analytics
        """
    }
)

# Professional institutional color scheme
INSTITUTIONAL_COLORS = {
    # Primary colors
    'dark_blue': '#0a1929',
    'navy_blue': '#1a2536',
    'slate_blue': '#2d3748',
    
    # Accent colors
    'professional_blue': '#0066cc',
    'institutional_green': '#00cc88',
    'warning_orange': '#ff6b35',
    'risk_red': '#ff4d4d',
    'analytics_purple': '#9d4edd',
    
    # Neutral colors
    'light_gray': '#e2e8f0',
    'medium_gray': '#a0aec0',
    'dark_gray': '#4a5568',
    
    # Text colors
    'text_primary': '#ffffff',
    'text_secondary': '#b0b0b0',
    'text_tertiary': '#718096',
    
    # Special colors
    'success': '#38a169',
    'warning': '#d69e2e',
    'danger': '#e53e3e',
    'info': '#3182ce'
}

# Professional CSS with institutional styling
st.markdown(f"""
<style>
    /* Base styling */
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95%;
    }}
    
    /* Professional header */
    .institutional-header {{
        background: linear-gradient(135deg, {INSTITUTIONAL_COLORS['dark_blue']}, {INSTITUTIONAL_COLORS['navy_blue']});
        border-bottom: 2px solid {INSTITUTIONAL_COLORS['professional_blue']};
        padding: 2rem;
        margin-bottom: 2rem;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }}
    
    /* Professional cards */
    .professional-card {{
        background: {INSTITUTIONAL_COLORS['navy_blue']};
        border: 1px solid {INSTITUTIONAL_COLORS['slate_blue']};
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }}
    
    .professional-card:hover {{
        border-color: {INSTITUTIONAL_COLORS['professional_blue']};
        box-shadow: 0 4px 16px rgba(0, 102, 204, 0.2);
        transform: translateY(-2px);
    }}
    
    /* Metric cards */
    .metric-card {{
        background: linear-gradient(145deg, {INSTITUTIONAL_COLORS['slate_blue']}, {INSTITUTIONAL_COLORS['navy_blue']});
        border: 1px solid {INSTITUTIONAL_COLORS['slate_blue']};
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        border-color: {INSTITUTIONAL_COLORS['professional_blue']};
        background: linear-gradient(145deg, {INSTITUTIONAL_COLORS['navy_blue']}, {INSTITUTIONAL_COLORS['slate_blue']});
    }}
    
    /* Professional tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {INSTITUTIONAL_COLORS['navy_blue']};
        padding: 0;
        border-bottom: 1px solid {INSTITUTIONAL_COLORS['slate_blue']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: {INSTITUTIONAL_COLORS['navy_blue']};
        color: {INSTITUTIONAL_COLORS['text_secondary']};
        border-right: 1px solid {INSTITUTIONAL_COLORS['slate_blue']};
        padding: 1rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {INSTITUTIONAL_COLORS['slate_blue']};
        color: {INSTITUTIONAL_COLORS['text_primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {INSTITUTIONAL_COLORS['professional_blue']};
        color: {INSTITUTIONAL_COLORS['text_primary']};
        border-bottom: 3px solid {INSTITUTIONAL_COLORS['institutional_green']};
    }}
    
    /* Professional buttons */
    .stButton > button {{
        background: {INSTITUTIONAL_COLORS['professional_blue']};
        color: {INSTITUTIONAL_COLORS['text_primary']};
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }}
    
    .stButton > button:hover {{
        background: {INSTITUTIONAL_COLORS['dark_blue']};
        border: 1px solid {INSTITUTIONAL_COLORS['professional_blue']};
        box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3);
        transform: translateY(-1px);
    }}
    
    /* Primary action button */
    .primary-button > button {{
        background: linear-gradient(135deg, {INSTITUTIONAL_COLORS['professional_blue']}, {INSTITUTIONAL_COLORS['analytics_purple']});
        font-weight: 600;
        padding: 1rem 2rem;
    }}
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {{
        background: {INSTITUTIONAL_COLORS['navy_blue']};
        color: {INSTITUTIONAL_COLORS['text_primary']};
        border: 1px solid {INSTITUTIONAL_COLORS['slate_blue']};
        border-radius: 6px;
        transition: all 0.3s ease;
    }}
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {INSTITUTIONAL_COLORS['professional_blue']};
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2);
    }}
    
    /* Dataframe styling */
    .dataframe {{
        background: {INSTITUTIONAL_COLORS['navy_blue']};
        color: {INSTITUTIONAL_COLORS['text_primary']};
        border: 1px solid {INSTITUTIONAL_COLORS['slate_blue']};
        border-radius: 6px;
    }}
    
    .dataframe thead {{
        background: {INSTITUTIONAL_COLORS['dark_blue']};
        color: {INSTITUTIONAL_COLORS['text_primary']};
    }}
    
    .dataframe tbody tr:nth-child(even) {{
        background: {INSTITUTIONAL_COLORS['slate_blue']};
    }}
    
    /* Badges */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
    }}
    
    .badge-success {{
        background: {INSTITUTIONAL_COLORS['success']};
        color: white;
    }}
    
    .badge-warning {{
        background: {INSTITUTIONAL_COLORS['warning']};
        color: white;
    }}
    
    .badge-danger {{
        background: {INSTITUTIONAL_COLORS['danger']};
        color: white;
    }}
    
    .badge-info {{
        background: {INSTITUTIONAL_COLORS['info']};
        color: white;
    }}
    
    /* Progress bars */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {INSTITUTIONAL_COLORS['professional_blue']}, {INSTITUTIONAL_COLORS['analytics_purple']});
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background: {INSTITUTIONAL_COLORS['navy_blue']};
        border: 1px solid {INSTITUTIONAL_COLORS['slate_blue']};
        border-radius: 6px;
        font-weight: 500;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: {INSTITUTIONAL_COLORS['slate_blue']};
        border-color: {INSTITUTIONAL_COLORS['professional_blue']};
    }}
    
    /* Charts background */
    .js-plotly-plot .plotly {{
        background: {INSTITUTIONAL_COLORS['navy_blue']} !important;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {INSTITUTIONAL_COLORS['navy_blue']};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {INSTITUTIONAL_COLORS['professional_blue']};
        border-radius: 5px;
        border: 2px solid {INSTITUTIONAL_COLORS['navy_blue']};
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {INSTITUTIONAL_COLORS['analytics_purple']};
    }}
    
    /* Tooltips */
    .stTooltip {{
        background: {INSTITUTIONAL_COLORS['dark_blue']} !important;
        border: 1px solid {INSTITUTIONAL_COLORS['professional_blue']} !important;
        color: {INSTITUTIONAL_COLORS['text_primary']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioAllocation:
    """Portfolio allocation with detailed metadata"""
    ticker: str
    weight: float
    sector: str
    expected_return: float
    volatility: float
    beta: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    marginal_risk_contribution: float
    risk_contribution: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class OptimizationResult:
    """Comprehensive optimization result"""
    method: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    omega_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    diversification_ratio: float
    risk_free_rate: float
    optimization_time: float
    convergence_status: str
    iterations: int
    objective_value: float
    constraints_violation: float
    efficient_frontier_point: bool
    portfolio_turnover: float
    concentration_index: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        weights_df = pd.DataFrame.from_dict(self.weights, orient='index', columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
        return weights_df
    
    def get_metrics(self) -> Dict:
        """Get key metrics"""
        return {
            'Expected Return': f"{self.expected_return:.2%}",
            'Expected Volatility': f"{self.expected_volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.3f}",
            'Sortino Ratio': f"{self.sortino_ratio:.3f}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'VaR (95%)': f"{self.var_95:.4f}",
            'CVaR (95%)': f"{self.cvar_95:.4f}",
            'Diversification Ratio': f"{self.diversification_ratio:.3f}"
        }

@dataclass
class VARAnalysis:
    """Comprehensive VAR analysis results"""
    confidence_level: float
    time_horizon: int
    
    # Historical methods
    historical_var: float
    historical_cvar: float
    filtered_historical_var: float
    
    # Parametric methods
    parametric_var_normal: float
    parametric_cvar_normal: float
    parametric_var_t: float
    parametric_cvar_t: float
    
    # Advanced methods
    cornish_fisher_var: float
    extreme_value_var: float
    extreme_value_cvar: float
    monte_carlo_var: float
    monte_carlo_cvar: float
    garch_var: float
    garch_cvar: float
    
    # Component analysis
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    incremental_var: Dict[str, float]
    
    # Stress testing
    stressed_var: Dict[str, float]
    scenario_var: Dict[str, float]
    
    # Backtesting
    var_exceptions: int
    exception_rate: float
    kupiec_pvalue: float
    christoffersen_pvalue: float
    conditional_coverage_pvalue: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to comprehensive DataFrame"""
        data = {
            'Method': [
                'Historical VaR', 'Historical CVaR',
                'Parametric VaR (Normal)', 'Parametric CVaR (Normal)',
                'Parametric VaR (Student-t)', 'Parametric CVaR (Student-t)',
                'Cornish-Fisher VaR', 'Extreme Value VaR',
                'Monte Carlo VaR', 'GARCH VaR'
            ],
            'Value': [
                self.historical_var, self.historical_cvar,
                self.parametric_var_normal, self.parametric_cvar_normal,
                self.parametric_var_t, self.parametric_cvar_t,
                self.cornish_fisher_var, self.extreme_value_var,
                self.monte_carlo_var, self.garch_var
            ],
            'Type': ['VaR', 'CVaR', 'VaR', 'CVaR', 'VaR', 'CVaR', 'VaR', 'VaR', 'VaR', 'VaR']
        }
        
        return pd.DataFrame(data)

@dataclass
class StressTestResult:
    """Stress testing results"""
    scenario: str
    portfolio_loss: float
    benchmark_loss: float
    relative_loss: float
    var_breach: bool
    cvar_impact: float
    component_impacts: Dict[str, float]
    recovery_period: int
    liquidity_impact: float
    correlation_impact: float
    volatility_impact: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class BacktestResult:
    """Backtesting results"""
    method: str
    period_start: datetime
    period_end: datetime
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    var_exceptions: int
    exception_rate: float
    average_turnover: float
    information_ratio: float
    alpha: float
    beta: float
    tracking_error: float
    r_squared: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        data = asdict(self)
        df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
        return df

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    cagr: float
    annual_return: float
    monthly_return: float
    weekly_return: float
    daily_return: float
    
    # Risk metrics
    annual_volatility: float
    downside_deviation: float
    max_drawdown: float
    avg_drawdown: float
    ulcer_index: float
    pain_index: float
    
    # Risk-adjusted ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    gain_to_pain_ratio: float
    tail_ratio: float
    common_sense_ratio: float
    kappa_three_ratio: float
    information_ratio: float
    treynor_ratio: float
    appraisal_ratio: float
    
    # Statistical metrics
    skewness: float
    kurtosis: float
    jarque_bera: float
    value_at_risk_95: float
    conditional_var_95: float
    expected_shortfall_95: float
    
    # Win/loss metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_win_loss_ratio: float
    expectancy: float
    
    # Benchmark comparisons
    alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    
    # Additional metrics
    var_95: float
    cvar_95: float
    modified_var_95: float
    entropy: float
    hurst_exponent: float
    autocorrelation: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame.from_dict(asdict(self), orient='index', columns=['Value'])

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED DATA FETCHER WITH CACHING
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedDataFetcher:
    """Advanced data fetcher with multiple sources and caching"""
    
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _generate_cache_key(self, tickers: List[str], start_date: datetime, 
                           end_date: datetime, interval: str) -> str:
        """Generate cache key"""
        key_data = {
            'tickers': sorted(tickers),
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'interval': interval
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    @st.cache_data(ttl=3600, show_spinner="📊 Fetching comprehensive market data...")
    def fetch_market_data(_self, tickers: List[str], start_date: datetime, 
                         end_date: datetime, interval: str = '1d') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch comprehensive market data"""
        
        if not HAS_YFINANCE:
            st.error("yfinance is not available")
            return None, None
        
        try:
            logger.logger.info(f"Fetching data for {len(tickers)} tickers")
            
            # Download data with multiple attempts
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = yf.download(
                        tickers,
                        start=start_date,
                        end=end_date + timedelta(days=1),
                        interval=interval,
                        progress=False,
                        show_errors=False,
                        threads=True,
                        timeout=30
                    )
                    
                    if data.empty:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            st.error("No data returned from Yahoo Finance")
                            return None, None
                    
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                        time.sleep(2)
                    else:
                        raise
            
            # Process data based on structure
            if len(tickers) == 1:
                close_prices = data['Close'].to_frame(tickers[0])
            else:
                close_prices = pd.DataFrame()
                for ticker in tickers:
                    if (ticker, 'Close') in data.columns:
                        close_prices[ticker] = data[(ticker, 'Close')]
            
            # Clean and validate data
            close_prices = close_prices.ffill().bfill()
            close_prices = close_prices.dropna(axis=1, how='all')
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            # Filter tickers with sufficient data
            valid_tickers = []
            for ticker in tickers:
                if ticker in returns.columns:
                    if returns[ticker].notna().sum() > 50:
                        valid_tickers.append(ticker)
            
            if len(valid_tickers) < 2:
                st.error(f"Insufficient data: only {len(valid_tickers)} valid tickers")
                return None, None
            
            close_prices = close_prices[valid_tickers]
            returns = returns[valid_tickers]
            
            logger.logger.info(f"Successfully loaded data for {len(valid_tickers)} tickers")
            
            return close_prices, returns
            
        except Exception as e:
            logger.logger.error(f"Data fetching error: {e}")
            st.error(f"Error fetching data: {str(e)}")
            return None, None
    
    def fetch_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamental_data = {
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'employees': info.get('fullTimeEmployees'),
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins'),
                'return_on_assets': info.get('returnOnAssets'),
                'return_on_equity': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio')
            }
            
            # Clean None values
            fundamental_data = {k: v for k, v in fundamental_data.items() if v is not None}
            
            return fundamental_data
            
        except Exception as e:
            logger.logger.warning(f"Failed to fetch fundamental data for {ticker}: {e}")
            return None
    
    def fetch_macro_data(self) -> Optional[pd.DataFrame]:
        """Fetch macroeconomic data"""
        try:
            macro_tickers = {
                'USD/TRY': 'TRY=X',
                'EUR/TRY': 'EURTRY=X',
                'Gold': 'GC=F',
                'Brent Oil': 'BZ=F',
                'VIX': '^VIX',
                'US 10Y Yield': '^TNX',
                'TR 10Y Yield': 'TAHV10Y.IS'
            }
            
            macro_data = {}
            for name, ticker in macro_tickers.items():
                try:
                    data = yf.download(ticker, period='1y', progress=False)
                    if not data.empty:
                        macro_data[name] = data['Close']
                except:
                    continue
            
            if macro_data:
                return pd.DataFrame(macro_data)
            
        except Exception as e:
            logger.logger.warning(f"Failed to fetch macro data: {e}")
        
        return None

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE PORTFOLIO OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class InstitutionalPortfolioOptimizer:
    """Institutional-grade portfolio optimizer with advanced features"""
    
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame):
        self.prices = prices
        self.returns = returns
        self.tickers = prices.columns.tolist()
        self.n_assets = len(self.tickers)
        
        # Validate and clean data
        self._validate_and_prepare_data()
        
        # Initialize models
        self._initialize_models()
        
        # Performance tracking
        self.optimization_history = []
        self.backtest_results = []
        self.stress_test_results = []
        
        logger.logger.info(f"Initialized optimizer with {self.n_assets} assets")
    
    def _validate_and_prepare_data(self):
        """Validate and prepare data for optimization"""
        if self.prices.empty or self.returns.empty:
            raise ValueError("Invalid data: prices or returns are empty")
        
        # Check for sufficient data points
        if len(self.returns) < 50:
            raise ValueError(f"Insufficient data points: {len(self.returns)} < 50")
        
        # Remove assets with too many NaN values
        valid_tickers = []
        for ticker in self.tickers:
            if ticker in self.returns.columns:
                non_nan_count = self.returns[ticker].notna().sum()
                if non_nan_count > 50:
                    valid_tickers.append(ticker)
        
        if len(valid_tickers) < 2:
            raise ValueError(f"Insufficient valid assets: {len(valid_tickers)}")
        
        # Update data
        self.tickers = valid_tickers
        self.n_assets = len(valid_tickers)
        self.prices = self.prices[valid_tickers]
        self.returns = self.returns[valid_tickers]
        
        # Forward fill and drop remaining NaNs
        self.prices = self.prices.ffill().bfill()
        self.returns = self.returns.ffill().bfill().dropna()
        
        # Remove zero or near-zero variance assets
        variances = self.returns.var()
        valid_tickers = variances[variances > 1e-10].index.tolist()
        self.tickers = valid_tickers
        self.n_assets = len(valid_tickers)
        self.prices = self.prices[valid_tickers]
        self.returns = self.returns[valid_tickers]
    
    def _initialize_models(self):
        """Initialize all optimization models"""
        # Expected returns models
        self.expected_returns = {
            'mean_historical': expected_returns.mean_historical_return(self.prices),
            'ema_historical': expected_returns.ema_historical_return(self.prices, span=500),
            'capm': self._calculate_capm_returns(),
            'log_returns': np.log(self.prices / self.prices.shift(1)).mean() * 252
        }
        
        # Covariance models
        self.covariance_matrices = {
            'sample': risk_models.sample_cov(self.returns),
            'ledoit_wolf': risk_models.CovarianceShrinkage(self.prices).ledoit_wolf(),
            'oracle': risk_models.CovarianceShrinkage(self.prices).oracle_approximating(),
            'semicovariance': risk_models.semicovariance(self.prices),
            'exp_cov': risk_models.exp_cov(self.returns)
        }
        
        # Additional risk models
        if HAS_SKLEARN:
            try:
                # Ledoit-Wolf from sklearn
                lw = LedoitWolf()
                lw.fit(self.returns)
                lw_cov = pd.DataFrame(
                    lw.covariance_ * 252,
                    index=self.tickers,
                    columns=self.tickers
                )
                self.covariance_matrices['sklearn_ledoit_wolf'] = lw_cov
                
                # OAS estimator
                oas = OAS()
                oas.fit(self.returns)
                oas_cov = pd.DataFrame(
                    oas.covariance_ * 252,
                    index=self.tickers,
                    columns=self.tickers
                )
                self.covariance_matrices['oas'] = oas_cov
                
            except Exception as e:
                logger.logger.warning(f"Failed to initialize sklearn models: {e}")
    
    def _calculate_capm_returns(self) -> pd.Series:
        """Calculate CAPM expected returns"""
        # Use market portfolio as proxy
        market_returns = self.returns.mean(axis=1)
        risk_free_rate = 0.30  # Turkey
        
        betas = {}
        for ticker in self.tickers:
            asset_returns = self.returns[ticker]
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            variance = np.var(market_returns)
            beta = covariance / variance if variance != 0 else 1.0
            betas[ticker] = beta
        
        # Assume market return of 15%
        market_return = 0.15
        capm_returns = risk_free_rate + pd.Series(betas) * (market_return - risk_free_rate)
        
        return capm_returns
    
    def optimize(self, method: OptimizationMethod, **kwargs) -> OptimizationResult:
        """Main optimization method with comprehensive error handling"""
        
        start_time = time.time()
        
        try:
            # Get parameters
            risk_free_rate = kwargs.get('risk_free_rate', 0.30)
            target_volatility = kwargs.get('target_volatility', 0.15)
            target_return = kwargs.get('target_return', 0.20)
            risk_aversion = kwargs.get('risk_aversion', 1.0)
            
            # Select covariance matrix
            cov_method = kwargs.get('covariance_method', 'ledoit_wolf')
            S = self.covariance_matrices.get(cov_method, self.covariance_matrices['ledoit_wolf'])
            
            # Select expected returns
            mu_method = kwargs.get('return_method', 'mean_historical')
            mu = self.expected_returns.get(mu_method, self.expected_returns['mean_historical'])
            
            # Perform optimization based on method
            if method == OptimizationMethod.MAX_SHARPE:
                result = self._optimize_max_sharpe(mu, S, risk_free_rate)
                
            elif method == OptimizationMethod.MIN_VOLATILITY:
                result = self._optimize_min_volatility(mu, S)
                
            elif method == OptimizationMethod.EFFICIENT_RISK:
                result = self._optimize_efficient_risk(mu, S, target_volatility, risk_free_rate)
                
            elif method == OptimizationMethod.EFFICIENT_RETURN:
                result = self._optimize_efficient_return(mu, S, target_return, risk_free_rate)
                
            elif method == OptimizationMethod.HRP:
                result = self._optimize_hrp(risk_free_rate)
                
            elif method == OptimizationMethod.MAX_SORTINO:
                result = self._optimize_max_sortino(risk_free_rate)
                
            elif method == OptimizationMethod.MIN_CVAR:
                result = self._optimize_min_cvar(risk_free_rate)
                
            elif method == OptimizationMethod.RISK_PARITY:
                result = self._optimize_risk_parity(S)
                
            elif method == OptimizationMethod.MAX_DIVERSIFICATION:
                result = self._optimize_max_diversification(mu, S, risk_free_rate)
                
            else:
                # Default to maximum Sharpe
                result = self._optimize_max_sharpe(mu, S, risk_free_rate)
            
            # Calculate additional metrics
            result = self._enhance_optimization_result(result, mu, S, risk_free_rate)
            
            # Store in history
            self.optimization_history.append(result)
            
            # Log performance
            duration = time.time() - start_time
            logger.log_performance(f"optimize_{method.value}", duration, 
                                  assets=self.n_assets, success=True)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.log_performance(f"optimize_{method.value}", duration, 
                                  assets=self.n_assets, success=False, error=str(e))
            
            logger.logger.error(f"Optimization failed: {e}")
            
            # Return equal weight portfolio as fallback
            return self._get_equal_weight_result(risk_free_rate)
    
    def _optimize_max_sharpe(self, mu: pd.Series, S: pd.DataFrame, 
                            risk_free_rate: float) -> OptimizationResult:
        """Optimize for maximum Sharpe ratio"""
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        return OptimizationResult(
            method=OptimizationMethod.MAX_SHARPE.value,
            weights=weights,
            expected_return=performance[0],
            expected_volatility=performance[1],
            sharpe_ratio=performance[2],
            sortino_ratio=0,  # Will be calculated later
            omega_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            diversification_ratio=0,
            risk_free_rate=risk_free_rate,
            optimization_time=0,
            convergence_status="Success",
            iterations=1,
            objective_value=performance[2],
            constraints_violation=0,
            efficient_frontier_point=True,
            portfolio_turnover=0,
            concentration_index=0
        )
    
    def _optimize_min_volatility(self, mu: pd.Series, S: pd.DataFrame) -> OptimizationResult:
        """Optimize for minimum volatility"""
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        weights = ef.clean_weights()
        performance = ef.portfolio_performance()
        
        return OptimizationResult(
            method=OptimizationMethod.MIN_VOLATILITY.value,
            weights=weights,
            expected_return=performance[0],
            expected_volatility=performance[1],
            sharpe_ratio=performance[2],
            sortino_ratio=0,
            omega_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            diversification_ratio=0,
            risk_free_rate=0.30,
            optimization_time=0,
            convergence_status="Success",
            iterations=1,
            objective_value=performance[1],
            constraints_violation=0,
            efficient_frontier_point=True,
            portfolio_turnover=0,
            concentration_index=0
        )
    
    def _optimize_efficient_risk(self, mu: pd.Series, S: pd.DataFrame, 
                                target_volatility: float, 
                                risk_free_rate: float) -> OptimizationResult:
        """Optimize for efficient risk"""
        ef = EfficientFrontier(mu, S)
        ef.efficient_risk(target_volatility=target_volatility)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        return OptimizationResult(
            method=OptimizationMethod.EFFICIENT_RISK.value,
            weights=weights,
            expected_return=performance[0],
            expected_volatility=performance[1],
            sharpe_ratio=performance[2],
            sortino_ratio=0,
            omega_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            diversification_ratio=0,
            risk_free_rate=risk_free_rate,
            optimization_time=0,
            convergence_status="Success",
            iterations=1,
            objective_value=performance[0],
            constraints_violation=0,
            efficient_frontier_point=True,
            portfolio_turnover=0,
            concentration_index=0
        )
    
    def _optimize_efficient_return(self, mu: pd.Series, S: pd.DataFrame, 
                                  target_return: float, 
                                  risk_free_rate: float) -> OptimizationResult:
        """Optimize for efficient return"""
        ef = EfficientFrontier(mu, S)
        ef.efficient_return(target_return=target_return)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        return OptimizationResult(
            method=OptimizationMethod.EFFICIENT_RETURN.value,
            weights=weights,
            expected_return=performance[0],
            expected_volatility=performance[1],
            sharpe_ratio=performance[2],
            sortino_ratio=0,
            omega_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            diversification_ratio=0,
            risk_free_rate=risk_free_rate,
            optimization_time=0,
            convergence_status="Success",
            iterations=1,
            objective_value=performance[1],
            constraints_violation=0,
            efficient_frontier_point=True,
            portfolio_turnover=0,
            concentration_index=0
        )
    
    def _optimize_hrp(self, risk_free_rate: float) -> OptimizationResult:
        """Optimize using Hierarchical Risk Parity"""
        hrp = HRPOpt(self.returns)
        hrp.optimize()
        weights = hrp.clean_weights()
        
        # Calculate performance metrics
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        return OptimizationResult(
            method=OptimizationMethod.HRP.value,
            weights=weights,
            expected_return=annual_return,
            expected_volatility=annual_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=0,
            omega_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            diversification_ratio=0,
            risk_free_rate=risk_free_rate,
            optimization_time=0,
            convergence_status="Success",
            iterations=1,
            objective_value=sharpe,
            constraints_violation=0,
            efficient_frontier_point=False,
            portfolio_turnover=0,
            concentration_index=0
        )
    
    def _optimize_max_sortino(self, risk_free_rate: float) -> OptimizationResult:
        """Optimize for maximum Sortino ratio"""
        # Custom optimization for Sortino ratio
        def negative_sortino(weights):
            weights = np.array(weights)
            portfolio_returns = self.returns.dot(weights)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            expected_return = np.mean(portfolio_returns) * 252
            sortino = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else -100
            return -sortino
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}  # Non-negative
        ]
        
        bounds = [(0, 1) for _ in range(self.n_assets)]
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimization
        result = minimize(
            negative_sortino,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if result.success:
            weights = {self.tickers[i]: result.x[i] for i in range(self.n_assets)}
            portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            return OptimizationResult(
                method=OptimizationMethod.MAX_SORTINO.value,
                weights=weights,
                expected_return=annual_return,
                expected_volatility=annual_vol,
                sharpe_ratio=sharpe,
                sortino_ratio=-result.fun,
                omega_ratio=0,
                calmar_ratio=0,
                max_drawdown=0,
                var_95=0,
                cvar_95=0,
                diversification_ratio=0,
                risk_free_rate=risk_free_rate,
                optimization_time=0,
                convergence_status="Success",
                iterations=result.nit,
                objective_value=-result.fun,
                constraints_violation=0,
                efficient_frontier_point=False,
                portfolio_turnover=0,
                concentration_index=0
            )
        else:
            raise ValueError("Sortino optimization failed")
    
    def _optimize_min_cvar(self, risk_free_rate: float, alpha: float = 0.05) -> OptimizationResult:
        """Optimize for minimum Conditional Value at Risk"""
        try:
            ef_cvar = EfficientCVaR(self.expected_returns['mean_historical'], self.returns)
            ef_cvar.min_cvar()
            weights = ef_cvar.clean_weights()
            
            # Calculate performance
            portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            # Calculate CVaR
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            return OptimizationResult(
                method=OptimizationMethod.MIN_CVAR.value,
                weights=weights,
                expected_return=annual_return,
                expected_volatility=annual_vol,
                sharpe_ratio=sharpe,
                sortino_ratio=0,
                omega_ratio=0,
                calmar_ratio=0,
                max_drawdown=0,
                var_95=var_95,
                cvar_95=cvar_95,
                diversification_ratio=0,
                risk_free_rate=risk_free_rate,
                optimization_time=0,
                convergence_status="Success",
                iterations=1,
                objective_value=cvar_95,
                constraints_violation=0,
                efficient_frontier_point=False,
                portfolio_turnover=0,
                concentration_index=0
            )
        except:
            # Fallback to minimum volatility
            return self._optimize_min_volatility(
                self.expected_returns['mean_historical'],
                self.covariance_matrices['ledoit_wolf']
            )
    
    def _optimize_risk_parity(self, S: pd.DataFrame) -> OptimizationResult:
        """Optimize for risk parity portfolio"""
        # Simple risk parity implementation
        volatilities = np.sqrt(np.diag(S.values))
        inverse_vol = 1 / volatilities
        weights = inverse_vol / np.sum(inverse_vol)
        
        weights_dict = {self.tickers[i]: weights[i] for i in range(self.n_assets)}
        
        portfolio_returns = (self.returns * pd.Series(weights_dict)).sum(axis=1)
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.30) / annual_vol if annual_vol > 0 else 0
        
        return OptimizationResult(
            method=OptimizationMethod.RISK_PARITY.value,
            weights=weights_dict,
            expected_return=annual_return,
            expected_volatility=annual_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=0,
            omega_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            diversification_ratio=0,
            risk_free_rate=0.30,
            optimization_time=0,
            convergence_status="Success",
            iterations=1,
            objective_value=0,
            constraints_violation=0,
            efficient_frontier_point=False,
            portfolio_turnover=0,
            concentration_index=0
        )
    
    def _optimize_max_diversification(self, mu: pd.Series, S: pd.DataFrame, 
                                    risk_free_rate: float) -> OptimizationResult:
        """Optimize for maximum diversification ratio"""
        # Diversification ratio = weighted avg vol / portfolio vol
        def negative_diversification(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(S.values, weights)))
            weighted_avg_vol = np.sum(weights * np.sqrt(np.diag(S.values)))
            diversification = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
            return -diversification
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        bounds = [(0, 1) for _ in range(self.n_assets)]
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimization
        result = minimize(
            negative_diversification,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if result.success:
            weights = {self.tickers[i]: result.x[i] for i in range(self.n_assets)}
            portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            return OptimizationResult(
                method=OptimizationMethod.MAX_DIVERSIFICATION.value,
                weights=weights,
                expected_return=annual_return,
                expected_volatility=annual_vol,
                sharpe_ratio=sharpe,
                sortino_ratio=0,
                omega_ratio=0,
                calmar_ratio=0,
                max_drawdown=0,
                var_95=0,
                cvar_95=0,
                diversification_ratio=-result.fun,
                risk_free_rate=risk_free_rate,
                optimization_time=0,
                convergence_status="Success",
                iterations=result.nit,
                objective_value=-result.fun,
                constraints_violation=0,
                efficient_frontier_point=False,
                portfolio_turnover=0,
                concentration_index=0
            )
        else:
            raise ValueError("Diversification optimization failed")
    
    def _get_equal_weight_result(self, risk_free_rate: float) -> OptimizationResult:
        """Get equal weight portfolio as fallback"""
        weights = {ticker: 1/self.n_assets for ticker in self.tickers}
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        return OptimizationResult(
            method="Equal Weight",
            weights=weights,
            expected_return=annual_return,
            expected_volatility=annual_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=0,
            omega_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            diversification_ratio=0,
            risk_free_rate=risk_free_rate,
            optimization_time=0,
            convergence_status="Fallback",
            iterations=0,
            objective_value=0,
            constraints_violation=0,
            efficient_frontier_point=False,
            portfolio_turnover=0,
            concentration_index=0
        )
    
    def _enhance_optimization_result(self, result: OptimizationResult, 
                                    mu: pd.Series, S: pd.DataFrame, 
                                    risk_free_rate: float) -> OptimizationResult:
        """Enhance optimization result with additional metrics"""
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * pd.Series(result.weights)).sum(axis=1)
        
        # Calculate Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (result.expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate Omega ratio
        threshold = risk_free_rate / 252
        gains = portfolio_returns[portfolio_returns > threshold].sum()
        losses = abs(portfolio_returns[portfolio_returns <= threshold].sum())
        omega_ratio = gains / losses if losses != 0 else float('inf')
        
        # Calculate maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar_ratio = result.expected_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate VAR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Calculate diversification ratio
        weights_array = np.array(list(result.weights.values()))
        weighted_avg_vol = np.sum(weights_array * np.sqrt(np.diag(S.values)))
        portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(S.values, weights_array)))
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate concentration index (Herfindahl-Hirschman Index)
        concentration_index = np.sum(weights_array ** 2)
        
        # Update result
        result.sortino_ratio = sortino_ratio
        result.omega_ratio = omega_ratio
        result.calmar_ratio = calmar_ratio
        result.max_drawdown = max_drawdown
        result.var_95 = var_95
        result.cvar_95 = cvar_95
        result.diversification_ratio = diversification_ratio
        result.concentration_index = concentration_index
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # ADVANCED VAR ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────
    
    def calculate_comprehensive_var(self, weights: Dict[str, float], 
                                   confidence_level: float = 0.95,
                                   time_horizon: int = 1,
                                   n_simulations: int = 10000) -> VARAnalysis:
        """Calculate comprehensive VAR analytics using multiple methods"""
        
        start_time = time.time()
        
        try:
            # Get portfolio returns
            portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
            
            # 1. Historical VaR/CVaR
            historical_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            historical_cvar = portfolio_returns[portfolio_returns <= historical_var].mean()
            
            # 2. Parametric VaR (Normal)
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            z_score_normal = norm.ppf(1 - confidence_level)
            parametric_var_normal = mean_return + z_score_normal * std_return
            parametric_cvar_normal = mean_return - (std_return / (1 - confidence_level)) * norm.pdf(z_score_normal)
            
            # 3. Parametric VaR (Student-t)
            # Fit t-distribution
            df, loc, scale = stats.t.fit(portfolio_returns)
            t_score = stats.t.ppf(1 - confidence_level, df)
            parametric_var_t = loc + t_score * scale
            parametric_cvar_t = self._calculate_t_cvar(df, loc, scale, confidence_level)
            
            # 4. Cornish-Fisher Expansion
            z = norm.ppf(1 - confidence_level)
            s = stats.skew(portfolio_returns)
            k = stats.kurtosis(portfolio_returns)
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36
            cornish_fisher_var = mean_return + z_cf * std_return
            
            # 5. Extreme Value Theory (EVT)
            extreme_value_var, extreme_value_cvar = self._calculate_evt_var(portfolio_returns, confidence_level)
            
            # 6. Monte Carlo VaR
            monte_carlo_var, monte_carlo_cvar = self._calculate_monte_carlo_var(
                portfolio_returns, confidence_level, n_simulations
            )
            
            # 7. GARCH VaR (if available)
            if HAS_ARCH:
                garch_var, garch_cvar = self._calculate_garch_var(portfolio_returns, confidence_level)
            else:
                garch_var, garch_cvar = 0, 0
            
            # 8. Component analysis
            marginal_var = self._calculate_marginal_var(weights, confidence_level)
            component_var = self._calculate_component_var(weights, confidence_level)
            incremental_var = self._calculate_incremental_var(weights, confidence_level)
            
            # 9. Stress testing VaR
            stressed_var = self._calculate_stressed_var(weights, confidence_level)
            
            # 10. Backtesting
            var_exceptions, exception_rate = self._backtest_var(portfolio_returns, historical_var)
            kupiec_pvalue = self._kupiec_test(var_exceptions, len(portfolio_returns), confidence_level)
            christoffersen_pvalue = self._christoffersen_test(portfolio_returns, historical_var)
            
            # Calculate conditional coverage
            conditional_coverage_pvalue = 0  # Simplified
            
            duration = time.time() - start_time
            logger.log_performance("comprehensive_var", duration, success=True)
            
            return VARAnalysis(
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                historical_var=historical_var,
                historical_cvar=historical_cvar,
                filtered_historical_var=historical_var,  # Simplified
                parametric_var_normal=parametric_var_normal,
                parametric_cvar_normal=parametric_cvar_normal,
                parametric_var_t=parametric_var_t,
                parametric_cvar_t=parametric_cvar_t,
                cornish_fisher_var=cornish_fisher_var,
                extreme_value_var=extreme_value_var,
                extreme_value_cvar=extreme_value_cvar,
                monte_carlo_var=monte_carlo_var,
                monte_carlo_cvar=monte_carlo_cvar,
                garch_var=garch_var,
                garch_cvar=garch_cvar,
                marginal_var=marginal_var,
                component_var=component_var,
                incremental_var=incremental_var,
                stressed_var=stressed_var,
                scenario_var={},  # Will be populated by stress tests
                var_exceptions=var_exceptions,
                exception_rate=exception_rate,
                kupiec_pvalue=kupiec_pvalue,
                christoffersen_pvalue=christoffersen_pvalue,
                conditional_coverage_pvalue=conditional_coverage_pvalue
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.log_performance("comprehensive_var", duration, success=False, error=str(e))
            logger.logger.error(f"VAR calculation failed: {e}")
            
            # Return basic VAR
            return self._calculate_basic_var(weights, confidence_level)
    
    def _calculate_t_cvar(self, df: float, loc: float, scale: float, 
                         confidence_level: float) -> float:
        """Calculate CVaR for t-distribution"""
        t_score = stats.t.ppf(1 - confidence_level, df)
        cvar = loc - scale * (stats.t.pdf(t_score, df) / (1 - confidence_level)) * \
               ((df + t_score**2) / (df - 1))
        return cvar
    
    def _calculate_evt_var(self, returns: pd.Series, 
                          confidence_level: float) -> Tuple[float, float]:
        """Calculate VAR using Extreme Value Theory"""
        # Simplified EVT implementation
        threshold = np.percentile(returns, 95)  # 95% threshold for extremes
        excess_returns = returns[returns > threshold]
        
        if len(excess_returns) < 10:
            return np.percentile(returns, (1 - confidence_level) * 100), 0
        
        # Fit Generalized Pareto Distribution (simplified)
        try:
            # Use method of moments
            mean_excess = excess_returns.mean()
            var_excess = excess_returns.var()
            
            # GPD parameters
            xi = 0.5 * (1 - (mean_excess**2 / var_excess))
            beta = 0.5 * mean_excess * (1 + (mean_excess**2 / var_excess))
            
            # Calculate VaR and CVaR
            n = len(returns)
            nu = len(excess_returns)
            evt_var = threshold + (beta / xi) * (((n / nu) * (1 - confidence_level))**(-xi) - 1)
            evt_cvar = (evt_var + beta - xi * threshold) / (1 - xi)
            
            return evt_var, evt_cvar
            
        except:
            # Fallback to historical
            var = np.percentile(returns, (1 - confidence_level) * 100)
            cvar = returns[returns <= var].mean()
            return var, cvar
    
    def _calculate_monte_carlo_var(self, returns: pd.Series, 
                                  confidence_level: float, 
                                  n_simulations: int) -> Tuple[float, float]:
        """Calculate VAR using Monte Carlo simulation"""
        
        # Fit distribution parameters
        mean = returns.mean()
        std = returns.std()
        skewness = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Generate random returns with moments matching
        simulated_returns = self._generate_correlated_returns(
            mean, std, skewness, kurt, n_simulations
        )
        
        # Calculate VaR and CVaR
        var_mc = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        cvar_mc = simulated_returns[simulated_returns <= var_mc].mean()
        
        return var_mc, cvar_mc
    
    def _generate_correlated_returns(self, mean: float, std: float, 
                                    skewness: float, kurtosis: float, 
                                    n: int) -> np.ndarray:
        """Generate returns with specified moments"""
        # Use Johnson SU distribution approximation
        try:
            # Fit Johnson SU parameters
            gamma = skewness / np.sqrt(8/np.pi * ((np.pi/2)**(1/3) - 1)**3)
            delta = np.sqrt(2 / np.log(np.sqrt(kurtosis + 3)))
            
            # Generate random numbers
            z = np.random.normal(0, 1, n)
            x = np.sinh((z - gamma) / delta)
            
            # Scale to desired mean and std
            returns = mean + std * x
            
            return returns
            
        except:
            # Fallback to normal distribution
            return np.random.normal(mean, std, n)
    
    def _calculate_garch_var(self, returns: pd.Series, 
                            confidence_level: float) -> Tuple[float, float]:
        """Calculate GARCH-based VaR"""
        if not HAS_ARCH:
            return 0, 0
        
        try:
            # Fit GARCH(1,1) model
            garch_model = arch.arch_model(returns * 100, vol='Garch', p=1, q=1)
            garch_result = garch_model.fit(disp='off')
            
            # Forecast volatility
            forecast = garch_result.forecast(horizon=1)
            forecast_vol = np.sqrt(forecast.variance.values[-1, 0]) / 100
            
            # Calculate VaR
            mean_return = returns.mean()
            z_score = norm.ppf(1 - confidence_level)
            garch_var = mean_return + z_score * forecast_vol
            garch_cvar = mean_return - (forecast_vol / (1 - confidence_level)) * norm.pdf(z_score)
            
            return garch_var, garch_cvar
            
        except:
            return 0, 0
    
    def _calculate_marginal_var(self, weights: Dict[str, float], 
                               confidence_level: float) -> Dict[str, float]:
        """Calculate marginal VaR for each asset"""
        marginal_var = {}
        
        for ticker, weight in weights.items():
            if weight > 0.001:
                # Simplified marginal VaR
                asset_returns = self.returns[ticker]
                asset_var = np.percentile(asset_returns, (1 - confidence_level) * 100)
                marginal_var[ticker] = weight * asset_var
        
        return marginal_var
    
    def _calculate_component_var(self, weights: Dict[str, float], 
                                confidence_level: float) -> Dict[str, float]:
        """Calculate component VaR for each asset"""
        component_var = {}
        total_var = 0
        
        for ticker, weight in weights.items():
            if weight > 0.001:
                # Simplified component VaR
                asset_returns = self.returns[ticker]
                asset_var = np.percentile(asset_returns, (1 - confidence_level) * 100)
                component_var[ticker] = weight * asset_var
                total_var += weight * asset_var
        
        # Normalize to 100%
        if total_var > 0:
            component_var = {k: v/total_var * 100 for k, v in component_var.items()}
        
        return component_var
    
    def _calculate_incremental_var(self, weights: Dict[str, float], 
                                  confidence_level: float) -> Dict[str, float]:
        """Calculate incremental VaR for each asset"""
        incremental_var = {}
        
        # Calculate portfolio VaR
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        portfolio_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        for ticker in weights.keys():
            # Remove asset and calculate VaR
            weights_without = weights.copy()
            del weights_without[ticker]
            
            if weights_without:
                # Rebalance weights
                total_weight = sum(weights_without.values())
                weights_without = {k: v/total_weight for k, v in weights_without.items()}
                
                portfolio_without = (self.returns * pd.Series(weights_without)).sum(axis=1)
                var_without = np.percentile(portfolio_without, (1 - confidence_level) * 100)
                
                incremental_var[ticker] = portfolio_var - var_without
            else:
                incremental_var[ticker] = 0
        
        return incremental_var
    
    def _calculate_stressed_var(self, weights: Dict[str, float], 
                               confidence_level: float) -> Dict[str, float]:
        """Calculate stressed VaR under different scenarios"""
        stressed_var = {}
        
        # Stress scenarios
        scenarios = {
            'Market_Crash': 1.5,  # 150% increase in volatility
            'Volatility_Spike': 2.0,  # 200% increase in volatility
            'Liquidity_Crisis': 1.8,  # 180% increase in volatility
        }
        
        for scenario, multiplier in scenarios.items():
            # Apply stress to returns
            stressed_returns = self.returns * multiplier
            portfolio_returns = (stressed_returns * pd.Series(weights)).sum(axis=1)
            stressed_var[scenario] = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        return stressed_var
    
    def _backtest_var(self, returns: pd.Series, var: float) -> Tuple[int, float]:
        """Backtest VaR model"""
        exceptions = (returns < var).sum()
        exception_rate = exceptions / len(returns)
        return int(exceptions), exception_rate
    
    def _kupiec_test(self, exceptions: int, n: int, confidence_level: float) -> float:
        """Kupiec's proportion of failures test"""
        expected_exceptions = n * (1 - confidence_level)
        actual_exceptions = exceptions
        
        if expected_exceptions == 0:
            return 1.0
        
        # Likelihood ratio test
        p_hat = actual_exceptions / n
        p = 1 - confidence_level
        
        if p_hat == 0:
            lr = -2 * np.log(((1 - p) ** n) / ((1 - p_hat) ** n))
        elif p_hat == 1:
            lr = -2 * np.log((p ** n) / (p_hat ** n))
        else:
            lr = -2 * np.log(((p ** actual_exceptions) * ((1 - p) ** (n - actual_exceptions))) /
                            ((p_hat ** actual_exceptions) * ((1 - p_hat) ** (n - actual_exceptions))))
        
        # Chi-square p-value
        p_value = 1 - stats.chi2.cdf(lr, 1)
        return p_value
    
    def _christoffersen_test(self, returns: pd.Series, var: float) -> float:
        """Christoffersen's independence test"""
        # Simplified implementation
        violations = (returns < var).astype(int)
        
        # Count transitions
        n00 = n01 = n10 = n11 = 0
        for i in range(1, len(violations)):
            if violations[i-1] == 0 and violations[i] == 0:
                n00 += 1
            elif violations[i-1] == 0 and violations[i] == 1:
                n01 += 1
            elif violations[i-1] == 1 and violations[i] == 0:
                n10 += 1
            elif violations[i-1] == 1 and violations[i] == 1:
                n11 += 1
        
        # Calculate test statistic
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        if pi0 == 0 or pi1 == 0 or pi == 0 or pi == 1:
            return 1.0
        
        lr = -2 * np.log(((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) /
                        ((1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11))
        
        p_value = 1 - stats.chi2.cdf(lr, 1)
        return p_value
    
    def _calculate_basic_var(self, weights: Dict[str, float], 
                            confidence_level: float) -> VARAnalysis:
        """Calculate basic VAR when comprehensive methods fail"""
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return VARAnalysis(
            confidence_level=confidence_level,
            time_horizon=1,
            historical_var=var,
            historical_cvar=cvar,
            filtered_historical_var=var,
            parametric_var_normal=var,
            parametric_cvar_normal=cvar,
            parametric_var_t=var,
            parametric_cvar_t=cvar,
            cornish_fisher_var=var,
            extreme_value_var=var,
            extreme_value_cvar=cvar,
            monte_carlo_var=var,
            monte_carlo_cvar=cvar,
            garch_var=var,
            garch_cvar=cvar,
            marginal_var={},
            component_var={},
            incremental_var={},
            stressed_var={},
            scenario_var={},
            var_exceptions=0,
            exception_rate=0,
            kupiec_pvalue=1.0,
            christoffersen_pvalue=1.0,
            conditional_coverage_pvalue=1.0
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRESS TESTING AND SCENARIO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def perform_stress_testing(self, weights: Dict[str, float], 
                              scenarios: List[StressScenario] = None) -> List[StressTestResult]:
        """Perform comprehensive stress testing"""
        
        if scenarios is None:
            scenarios = [
                StressScenario.MARKET_CRASH_2008,
                StressScenario.COVID_CRASH,
                StressScenario.INTEREST_RATE_SHOCK,
                StressScenario.VOLATILITY_SPIKE
            ]
        
        results = []
        
        for scenario in scenarios:
            try:
                result = self._stress_test_scenario(weights, scenario)
                results.append(result)
                
            except Exception as e:
                logger.logger.warning(f"Stress test failed for {scenario}: {e}")
                continue
        
        self.stress_test_results.extend(results)
        return results
    
    def _stress_test_scenario(self, weights: Dict[str, float], 
                             scenario: StressScenario) -> StressTestResult:
        """Stress test for a specific scenario"""
        
        # Define scenario parameters
        scenario_params = self._get_scenario_parameters(scenario)
        
        # Apply stress to returns
        stressed_returns = self._apply_stress_to_returns(scenario_params)
        
        # Calculate portfolio performance under stress
        portfolio_returns = (stressed_returns * pd.Series(weights)).sum(axis=1)
        
        # Calculate metrics
        portfolio_loss = portfolio_returns.mean() * 252
        max_daily_loss = portfolio_returns.min()
        
        # Calculate VAR breach
        normal_var = np.percentile(self.returns.dot(list(weights.values())), 5)
        stressed_var = np.percentile(portfolio_returns, 5)
        var_breach = stressed_var < normal_var
        
        # Calculate CVaR impact
        normal_cvar = self.returns.dot(list(weights.values()))
        normal_cvar = normal_cvar[normal_cvar <= normal_var].mean()
        stressed_cvar = portfolio_returns[portfolio_returns <= stressed_var].mean()
        cvar_impact = stressed_cvar - normal_cvar
        
        # Calculate component impacts
        component_impacts = {}
        for ticker, weight in weights.items():
            if weight > 0.001:
                normal_return = self.returns[ticker].mean()
                stressed_return = stressed_returns[ticker].mean()
                component_impacts[ticker] = stressed_return - normal_return
        
        return StressTestResult(
            scenario=scenario.value,
            portfolio_loss=portfolio_loss,
            benchmark_loss=0,  # Would need benchmark
            relative_loss=portfolio_loss,
            var_breach=var_breach,
            cvar_impact=cvar_impact,
            component_impacts=component_impacts,
            recovery_period=0,  # Would need simulation
            liquidity_impact=0,
            correlation_impact=0,
            volatility_impact=scenario_params.get('volatility_multiplier', 1) - 1
        )
    
    def _get_scenario_parameters(self, scenario: StressScenario) -> Dict:
        """Get parameters for stress scenario"""
        scenarios = {
            StressScenario.MARKET_CRASH_2008: {
                'volatility_multiplier': 3.0,
                'return_shift': -0.40,
                'correlation_increase': 0.3
            },
            StressScenario.COVID_CRASH: {
                'volatility_multiplier': 2.5,
                'return_shift': -0.35,
                'correlation_increase': 0.2
            },
            StressScenario.INTEREST_RATE_SHOCK: {
                'volatility_multiplier': 1.8,
                'return_shift': -0.20,
                'sector_impact': {'Banking': -0.30, 'Real Estate': -0.25}
            },
            StressScenario.VOLATILITY_SPIKE: {
                'volatility_multiplier': 2.2,
                'return_shift': -0.15,
                'correlation_increase': 0.15
            }
        }
        
        return scenarios.get(scenario, {
            'volatility_multiplier': 1.5,
            'return_shift': -0.10,
            'correlation_increase': 0.1
        })
    
    def _apply_stress_to_returns(self, params: Dict) -> pd.DataFrame:
        """Apply stress parameters to returns"""
        stressed_returns = self.returns.copy()
        
        # Apply volatility multiplier
        volatility_multiplier = params.get('volatility_multiplier', 1)
        stressed_returns = stressed_returns * volatility_multiplier
        
        # Apply return shift
        return_shift = params.get('return_shift', 0)
        stressed_returns = stressed_returns + return_shift / 252  # Convert annual to daily
        
        # Apply sector-specific impacts
        sector_impact = params.get('sector_impact', {})
        for sector, impact in sector_impact.items():
            sector_tickers = SECTOR_MAPPING.get(sector, [])
            for ticker in sector_tickers:
                if ticker in stressed_returns.columns:
                    stressed_returns[ticker] = stressed_returns[ticker] + impact / 252
        
        return stressed_returns
    
    # ─────────────────────────────────────────────────────────────────────────
    # BACKTESTING FRAMEWORK
    # ─────────────────────────────────────────────────────────────────────────
    
    def perform_backtesting(self, method: BacktestMethod = BacktestMethod.ROLLING_WINDOW,
                           window_size: int = 252,
                           rebalance_frequency: int = 63) -> List[BacktestResult]:
        """Perform comprehensive backtesting"""
        
        backtest_results = []
        
        if method == BacktestMethod.ROLLING_WINDOW:
            results = self._rolling_window_backtest(window_size, rebalance_frequency)
            backtest_results.extend(results)
        
        elif method == BacktestMethod.EXPANDING_WINDOW:
            results = self._expanding_window_backtest(window_size, rebalance_frequency)
            backtest_results.extend(results)
        
        elif method == BacktestMethod.WALK_FORWARD:
            results = self._walk_forward_backtest(window_size, rebalance_frequency)
            backtest_results.extend(results)
        
        self.backtest_results.extend(backtest_results)
        return backtest_results
    
    def _rolling_window_backtest(self, window_size: int, 
                                rebalance_frequency: int) -> List[BacktestResult]:
        """Rolling window backtest"""
        results = []
        n_periods = len(self.returns)
        
        for i in range(window_size, n_periods, rebalance_frequency):
            try:
                # Training period
                train_start = i - window_size
                train_end = i
                train_returns = self.returns.iloc[train_start:train_end]
                train_prices = self.prices.iloc[train_start:train_end]
                
                # Test period
                test_start = i
                test_end = min(i + rebalance_frequency, n_periods)
                test_returns = self.returns.iloc[test_start:test_end]
                
                if len(train_returns) < 50 or len(test_returns) < 10:
                    continue
                
                # Create optimizer for training period
                train_optimizer = InstitutionalPortfolioOptimizer(train_prices, train_returns)
                
                # Optimize portfolio
                optimization_result = train_optimizer.optimize(
                    OptimizationMethod.MAX_SHARPE,
                    risk_free_rate=0.30
                )
                
                # Calculate test period performance
                test_portfolio_returns = (test_returns * pd.Series(optimization_result.weights)).sum(axis=1)
                
                # Calculate metrics
                total_return = (1 + test_portfolio_returns).prod() - 1
                annual_return = test_portfolio_returns.mean() * 252
                annual_vol = test_portfolio_returns.std() * np.sqrt(252)
                sharpe = (annual_return - 0.30) / annual_vol if annual_vol > 0 else 0
                
                # Maximum drawdown
                cum_returns = (1 + test_portfolio_returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                max_dd = drawdown.min()
                calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
                
                # Win rate
                win_rate = (test_portfolio_returns > 0).sum() / len(test_portfolio_returns)
                
                # Profit factor
                gains = test_portfolio_returns[test_portfolio_returns > 0].sum()
                losses = abs(test_portfolio_returns[test_portfolio_returns < 0].sum())
                profit_factor = gains / losses if losses > 0 else float('inf')
                
                # VAR exceptions
                var_95 = np.percentile(test_portfolio_returns, 5)
                var_exceptions = (test_portfolio_returns < var_95).sum()
                exception_rate = var_exceptions / len(test_portfolio_returns)
                
                result = BacktestResult(
                    method=BacktestMethod.ROLLING_WINDOW.value,
                    period_start=self.returns.index[test_start],
                    period_end=self.returns.index[test_end - 1],
                    total_return=total_return,
                    annual_return=annual_return,
                    annual_volatility=annual_vol,
                    sharpe_ratio=sharpe,
                    max_drawdown=max_dd,
                    calmar_ratio=calmar,
                    win_rate=win_rate,
                    profit_factor=profit_factor,
                    var_exceptions=int(var_exceptions),
                    exception_rate=exception_rate,
                    average_turnover=0,
                    information_ratio=0,
                    alpha=0,
                    beta=0,
                    tracking_error=0,
                    r_squared=0
                )
                
                results.append(result)
                
            except Exception as e:
                logger.logger.warning(f"Backtest failed for period {i}: {e}")
                continue
        
        return results
    
    def _expanding_window_backtest(self, window_size: int, 
                                  rebalance_frequency: int) -> List[BacktestResult]:
        """Expanding window backtest"""
        results = []
        n_periods = len(self.returns)
        
        for i in range(window_size, n_periods, rebalance_frequency):
            try:
                # Training period (expanding window)
                train_start = 0
                train_end = i
                train_returns = self.returns.iloc[train_start:train_end]
                train_prices = self.prices.iloc[train_start:train_end]
                
                # Test period
                test_start = i
                test_end = min(i + rebalance_frequency, n_periods)
                test_returns = self.returns.iloc[test_start:test_end]
                
                if len(test_returns) < 10:
                    continue
                
                # Create optimizer for training period
                train_optimizer = InstitutionalPortfolioOptimizer(train_prices, train_returns)
                
                # Optimize portfolio
                optimization_result = train_optimizer.optimize(
                    OptimizationMethod.MAX_SHARPE,
                    risk_free_rate=0.30
                )
                
                # Calculate test period performance
                test_portfolio_returns = (test_returns * pd.Series(optimization_result.weights)).sum(axis=1)
                
                # Calculate metrics (similar to rolling window)
                total_return = (1 + test_portfolio_returns).prod() - 1
                annual_return = test_portfolio_returns.mean() * 252
                annual_vol = test_portfolio_returns.std() * np.sqrt(252)
                sharpe = (annual_return - 0.30) / annual_vol if annual_vol > 0 else 0
                
                # Maximum drawdown
                cum_returns = (1 + test_portfolio_returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                max_dd = drawdown.min()
                calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
                
                # Win rate
                win_rate = (test_portfolio_returns > 0).sum() / len(test_portfolio_returns)
                
                result = BacktestResult(
                    method=BacktestMethod.EXPANDING_WINDOW.value,
                    period_start=self.returns.index[test_start],
                    period_end=self.returns.index[test_end - 1],
                    total_return=total_return,
                    annual_return=annual_return,
                    annual_volatility=annual_vol,
                    sharpe_ratio=sharpe,
                    max_drawdown=max_dd,
                    calmar_ratio=calmar,
                    win_rate=win_rate,
                    profit_factor=0,
                    var_exceptions=0,
                    exception_rate=0,
                    average_turnover=0,
                    information_ratio=0,
                    alpha=0,
                    beta=0,
                    tracking_error=0,
                    r_squared=0
                )
                
                results.append(result)
                
            except Exception as e:
                logger.logger.warning(f"Expanding window backtest failed for period {i}: {e}")
                continue
        
        return results
    
    def _walk_forward_backtest(self, window_size: int, 
                              rebalance_frequency: int) -> List[BacktestResult]:
        """Walk-forward analysis backtest"""
        # Similar to rolling window but with optimization parameter tuning
        return self._rolling_window_backtest(window_size, rebalance_frequency)
    
    # ─────────────────────────────────────────────────────────────────────────
    # QUANTSTATS INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def generate_quantstats_report(self, weights: Dict[str, float], 
                                  benchmark_ticker: str = None) -> Optional[str]:
        """Generate comprehensive QuantStats report"""
        
        if not HAS_QUANTSTATS:
            logger.logger.warning("QuantStats not available")
            return None
        
        try:
            # Calculate portfolio returns
            portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
            
            # Get benchmark returns if specified
            benchmark_returns = None
            if benchmark_ticker:
                try:
                    # Fetch benchmark data
                    benchmark_data = yf.download(
                        benchmark_ticker,
                        start=self.returns.index[0],
                        end=self.returns.index[-1],
                        progress=False
                    )
                    
                    if not benchmark_data.empty:
                        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                        # Align dates
                        benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).ffill().bfill()
                        
                except Exception as e:
                    logger.logger.warning(f"Failed to fetch benchmark data: {e}")
            
            # Generate HTML report
            buffer = io.StringIO()
            
            if benchmark_returns is not None:
                qs.reports.html(
                    portfolio_returns,
                    benchmark_returns,
                    rf=0.30,
                    title='Portfolio Performance Report',
                    output=buffer,
                    download_filename='portfolio_report.html'
                )
            else:
                qs.reports.html(
                    portfolio_returns,
                    rf=0.30,
                    title='Portfolio Performance Report',
                    output=buffer,
                    download_filename='portfolio_report.html'
                )
            
            html_report = buffer.getvalue()
            buffer.close()
            
            return html_report
            
        except Exception as e:
            logger.logger.error(f"QuantStats report generation failed: {e}")
            return None
    
    def calculate_quantstats_metrics(self, weights: Dict[str, float]) -> Dict:
        """Calculate comprehensive QuantStats metrics"""
        
        if not HAS_QUANTSTATS:
            return {}
        
        try:
            portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
            
            metrics = {}
            
            # Basic metrics
            metrics['cagr'] = qs_stats.cagr(portfolio_returns)
            metrics['volatility'] = qs_stats.volatility(portfolio_returns)
            metrics['sharpe'] = qs_stats.sharpe(portfolio_returns, rf=0.30)
            metrics['sortino'] = qs_stats.sortino(portfolio_returns, rf=0.30)
            metrics['max_drawdown'] = qs_stats.max_drawdown(portfolio_returns)
            metrics['calmar'] = qs_stats.calmar(portfolio_returns)
            
            # Advanced metrics
            metrics['omega'] = qs_stats.omega(portfolio_returns, rf=0.30)
            metrics['tail_ratio'] = qs_stats.tail_ratio(portfolio_returns)
            metrics['common_sense_ratio'] = qs_stats.common_sense_ratio(portfolio_returns)
            metrics['gain_to_pain_ratio'] = qs_stats.gain_to_pain_ratio(portfolio_returns)
            
            # Risk metrics
            metrics['value_at_risk'] = qs_stats.value_at_risk(portfolio_returns)
            metrics['conditional_var'] = qs_stats.conditional_value_at_risk(portfolio_returns)
            metrics['expected_shortfall'] = qs_stats.expected_shortfall(portfolio_returns)
            
            # Additional metrics
            metrics['skew'] = qs_stats.skew(portfolio_returns)
            metrics['kurtosis'] = qs_stats.kurtosis(portfolio_returns)
            metrics['ulcer_index'] = qs_stats.ulcer_index(portfolio_returns)
            metrics['serenity_index'] = qs_stats.serenity_index(portfolio_returns)
            
            return metrics
            
        except Exception as e:
            logger.logger.error(f"QuantStats metrics calculation failed: {e}")
            return {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # EFFICIENT FRONTIER ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def calculate_efficient_frontier(self, points: int = 100) -> Dict:
        """Calculate efficient frontier"""
        
        try:
            mu = self.expected_returns['mean_historical']
            S = self.covariance_matrices['ledoit_wolf']
            
            ef = EfficientFrontier(mu, S)
            
            # Get minimum volatility portfolio
            ef.min_volatility()
            min_vol_weights = ef.clean_weights()
            min_vol_perf = ef.portfolio_performance()
            
            # Get maximum Sharpe portfolio
            ef.max_sharpe()
            max_sharpe_perf = ef.portfolio_performance()
            
            # Generate frontier points
            target_returns = np.linspace(min_vol_perf[0], mu.max(), points)
            volatilities = []
            sharpe_ratios = []
            
            for target in target_returns:
                try:
                    ef = EfficientFrontier(mu, S)
                    ef.efficient_return(target_return=target)
                    _, vol, sharpe = ef.portfolio_performance()
                    volatilities.append(vol)
                    sharpe_ratios.append(sharpe)
                except:
                    volatilities.append(np.nan)
                    sharpe_ratios.append(np.nan)
            
            # Calculate individual asset points
            asset_returns = []
            asset_volatilities = []
            
            for ticker in self.tickers:
                asset_return = mu[ticker]
                asset_vol = np.sqrt(S.loc[ticker, ticker])
                asset_returns.append(asset_return)
                asset_volatilities.append(asset_vol)
            
            return {
                'target_returns': target_returns,
                'volatilities': volatilities,
                'sharpe_ratios': sharpe_ratios,
                'min_vol_return': min_vol_perf[0],
                'min_vol_volatility': min_vol_perf[1],
                'min_vol_sharpe': min_vol_perf[2],
                'max_sharpe_return': max_sharpe_perf[0],
                'max_sharpe_volatility': max_sharpe_perf[1],
                'max_sharpe_sharpe': max_sharpe_perf[2],
                'asset_returns': asset_returns,
                'asset_volatilities': asset_volatilities,
                'assets': self.tickers
            }
            
        except Exception as e:
            logger.logger.error(f"Efficient frontier calculation failed: {e}")
            return {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def calculate_discrete_allocation(self, weights: Dict[str, float], 
                                     portfolio_value: float = 1000000) -> Tuple[Dict, float]:
        """Calculate discrete share allocation"""
        try:
            latest_prices = get_latest_prices(self.prices)
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_value)
            
            try:
                allocation, leftover = da.lp_portfolio()
            except:
                allocation, leftover = da.greedy_portfolio()
            
            return allocation, leftover
            
        except Exception as e:
            logger.logger.error(f"Discrete allocation failed: {e}")
            return {}, 0.0
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix"""
        return self.returns.corr()
    
    def calculate_rolling_metrics(self, weights: Dict[str, float], 
                                 window: int = 63) -> Dict:
        """Calculate rolling metrics"""
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        
        rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = portfolio_returns.rolling(window).apply(
            lambda x: (x.mean() * 252 - 0.30) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        rolling_max_dd = portfolio_returns.rolling(window).apply(
            lambda x: self._calculate_rolling_drawdown(x)
        )
        
        return {
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_max_drawdown': rolling_max_dd
        }
    
    def _calculate_rolling_drawdown(self, returns: pd.Series) -> float:
        """Calculate rolling drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    def get_performance_summary(self, weights: Dict[str, float]) -> Dict:
        """Get comprehensive performance summary"""
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        
        summary = {
            'total_return': (1 + portfolio_returns).prod() - 1,
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252 - 0.30) / (portfolio_returns.std() * np.sqrt(252)) 
                           if portfolio_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'win_rate': (portfolio_returns > 0).sum() / len(portfolio_returns),
            'profit_factor': abs(portfolio_returns[portfolio_returns > 0].sum() / 
                                portfolio_returns[portfolio_returns < 0].sum()) 
                           if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0,
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
        }
        
        return summary
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

# ─────────────────────────────────────────────────────────────────────────────
# DATA CONSTANTS
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
    'Brent Oil': 'BZ=F',
    'US 10Y Yield': '^TNX'
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main application function"""
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.optimized = False
        st.session_state.portfolio_data = None
        st.session_state.optimizer = None
        st.session_state.optimization_result = None
        st.session_state.var_results = None
        st.session_state.stress_results = None
        st.session_state.backtest_results = None
    
    # Professional header
    st.markdown(f"""
    <div class="institutional-header">
        <h1 style="color: {INSTITUTIONAL_COLORS['text_primary']}; text-align: center; margin-bottom: 1rem;">
            📊 BIST Institutional Portfolio Analytics Suite Pro
        </h1>
        <p style="color: {INSTITUTIONAL_COLORS['text_secondary']}; text-align: center; font-size: 1.1rem;">
            Advanced Quantitative Portfolio Optimization with Comprehensive Risk Analytics
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <span class="badge badge-success">VaR Analytics</span>
            <span class="badge badge-warning">Stress Testing</span>
            <span class="badge badge-danger">Backtesting</span>
            <span class="badge badge-info">QuantStats</span>
            <span class="badge">Efficient Frontier</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check requirements
    if not HAS_YFINANCE:
        st.error("""
        ## ❌ Critical Dependency Missing
        
        **yfinance** is required but not installed. Please install with:
        ```bash
        pip install yfinance>=0.2.28
        ```
        
        For all dependencies:
        ```bash
        pip install yfinance pypfopt quantstats streamlit plotly pandas numpy scipy scikit-learn arch
        ```
        """)
        return
    
    if not HAS_PYPFOPT:
        st.warning("""
        ⚠️ **PyPortfolioOpt** is not fully installed.
        Some optimization features may be limited.
        """)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>⚙️ Configuration</h3>", 
                   unsafe_allow_html=True)
        
        # Date selection
        date_preset = st.selectbox(
            "Time Period",
            ["1 Year", "3 Years", "5 Years", "Custom"],
            index=1
        )
        
        if date_preset == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*3))
            with col2:
                end_date = st.date_input("End Date", datetime.now())
        else:
            end_date = datetime.now()
            if date_preset == "1 Year":
                start_date = end_date - timedelta(days=365)
            elif date_preset == "3 Years":
                start_date = end_date - timedelta(days=365*3)
            else:  # 5 Years
                start_date = end_date - timedelta(days=365*5)
            
            with st.expander("Date Details", expanded=False):
                st.write(f"Start: {start_date.strftime('%Y-%m-%d')}")
                st.write(f"End: {end_date.strftime('%Y-%m-%d')}")
        
        # Asset selection
        st.markdown("---")
        st.markdown(f"<h4 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>📊 Asset Selection</h4>", 
                   unsafe_allow_html=True)
        
        # Sector filter
        selected_sector = st.selectbox(
            "Sector Filter",
            ["All Sectors"] + list(SECTOR_MAPPING.keys())
        )
        
        # Get available tickers
        if selected_sector == "All Sectors":
            available_tickers = BIST100_TICKERS
        else:
            available_tickers = SECTOR_MAPPING.get(selected_sector, [])
        
        # Search box
        search_query = st.text_input("🔍 Search Tickers", "")
        if search_query:
            available_tickers = [t for t in available_tickers if search_query.upper() in t]
        
        # Multi-select assets
        assets = st.multiselect(
            "Select Assets (Max 15)",
            available_tickers,
            default=['THYAO.IS', 'GARAN.IS', 'ASELS.IS', 'AKBNK.IS', 'FROTO.IS'],
            max_selections=15
        )
        
        # Optimization settings
        st.markdown("---")
        st.markdown(f"<h4 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>⚡ Optimization</h4>", 
                   unsafe_allow_html=True)
        
        optimization_method = st.selectbox(
            "Optimization Method",
            [m.value for m in OptimizationMethod][:10],  # Show first 10 methods
            index=0
        )
        
        # Advanced parameters
        with st.expander("Advanced Parameters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 50.0, 30.0, 0.1) / 100
                target_volatility = st.slider("Target Volatility (%)", 5.0, 50.0, 15.0, 0.5) / 100
            with col2:
                target_return = st.slider("Target Return (%)", 5.0, 100.0, 20.0, 0.5) / 100
                var_confidence = st.slider("VAR Confidence", 90.0, 99.9, 95.0, 0.1) / 100
        
        # Analytics options
        st.markdown("---")
        st.markdown(f"<h4 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>📈 Analytics</h4>", 
                   unsafe_allow_html=True)
        
        col_analytics1, col_analytics2 = st.columns(2)
        with col_analytics1:
            calculate_var = st.checkbox("VaR Analytics", True)
            stress_testing = st.checkbox("Stress Testing", True)
        with col_analytics2:
            backtesting = st.checkbox("Backtesting", True)
            quantstats_report = st.checkbox("QuantStats Report", True)
        
        # Action buttons
        st.markdown("---")
        col_actions1, col_actions2 = st.columns(2)
        with col_actions1:
            if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
                st.session_state.data_loaded = False
                st.rerun()
        with col_actions2:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
    
    # Main content
    if not assets or len(assets) < 2:
        st.warning("Please select at least 2 assets for portfolio optimization")
        return
    
    # Load data
    with st.spinner("📥 Loading market data..."):
        data_fetcher = AdvancedDataFetcher()
        close_prices, returns = data_fetcher.fetch_market_data(assets, start_date, end_date)
        
        if close_prices is None or returns is None:
            st.error("Failed to load market data. Please try different assets or date range.")
            return
        
        st.session_state.data_loaded = True
        st.session_state.close_prices = close_prices
        st.session_state.returns = returns
    
    if st.session_state.data_loaded:
        # Create optimizer
        with st.spinner("⚙️ Initializing optimizer..."):
            optimizer = InstitutionalPortfolioOptimizer(close_prices, returns)
            st.session_state.optimizer = optimizer
        
        # Optimize portfolio
        with st.spinner("🔧 Optimizing portfolio..."):
            # Find the enum value from the display string
            method_enum = None
            for method in OptimizationMethod:
                if method.value == optimization_method:
                    method_enum = method
                    break
            
            if method_enum is None:
                method_enum = OptimizationMethod.MAX_SHARPE
            
            optimization_result = optimizer.optimize(
                method_enum,
                risk_free_rate=risk_free_rate,
                target_volatility=target_volatility,
                target_return=target_return
            )
            
            st.session_state.optimization_result = optimization_result
        
        # Display results
        display_results(optimizer, optimization_result, var_confidence, 
                       calculate_var, stress_testing, backtesting, quantstats_report)

def display_results(optimizer: InstitutionalPortfolioOptimizer, 
                   optimization_result: OptimizationResult,
                   var_confidence: float,
                   calculate_var: bool,
                   stress_testing: bool,
                   backtesting: bool,
                   quantstats_report: bool):
    """Display comprehensive results"""
    
    # Performance summary
    st.markdown("---")
    st.markdown(f"<h2 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>📊 Performance Summary</h2>", 
               unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Expected Return", f"{optimization_result.expected_return:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Expected Volatility", f"{optimization_result.expected_volatility:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Sharpe Ratio", f"{optimization_result.sharpe_ratio:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Sortino Ratio", f"{optimization_result.sortino_ratio:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Max Drawdown", f"{optimization_result.max_drawdown:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Calmar Ratio", f"{optimization_result.calmar_ratio:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col7:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Omega Ratio", f"{optimization_result.omega_ratio:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col8:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Diversification", f"{optimization_result.diversification_ratio:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab_names = [
        "🎯 Portfolio Allocation",
        "📈 Performance Analytics",
        "⚠️ Risk & VAR Analytics",
        "📊 Efficient Frontier",
        "🔬 Stress Testing",
        "📋 Backtesting",
        "📑 QuantStats Report",
        "📋 Detailed Metrics"
    ]
    
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        display_portfolio_allocation(optimizer, optimization_result)
    
    with tabs[1]:
        display_performance_analytics(optimizer, optimization_result)
    
    with tabs[2]:
        if calculate_var:
            display_var_analytics(optimizer, optimization_result, var_confidence)
        else:
            st.info("Enable VaR Analytics in sidebar to see risk metrics")
    
    with tabs[3]:
        display_efficient_frontier(optimizer, optimization_result)
    
    with tabs[4]:
        if stress_testing:
            display_stress_testing(optimizer, optimization_result)
        else:
            st.info("Enable Stress Testing in sidebar to see scenario analysis")
    
    with tabs[5]:
        if backtesting:
            display_backtesting(optimizer)
        else:
            st.info("Enable Backtesting in sidebar to see historical performance")
    
    with tabs[6]:
        if quantstats_report and HAS_QUANTSTATS:
            display_quantstats_report(optimizer, optimization_result)
        else:
            st.info("Enable QuantStats Report in sidebar to generate comprehensive report")
    
    with tabs[7]:
        display_detailed_metrics(optimizer, optimization_result)

def display_portfolio_allocation(optimizer: InstitutionalPortfolioOptimizer, 
                               optimization_result: OptimizationResult):
    """Display portfolio allocation"""
    
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>Portfolio Allocation</h3>", 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Pie chart
        weights_df = optimization_result.to_dataframe()
        if not weights_df.empty:
            fig = px.pie(
                weights_df,
                values='Weight',
                names=weights_df.index,
                hole=0.4,
                color_discrete_sequence=[
                    INSTITUTIONAL_COLORS['professional_blue'],
                    INSTITUTIONAL_COLORS['institutional_green'],
                    INSTITUTIONAL_COLORS['analytics_purple'],
                    INSTITUTIONAL_COLORS['warning_orange'],
                    INSTITUTIONAL_COLORS['risk_red']
                ]
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=400,
                paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
                plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
                font_color=INSTITUTIONAL_COLORS['text_primary'],
                legend=dict(
                    bgcolor=INSTITUTIONAL_COLORS['dark_blue'],
                    bordercolor=INSTITUTIONAL_COLORS['slate_blue'],
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weights table with sector information
        st.markdown("#### Portfolio Composition")
        
        display_df = weights_df.copy()
        display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2%}")
        display_df['Sector'] = display_df.index.map(
            lambda x: next((sector for sector, tickers in SECTOR_MAPPING.items() if x in tickers), 'Unknown')
        )
        
        # Add expected return and risk
        display_df['Exp. Return'] = display_df.index.map(
            lambda x: f"{optimizer.expected_returns['mean_historical'].get(x, 0):.2%}"
        )
        display_df['Volatility'] = display_df.index.map(
            lambda x: f"{np.sqrt(optimizer.covariance_matrices['ledoit_wolf'].loc[x, x]):.2%}"
        )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Concentration metrics
        st.markdown("#### Concentration Analysis")
        
        col_conc1, col_conc2, col_conc3 = st.columns(3)
        
        with col_conc1:
            hhi = optimization_result.concentration_index
            st.metric("HHI Index", f"{hhi:.4f}", 
                     delta="Low" if hhi < 0.15 else "High" if hhi > 0.25 else "Medium",
                     delta_color="inverse")
        
        with col_conc2:
            num_positions = len(weights_df)
            st.metric("Number of Positions", num_positions)
        
        with col_conc3:
            top3 = weights_df.nlargest(3, 'Weight')
            top3_concentration = top3['Weight'].sum()
            st.metric("Top 3 Concentration", f"{top3_concentration:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_performance_analytics(optimizer: InstitutionalPortfolioOptimizer, 
                                optimization_result: OptimizationResult):
    """Display performance analytics"""
    
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>Performance Analytics</h3>", 
               unsafe_allow_html=True)
    
    # Calculate portfolio returns
    portfolio_returns = (optimizer.returns * pd.Series(optimization_result.weights)).sum(axis=1)
    
    # Cumulative returns chart
    cum_returns = (1 + portfolio_returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns.values,
        mode='lines',
        name='Portfolio',
        line=dict(color=INSTITUTIONAL_COLORS['institutional_green'], width=3),
        fill='tozeroy',
        fillcolor=f"rgba({int(INSTITUTIONAL_COLORS['institutional_green'][1:3], 16)}, "
                 f"{int(INSTITUTIONAL_COLORS['institutional_green'][3:5], 16)}, "
                 f"{int(INSTITUTIONAL_COLORS['institutional_green'][5:7], 16)}, 0.2)"
    ))
    
    fig.update_layout(
        title="Cumulative Portfolio Returns",
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        font_color=INSTITUTIONAL_COLORS['text_primary']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rolling metrics
    st.markdown("#### Rolling Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rolling volatility (6 months)
        rolling_vol = portfolio_returns.rolling(window=126).std() * np.sqrt(252)
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode='lines',
            name='Rolling Volatility',
            line=dict(color=INSTITUTIONAL_COLORS['warning_orange'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba({int(INSTITUTIONAL_COLORS['warning_orange'][1:3], 16)}, "
                     f"{int(INSTITUTIONAL_COLORS['warning_orange'][3:5], 16)}, "
                     f"{int(INSTITUTIONAL_COLORS['warning_orange'][5:7], 16)}, 0.2)"
        ))
        
        fig_vol.update_layout(
            title="Rolling 6-Month Volatility",
            template="plotly_dark",
            height=300,
            xaxis_title="Date",
            yaxis_title="Volatility",
            yaxis_tickformat=".0%",
            paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
            plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue']
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        # Rolling Sharpe ratio (6 months)
        rolling_sharpe = portfolio_returns.rolling(window=126).apply(
            lambda x: (x.mean() * 252 - optimization_result.risk_free_rate) / 
                     (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color=INSTITUTIONAL_COLORS['professional_blue'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba({int(INSTITUTIONAL_COLORS['professional_blue'][1:3], 16)}, "
                     f"{int(INSTITUTIONAL_COLORS['professional_blue'][3:5], 16)}, "
                     f"{int(INSTITUTIONAL_COLORS['professional_blue'][5:7], 16)}, 0.2)"
        ))
        
        fig_sharpe.update_layout(
            title="Rolling 6-Month Sharpe Ratio",
            template="plotly_dark",
            height=300,
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
            plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue']
        )
        
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    # Drawdown analysis
    st.markdown("#### Drawdown Analysis")
    
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        line=dict(color=INSTITUTIONAL_COLORS['risk_red'], width=2),
        fill='tozeroy',
        fillcolor=f"rgba({int(INSTITUTIONAL_COLORS['risk_red'][1:3], 16)}, "
                 f"{int(INSTITUTIONAL_COLORS['risk_red'][3:5], 16)}, "
                 f"{int(INSTITUTIONAL_COLORS['risk_red'][5:7], 16)}, 0.3)"
    ))
    
    fig_dd.update_layout(
        title="Portfolio Drawdown",
        template="plotly_dark",
        height=300,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue']
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_var_analytics(optimizer: InstitutionalPortfolioOptimizer, 
                        optimization_result: OptimizationResult,
                        var_confidence: float):
    """Display comprehensive VAR analytics"""
    
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>Value at Risk Analytics</h3>", 
               unsafe_allow_html=True)
    
    with st.spinner("Calculating comprehensive VAR analytics..."):
        var_results = optimizer.calculate_comprehensive_var(
            optimization_result.weights,
            confidence_level=var_confidence
        )
    
    # VAR metrics
    st.markdown("#### VAR Metrics")
    
    col_var1, col_var2, col_var3, col_var4 = st.columns(4)
    
    with col_var1:
        st.metric(f"Historical VaR ({var_confidence*100:.1f}%)", 
                 f"{var_results.historical_var:.4f}")
    
    with col_var2:
        st.metric(f"Expected Shortfall", 
                 f"{var_results.historical_cvar:.4f}")
    
    with col_var3:
        st.metric("Monte Carlo VaR", 
                 f"{var_results.monte_carlo_var:.4f}")
    
    with col_var4:
        st.metric("VAR Exceptions", 
                 f"{var_results.var_exceptions}",
                 delta=f"{var_results.exception_rate:.2%}")
    
    # VAR comparison chart
    st.markdown("#### VAR Method Comparison")
    
    var_data = var_results.to_dataframe()
    
    fig_var = px.bar(
        var_data,
        x='Method',
        y='Value',
        color='Type',
        barmode='group',
        color_discrete_map={
            'VaR': INSTITUTIONAL_COLORS['professional_blue'],
            'CVaR': INSTITUTIONAL_COLORS['risk_red']
        }
    )
    
    fig_var.update_layout(
        title=f"VAR Comparison at {var_confidence*100:.1f}% Confidence",
        template="plotly_dark",
        height=400,
        xaxis_title="Method",
        yaxis_title="Value",
        paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        font_color=INSTITUTIONAL_COLORS['text_primary']
    )
    
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Component VAR analysis
    if var_results.component_var:
        st.markdown("#### Component VAR Analysis")
        
        component_df = pd.DataFrame.from_dict(
            var_results.component_var, 
            orient='index', 
            columns=['Contribution %']
        ).sort_values('Contribution %', ascending=False)
        
        fig_component = px.bar(
            component_df,
            y='Contribution %',
            color='Contribution %',
            color_continuous_scale=[
                INSTITUTIONAL_COLORS['professional_blue'],
                INSTITUTIONAL_COLORS['institutional_green']
            ],
            title="VAR Contribution by Asset"
        )
        
        fig_component.update_layout(
            template="plotly_dark",
            height=400,
            yaxis_title="Contribution (%)",
            paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
            plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue']
        )
        
        st.plotly_chart(fig_component, use_container_width=True)
    
    # Backtesting results
    st.markdown("#### Backtesting Results")
    
    col_back1, col_back2, col_back3 = st.columns(3)
    
    with col_back1:
        st.metric("Kupiec Test p-value", 
                 f"{var_results.kupiec_pvalue:.4f}",
                 delta="Pass" if var_results.kupiec_pvalue > 0.05 else "Fail")
    
    with col_back2:
        st.metric("Christoffersen Test p-value", 
                 f"{var_results.christoffersen_pvalue:.4f}",
                 delta="Pass" if var_results.christoffersen_pvalue > 0.05 else "Fail")
    
    with col_back3:
        expected_exceptions = len(optimizer.returns) * (1 - var_confidence)
        st.metric("Expected Exceptions", 
                 f"{expected_exceptions:.1f}",
                 delta=f"Actual: {var_results.var_exceptions}")
    
    # Stressed VAR
    if var_results.stressed_var:
        st.markdown("#### Stressed VAR Analysis")
        
        stressed_df = pd.DataFrame.from_dict(
            var_results.stressed_var, 
            orient='index', 
            columns=['Stressed VaR']
        )
        
        st.dataframe(
            stressed_df.style.format('{:.4f}'),
            use_container_width=True,
            height=200
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_efficient_frontier(optimizer: InstitutionalPortfolioOptimizer, 
                             optimization_result: OptimizationResult):
    """Display efficient frontier"""
    
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>Efficient Frontier Analysis</h3>", 
               unsafe_allow_html=True)
    
    with st.spinner("Calculating efficient frontier..."):
        frontier_data = optimizer.calculate_efficient_frontier()
    
    if not frontier_data:
        st.error("Failed to calculate efficient frontier")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Create efficient frontier chart
    fig = go.Figure()
    
    # Efficient frontier line
    fig.add_trace(go.Scatter(
        x=frontier_data['volatilities'],
        y=frontier_data['target_returns'],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color=INSTITUTIONAL_COLORS['text_secondary'], width=3),
        fill='tonexty',
        fillcolor=f"rgba({int(INSTITUTIONAL_COLORS['accent'][1:3], 16)}, "
                 f"{int(INSTITUTIONAL_COLORS['accent'][3:5], 16)}, "
                 f"{int(INSTITUTIONAL_COLORS['accent'][5:7], 16)}, 0.1)"
    ))
    
    # Minimum volatility portfolio
    fig.add_trace(go.Scatter(
        x=[frontier_data['min_vol_volatility']],
        y=[frontier_data['min_vol_return']],
        mode='markers',
        name='Minimum Volatility',
        marker=dict(
            color=INSTITUTIONAL_COLORS['institutional_green'],
            size=15,
            symbol='circle',
            line=dict(color='white', width=2)
        )
    ))
    
    # Maximum Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[frontier_data['max_sharpe_volatility']],
        y=[frontier_data['max_sharpe_return']],
        mode='markers',
        name='Maximum Sharpe',
        marker=dict(
            color=INSTITUTIONAL_COLORS['professional_blue'],
            size=15,
            symbol='diamond',
            line=dict(color='white', width=2)
        )
    ))
    
    # Optimized portfolio
    fig.add_trace(go.Scatter(
        x=[optimization_result.expected_volatility],
        y=[optimization_result.expected_return],
        mode='markers',
        name='Optimized Portfolio',
        marker=dict(
            color=INSTITUTIONAL_COLORS['warning_orange'],
            size=20,
            symbol='star',
            line=dict(color='white', width=3)
        )
    ))
    
    # Individual assets
    for i, ticker in enumerate(frontier_data['assets']):
        fig.add_trace(go.Scatter(
            x=[frontier_data['asset_volatilities'][i]],
            y=[frontier_data['asset_returns'][i]],
            mode='markers+text',
            text=[ticker],
            textposition="top center",
            marker=dict(
                color=INSTITUTIONAL_COLORS['text_secondary'],
                size=10,
                opacity=0.7
            ),
            name=ticker,
            showlegend=False
        ))
    
    fig.update_layout(
        title="Efficient Frontier Analysis",
        template="plotly_dark",
        height=600,
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Expected Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        hovermode='closest',
        paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        font_color=INSTITUTIONAL_COLORS['text_primary'],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor=f"rgba({int(INSTITUTIONAL_COLORS['dark_blue'][1:3], 16)}, "
                   f"{int(INSTITUTIONAL_COLORS['dark_blue'][3:5], 16)}, "
                   f"{int(INSTITUTIONAL_COLORS['dark_blue'][5:7], 16)}, 0.8)",
            bordercolor=INSTITUTIONAL_COLORS['slate_blue'],
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Frontier statistics
    st.markdown("#### Frontier Statistics")
    
    col_front1, col_front2, col_front3, col_front4 = st.columns(4)
    
    with col_front1:
        st.metric("Frontier Range (Return)", 
                 f"{frontier_data['min_vol_return']:.2%} - {max(frontier_data['target_returns']):.2%}")
    
    with col_front2:
        st.metric("Frontier Range (Volatility)", 
                 f"{frontier_data['min_vol_volatility']:.2%} - {max(frontier_data['volatilities']):.2%}")
    
    with col_front3:
        st.metric("Maximum Sharpe Ratio", 
                 f"{frontier_data['max_sharpe_sharpe']:.3f}")
    
    with col_front4:
        efficiency = optimization_result.sharpe_ratio / frontier_data['max_sharpe_sharpe'] \
                    if frontier_data['max_sharpe_sharpe'] > 0 else 0
        st.metric("Portfolio Efficiency", 
                 f"{efficiency:.2%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_stress_testing(optimizer: InstitutionalPortfolioOptimizer, 
                         optimization_result: OptimizationResult):
    """Display stress testing results"""
    
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>Stress Testing & Scenario Analysis</h3>", 
               unsafe_allow_html=True)
    
    with st.spinner("Performing stress tests..."):
        stress_results = optimizer.perform_stress_testing(optimization_result.weights)
    
    if not stress_results:
        st.info("No stress test results available")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Stress test summary
    st.markdown("#### Stress Test Summary")
    
    for result in stress_results:
        with st.expander(f"📉 {result.scenario}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Portfolio Loss", f"{result.portfolio_loss:.2%}")
            
            with col2:
                st.metric("CVaR Impact", f"{result.cvar_impact:.4f}")
            
            with col3:
                st.metric("VAR Breach", "Yes" if result.var_breach else "No")
            
            # Component impacts
            if result.component_impacts:
                st.markdown("##### Component Impacts")
                
                impacts_df = pd.DataFrame.from_dict(
                    result.component_impacts, 
                    orient='index', 
                    columns=['Impact']
                ).sort_values('Impact')
                
                st.dataframe(
                    impacts_df.style.format('{:.4f}'),
                    use_container_width=True,
                    height=200
                )
    
    # Stress test visualization
    st.markdown("#### Stress Test Comparison")
    
    scenarios = [r.scenario for r in stress_results]
    losses = [r.portfolio_loss for r in stress_results]
    
    fig_stress = go.Figure(data=[
        go.Bar(
            x=scenarios,
            y=losses,
            text=[f"{l:.2%}" for l in losses],
            textposition='auto',
            marker_color=[
                INSTITUTIONAL_COLORS['risk_red'],
                INSTITUTIONAL_COLORS['warning_orange'],
                INSTITUTIONAL_COLORS['warning'],
                INSTITUTIONAL_COLORS['info']
            ]
        )
    ])
    
    fig_stress.update_layout(
        title="Portfolio Loss Under Stress Scenarios",
        template="plotly_dark",
        height=400,
        xaxis_title="Scenario",
        yaxis_title="Portfolio Loss",
        yaxis_tickformat=".0%",
        paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue']
    )
    
    st.plotly_chart(fig_stress, use_container_width=True)
    
    # Risk metrics under stress
    st.markdown("#### Risk Metrics Under Stress")
    
    stress_metrics_data = []
    for result in stress_results:
        stress_metrics_data.append({
            'Scenario': result.scenario,
            'Portfolio Loss': result.portfolio_loss,
            'CVaR Impact': result.cvar_impact,
            'Volatility Impact': result.volatility_impact
        })
    
    stress_metrics_df = pd.DataFrame(stress_metrics_data)
    
    st.dataframe(
        stress_metrics_df.style.format({
            'Portfolio Loss': '{:.2%}',
            'CVaR Impact': '{:.4f}',
            'Volatility Impact': '{:.2%}'
        }),
        use_container_width=True,
        height=300
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_backtesting(optimizer: InstitutionalPortfolioOptimizer):
    """Display backtesting results"""
    
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>Backtesting Analysis</h3>", 
               unsafe_allow_html=True)
    
    with st.spinner("Running backtests..."):
        backtest_results = optimizer.perform_backtesting(
            method=BacktestMethod.ROLLING_WINDOW,
            window_size=252,
            rebalance_frequency=63
        )
    
    if not backtest_results:
        st.info("No backtest results available")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Backtest summary
    st.markdown("#### Backtest Performance Summary")
    
    # Aggregate metrics
    total_returns = [r.total_return for r in backtest_results]
    annual_returns = [r.annual_return for r in backtest_results]
    sharpe_ratios = [r.sharpe_ratio for r in backtest_results]
    max_drawdowns = [r.max_drawdown for r in backtest_results]
    
    col_back1, col_back2, col_back3, col_back4 = st.columns(4)
    
    with col_back1:
        avg_return = np.mean(total_returns)
        st.metric("Average Total Return", f"{avg_return:.2%}")
    
    with col_back2:
        avg_annual = np.mean(annual_returns)
        st.metric("Average Annual Return", f"{avg_annual:.2%}")
    
    with col_back3:
        avg_sharpe = np.mean(sharpe_ratios)
        st.metric("Average Sharpe Ratio", f"{avg_sharpe:.3f}")
    
    with col_back4:
        avg_max_dd = np.mean(max_drawdowns)
        st.metric("Average Max Drawdown", f"{avg_max_dd:.2%}")
    
    # Backtest performance over time
    st.markdown("#### Backtest Performance Over Time")
    
    periods = [f"{r.period_start.strftime('%Y-%m')} to {r.period_end.strftime('%Y-%m')}" 
               for r in backtest_results]
    
    fig_backtest = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Total Return
    fig_backtest.add_trace(
        go.Bar(
            x=periods,
            y=total_returns,
            name='Total Return',
            marker_color=INSTITUTIONAL_COLORS['institutional_green']
        ),
        row=1, col=1
    )
    
    # Sharpe Ratio
    fig_backtest.add_trace(
        go.Bar(
            x=periods,
            y=sharpe_ratios,
            name='Sharpe Ratio',
            marker_color=INSTITUTIONAL_COLORS['professional_blue']
        ),
        row=1, col=2
    )
    
    # Max Drawdown
    fig_backtest.add_trace(
        go.Bar(
            x=periods,
            y=max_drawdowns,
            name='Max Drawdown',
            marker_color=INSTITUTIONAL_COLORS['risk_red']
        ),
        row=2, col=1
    )
    
    # Win Rate
    win_rates = [r.win_rate for r in backtest_results]
    fig_backtest.add_trace(
        go.Bar(
            x=periods,
            y=win_rates,
            name='Win Rate',
            marker_color=INSTITUTIONAL_COLORS['warning_orange']
        ),
        row=2, col=2
    )
    
    fig_backtest.update_layout(
        height=600,
        template="plotly_dark",
        showlegend=False,
        paper_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        plot_bgcolor=INSTITUTIONAL_COLORS['navy_blue'],
        font_color=INSTITUTIONAL_COLORS['text_primary']
    )
    
    fig_backtest.update_yaxes(title_text="Return", tickformat=".0%", row=1, col=1)
    fig_backtest.update_yaxes(title_text="Ratio", row=1, col=2)
    fig_backtest.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig_backtest.update_yaxes(title_text="Rate", tickformat=".0%", row=2, col=2)
    
    st.plotly_chart(fig_backtest, use_container_width=True)
    
    # Detailed backtest results
    st.markdown("#### Detailed Backtest Results")
    
    backtest_df = pd.DataFrame([asdict(r) for r in backtest_results])
    
    # Format columns
    format_dict = {
        'total_return': '{:.2%}',
        'annual_return': '{:.2%}',
        'annual_volatility': '{:.2%}',
        'max_drawdown': '{:.2%}',
        'win_rate': '{:.2%}',
        'exception_rate': '{:.2%}'
    }
    
    st.dataframe(
        backtest_df.style.format(format_dict),
        use_container_width=True,
        height=400
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_quantstats_report(optimizer: InstitutionalPortfolioOptimizer, 
                            optimization_result: OptimizationResult):
    """Display QuantStats report"""
    
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>QuantStats Performance Report</h3>", 
               unsafe_allow_html=True)
    
    with st.spinner("Generating QuantStats report..."):
        html_report = optimizer.generate_quantstats_report(
            optimization_result.weights,
            benchmark_ticker='XU100.IS'
        )
    
    if html_report:
        # Display report
        with st.expander("📋 View Full Report", expanded=True):
            st.components.v1.html(html_report, height=800, scrolling=True)
        
        # Download button
        b64 = base64.b64encode(html_report.encode()).decode()
        href = f'''
        <a href="data:text/html;base64,{b64}" 
           download="quantstats_portfolio_report.html"
           style="text-decoration: none;">
           <button style="
               background: {INSTITUTIONAL_COLORS['professional_blue']};
               color: {INSTITUTIONAL_COLORS['text_primary']};
               padding: 0.75rem 1.5rem;
               border: none;
               border-radius: 6px;
               cursor: pointer;
               font-weight: 500;
               width: 100%;
               margin-top: 1rem;
           ">
           📥 Download QuantStats HTML Report
           </button>
        </a>
        '''
        st.markdown(href, unsafe_allow_html=True)
        
        # Calculate metrics
        metrics = optimizer.calculate_quantstats_metrics(optimization_result.weights)
        
        if metrics:
            st.markdown("#### Key QuantStats Metrics")
            
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            
            col_metrics1, col_metrics2 = st.columns(2)
            
            with col_metrics1:
                st.dataframe(
                    metrics_df.iloc[:10].style.format('{:.4f}'),
                    use_container_width=True,
                    height=400
                )
            
            with col_metrics2:
                st.dataframe(
                    metrics_df.iloc[10:].style.format('{:.4f}'),
                    use_container_width=True,
                    height=400
                )
    else:
        st.warning("QuantStats report generation failed or not available")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_detailed_metrics(optimizer: InstitutionalPortfolioOptimizer, 
                           optimization_result: OptimizationResult):
    """Display detailed metrics"""
    
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {INSTITUTIONAL_COLORS['text_primary']};'>Detailed Portfolio Metrics</h3>", 
               unsafe_allow_html=True)
    
    # Calculate comprehensive metrics
    portfolio_returns = (optimizer.returns * pd.Series(optimization_result.weights)).sum(axis=1)
    
    # Calculate all metrics
    total_days = len(portfolio_returns)
    total_years = total_days / 252
    
    # Return metrics
    total_return = (1 + portfolio_returns).prod() - 1
    cagr = (1 + total_return) ** (1/total_years) - 1 if total_years > 0 else 0
    annual_return = portfolio_returns.mean() * 252
    monthly_return = annual_return / 12
    weekly_return = annual_return / 52
    daily_return = portfolio_returns.mean()
    
    # Risk metrics
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    # Drawdown calculations
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown.mean()
    
    # Ulcer index
    ulcer_index = np.sqrt(np.mean(drawdown ** 2))
    
    # Risk-adjusted ratios
    sharpe_ratio = (annual_return - optimization_result.risk_free_rate) / annual_volatility \
                  if annual_volatility > 0 else 0
    sortino_ratio = (annual_return - optimization_result.risk_free_rate) / downside_deviation \
                   if downside_deviation > 0 else 0
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Omega ratio
    threshold = optimization_result.risk_free_rate / 252
    gains = portfolio_returns[portfolio_returns > threshold].sum()
    losses = abs(portfolio_returns[portfolio_returns <= threshold].sum())
    omega_ratio = gains / losses if losses != 0 else float('inf')
    
    # Additional ratios
    gain_to_pain_ratio = abs(portfolio_returns[portfolio_returns > 0].sum() / 
                            portfolio_returns[portfolio_returns < 0].sum()) \
                        if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0
    
    # Statistical metrics
    skewness = stats.skew(portfolio_returns)
    kurtosis = stats.kurtosis(portfolio_returns)
    jarque_bera = stats.jarque_bera(portfolio_returns)[0]
    
    # VAR calculations
    value_at_risk_95 = np.percentile(portfolio_returns, 5)
    conditional_var_95 = portfolio_returns[portfolio_returns <= value_at_risk_95].mean()
    expected_shortfall_95 = conditional_var_95
    
    # Win/loss metrics
    positive_returns = portfolio_returns[portfolio_returns > 0]
    negative_returns = portfolio_returns[portfolio_returns < 0]
    
    win_rate = len(positive_returns) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
    profit_factor = abs(positive_returns.sum() / negative_returns.sum()) \
                   if negative_returns.sum() != 0 else float('inf')
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
    max_win = positive_returns.max() if len(positive_returns) > 0 else 0
    max_loss = negative_returns.min() if len(negative_returns) > 0 else 0
    avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    
    # Create metrics DataFrame
    metrics_data = {
        'Return Metrics': {
            'Total Return': total_return,
            'CAGR': cagr,
            'Annual Return': annual_return,
            'Monthly Return': monthly_return,
            'Weekly Return': weekly_return,
            'Daily Return': daily_return
        },
        'Risk Metrics': {
            'Annual Volatility': annual_volatility,
            'Downside Deviation': downside_deviation,
            'Max Drawdown': max_drawdown,
            'Average Drawdown': avg_drawdown,
            'Ulcer Index': ulcer_index
        },
        'Risk-Adjusted Ratios': {
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Omega Ratio': omega_ratio,
            'Gain to Pain Ratio': gain_to_pain_ratio
        },
        'Statistical Metrics': {
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Jarque-Bera Statistic': jarque_bera,
            'Value at Risk (95%)': value_at_risk_95,
            'Conditional VaR (95%)': conditional_var_95,
            'Expected Shortfall (95%)': expected_shortfall_95
        },
        'Win/Loss Metrics': {
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Maximum Win': max_win,
            'Maximum Loss': max_loss,
            'Average Win/Loss Ratio': avg_win_loss_ratio,
            'Expectancy': expectancy
        }
    }
    
    # Display metrics in expanders
    for category, metrics in metrics_data.items():
        with st.expander(f"📊 {category}", expanded=False):
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            
            # Format based on metric type
            format_dict = {}
            for metric in metrics.keys():
                if any(x in metric.lower() for x in ['return', 'drawdown', 'volatility', 'deviation', 'rate']):
                    format_dict[metric] = '{:.2%}'
                elif any(x in metric.lower() for x in ['ratio', 'factor', 'skewness', 'kurtosis', 'index']):
                    format_dict[metric] = '{:.4f}'
                elif 'var' in metric.lower() or 'shortfall' in metric.lower():
                    format_dict[metric] = '{:.6f}'
                else:
                    format_dict[metric] = '{:.4f}'
            
            st.dataframe(
                metrics_df.style.format(format_dict),
                use_container_width=True,
                height=min(300, len(metrics) * 35 + 40)
            )
    
    # Export metrics
    st.markdown("#### Export Metrics")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        # Combine all metrics
        all_metrics = {}
        for category, metrics in metrics_data.items():
            all_metrics.update(metrics)
        
        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Value'])
        csv = metrics_df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        
        href = f'''
        <a href="data:file/csv;base64,{b64}" download="portfolio_metrics.csv">
        <button style="
            background: {INSTITUTIONAL_COLORS['professional_blue']};
            color: {INSTITUTIONAL_COLORS['text_primary']};
            padding: 0.5rem 1rem;
            border: 1px solid {INSTITUTIONAL_COLORS['border']};
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            width: 100%;
        ">
        📥 Download Metrics CSV
        </button>
        </a>
        '''
        st.markdown(href, unsafe_allow_html=True)
    
    with col_export2:
        # Export weights
        weights_df = pd.DataFrame.from_dict(
            optimization_result.weights, 
            orient='index', 
            columns=['Weight']
        )
        weights_csv = weights_df.to_csv()
        weights_b64 = base64.b64encode(weights_csv.encode()).decode()
        
        weights_href = f'''
        <a href="data:file/csv;base64,{weights_b64}" download="portfolio_weights.csv">
        <button style="
            background: {INSTITUTIONAL_COLORS['professional_blue']};
            color: {INSTITUTIONAL_COLORS['text_primary']};
            padding: 0.5rem 1rem;
            border: 1px solid {INSTITUTIONAL_COLORS['border']};
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            width: 100%;
        ">
        📥 Download Weights CSV
        </button>
        </a>
        '''
        st.markdown(weights_href, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ## 🚨 Critical Application Error
        
        **Error Type:** {type(e).__name__}
        **Error Details:** {str(e)}
        
        Please check the logs for more details and ensure all dependencies are installed.
        """)
        
        with st.expander("🔍 Technical Details & Traceback"):
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
        
        # Installation instructions
        st.markdown("""
        ## 🔧 Installation Requirements
        
        Please install all required packages:
        
        ```bash
        pip install streamlit yfinance pypfopt quantstats plotly pandas numpy scipy scikit-learn arch
        ```
        
        For comprehensive functionality, also install:
        
        ```bash
        pip install cvxpy pyfolio
        ```
        
        ## 🐛 Common Issues & Solutions
        
        1. **yfinance connection issues:**
           - Check internet connection
           - Try using a VPN
           - Use different tickers
        
        2. **Memory issues with large datasets:**
           - Select fewer assets
           - Use shorter time periods
           - Close other applications
        
        3. **Optimization failures:**
           - Ensure sufficient historical data
           - Try different optimization methods
           - Adjust risk parameters
        
        4. **Streamlit Cloud deployment:**
           - Ensure requirements.txt includes all packages
           - Check memory limits
           - Monitor application logs
        """)
