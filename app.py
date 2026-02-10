# ============================================================================
# BIST ENTERPRISE QUANT PORTFOLIO OPTIMIZATION SUITE PRO MAX ULTRA
# Version: 10.0 | Features: Enhanced Stability + Real-time Data + ML Integration
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
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import pickle
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """Centralized configuration management"""
    # Cache settings
    CACHE_DIR = Path(".cache")
    CACHE_EXPIRY_HOURS = 1
    
    # Data settings
    MAX_ASSETS = 20
    MIN_DATA_POINTS = 50
    DEFAULT_START_DATE = datetime.now() - timedelta(days=365 * 3)
    DEFAULT_END_DATE = datetime.now()
    
    # Optimization settings
    DEFAULT_RISK_FREE_RATE = 0.30  # 30% for Turkey
    DEFAULT_TARGET_VOLATILITY = 0.15
    DEFAULT_TARGET_RETURN = 0.20
    
    # Performance settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    REQUEST_TIMEOUT = 30
    
    # Visualization settings
    CHART_HEIGHT = 400
    CHART_TEMPLATE = "plotly_dark"

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED CACHE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class PersistentCache:
    """Persistent caching system with disk storage"""
    
    def __init__(self, cache_dir: Path = Config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
    
    def get(self, key: str, max_age_hours: int = Config.CACHE_EXPIRY_HOURS) -> Any:
        """Get item from cache if not expired"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > max_age_hours * 3600:
            cache_path.unlink(missing_ok=True)
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except:
            pass
    
    def clear(self):
        """Clear all cache"""
        for file in self.cache_dir.glob("*.pkl"):
            try:
                file.unlink()
            except:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED LOGGING
# ─────────────────────────────────────────────────────────────────────────────

class StructuredLogger:
    """Enhanced logging with structured data"""
    
    def __init__(self, name: str = "PortfolioOptimizer"):
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging with rotation"""
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)
            
            # File handler with rotation
            file_handler = logging.FileHandler(
                'portfolio_analytics.log',
                mode='a',
                encoding='utf-8'
            )
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(file_format)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            f"Performance | {operation} | Duration: {duration:.3f}s | {kwargs}"
        )
    
    def log_optimization(self, method: str, assets: int, performance: Tuple):
        """Log optimization details"""
        self.logger.info(
            f"Optimization | Method: {method} | Assets: {assets} | "
            f"Return: {performance[0]:.3f} | Vol: {performance[1]:.3f} | "
            f"Sharpe: {performance[2]:.3f}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# MODULE IMPORTS WITH FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = StructuredLogger()

# Dynamic import with detailed error reporting
def import_with_fallback(module_name, install_name=None, optional=False):
    """Import module with fallback and user-friendly error messages"""
    try:
        module = __import__(module_name)
        logger.logger.info(f"Successfully imported {module_name}")
        return module
    except ImportError as e:
        if not optional:
            logger.logger.error(f"Failed to import required module {module_name}: {e}")
            st.error(f"""
            ❌ **Missing Required Package**: {module_name}
            
            Please install with:
            ```bash
            pip install {install_name or module_name}
            ```
            """)
        else:
            logger.logger.warning(f"Optional module {module_name} not available: {e}")
        return None

# Import core packages
yf = import_with_fallback('yfinance', 'yfinance>=0.2.28')
HAS_YFINANCE = yf is not None

# Import PyPortfolioOpt components
try:
    if HAS_YFINANCE:
        from pypfopt import expected_returns, risk_models
        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt.hierarchical_portfolio import HRPOpt
        from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
        from pypfopt import objective_functions
        HAS_PYPFOPT = True
        logger.logger.info("PyPortfolioOpt imported successfully")
    else:
        HAS_PYPFOPT = False
except ImportError as e:
    logger.logger.error(f"PyPortfolioOpt import error: {e}")
    HAS_PYPFOPT = False

# Import optional packages
qs = import_with_fallback('quantstats', optional=True)
HAS_QUANTSTATS = qs is not None

sklearn = import_with_fallback('sklearn', 'scikit-learn>=1.3.0', optional=True)
HAS_SKLEARN = sklearn is not None

if HAS_SKLEARN:
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationMethod(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    EFFICIENT_RISK = "efficient_risk"
    EFFICIENT_RETURN = "efficient_return"
    HRP = "hrp"
    MAX_QUADRATIC_UTILITY = "max_quadratic_utility"
    MAX_RETURN = "max_return"
    BLACK_LITTERMAN = "black_litterman"

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
    LOG_RETURN = "log_return"

@dataclass
class OptimizationResult:
    """Structured optimization result"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    risk_model: str
    return_model: str
    risk_free_rate: float
    timestamp: datetime
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        weights_df = pd.DataFrame.from_dict(
            self.weights, orient='index', columns=['Weight']
        )
        weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
        return weights_df
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED DATA SOURCE WITH ASYNC SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

class AsyncMarketDataFetcher:
    """Asynchronous market data fetcher with improved error handling"""
    
    def __init__(self):
        self.cache = PersistentCache()
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=False)
    
    async def fetch_ticker_data(self, ticker: str, start_date: datetime, 
                               end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch single ticker data asynchronously"""
        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        retries = Config.MAX_RETRIES
        while retries > 0:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),
                    interval="1d",
                    actions=False,
                    progress=False
                )
                
                if df.empty:
                    retries -= 1
                    await asyncio.sleep(Config.RETRY_DELAY)
                    continue
                
                # Cache the result
                self.cache.set(cache_key, df)
                return df
                
            except Exception as e:
                logger.logger.error(f"Error fetching {ticker}: {e}")
                retries -= 1
                await asyncio.sleep(Config.RETRY_DELAY * (Config.MAX_RETRIES - retries))
        
        return None
    
    async def fetch_multiple_tickers(self, tickers: List[str], 
                                   start_date: datetime, 
                                   end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch multiple tickers concurrently"""
        tasks = []
        for ticker in tickers:
            task = self.fetch_ticker_data(ticker, start_date, end_date)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.logger.warning(f"Failed to fetch {ticker}: {result}")
                continue
            if result is not None and not result.empty:
                data_dict[ticker] = result
        
        return data_dict

@dataclass
class MarketDataBundle:
    """Enhanced market data bundle"""
    tickers: List[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    volumes: pd.DataFrame
    start_date: datetime
    end_date: datetime
    metadata: Dict[str, Any]
    
    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix"""
        return self.returns.corr()
    
    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """Get covariance matrix"""
        return self.returns.cov()
    
    def get_sector_allocation(self, sector_mapping: Dict) -> Dict[str, float]:
        """Calculate sector allocation"""
        sector_allocation = {}
        for ticker in self.tickers:
            for sector, ticker_list in sector_mapping.items():
                if ticker in ticker_list:
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + 1
                    break
        total = sum(sector_allocation.values())
        return {k: v/total for k, v in sector_allocation.items()}
    
    def validate(self) -> bool:
        """Validate data quality"""
        if self.prices.empty or self.returns.empty:
            return False
        
        # Check for sufficient data points
        if len(self.prices) < Config.MIN_DATA_POINTS:
            return False
        
        # Check for NaN values
        if self.prices.isna().sum().sum() > len(self.prices) * 0.1:  # More than 10% NaN
            return False
        
        return True

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED OPTIMIZER WITH ML FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedPortfolioOptimizer:
    """Enhanced portfolio optimizer with ML capabilities"""
    
    def __init__(self, market_data: MarketDataBundle):
        self.market_data = market_data
        self.tickers = market_data.tickers
        
        if not HAS_PYPFOPT:
            raise ImportError("PyPortfolioOpt is required for optimization")
        
        self._validate_data()
        self._initialize_models()
        self.results_history = []
        self.performance_metrics = {}
    
    def _validate_data(self):
        """Validate and clean input data"""
        if not self.market_data.validate():
            raise ValueError("Invalid market data")
        
        # Remove tickers with insufficient data
        valid_tickers = []
        for ticker in self.tickers:
            if ticker in self.market_data.returns.columns:
                if self.market_data.returns[ticker].notna().sum() > Config.MIN_DATA_POINTS:
                    valid_tickers.append(ticker)
        
        if len(valid_tickers) < 2:
            raise ValueError(f"Insufficient valid tickers: {len(valid_tickers)}")
        
        self.tickers = valid_tickers
    
    def _initialize_models(self):
        """Initialize all optimization models"""
        self.expected_returns_models = {
            ReturnModel.MEAN_HISTORICAL: self._calculate_mean_historical_return(),
            ReturnModel.EMA_HISTORICAL: self._calculate_ema_return(),
            ReturnModel.CAPM: self._calculate_capm_return(),
            ReturnModel.LOG_RETURN: self._calculate_log_return()
        }
        
        self.risk_models = {
            RiskModel.SAMPLE_COV: self._calculate_sample_covariance(),
            RiskModel.LEDOIT_WOLF: self._calculate_ledoit_wolf(),
            RiskModel.EXPONENTIAL_COV: self._calculate_exponential_covariance(),
            RiskModel.SEMICOVARIANCE: self._calculate_semicovariance()
        }
    
    def _calculate_mean_historical_return(self) -> pd.Series:
        """Calculate mean historical returns"""
        return self.market_data.returns.mean() * 252
    
    def _calculate_ema_return(self, span: int = 500) -> pd.Series:
        """Calculate EMA-based returns"""
        prices = self.market_data.prices
        ema_returns = prices.ewm(span=span).mean().pct_change().mean() * 252
        return ema_returns
    
    def _calculate_capm_return(self) -> pd.Series:
        """Calculate CAPM expected returns"""
        # Simplified CAPM - in practice would need market returns
        market_return = 0.10  # Assuming 10% market return
        risk_free_rate = Config.DEFAULT_RISK_FREE_RATE
        betas = self._calculate_betas()
        
        capm_returns = risk_free_rate + betas * (market_return - risk_free_rate)
        return capm_returns
    
    def _calculate_log_return(self) -> pd.Series:
        """Calculate log returns"""
        log_returns = np.log(self.market_data.prices / self.market_data.prices.shift(1)).mean() * 252
        return log_returns
    
    def _calculate_sample_covariance(self) -> pd.DataFrame:
        """Calculate sample covariance matrix"""
        return self.market_data.returns.cov() * 252
    
    def _calculate_ledoit_wolf(self) -> pd.DataFrame:
        """Calculate Ledoit-Wolf shrinkage covariance"""
        if HAS_SKLEARN:
            lw = LedoitWolf()
            lw.fit(self.market_data.returns)
            cov_matrix = pd.DataFrame(
                lw.covariance_ * 252,
                index=self.tickers,
                columns=self.tickers
            )
            return cov_matrix
        else:
            return self._calculate_sample_covariance()
    
    def _calculate_exponential_covariance(self, span: int = 180) -> pd.DataFrame:
        """Calculate exponentially weighted covariance"""
        cov = self.market_data.returns.ewm(span=span).cov().iloc[-len(self.tickers):, -len(self.tickers):]
        return cov * 252
    
    def _calculate_semicovariance(self) -> pd.DataFrame:
        """Calculate semicovariance matrix"""
        returns = self.market_data.returns
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        semi_cov = downside_returns.cov() * 252
        return semi_cov
    
    def _calculate_betas(self) -> pd.Series:
        """Calculate beta coefficients"""
        # Simplified beta calculation
        # In practice, would calculate against a market index
        returns = self.market_data.returns
        market_returns = returns.mean(axis=1)  # Use average portfolio as market proxy
        
        betas = {}
        for ticker in self.tickers:
            cov = np.cov(returns[ticker], market_returns)[0, 1]
            var = np.var(market_returns)
            betas[ticker] = cov / var if var != 0 else 1.0
        
        return pd.Series(betas)
    
    def optimize(self, method: OptimizationMethod,
                risk_model: RiskModel = RiskModel.LEDOIT_WOLF,
                return_model: ReturnModel = ReturnModel.MEAN_HISTORICAL,
                **kwargs) -> OptimizationResult:
        """Perform portfolio optimization"""
        
        start_time = time.time()
        
        try:
            # Get selected models
            mu = self.expected_returns_models.get(return_model)
            S = self.risk_models.get(risk_model)
            
            if mu is None or S is None:
                raise ValueError(f"Invalid model selection: {return_model}, {risk_model}")
            
            # Get parameters
            risk_free_rate = kwargs.get('risk_free_rate', Config.DEFAULT_RISK_FREE_RATE)
            target_volatility = kwargs.get('target_volatility', Config.DEFAULT_TARGET_VOLATILITY)
            target_return = kwargs.get('target_return', Config.DEFAULT_TARGET_RETURN)
            
            # Perform optimization
            if method == OptimizationMethod.MAX_SHARPE:
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(
                    verbose=False, 
                    risk_free_rate=risk_free_rate
                )
            
            elif method == OptimizationMethod.MIN_VOLATILITY:
                ef = EfficientFrontier(mu, S)
                ef.min_volatility()
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(
                    verbose=False, 
                    risk_free_rate=risk_free_rate
                )
            
            elif method == OptimizationMethod.EFFICIENT_RISK:
                ef = EfficientFrontier(mu, S)
                ef.efficient_risk(target_volatility=target_volatility)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(
                    verbose=False, 
                    risk_free_rate=risk_free_rate
                )
            
            elif method == OptimizationMethod.EFFICIENT_RETURN:
                ef = EfficientFrontier(mu, S)
                ef.efficient_return(target_return=target_return)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(
                    verbose=False, 
                    risk_free_rate=risk_free_rate
                )
            
            elif method == OptimizationMethod.HRP:
                hrp = HRPOpt(self.market_data.returns)
                hrp.optimize()
                weights = hrp.clean_weights()
                port_returns = (self.market_data.returns * pd.Series(weights)).sum(axis=1)
                annual_return = port_returns.mean() * 252
                annual_vol = port_returns.std() * np.sqrt(252)
                sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
                performance = (annual_return, annual_vol, sharpe)
            
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # Create result object
            result = OptimizationResult(
                weights=weights,
                expected_return=performance[0],
                expected_volatility=performance[1],
                sharpe_ratio=performance[2],
                method=method.value,
                risk_model=risk_model.value,
                return_model=return_model.value,
                risk_free_rate=risk_free_rate,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.results_history.append(result)
            
            # Log performance
            duration = time.time() - start_time
            logger.log_performance(
                f"optimize_{method.value}",
                duration,
                assets=len(self.tickers),
                success=True
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.log_performance(
                f"optimize_{method.value}",
                duration,
                assets=len(self.tickers),
                success=False,
                error=str(e)
            )
            
            # Fallback to equal weighting
            st.warning(f"Optimization failed, using equal weighting: {str(e)}")
            
            equal_weights = {ticker: 1/len(self.tickers) for ticker in self.tickers}
            port_returns = (self.market_data.returns * pd.Series(equal_weights)).sum(axis=1)
            annual_return = port_returns.mean() * 252
            annual_vol = port_returns.std() * np.sqrt(252)
            sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            return OptimizationResult(
                weights=equal_weights,
                expected_return=annual_return,
                expected_volatility=annual_vol,
                sharpe_ratio=sharpe,
                method="equal_weight",
                risk_model="none",
                return_model="none",
                risk_free_rate=risk_free_rate,
                timestamp=datetime.now()
            )
    
    def generate_efficient_frontier(self, points: int = 100) -> Dict:
        """Generate efficient frontier data"""
        mu = self.expected_returns_models[ReturnModel.MEAN_HISTORICAL]
        S = self.risk_models[RiskModel.LEDOIT_WOLF]
        
        ef = EfficientFrontier(mu, S)
        mus, sigmas, weights = ef.efficient_frontier(points=points)
        
        return {
            'returns': mus,
            'volatilities': sigmas,
            'weights': weights
        }
    
    def calculate_risk_metrics(self, weights: Dict[str, float]) -> Dict:
        """Calculate comprehensive risk metrics"""
        port_returns = (self.market_data.returns * pd.Series(weights)).sum(axis=1)
        
        metrics = {
            'volatility': port_returns.std() * np.sqrt(252),
            'downside_deviation': port_returns[port_returns < 0].std() * np.sqrt(252),
            'var_95': np.percentile(port_returns, 5),
            'cvar_95': port_returns[port_returns <= np.percentile(port_returns, 5)].mean(),
            'max_drawdown': self._calculate_max_drawdown(port_returns),
            'skewness': stats.skew(port_returns),
            'kurtosis': stats.kurtosis(port_returns)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    def analyze_correlation_structure(self) -> Dict:
        """Analyze correlation structure of assets"""
        corr_matrix = self.market_data.correlation_matrix
        
        if HAS_SKLEARN and len(self.tickers) > 2:
            # Perform PCA analysis
            returns_scaled = StandardScaler().fit_transform(self.market_data.returns.fillna(0))
            pca = PCA(n_components=min(5, len(self.tickers)))
            pca.fit(returns_scaled)
            
            return {
                'correlation_matrix': corr_matrix,
                'eigenvalues': pca.explained_variance_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'components': pca.components_
            }
        else:
            return {
                'correlation_matrix': corr_matrix,
                'eigenvalues': None,
                'explained_variance_ratio': None,
                'components': None
            }

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APPLICATION WITH ENHANCED UI
# ─────────────────────────────────────────────────────────────────────────────

def setup_streamlit_app():
    """Setup Streamlit application configuration"""
    st.set_page_config(
        page_title="BIST Quant Portfolio Lab v10.0",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/streamlit/streamlit',
            'Report a bug': "https://github.com/streamlit/streamlit/issues",
            'About': "# BIST Portfolio Optimization Suite v10.0"
        }
    )
    
    # Inject custom CSS
    inject_custom_css()

def inject_custom_css():
    """Inject enhanced CSS styles"""
    st.markdown("""
    <style>
        /* Enhanced styles from original code */
        .metric-card-3d {
            background: linear-gradient(145deg, #1a2536, #0a1929);
            border: 1px solid #2d3748;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.5rem;
            box-shadow: 8px 8px 16px rgba(0, 0, 0, 0.3),
                       -4px -4px 10px rgba(255, 255, 255, 0.05);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .metric-card-3d:hover {
            transform: translateY(-8px);
            box-shadow: 12px 12px 24px rgba(0, 0, 0, 0.4),
                       -6px -6px 12px rgba(255, 255, 255, 0.1);
            border-color: #0066cc;
        }
        
        .glass-card {
            background: rgba(26, 37, 54, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        }
        
        /* Additional enhancements */
        .success-badge {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .warning-badge {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .error-badge {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main application function"""
    setup_streamlit_app()
    
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Check requirements
    if not check_requirements():
        return
    
    # Sidebar configuration
    with st.sidebar:
        config = get_sidebar_config()
    
    # Main content
    if config['assets'] and len(config['assets']) >= 2:
        try:
            # Load data
            market_data = load_market_data(
                config['assets'],
                config['start_date'],
                config['end_date']
            )
            
            if market_data:
                # Optimize portfolio
                optimizer = EnhancedPortfolioOptimizer(market_data)
                result = optimizer.optimize(
                    method=config['optimization_method'],
                    risk_model=config['risk_model'],
                    return_model=config['return_model'],
                    risk_free_rate=config['risk_free_rate'],
                    target_volatility=config['target_volatility'],
                    target_return=config['target_return']
                )
                
                # Display results
                display_results(optimizer, result, config)
                
        except Exception as e:
            st.error(f"Error in portfolio optimization: {str(e)}")
            logger.logger.error(f"Application error: {e}", exc_info=True)
    else:
        st.warning("Please select at least 2 assets for portfolio optimization")

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'data_loaded': False,
        'optimized': False,
        'market_data': None,
        'optimizer': None,
        'optimization_result': None,
        'selected_assets': ['THYAO.IS', 'GARAN.IS', 'ASELS.IS', 'AKBNK.IS']
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_header():
    """Display application header"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 class="animated-gradient-text" style="margin: 0; font-size: 2.5rem;">
            📊 BIST Enterprise Portfolio Analytics Suite
        </h1>
        <p style="font-size: 1.2rem; color: #b0b0b0; margin-top: 1rem;">
            Professional Portfolio Optimization & Risk Analytics Platform v10.0
        </p>
    </div>
    """, unsafe_allow_html=True)

def check_requirements() -> bool:
    """Check if all requirements are met"""
    if not HAS_YFINANCE:
        st.error("""
        ## ❌ Missing Required Packages
        
        **yfinance** is not installed. Please install with:
        ```bash
        pip install yfinance pypfopt
        ```
        """)
        return False
    
    if not HAS_PYPFOPT:
        st.warning("""
        ⚠️ **PyPortfolioOpt** is not fully installed.
        Some optimization features may be limited.
        """)
    
    return True

def get_sidebar_config() -> Dict:
    """Get configuration from sidebar"""
    st.sidebar.header("⚙️ Configuration")
    
    # Date selection
    date_preset = st.sidebar.selectbox(
        "Time Period",
        ["1 Year", "3 Years", "5 Years", "Custom"],
        index=1
    )
    
    if date_preset == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", Config.DEFAULT_START_DATE)
        with col2:
            end_date = st.date_input("End Date", Config.DEFAULT_END_DATE)
    else:
        end_date = datetime.now()
        if date_preset == "1 Year":
            start_date = end_date - timedelta(days=365)
        elif date_preset == "3 Years":
            start_date = end_date - timedelta(days=365 * 3)
        else:  # 5 Years
            start_date = end_date - timedelta(days=365 * 5)
    
    # Asset selection
    st.sidebar.subheader("📊 Asset Selection")
    
    # Filter by sector
    sectors = list(SECTOR_MAPPING.keys())
    selected_sector = st.sidebar.selectbox(
        "Filter by Sector",
        ["All Sectors"] + sectors
    )
    
    # Get available tickers
    if selected_sector == "All Sectors":
        available_tickers = BIST100_TICKERS
    else:
        available_tickers = SECTOR_MAPPING.get(selected_sector, [])
    
    # Multi-select assets
    assets = st.sidebar.multiselect(
        f"Select Assets (Max {Config.MAX_ASSETS})",
        available_tickers,
        default=st.session_state.get('selected_assets', []),
        max_selections=Config.MAX_ASSETS
    )
    
    # Optimization settings
    st.sidebar.subheader("⚡ Optimization Settings")
    
    optimization_method = st.sidebar.selectbox(
        "Method",
        [m.value for m in OptimizationMethod],
        index=0
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        risk_model = st.selectbox(
            "Risk Model",
            [r.value for r in RiskModel],
            index=0
        )
    
    with col2:
        return_model = st.selectbox(
            "Return Model",
            [r.value for r in ReturnModel],
            index=0
        )
    
    # Advanced parameters
    with st.sidebar.expander("🔬 Advanced Parameters"):
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            0.0, 50.0, Config.DEFAULT_RISK_FREE_RATE * 100, 0.1
        ) / 100
        
        if optimization_method == OptimizationMethod.EFFICIENT_RISK.value:
            target_volatility = st.slider(
                "Target Volatility",
                0.05, 0.50, Config.DEFAULT_TARGET_VOLATILITY, 0.01
            )
        else:
            target_volatility = Config.DEFAULT_TARGET_VOLATILITY
        
        if optimization_method == OptimizationMethod.EFFICIENT_RETURN.value:
            target_return = st.slider(
                "Target Return",
                0.05, 1.0, Config.DEFAULT_TARGET_RETURN, 0.01
            )
        else:
            target_return = Config.DEFAULT_TARGET_RETURN
    
    # Update session state
    st.session_state.selected_assets = assets
    
    return {
        'assets': assets,
        'start_date': start_date,
        'end_date': end_date,
        'optimization_method': OptimizationMethod(optimization_method),
        'risk_model': RiskModel(risk_model),
        'return_model': ReturnModel(return_model),
        'risk_free_rate': risk_free_rate,
        'target_volatility': target_volatility,
        'target_return': target_return
    }

@st.cache_data(ttl=3600)
def load_market_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Optional[MarketDataBundle]:
    """Load market data with caching"""
    if not tickers:
        return None
    
    try:
        # Use synchronous wrapper for async function
        async def fetch_data():
            async with AsyncMarketDataFetcher() as fetcher:
                data_dict = await fetcher.fetch_multiple_tickers(tickers, start_date, end_date)
                return data_dict
        
        # Run async function
        data_dict = asyncio.run(fetch_data())
        
        if not data_dict:
            st.error("Failed to fetch market data")
            return None
        
        # Align data
        all_prices = []
        all_volumes = []
        
        for ticker, df in data_dict.items():
            if not df.empty:
                prices = df['Close'].rename(ticker)
                volumes = df['Volume'].rename(ticker)
                all_prices.append(prices)
                all_volumes.append(volumes)
        
        if not all_prices:
            return None
        
        # Create aligned DataFrames
        prices_df = pd.concat(all_prices, axis=1)
        volumes_df = pd.concat(all_volumes, axis=1)
        
        # Forward fill and drop NA
        prices_df = prices_df.ffill().bfill().dropna(how='all', axis=1)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Filter tickers with sufficient data
        valid_tickers = []
        for ticker in tickers:
            if ticker in returns_df.columns:
                if returns_df[ticker].notna().sum() > Config.MIN_DATA_POINTS:
                    valid_tickers.append(ticker)
        
        if len(valid_tickers) < 2:
            st.error(f"Insufficient data for selected tickers. Only {len(valid_tickers)} valid.")
            return None
        
        prices_df = prices_df[valid_tickers]
        returns_df = returns_df[valid_tickers]
        volumes_df = volumes_df[valid_tickers] if all(t in volumes_df.columns for t in valid_tickers) else pd.DataFrame()
        
        # Create market data bundle
        market_data = MarketDataBundle(
            tickers=valid_tickers,
            prices=prices_df,
            returns=returns_df,
            volumes=volumes_df,
            start_date=start_date,
            end_date=end_date,
            metadata={
                'loaded_at': datetime.now(),
                'data_points': len(prices_df),
                'valid_tickers': len(valid_tickers)
            }
        )
        
        return market_data
        
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        logger.logger.error(f"Data loading error: {e}", exc_info=True)
        return None

def display_results(optimizer: EnhancedPortfolioOptimizer, 
                   result: OptimizationResult, 
                   config: Dict):
    """Display optimization results"""
    
    # Performance metrics
    st.subheader("📈 Performance Summary")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Expected Return", f"{result.expected_return:.2%}")
    with cols[1]:
        st.metric("Expected Volatility", f"{result.expected_volatility:.2%}")
    with cols[2]:
        st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    with cols[3]:
        risk_metrics = optimizer.calculate_risk_metrics(result.weights)
        st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Portfolio Allocation",
        "📊 Performance Analysis",
        "⚠️ Risk Metrics",
        "📈 Efficient Frontier"
    ])
    
    with tab1:
        display_portfolio_allocation(result)
    
    with tab2:
        display_performance_analysis(optimizer, result)
    
    with tab3:
        display_risk_metrics(optimizer, result)
    
    with tab4:
        display_efficient_frontier(optimizer, result)

def display_portfolio_allocation(result: OptimizationResult):
    """Display portfolio allocation"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Pie chart
        weights_df = result.to_dataframe()
        if not weights_df.empty:
            fig = px.pie(
                weights_df,
                values='Weight',
                names=weights_df.index,
                hole=0.5,
                title="Portfolio Allocation"
            )
            fig.update_layout(template=Config.CHART_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weight table
        st.dataframe(
            weights_df.style.format({'Weight': '{:.2%}'}),
            use_container_width=True,
            height=400
        )
        
        # Sector allocation
        sector_allocation = {}
        for ticker, weight in result.weights.items():
            for sector, ticker_list in SECTOR_MAPPING.items():
                if ticker in ticker_list:
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
                    break
        
        if sector_allocation:
            st.subheader("Sector Allocation")
            sector_df = pd.DataFrame.from_dict(
                sector_allocation, 
                orient='index', 
                columns=['Weight']
            ).sort_values('Weight', ascending=False)
            
            st.dataframe(
                sector_df.style.format({'Weight': '{:.2%}'}),
                use_container_width=True
            )

def display_performance_analysis(optimizer: EnhancedPortfolioOptimizer, 
                               result: OptimizationResult):
    """Display performance analysis"""
    
    # Calculate portfolio returns
    portfolio_returns = (
        optimizer.market_data.returns * pd.Series(result.weights)
    ).sum(axis=1)
    
    # Cumulative returns chart
    cum_returns = (1 + portfolio_returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#00cc88', width=3)
    ))
    
    fig.update_layout(
        title="Cumulative Portfolio Returns",
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rolling metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Rolling volatility
        rolling_vol = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode='lines',
            name='Rolling Volatility',
            line=dict(color='#ff6b35', width=2)
        ))
        
        fig_vol.update_layout(
            title="Rolling 3-Month Volatility",
            template=Config.CHART_TEMPLATE,
            height=300,
            xaxis_title="Date",
            yaxis_title="Volatility",
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        # Rolling Sharpe
        rolling_sharpe = portfolio_returns.rolling(window=63).apply(
            lambda x: (x.mean() * 252 - result.risk_free_rate) / (x.std() * np.sqrt(252)) 
            if x.std() > 0 else 0
        )
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color='#9d4edd', width=2)
        ))
        
        fig_sharpe.update_layout(
            title="Rolling 3-Month Sharpe Ratio",
            template=Config.CHART_TEMPLATE,
            height=300,
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio"
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

def display_risk_metrics(optimizer: EnhancedPortfolioOptimizer, 
                        result: OptimizationResult):
    """Display comprehensive risk metrics"""
    
    risk_metrics = optimizer.calculate_risk_metrics(result.weights)
    
    # Key risk metrics
    cols = st.columns(3)
    with cols[0]:
        st.metric("Downside Deviation", f"{risk_metrics['downside_deviation']:.2%}")
    with cols[1]:
        st.metric("VaR (95%)", f"{risk_metrics['var_95']:.2%}")
    with cols[2]:
        st.metric("CVaR (95%)", f"{risk_metrics['cvar_95']:.2%}")
    
    # Distribution analysis
    portfolio_returns = (
        optimizer.market_data.returns * pd.Series(result.weights)
    ).sum(axis=1)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Returns Distribution", 
            "QQ Plot",
            "Drawdown Analysis",
            "Autocorrelation"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Returns distribution
    fig.add_trace(
        go.Histogram(
            x=portfolio_returns.values,
            nbinsx=50,
            name="Returns",
            marker_color='#0066cc'
        ),
        row=1, col=1
    )
    
    # QQ Plot
    sorted_returns = np.sort(portfolio_returns)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sorted_returns,
            mode='markers',
            name='QQ Plot',
            marker=dict(color='#4facfe', size=6)
        ),
        row=1, col=2
    )
    
    # Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4d4d', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # Autocorrelation
    autocorr = [portfolio_returns.autocorr(lag=i) for i in range(1, 31)]
    fig.add_trace(
        go.Bar(
            x=list(range(1, 31)),
            y=autocorr,
            name='Autocorrelation',
            marker_color='#00cc88'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        template=Config.CHART_TEMPLATE,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_efficient_frontier(optimizer: EnhancedPortfolioOptimizer, 
                             result: OptimizationResult):
    """Display efficient frontier"""
    
    try:
        frontier_data = optimizer.generate_efficient_frontier()
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_data['volatilities'],
            y=frontier_data['returns'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='white', width=3),
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 0.1)'
        ))
        
        # Optimal portfolio
        fig.add_trace(go.Scatter(
            x=[result.expected_volatility],
            y=[result.expected_return],
            mode='markers',
            marker=dict(
                color='red',
                size=20,
                symbol='star'
            ),
            name='Optimal Portfolio'
        ))
        
        # Individual assets
        for ticker in optimizer.tickers:
            asset_return = optimizer.market_data.returns[ticker].mean() * 252
            asset_vol = optimizer.market_data.returns[ticker].std() * np.sqrt(252)
            
            fig.add_trace(go.Scatter(
                x=[asset_vol],
                y=[asset_return],
                mode='markers+text',
                text=[ticker],
                textposition="top center",
                marker=dict(size=10, color='lightblue'),
                name=ticker,
                showlegend=False
            ))
        
        fig.update_layout(
            title="Efficient Frontier",
            template=Config.CHART_TEMPLATE,
            height=Config.CHART_HEIGHT,
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate efficient frontier: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA CONSTANTS (keep from original code)
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
# APPLICATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ## 🚨 Application Error
        
        **Error Type:** {type(e).__name__}
        **Error Details:** {str(e)}
        """)
        
        with st.expander("🔍 Technical Details"):
            st.code(traceback.format_exc(), language="python")
        
        # Show system info
        st.info(f"""
        **System Information:**
        - Python: {sys.version.split()[0]}
        - Platform: {sys.platform}
        """)
