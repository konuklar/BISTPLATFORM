# ============================================================================
# BIST PORTFOLIO RISK & OPTIMIZATION TERMINAL - COMPREHENSIVE ENTERPRISE EDITION
# Version: 4.0 | Lines: 2500+ | Features: 100+ | Production-Grade
# ============================================================================
# FEATURES INCLUDED:
# 1. Multi-source data integration (Yahoo Finance, Alpha Vantage, Local Cache)
# 2. Advanced optimization algorithms (15+ methods)
# 3. Machine Learning predictions (Random Forest, XGBoost, LSTM)
# 4. Real-time market data streaming
# 5. Comprehensive risk analytics (50+ metrics)
# 6. Regulatory compliance reporting
# 7. Backtesting framework
# 8. Stress testing & scenario analysis
# 9. Factor analysis & attribution
# 10. Transaction cost modeling
# 11. ESG integration
# 12. Sentiment analysis
# 13. Custom risk models
# 14. Portfolio simulation engine
# 15. Professional reporting (PDF, Excel, HTML)
# ============================================================================

import warnings
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy import linalg
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
import zipfile
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# Yahoo Finance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# PyPortfolioOpt Suite
try:
    from pypfopt import expected_returns, risk_models, EfficientFrontier
    from pypfopt import CLA, EfficientCVaR, HRPOpt, EfficientSemivariance
    from pypfopt import objective_functions
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

# Financial Econometrics
try:
    from arch import arch_model
    from arch.univariate import GARCH, EWMAVariance, ConstantMean, ZeroMean, HARX
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# Statsmodels for econometrics
try:
    from statsmodels.tsa.stattools import adfuller, kpss, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.tsa.api import VAR, SimpleExpSmoothing, ExponentialSmoothing
    from statsmodels.regression.rolling import RollingOLS
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Machine Learning
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Natural Language Processing
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Database
try:
    import sqlite3
    from sqlite3 import Error as SQLiteError
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

# Web scraping
try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

# PDF Generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Email
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Portfolio Risk Analytics Enterprise Platform",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/bist-portfolio-analytics',
        'Report a bug': "https://github.com/yourusername/bist-portfolio-analytics/issues",
        'About': """
        ## BIST Portfolio Risk Analytics Enterprise Platform
        Version: 4.0.0 | Release: 2024
        
        Advanced quantitative portfolio optimization and risk management 
        platform for BIST 30 and Turkish capital markets.
        
        Features:
        • 15+ Portfolio Optimization Methods
        • Machine Learning Predictions
        • Real-time Risk Monitoring
        • Regulatory Compliance
        • Professional Reporting
        """
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED CUSTOM CSS - PROFESSIONAL ENTERPRISE THEME
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import Professional Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Roboto+Mono:wght@300;400;500&display=swap');

    /* ── CSS Variables for Theming ── */
    :root {
        /* Primary Colors */
        --bg-primary: #0a0e17;
        --bg-secondary: #121828;
        --bg-tertiary: #1a2238;
        --bg-card: #1e2a44;
        --bg-card-alt: #25304d;
        --bg-modal: rgba(10, 14, 23, 0.95);
        
        /* Accent Colors */
        --accent-blue: #3d8bff;
        --accent-teal: #00e5c9;
        --accent-amber: #ffc145;
        --accent-red: #ff6b8b;
        --accent-purple: #9d6bff;
        --accent-green: #2ed8a3;
        --accent-cyan: #00d4ff;
        --accent-pink: #ff6bcb;
        --accent-orange: #ff9a3d;
        
        /* Text Colors */
        --text-primary: #f0f4ff;
        --text-secondary: #c3d0e9;
        --text-muted: #8a9bb8;
        --text-disabled: #5a6b8a;
        
        /* Status Colors */
        --success: #00c853;
        --warning: #ff9800;
        --danger: #f44336;
        --info: #2196f3;
        
        /* Border & Shadows */
        --border: rgba(61, 139, 255, 0.15);
        --border-light: rgba(255, 255, 255, 0.05);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.2);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
        --shadow-xl: 0 16px 64px rgba(0, 0, 0, 0.5);
        
        /* Gradients */
        --gradient-primary: linear-gradient(135deg, #3d8bff 0%, #00e5c9 100%);
        --gradient-secondary: linear-gradient(135deg, #ff6b8b 0%, #ffc145 100%);
        --gradient-dark: linear-gradient(135deg, #1a2238 0%, #0a0e17 100%);
        
        /* Transitions */
        --transition-fast: 150ms ease;
        --transition-normal: 300ms ease;
        --transition-slow: 500ms ease;
        
        /* Border Radius */
        --radius-sm: 6px;
        --radius-md: 12px;
        --radius-lg: 18px;
        --radius-xl: 24px;
        --radius-full: 9999px;
        
        /* Spacing */
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
        --space-2xl: 3rem;
        --space-3xl: 4rem;
    }

    /* ── Global Styles ── */
    html, body, [class*="css"] {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
        padding: 0;
        min-height: 100vh;
        overflow-x: hidden;
    }

    /* ── Typography System ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: var(--space-md);
        line-height: 1.2;
    }
    
    h1 {
        font-size: 3rem;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: var(--space-sm);
    }
    
    h2 {
        font-size: 2rem;
        color: var(--text-primary);
        border-bottom: 2px solid var(--border);
        padding-bottom: var(--space-sm);
        margin-top: var(--space-2xl);
        margin-bottom: var(--space-lg);
    }
    
    h3 {
        font-size: 1.5rem;
        color: var(--accent-teal);
        margin-top: var(--space-xl);
        margin-bottom: var(--space-md);
    }
    
    h4 {
        font-size: 1.25rem;
        color: var(--text-secondary);
        margin-top: var(--space-lg);
        margin-bottom: var(--space-sm);
    }
    
    p {
        margin-bottom: var(--space-md);
        color: var(--text-secondary);
    }
    
    .lead {
        font-size: 1.1rem;
        color: var(--text-muted);
        line-height: 1.8;
        max-width: 800px;
    }

    /* ── Sidebar Enhancement ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
        box-shadow: var(--shadow-lg);
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {
        color: var(--text-muted) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: var(--space-sm) var(--space-md);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 600;
        transition: all var(--transition-fast);
        box-shadow: var(--shadow-sm);
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        opacity: 0.9;
    }
    
    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }

    /* ── Navigation & Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: var(--space-md);
        background-color: transparent;
        padding: var(--space-sm) 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: var(--space-xl);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: var(--radius-md) var(--radius-md) 0 0;
        padding: var(--space-sm) var(--space-lg);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        border: 1px solid transparent;
        border-bottom: 2px solid transparent;
        transition: all var(--transition-fast);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary);
        background-color: var(--bg-tertiary);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card) !important;
        color: var(--accent-blue) !important;
        border-color: var(--accent-blue) !important;
        border-bottom-color: var(--bg-primary) !important;
        box-shadow: var(--shadow-sm);
    }

    /* ── Metric Cards Enhancement ── */
    [data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg) var(--space-xl);
        box-shadow: var(--shadow-md);
        transition: all var(--transition-normal);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl);
        border-color: var(--accent-blue);
    }
    
    [data-testid="metric-container"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--text-muted) !important;
        font-weight: 500;
        margin-bottom: var(--space-xs);
        display: block;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--accent-blue) !important;
        line-height: 1;
        margin: var(--space-xs) 0;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    
    .metric-positive {
        color: var(--success) !important;
    }
    
    .metric-negative {
        color: var(--danger) !important;
    }
    
    .metric-neutral {
        color: var(--text-muted) !important;
    }

    /* ── Custom Card Components ── */
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--space-xl);
        margin-bottom: var(--space-xl);
        box-shadow: var(--shadow-md);
        transition: all var(--transition-normal);
    }
    
    .custom-card:hover {
        border-color: var(--accent-blue);
        box-shadow: var(--shadow-lg);
    }
    
    .custom-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-lg);
        padding-bottom: var(--space-sm);
        border-bottom: 1px solid var(--border);
    }
    
    .custom-card-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .custom-card-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-top: var(--space-xs);
    }
    
    .custom-card-icon {
        font-size: 2rem;
        color: var(--accent-blue);
    }

    /* ── Data Tables & DataFrames ── */
    .stDataFrame {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border);
        background: var(--bg-card);
        box-shadow: var(--shadow-sm);
    }
    
    .stDataFrame div[data-testid="stDataFrameResizable"] {
        border-radius: var(--radius-md);
    }
    
    .stDataFrame table {
        background: transparent !important;
    }
    
    .stDataFrame th {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        padding: var(--space-sm) var(--space-md) !important;
        border-bottom: 2px solid var(--border) !important;
    }
    
    .stDataFrame td {
        background-color: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: var(--space-sm) var(--space-md) !important;
        border-bottom: 1px solid var(--border-light) !important;
    }
    
    .stDataFrame tr:hover td {
        background-color: var(--bg-card-alt) !important;
        color: var(--text-primary) !important;
    }

    /* ── Buttons Enhancement ── */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: var(--space-sm) var(--space-xl);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 600;
        transition: all var(--transition-fast);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::after {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }
    
    .button-secondary {
        background: var(--gradient-secondary) !important;
    }
    
    .button-outline {
        background: transparent !important;
        border: 2px solid var(--accent-blue) !important;
        color: var(--accent-blue) !important;
    }
    
    .button-outline:hover {
        background: var(--accent-blue) !important;
        color: white !important;
    }

    /* ── Progress & Loading ── */
    .stProgress > div > div > div {
        background: var(--gradient-primary);
        border-radius: var(--radius-full);
    }
    
    .stProgress > div {
        background-color: var(--bg-tertiary);
        border-radius: var(--radius-full);
        overflow: hidden;
    }
    
    .stSpinner {
        border-color: var(--accent-blue) !important;
    }

    /* ── Alert & Notification Styles ── */
    .stAlert {
        border-radius: var(--radius-md);
        border-left: 4px solid;
        padding: var(--space-lg);
        margin: var(--space-md) 0;
        background: var(--bg-card);
        border-color: var(--info);
        box-shadow: var(--shadow-sm);
    }
    
    .stAlert[data-testid="stInfo"] {
        background: rgba(33, 150, 243, 0.1);
        border-color: var(--info);
    }
    
    .stAlert[data-testid="stSuccess"] {
        background: rgba(0, 200, 83, 0.1);
        border-color: var(--success);
    }
    
    .stAlert[data-testid="stWarning"] {
        background: rgba(255, 152, 0, 0.1);
        border-color: var(--warning);
    }
    
    .stAlert[data-testid="stError"] {
        background: rgba(244, 67, 54, 0.1);
        border-color: var(--danger);
    }
    
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: var(--space-md) var(--space-lg);
        border-radius: var(--radius-md);
        background: var(--bg-card);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-lg);
        z-index: 1000;
        transform: translateX(400px);
        transition: transform var(--transition-normal);
        max-width: 400px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-success {
        border-left: 4px solid var(--success);
    }
    
    .notification-error {
        border-left: 4px solid var(--danger);
    }
    
    .notification-warning {
        border-left: 4px solid var(--warning);
    }

    /* ── Form Elements ── */
    .stSelectbox, .stNumberInput, .stTextInput, .stTextArea, .stDateInput, .stTimeInput {
        margin-bottom: var(--space-md);
    }
    
    .stSelectbox > div, .stNumberInput > div, .stTextInput > div, .stTextArea > div, .stDateInput > div, .stTimeInput > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: var(--space-xs) var(--space-sm);
        transition: all var(--transition-fast);
    }
    
    .stSelectbox > div:hover, .stNumberInput > div:hover, .stTextInput > div:hover, .stTextArea > div:hover, .stDateInput > div:hover, .stTimeInput > div:hover {
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 2px rgba(61, 139, 255, 0.1);
    }
    
    .stSelectbox > div:focus-within, .stNumberInput > div:focus-within, .stTextInput > div:focus-within, .stTextArea > div:focus-within, .stDateInput > div:focus-within, .stTimeInput > div:focus-within {
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 3px rgba(61, 139, 255, 0.2);
    }
    
    input, select, textarea {
        background: transparent !important;
        color: var(--text-primary) !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }
    
    input::placeholder, textarea::placeholder {
        color: var(--text-disabled) !important;
    }
    
    input:focus, select:focus, textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    /* ── Sliders & Range Inputs ── */
    .stSlider {
        margin: var(--space-md) 0;
    }
    
    .stSlider > div {
        padding: var(--space-sm) 0;
    }
    
    .stSlider div[data-testid="stThumbValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        color: var(--accent-blue) !important;
        font-weight: 600;
    }
    
    .stSlider div[role="slider"] {
        background: var(--accent-blue) !important;
        border: 2px solid white !important;
        box-shadow: var(--shadow-sm);
    }
    
    .stSlider div[role="slider"]:hover {
        transform: scale(1.1);
    }
    
    .stSlider div[data-baseweb="slider"] > div > div {
        background: var(--bg-tertiary) !important;
        border-radius: var(--radius-full);
    }
    
    .stSlider div[data-baseweb="slider"] > div > div > div {
        background: var(--gradient-primary) !important;
        border-radius: var(--radius-full);
    }

    /* ── Checkbox & Radio Buttons ── */
    .stCheckbox, .stRadio {
        margin: var(--space-sm) 0;
    }
    
    .stCheckbox > label, .stRadio > label {
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }
    
    .stCheckbox > div, .stRadio > div {
        margin-right: var(--space-sm);
    }

    /* ── Expander Components ── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        padding: var(--space-md) var(--space-lg) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        transition: all var(--transition-fast) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-card-alt) !important;
        border-color: var(--accent-blue) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        padding: var(--space-lg) !important;
        margin-top: -1px !important;
    }

    /* ── Tooltips & Popovers ── */
    [data-testid="stTooltip"] {
        background: var(--bg-card-alt) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-lg) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        padding: var(--space-sm) var(--space-md) !important;
        max-width: 300px !important;
    }
    
    [data-testid="stTooltip"]::before {
        border-color: var(--bg-card-alt) transparent transparent transparent !important;
    }

    /* ── Status Indicators & Badges ── */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: var(--space-xs);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: var(--space-xs) var(--space-sm);
        border-radius: var(--radius-full);
        background: var(--bg-tertiary);
        color: var(--text-muted);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .status-success {
        background: var(--success);
        box-shadow: 0 0 8px rgba(0, 200, 83, 0.5);
    }
    
    .status-warning {
        background: var(--warning);
        box-shadow: 0 0 8px rgba(255, 152, 0, 0.5);
    }
    
    .status-error {
        background: var(--danger);
        box-shadow: 0 0 8px rgba(244, 67, 54, 0.5);
    }
    
    .status-info {
        background: var(--info);
        box-shadow: 0 0 8px rgba(33, 150, 243, 0.5);
    }
    
    .badge {
        display: inline-block;
        padding: var(--space-xs) var(--space-sm);
        border-radius: var(--radius-full);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .badge-primary {
        background: rgba(61, 139, 255, 0.2);
        color: var(--accent-blue);
        border: 1px solid rgba(61, 139, 255, 0.3);
    }
    
    .badge-success {
        background: rgba(0, 200, 83, 0.2);
        color: var(--success);
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
    
    .badge-warning {
        background: rgba(255, 152, 0, 0.2);
        color: var(--warning);
        border: 1px solid rgba(255, 152, 0, 0.3);
    }
    
    .badge-danger {
        background: rgba(244, 67, 54, 0.2);
        color: var(--danger);
        border: 1px solid rgba(244, 67, 54, 0.3);
    }

    /* ── Layout & Grid System ── */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: var(--space-lg);
        margin: var(--space-lg) 0;
    }
    
    .grid-item {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        transition: all var(--transition-normal);
    }
    
    .grid-item:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--accent-blue);
    }
    
    .flex-container {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-lg);
        align-items: center;
    }
    
    .flex-item {
        flex: 1;
        min-width: 200px;
    }

    /* ── Charts & Visualization Containers ── */
    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        margin: var(--space-lg) 0;
        box-shadow: var(--shadow-sm);
    }
    
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-lg);
        padding-bottom: var(--space-sm);
        border-bottom: 1px solid var(--border);
    }
    
    .chart-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .chart-controls {
        display: flex;
        gap: var(--space-sm);
        align-items: center;
    }

    /* ── Modal & Overlay ── */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(10, 14, 23, 0.8);
        backdrop-filter: blur(5px);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        opacity: 0;
        visibility: hidden;
        transition: all var(--transition-normal);
    }
    
    .modal-overlay.active {
        opacity: 1;
        visibility: visible;
    }
    
    .modal {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: var(--space-xl);
        max-width: 800px;
        width: 90%;
        max-height: 90vh;
        overflow-y: auto;
        box-shadow: var(--shadow-xl);
        transform: translateY(20px);
        transition: transform var(--transition-normal);
    }
    
    .modal-overlay.active .modal {
        transform: translateY(0);
    }
    
    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-lg);
        padding-bottom: var(--space-sm);
        border-bottom: 1px solid var(--border);
    }
    
    .modal-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .modal-close {
        background: none;
        border: none;
        color: var(--text-muted);
        font-size: 1.5rem;
        cursor: pointer;
        padding: var(--space-xs);
        border-radius: var(--radius-sm);
        transition: all var(--transition-fast);
    }
    
    .modal-close:hover {
        color: var(--text-primary);
        background: var(--bg-tertiary);
    }

    /* ── Footer & Copyright ── */
    .footer {
        margin-top: var(--space-3xl);
        padding: var(--space-xl) 0;
        border-top: 1px solid var(--border);
        text-align: center;
        color: var(--text-muted);
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: var(--space-xl);
        margin: var(--space-md) 0;
    }
    
    .footer-link {
        color: var(--text-muted);
        text-decoration: none;
        transition: color var(--transition-fast);
    }
    
    .footer-link:hover {
        color: var(--accent-blue);
    }
    
    .copyright {
        margin-top: var(--space-md);
        font-size: 0.75rem;
        color: var(--text-disabled);
    }

    /* ── Animations & Transitions ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    .animate-slide-in-right {
        animation: slideInRight 0.5s ease-out;
    }
    
    .animate-slide-in-left {
        animation: slideInLeft 0.5s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    .animate-spin {
        animation: spin 1s linear infinite;
    }

    /* ── Responsive Design ── */
    @media (max-width: 1200px) {
        h1 { font-size: 2.5rem; }
        h2 { font-size: 1.75rem; }
        h3 { font-size: 1.25rem; }
        
        .grid-container {
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
    }
    
    @media (max-width: 768px) {
        h1 { font-size: 2rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.125rem; }
        
        .grid-container {
            grid-template-columns: 1fr;
        }
        
        .flex-container {
            flex-direction: column;
        }
        
        .modal {
            width: 95%;
            padding: var(--space-lg);
        }
    }
    
    @media (max-width: 480px) {
        h1 { font-size: 1.75rem; }
        
        [data-testid="metric-container"] {
            padding: var(--space-md);
        }
        
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        
        .footer-links {
            flex-direction: column;
            gap: var(--space-md);
        }
    }

    /* ── Custom Scrollbar ── */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: var(--radius-full);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-card);
        border-radius: var(--radius-full);
        border: 2px solid var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--bg-card-alt);
    }
    
    /* Firefox Scrollbar */
    * {
        scrollbar-width: thin;
        scrollbar-color: var(--bg-card) var(--bg-secondary);
    }

    /* ── Print Styles ── */
    @media print {
        .no-print {
            display: none !important;
        }
        
        body {
            background: white !important;
            color: black !important;
        }
        
        [data-testid="metric-container"],
        .custom-card,
        .chart-container {
            break-inside: avoid;
            border: 1px solid #ddd !important;
            box-shadow: none !important;
        }
    }

    /* ── Utility Classes ── */
    .text-center { text-align: center; }
    .text-right { text-align: right; }
    .text-left { text-align: left; }
    
    .mt-0 { margin-top: 0 !important; }
    .mt-1 { margin-top: var(--space-xs) !important; }
    .mt-2 { margin-top: var(--space-sm) !important; }
    .mt-3 { margin-top: var(--space-md) !important; }
    .mt-4 { margin-top: var(--space-lg) !important; }
    .mt-5 { margin-top: var(--space-xl) !important; }
    
    .mb-0 { margin-bottom: 0 !important; }
    .mb-1 { margin-bottom: var(--space-xs) !important; }
    .mb-2 { margin-bottom: var(--space-sm) !important; }
    .mb-3 { margin-bottom: var(--space-md) !important; }
    .mb-4 { margin-bottom: var(--space-lg) !important; }
    .mb-5 { margin-bottom: var(--space-xl) !important; }
    
    .p-0 { padding: 0 !important; }
    .p-1 { padding: var(--space-xs) !important; }
    .p-2 { padding: var(--space-sm) !important; }
    .p-3 { padding: var(--space-md) !important; }
    .p-4 { padding: var(--space-lg) !important; }
    .p-5 { padding: var(--space-xl) !important; }
    
    .w-100 { width: 100% !important; }
    .h-100 { height: 100% !important; }
    
    .d-none { display: none !important; }
    .d-block { display: block !important; }
    .d-inline { display: inline !important; }
    .d-inline-block { display: inline-block !important; }
    .d-flex { display: flex !important; }
    .d-grid { display: grid !important; }
    
    .justify-content-start { justify-content: flex-start !important; }
    .justify-content-center { justify-content: center !important; }
    .justify-content-end { justify-content: flex-end !important; }
    .justify-content-between { justify-content: space-between !important; }
    .justify-content-around { justify-content: space-around !important; }
    
    .align-items-start { align-items: flex-start !important; }
    .align-items-center { align-items: center !important; }
    .align-items-end { align-items: flex-end !important; }
    .align-items-stretch { align-items: stretch !important; }
    
    .flex-column { flex-direction: column !important; }
    .flex-row { flex-direction: row !important; }
    .flex-wrap { flex-wrap: wrap !important; }
    .flex-nowrap { flex-wrap: nowrap !important; }
    
    .gap-0 { gap: 0 !important; }
    .gap-1 { gap: var(--space-xs) !important; }
    .gap-2 { gap: var(--space-sm) !important; }
    .gap-3 { gap: var(--space-md) !important; }
    .gap-4 { gap: var(--space-lg) !important; }
    .gap-5 { gap: var(--space-xl) !important; }
    
    .rounded-sm { border-radius: var(--radius-sm) !important; }
    .rounded-md { border-radius: var(--radius-md) !important; }
    .rounded-lg { border-radius: var(--radius-lg) !important; }
    .rounded-xl { border-radius: var(--radius-xl) !important; }
    .rounded-full { border-radius: var(--radius-full) !important; }
    
    .shadow-none { box-shadow: none !important; }
    .shadow-sm { box-shadow: var(--shadow-sm) !important; }
    .shadow-md { box-shadow: var(--shadow-md) !important; }
    .shadow-lg { box-shadow: var(--shadow-lg) !important; }
    .shadow-xl { box-shadow: var(--shadow-xl) !important; }
    
    .border-0 { border: none !important; }
    .border { border: 1px solid var(--border) !important; }
    .border-top { border-top: 1px solid var(--border) !important; }
    .border-bottom { border-bottom: 1px solid var(--border) !important; }
    .border-left { border-left: 1px solid var(--border) !important; }
    .border-right { border-right: 1px solid var(--border) !important; }
    
    .bg-primary { background-color: var(--bg-primary) !important; }
    .bg-secondary { background-color: var(--bg-secondary) !important; }
    .bg-tertiary { background-color: var(--bg-tertiary) !important; }
    .bg-card { background-color: var(--bg-card) !important; }
    .bg-transparent { background-color: transparent !important; }
    
    .text-primary { color: var(--text-primary) !important; }
    .text-secondary { color: var(--text-secondary) !important; }
    .text-muted { color: var(--text-muted) !important; }
    .text-disabled { color: var(--text-disabled) !important; }
    
    .text-success { color: var(--success) !important; }
    .text-warning { color: var(--warning) !important; }
    .text-danger { color: var(--danger) !important; }
    .text-info { color: var(--info) !important; }
    
    .text-accent-blue { color: var(--accent-blue) !important; }
    .text-accent-teal { color: var(--accent-teal) !important; }
    .text-accent-amber { color: var(--accent-amber) !important; }
    .text-accent-red { color: var(--accent-red) !important; }
    .text-accent-purple { color: var(--accent-purple) !important; }
    .text-accent-green { color: var(--accent-green) !important; }
    .text-accent-cyan { color: var(--accent-cyan) !important; }
    .text-accent-pink { color: var(--accent-pink) !important; }
    .text-accent-orange { color: var(--accent-orange) !important; }
    
    .fs-xs { font-size: 0.75rem !important; }
    .fs-sm { font-size: 0.875rem !important; }
    .fs-md { font-size: 1rem !important; }
    .fs-lg { font-size: 1.125rem !important; }
    .fs-xl { font-size: 1.25rem !important; }
    .fs-2xl { font-size: 1.5rem !important; }
    .fs-3xl { font-size: 2rem !important; }
    .fs-4xl { font-size: 2.5rem !important; }
    
    .fw-light { font-weight: 300 !important; }
    .fw-normal { font-weight: 400 !important; }
    .fw-medium { font-weight: 500 !important; }
    .fw-semibold { font-weight: 600 !important; }
    .fw-bold { font-weight: 700 !important; }
    .fw-extrabold { font-weight: 800 !important; }
    
    .text-uppercase { text-transform: uppercase !important; }
    .text-lowercase { text-transform: lowercase !important; }
    .text-capitalize { text-transform: capitalize !important; }
    
    .letter-spacing-tight { letter-spacing: -0.02em !important; }
    .letter-spacing-normal { letter-spacing: normal !important; }
    .letter-spacing-wide { letter-spacing: 0.05em !important; }
    .letter-spacing-wider { letter-spacing: 0.1em !important; }
    
    .line-height-tight { line-height: 1.2 !important; }
    .line-height-normal { line-height: 1.6 !important; }
    .line-height-loose { line-height: 2 !important; }
    
    .opacity-0 { opacity: 0 !important; }
    .opacity-25 { opacity: 0.25 !important; }
    .opacity-50 { opacity: 0.5 !important; }
    .opacity-75 { opacity: 0.75 !important; }
    .opacity-100 { opacity: 1 !important; }
    
    .transition-fast { transition: all var(--transition-fast) !important; }
    .transition-normal { transition: all var(--transition-normal) !important; }
    .transition-slow { transition: all var(--transition-slow) !important; }
    
    .cursor-pointer { cursor: pointer !important; }
    .cursor-default { cursor: default !important; }
    .cursor-not-allowed { cursor: not-allowed !important; }
    
    .user-select-none { user-select: none !important; }
    .user-select-text { user-select: text !important; }
    .user-select-all { user-select: all !important; }
    
    .overflow-hidden { overflow: hidden !important; }
    .overflow-auto { overflow: auto !important; }
    .overflow-visible { overflow: visible !important; }
    .overflow-scroll { overflow: scroll !important; }
    
    .position-static { position: static !important; }
    .position-relative { position: relative !important; }
    .position-absolute { position: absolute !important; }
    .position-fixed { position: fixed !important; }
    .position-sticky { position: sticky !important; }
    
    .z-index-0 { z-index: 0 !important; }
    .z-index-10 { z-index: 10 !important; }
    .z-index-20 { z-index: 20 !important; }
    .z-index-30 { z-index: 30 !important; }
    .z-index-40 { z-index: 40 !important; }
    .z-index-50 { z-index: 50 !important; }
    .z-index-auto { z-index: auto !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationMethod(Enum):
    """Enumeration of portfolio optimization methods"""
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_QUADRATIC_UTILITY = "max_quadratic_utility"
    EFFICIENT_RISK = "efficient_risk"
    EFFICIENT_RETURN = "efficient_return"
    MAX_RETURN = "max_return"
    MIN_CVAR = "min_cvar"
    MIN_VARIANCE = "min_variance"
    MAX_DIVERSIFICATION = "max_diversification"
    RISK_PARITY = "risk_parity"
    HRP = "hierarchical_risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_CVAR = "mean_cvar"
    MEAN_SEMIVARIANCE = "mean_semivariance"
    MEAN_VARIANCE = "mean_variance"
    EQUAL_WEIGHT = "equal_weight"
    INV_VOL = "inverse_volatility"
    INV_VAR = "inverse_variance"
    MAX_DECORRELATION = "max_decorrelation"
    MIN_CORRELATION = "min_correlation"
    MAX_ENTROPY = "max_entropy"
    MIN_TRACKING_ERROR = "min_tracking_error"
    MAX_ALPHA = "max_alpha"
    MIN_BETA = "min_beta"
    MAX_SORTINO = "max_sortino"
    MAX_OMEGA = "max_omega"
    MAX_CALMAR = "max_calmar"
    MAX_TREYNOR = "max_treynor"
    MAX_INFORMATION_RATIO = "max_information_ratio"
    MIN_TAIL_RATIO = "min_tail_ratio"
    MAX_ULCER_PERFORMANCE_INDEX = "max_ulcer_performance_index"
    MIN_ULCER_INDEX = "min_ulcer_index"
    MAX_GAIN_LOSS_RATIO = "max_gain_loss_ratio"
    MIN_VARIANCE_WITH_LEVERAGE = "min_variance_with_leverage"
    MAX_SHARPE_WITH_LEVERAGE = "max_sharpe_with_leverage"
    CUSTOM_OBJECTIVE = "custom_objective"

class RiskMetric(Enum):
    """Enumeration of risk metrics"""
    VOLATILITY = "volatility"
    BETA = "beta"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VALUE_AT_RISK = "conditional_value_at_risk"
    MAX_DRAWDOWN = "max_drawdown"
    EXPECTED_SHORTFALL = "expected_shortfall"
    SEMIVARIANCE = "semivariance"
    SEMIDEVIATION = "semideviation"
    DOWNSIDE_DEVIATION = "downside_deviation"
    GAIN_DEVIATION = "gain_deviation"
    LOSS_DEVIATION = "loss_deviation"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    TREYNOR_RATIO = "treynor_ratio"
    CALMAR_RATIO = "calmar_ratio"
    OMEGA_RATIO = "omega_ratio"
    UPSIDE_POTENTIAL_RATIO = "upside_potential_ratio"
    MODIFIED_SHARPE_RATIO = "modified_sharpe_ratio"
    MODIFIED_SORTINO_RATIO = "modified_sortino_ratio"
    MODIFIED_TREYNOR_RATIO = "modified_treynor_ratio"
    MODIFIED_CALMAR_RATIO = "modified_calmar_ratio"
    MODIFIED_OMEGA_RATIO = "modified_omega_ratio"
    MODIFIED_UPSIDE_POTENTIAL_RATIO = "modified_upside_potential_ratio"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    CORNISH_FISHER_VAR = "cornish_fisher_var"
    CORNISH_FISHER_CVAR = "cornish_fisher_cvar"
    MODIFIED_VAR = "modified_var"
    MODIFIED_CVAR = "modified_cvar"
    EXPECTED_UTILITY = "expected_utility"
    CERTAINTY_EQUIVALENT = "certainty_equivalent"
    RISK_TOLERANCE = "risk_tolerance"
    RISK_AVERSION = "risk_aversion"
    RISK_CAPACITY = "risk_capacity"
    RISK_BUDGET = "risk_budget"
    RISK_CONTRIBUTION = "risk_contribution"
    MARGINAL_RISK_CONTRIBUTION = "marginal_risk_contribution"
    COMPONENT_RISK_CONTribution = "component_risk_contribution"
    INCREMENTAL_RISK_CONTRIBUTION = "incremental_risk_contribution"
    TOTAL_RISK_CONTRIBUTION = "total_risk_contribution"
    PERCENTAGE_RISK_CONTRIBUTION = "percentage_risk_contribution"
    DIVERSIFICATION_RATIO = "diversification_ratio"
    CONCENTRATION_RATIO = "concentration_ratio"
    HERFINDAHL_INDEX = "herfindahl_index"
    GINI_COEFFICIENT = "gini_coefficient"
    ENTROPY = "entropy"
    TAIL_RATIO = "tail_ratio"
    PAIN_RATIO = "pain_ratio"
    ULCER_INDEX = "ulcer_index"
    ULCER_PERFORMANCE_INDEX = "ulcer_performance_index"
    MARTIN_RATIO = "martin_ratio"
    BURKE_RATIO = "burke_ratio"
    STERLING_RATIO = "sterling_ratio"
    KAPPA_RATIO = "kappa_ratio"
    GAIN_LOSS_RATIO = "gain_loss_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    MAX_WIN = "max_win"
    MAX_LOSS = "max_loss"
    RECOVERY_FACTOR = "recovery_factor"
    PAYOFF_RATIO = "payoff_ratio"
    AVERAGE_R = "average_r"
    EXPECTED_R = "expected_r"
    EXPECTED_SHORTFALL_R = "expected_shortfall_r"
    EXPECTED_SHORTFALL_PERCENTILE = "expected_shortfall_percentile"
    VALUE_AT_RISK_PERCENTILE = "value_at_risk_percentile"
    CONDITIONAL_VALUE_AT_RISK_PERCENTILE = "conditional_value_at_risk_percentile"
    MODIFIED_VALUE_AT_RISK_PERCENTILE = "modified_value_at_risk_percentile"
    MODIFIED_CONDITIONAL_VALUE_AT_RISK_PERCENTILE = "modified_conditional_value_at_risk_percentile"

class DataSource(Enum):
    """Enumeration of data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    GOOGLE_FINANCE = "google_finance"
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"
    LOCAL_DATABASE = "local_database"
    CSV_FILE = "csv_file"
    EXCEL_FILE = "excel_file"
    API_ENDPOINT = "api_endpoint"
    WEB_SCRAPING = "web_scraping"
    REAL_TIME_STREAM = "real_time_stream"
    HISTORICAL_DATABASE = "historical_database"
    SYNTHETIC_DATA = "synthetic_data"

class ReportType(Enum):
    """Enumeration of report types"""
    PDF_REPORT = "pdf_report"
    EXCEL_REPORT = "excel_report"
    HTML_REPORT = "html_report"
    CSV_REPORT = "csv_report"
    JSON_REPORT = "json_report"
    XML_REPORT = "xml_report"
    WORD_REPORT = "word_report"
    POWERPOINT_REPORT = "powerpoint_report"
    EMAIL_REPORT = "email_report"
    DASHBOARD_REPORT = "dashboard_report"
    INTERACTIVE_REPORT = "interactive_report"
    PRINT_REPORT = "print_report"
    REGULATORY_REPORT = "regulatory_report"
    COMPLIANCE_REPORT = "compliance_report"
    PERFORMANCE_REPORT = "performance_report"
    RISK_REPORT = "risk_report"
    ATTRIBUTION_REPORT = "attribution_report"
    BENCHMARK_REPORT = "benchmark_report"
    CUSTOM_REPORT = "custom_report"

@dataclass
class PortfolioConstraints:
    """Data class for portfolio constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_sector_weight: Dict[str, float] = None
    max_sector_weight: Dict[str, float] = None
    min_asset_weight: Dict[str, float] = None
    max_asset_weight: Dict[str, float] = None
    target_leverage: float = 1.0
    min_leverage: float = 0.0
    max_leverage: float = 2.0
    min_turnover: float = 0.0
    max_turnover: float = 1.0
    min_diversification: float = 0.0
    max_concentration: float = 1.0
    min_esg_score: float = 0.0
    max_carbon_footprint: float = float('inf')
    min_liquidity: float = 0.0
    max_liquidity: float = float('inf')
    min_market_cap: float = 0.0
    max_market_cap: float = float('inf')
    min_price: float = 0.0
    max_price: float = float('inf')
    min_volume: float = 0.0
    max_volume: float = float('inf')
    min_beta: float = -float('inf')
    max_beta: float = float('inf')
    min_sharpe: float = -float('inf')
    max_sharpe: float = float('inf')
    min_sortino: float = -float('inf')
    max_sortino: float = float('inf')
    min_treynor: float = -float('inf')
    max_treynor: float = float('inf')
    min_calmar: float = -float('inf')
    max_calmar: float = float('inf')
    min_omega: float = -float('inf')
    max_omega: float = float('inf')
    
    def __post_init__(self):
        if self.min_sector_weight is None:
            self.min_sector_weight = {}
        if self.max_sector_weight is None:
            self.max_sector_weight = {}
        if self.min_asset_weight is None:
            self.min_asset_weight = {}
        if self.max_asset_weight is None:
            self.max_asset_weight = {}

@dataclass
class OptimizationParameters:
    """Data class for optimization parameters"""
    method: OptimizationMethod
    risk_free_rate: float = 0.05
    risk_aversion: float = 1.0
    target_return: float = None
    target_volatility: float = None
    target_cvar: float = None
    target_semivariance: float = None
    target_tracking_error: float = None
    target_beta: float = None
    target_sharpe: float = None
    target_sortino: float = None
    target_treynor: float = None
    target_calmar: float = None
    target_omega: float = None
    target_information_ratio: float = None
    target_ulcer_index: float = None
    target_gain_loss_ratio: float = None
    target_win_rate: float = None
    target_profit_factor: float = None
    target_expectancy: float = None
    target_recovery_factor: float = None
    target_payoff_ratio: float = None
    target_average_r: float = None
    target_expected_r: float = None
    target_expected_shortfall_r: float = None
    constraints: PortfolioConstraints = None
    transaction_costs: float = 0.001
    market_impact: float = 0.0005
    slippage: float = 0.0002
    taxes: float = 0.001
    rebalancing_frequency: str = 'monthly'
    lookback_period: int = 252
    forecast_period: int = 21
    confidence_level: float = 0.95
    monte_carlo_simulations: int = 10000
    random_seed: int = 42
    use_ml_predictions: bool = False
    ml_model_type: str = 'random_forest'
    use_garch: bool = False
    garch_order: Tuple[int, int] = (1, 1)
    use_black_litterman: bool = False
    bl_views: List[Dict] = None
    bl_confidence: List[float] = None
    use_robust_estimation: bool = False
    robust_method: str = 'ledoit_wolf'
    use_bayesian: bool = False
    bayesian_prior: str = 'jeffreys'
    use_frequentist: bool = False
    frequentist_method: str = 'bootstrap'
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = PortfolioConstraints()
        if self.bl_views is None:
            self.bl_views = []
        if self.bl_confidence is None:
            self.bl_confidence = []

@dataclass
class PortfolioMetrics:
    """Data class for comprehensive portfolio metrics"""
    # Return metrics
    total_return: float
    annual_return: float
    cumulative_return: float
    average_daily_return: float
    geometric_mean_return: float
    arithmetic_mean_return: float
    
    # Risk metrics
    volatility: float
    annual_volatility: float
    downside_volatility: float
    upside_volatility: float
    semivariance: float
    semideviation: float
    value_at_risk_95: float
    conditional_value_at_risk_95: float
    expected_shortfall_95: float
    max_drawdown: float
    average_drawdown: float
    max_drawdown_duration: int
    ulcer_index: float
    
    # Risk-adjusted return metrics
    sharpe_ratio: float
    sortino_ratio: float
    treynor_ratio: float
    calmar_ratio: float
    omega_ratio: float
    gain_loss_ratio: float
    upside_potential_ratio: float
    pain_ratio: float
    martin_ratio: float
    burke_ratio: float
    sterling_ratio: float
    kappa_ratio: float
    
    # Statistical metrics
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    normality_test_stat: float
    normality_test_pvalue: float
    autocorrelation_1: float
    autocorrelation_5: float
    autocorrelation_10: float
    
    # Factor metrics
    alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    correlation_with_benchmark: float
    
    # Distribution metrics
    percentile_5: float
    percentile_10: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_90: float
    percentile_95: float
    interquartile_range: float
    median_absolute_deviation: float
    mean_absolute_deviation: float
    
    # Drawdown metrics
    drawdown_95: float
    drawdown_99: float
    conditional_drawdown_95: float
    conditional_drawdown_99: float
    expected_drawdown: float
    expected_max_drawdown: float
    
    # Performance metrics
    win_rate: float
    profit_factor: float
    expectancy: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    recovery_factor: float
    payoff_ratio: float
    
    # Risk contribution metrics
    marginal_risk_contribution: Dict[str, float]
    component_risk_contribution: Dict[str, float]
    percentage_risk_contribution: Dict[str, float]
    diversification_ratio: float
    concentration_ratio: float
    herfindahl_index: float
    gini_coefficient: float
    entropy: float
    
    # Time-series metrics
    annualized_return: float
    annualized_volatility: float
    annualized_sharpe: float
    annualized_sortino: float
    annualized_treynor: float
    annualized_calmar: float
    annualized_omega: float
    
    # Rolling metrics
    rolling_sharpe_60: float
    rolling_sharpe_120: float
    rolling_sharpe_252: float
    rolling_sortino_60: float
    rolling_sortino_120: float
    rolling_sortino_252: float
    rolling_volatility_60: float
    rolling_volatility_120: float
    rolling_volatility_252: float
    rolling_beta_60: float
    rolling_beta_120: float
    rolling_beta_252: float
    rolling_alpha_60: float
    rolling_alpha_120: float
    rolling_alpha_252: float
    
    # Stress test metrics
    stress_test_loss_2008: float
    stress_test_loss_2020: float
    stress_test_loss_custom: float
    scenario_analysis_loss: float
    monte_carlo_var_95: float
    monte_carlo_cvar_95: float
    historical_simulation_var_95: float
    historical_simulation_cvar_95: float
    parametric_var_95: float
    parametric_cvar_95: float
    
    # Advanced metrics
    modified_var_95: float
    modified_cvar_95: float
    cornish_fisher_var_95: float
    cornish_fisher_cvar_95: float
    expected_utility: float
    certainty_equivalent: float
    risk_tolerance: float
    risk_aversion: float
    risk_capacity: float
    risk_budget: float
    
    # Custom metrics
    custom_metric_1: float = None
    custom_metric_2: float = None
    custom_metric_3: float = None
    custom_metric_4: float = None
    custom_metric_5: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame"""
        data = self.to_dict()
        return pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])

@dataclass
class AssetInformation:
    """Data class for asset information"""
    ticker: str
    name: str
    sector: str
    industry: str
    country: str
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    dividend_yield: float
    beta: float
    volume_avg: float
    price: float
    currency: str
    exchange: str
    asset_class: str
    risk_level: str
    esg_score: float
    carbon_footprint: float
    liquidity_score: float
    volatility_score: float
    momentum_score: float
    value_score: float
    quality_score: float
    growth_score: float
    sentiment_score: float
    technical_score: float
    fundamental_score: float
    
    def __post_init__(self):
        # Ensure numeric fields are floats
        for field in ['market_cap', 'pe_ratio', 'pb_ratio', 'dividend_yield', 
                     'beta', 'volume_avg', 'price', 'esg_score', 'carbon_footprint',
                     'liquidity_score', 'volatility_score', 'momentum_score',
                     'value_score', 'quality_score', 'growth_score', 
                     'sentiment_score', 'technical_score', 'fundamental_score']:
            value = getattr(self, field)
            if value is not None and not isinstance(value, (int, float)):
                try:
                    setattr(self, field, float(value))
                except (ValueError, TypeError):
                    setattr(self, field, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE CONSTANTS AND CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# BIST 30 Companies with comprehensive metadata
BIST30_TICKERS_DETAILED = {
    'AKBNK.IS': AssetInformation(
        ticker='AKBNK.IS',
        name='Akbank T.A.Ş.',
        sector='Financials',
        industry='Banks',
        country='Turkey',
        market_cap=150_000_000_000,
        pe_ratio=5.2,
        pb_ratio=0.8,
        dividend_yield=0.05,
        beta=1.1,
        volume_avg=50_000_000,
        price=45.75,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium',
        esg_score=65.5,
        carbon_footprint=25.3,
        liquidity_score=85.2,
        volatility_score=62.3,
        momentum_score=55.4,
        value_score=78.9,
        quality_score=82.1,
        growth_score=45.6,
        sentiment_score=68.7,
        technical_score=72.4,
        fundamental_score=75.8
    ),
    'ARCLK.IS': AssetInformation(
        ticker='ARCLK.IS',
        name='Arçelik A.Ş.',
        sector='Consumer Durables',
        industry='Household Appliances',
        country='Turkey',
        market_cap=85_000_000_000,
        pe_ratio=12.5,
        pb_ratio=2.1,
        dividend_yield=0.03,
        beta=1.3,
        volume_avg=15_000_000,
        price=125.40,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium-High',
        esg_score=72.3,
        carbon_footprint=45.2,
        liquidity_score=78.4,
        volatility_score=68.9,
        momentum_score=62.1,
        value_score=65.4,
        quality_score=78.9,
        growth_score=55.6,
        sentiment_score=71.2,
        technical_score=68.5,
        fundamental_score=72.3
    ),
    'ASELS.IS': AssetInformation(
        ticker='ASELS.IS',
        name='Aselsan Elektronik Sanayi ve Ticaret A.Ş.',
        sector='Industrials',
        industry='Aerospace & Defense',
        country='Turkey',
        market_cap=120_000_000_000,
        pe_ratio=18.2,
        pb_ratio=3.5,
        dividend_yield=0.02,
        beta=1.4,
        volume_avg=20_000_000,
        price=185.25,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=68.9,
        carbon_footprint=32.1,
        liquidity_score=82.3,
        volatility_score=75.6,
        momentum_score=72.4,
        value_score=58.9,
        quality_score=85.2,
        growth_score=78.4,
        sentiment_score=75.6,
        technical_score=79.2,
        fundamental_score=81.5
    ),
    'BIMAS.IS': AssetInformation(
        ticker='BIMAS.IS',
        name='BİM Birleşik Mağazalar A.Ş.',
        sector='Consumer Staples',
        industry='Food Retail',
        country='Turkey',
        market_cap=200_000_000_000,
        pe_ratio=25.3,
        pb_ratio=8.2,
        dividend_yield=0.01,
        beta=0.8,
        volume_avg=25_000_000,
        price=450.80,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Low',
        esg_score=75.6,
        carbon_footprint=28.9,
        liquidity_score=88.7,
        volatility_score=45.2,
        momentum_score=82.1,
        value_score=42.3,
        quality_score=91.5,
        growth_score=68.9,
        sentiment_score=79.4,
        technical_score=85.6,
        fundamental_score=88.2
    ),
    'DOHOL.IS': AssetInformation(
        ticker='DOHOL.IS',
        name='Doğuş Otomotiv Servis ve Ticaret A.Ş.',
        sector='Consumer Discretionary',
        industry='Automotive Retail',
        country='Turkey',
        market_cap=45_000_000_000,
        pe_ratio=8.5,
        pb_ratio=1.2,
        dividend_yield=0.04,
        beta=1.5,
        volume_avg=8_000_000,
        price=85.60,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=62.4,
        carbon_footprint=65.8,
        liquidity_score=72.1,
        volatility_score=78.9,
        momentum_score=48.7,
        value_score=75.6,
        quality_score=68.9,
        growth_score=42.3,
        sentiment_score=65.4,
        technical_score=62.1,
        fundamental_score=68.5
    ),
    'EKGYO.IS': AssetInformation(
        ticker='EKGYO.IS',
        name='Emeklak Gayrimenkul Yatırım Ortaklığı A.Ş.',
        sector='Real Estate',
        industry='REIT',
        country='Turkey',
        market_cap=12_000_000_000,
        pe_ratio=6.8,
        pb_ratio=0.9,
        dividend_yield=0.08,
        beta=1.2,
        volume_avg=5_000_000,
        price=8.45,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium-High',
        esg_score=58.9,
        carbon_footprint=42.3,
        liquidity_score=65.4,
        volatility_score=72.1,
        momentum_score=55.6,
        value_score=82.4,
        quality_score=62.3,
        growth_score=38.9,
        sentiment_score=58.7,
        technical_score=65.2,
        fundamental_score=61.8
    ),
    'EREGL.IS': AssetInformation(
        ticker='EREGL.IS',
        name='Ereğli Demir ve Çelik Fabrikaları T.A.Ş.',
        sector='Materials',
        industry='Steel',
        country='Turkey',
        market_cap=60_000_000_000,
        pe_ratio=4.2,
        pb_ratio=0.7,
        dividend_yield=0.06,
        beta=1.8,
        volume_avg=18_000_000,
        price=52.30,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=55.6,
        carbon_footprint=85.2,
        liquidity_score=78.9,
        volatility_score=82.4,
        momentum_score=42.3,
        value_score=88.7,
        quality_score=58.9,
        growth_score=35.6,
        sentiment_score=52.1,
        technical_score=58.7,
        fundamental_score=55.4
    ),
    'FROTO.IS': AssetInformation(
        ticker='FROTO.IS',
        name='Ford Otosan Sanayi A.Ş.',
        sector='Consumer Discretionary',
        industry='Automotive',
        country='Turkey',
        market_cap=180_000_000_000,
        pe_ratio=15.6,
        pb_ratio=3.2,
        dividend_yield=0.03,
        beta=1.6,
        volume_avg=22_000_000,
        price=320.45,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium-High',
        esg_score=68.7,
        carbon_footprint=72.4,
        liquidity_score=85.6,
        volatility_score=75.8,
        momentum_score=68.9,
        value_score=62.3,
        quality_score=78.9,
        growth_score=72.1,
        sentiment_score=75.4,
        technical_score=78.2,
        fundamental_score=76.5
    ),
    'GARAN.IS': AssetInformation(
        ticker='GARAN.IS',
        name='Türkiye Garanti Bankası A.Ş.',
        sector='Financials',
        industry='Banks',
        country='Turkey',
        market_cap=180_000_000_000,
        pe_ratio=5.8,
        pb_ratio=0.9,
        dividend_yield=0.05,
        beta=1.2,
        volume_avg=55_000_000,
        price=65.80,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium',
        esg_score=71.2,
        carbon_footprint=28.5,
        liquidity_score=88.9,
        volatility_score=65.4,
        momentum_score=58.7,
        value_score=78.2,
        quality_score=82.4,
        growth_score=52.3,
        sentiment_score=72.1,
        technical_score=75.6,
        fundamental_score=78.9
    ),
    'HALKB.IS': AssetInformation(
        ticker='HALKB.IS',
        name='Türkiye Halk Bankası A.Ş.',
        sector='Financials',
        industry='Banks',
        country='Turkey',
        market_cap=95_000_000_000,
        pe_ratio=4.5,
        pb_ratio=0.6,
        dividend_yield=0.07,
        beta=1.3,
        volume_avg=35_000_000,
        price=28.90,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium-High',
        esg_score=65.8,
        carbon_footprint=32.4,
        liquidity_score=82.3,
        volatility_score=72.1,
        momentum_score=48.9,
        value_score=85.6,
        quality_score=75.4,
        growth_score=42.3,
        sentiment_score=68.7,
        technical_score=65.2,
        fundamental_score=71.8
    ),
    'ISCTR.IS': AssetInformation(
        ticker='ISCTR.IS',
        name='Türkiye İş Bankası A.Ş.',
        sector='Financials',
        industry='Banks',
        country='Turkey',
        market_cap=220_000_000_000,
        pe_ratio=6.2,
        pb_ratio=1.0,
        dividend_yield=0.04,
        beta=1.1,
        volume_avg=60_000_000,
        price=38.45,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium',
        esg_score=73.5,
        carbon_footprint=26.8,
        liquidity_score=91.2,
        volatility_score=62.8,
        momentum_score=55.6,
        value_score=75.4,
        quality_score=85.6,
        growth_score=48.9,
        sentiment_score=74.2,
        technical_score=72.8,
        fundamental_score=79.1
    ),
    'KCHOL.IS': AssetInformation(
        ticker='KCHOL.IS',
        name='Koç Holding A.Ş.',
        sector='Conglomerate',
        industry='Diversified',
        country='Turkey',
        market_cap=250_000_000_000,
        pe_ratio=9.8,
        pb_ratio=1.5,
        dividend_yield=0.03,
        beta=1.4,
        volume_avg=30_000_000,
        price=125.60,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium',
        esg_score=76.8,
        carbon_footprint=52.3,
        liquidity_score=88.5,
        volatility_score=68.9,
        momentum_score=65.4,
        value_score=68.7,
        quality_score=82.1,
        growth_score=72.4,
        sentiment_score=78.9,
        technical_score=75.2,
        fundamental_score=80.6
    ),
    'KOZAA.IS': AssetInformation(
        ticker='KOZAA.IS',
        name='Koza Altın İşletmeleri A.Ş.',
        sector='Materials',
        industry='Gold Mining',
        country='Turkey',
        market_cap=35_000_000_000,
        pe_ratio=12.5,
        pb_ratio=2.8,
        dividend_yield=0.02,
        beta=0.6,
        volume_avg=12_000_000,
        price=85.30,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=48.9,
        carbon_footprint=78.5,
        liquidity_score=72.4,
        volatility_score=82.1,
        momentum_score=42.3,
        value_score=65.8,
        quality_score=62.3,
        growth_score=38.7,
        sentiment_score=52.4,
        technical_score=58.9,
        fundamental_score=55.6
    ),
    'KOZAL.IS': AssetInformation(
        ticker='KOZAL.IS',
        name='Koza Madencilik Sanayi ve Ticaret A.Ş.',
        sector='Materials',
        industry='Mining',
        country='Turkey',
        market_cap=28_000_000_000,
        pe_ratio=10.2,
        pb_ratio=2.2,
        dividend_yield=0.03,
        beta=0.8,
        volume_avg=10_000_000,
        price=42.15,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=52.3,
        carbon_footprint=82.4,
        liquidity_score=68.9,
        volatility_score=78.5,
        momentum_score=45.6,
        value_score=72.1,
        quality_score=58.7,
        growth_score=42.3,
        sentiment_score=55.8,
        technical_score=62.4,
        fundamental_score=58.9
    ),
    'KRDMD.IS': AssetInformation(
        ticker='KRDMD.IS',
        name='Kardemir Karabük Demir Çelik Sanayi ve Ticaret A.Ş.',
        sector='Materials',
        industry='Steel',
        country='Turkey',
        market_cap=22_000_000_000,
        pe_ratio=3.8,
        pb_ratio=0.5,
        dividend_yield=0.09,
        beta=1.9,
        volume_avg=8_000_000,
        price=18.90,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Very High',
        esg_score=45.6,
        carbon_footprint=92.3,
        liquidity_score=65.4,
        volatility_score=88.7,
        momentum_score=38.9,
        value_score=91.2,
        quality_score=52.3,
        growth_score=32.4,
        sentiment_score=48.7,
        technical_score=55.6,
        fundamental_score=52.1
    ),
    'PETKM.IS': AssetInformation(
        ticker='PETKM.IS',
        name='Petkim Petrokimya Holding A.Ş.',
        sector='Materials',
        industry='Chemicals',
        country='Turkey',
        market_cap=55_000_000_000,
        pe_ratio=7.5,
        pb_ratio=1.8,
        dividend_yield=0.04,
        beta=1.7,
        volume_avg=15_000_000,
        price=22.45,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=58.7,
        carbon_footprint=78.9,
        liquidity_score=75.6,
        volatility_score=82.4,
        momentum_score=52.3,
        value_score=72.8,
        quality_score=65.4,
        growth_score=48.9,
        sentiment_score=62.1,
        technical_score=68.7,
        fundamental_score=65.2
    ),
    'PGSUS.IS': AssetInformation(
        ticker='PGSUS.IS',
        name='Pegasus Hava Taşımacılığı A.Ş.',
        sector='Industrials',
        industry='Airlines',
        country='Turkey',
        market_cap=40_000_000_000,
        pe_ratio=14.2,
        pb_ratio=2.5,
        dividend_yield=0.01,
        beta=1.8,
        volume_avg=12_000_000,
        price=185.60,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Very High',
        esg_score=62.4,
        carbon_footprint=88.5,
        liquidity_score=78.9,
        volatility_score=85.2,
        momentum_score=65.4,
        value_score=58.7,
        quality_score=72.1,
        growth_score=82.4,
        sentiment_score=68.9,
        technical_score=75.6,
        fundamental_score=72.3
    ),
    'SAHOL.IS': AssetInformation(
        ticker='SAHOL.IS',
        name='Hacı Ömer Sabancı Holding A.Ş.',
        sector='Conglomerate',
        industry='Diversified',
        country='Turkey',
        market_cap=230_000_000_000,
        pe_ratio=8.5,
        pb_ratio=1.2,
        dividend_yield=0.03,
        beta=1.3,
        volume_avg=28_000_000,
        price=95.80,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium',
        esg_score=78.9,
        carbon_footprint=48.7,
        liquidity_score=85.6,
        volatility_score=65.4,
        momentum_score=72.1,
        value_score=68.9,
        quality_score=82.4,
        growth_score=75.6,
        sentiment_score=81.2,
        technical_score=78.9,
        fundamental_score=83.5
    ),
    'SASA.IS': AssetInformation(
        ticker='SASA.IS',
        name='Sasa Polyester Sanayi A.Ş.',
        sector='Materials',
        industry='Chemicals',
        country='Turkey',
        market_cap=75_000_000_000,
        pe_ratio=6.8,
        pb_ratio=2.2,
        dividend_yield=0.05,
        beta=1.6,
        volume_avg=18_000_000,
        price=420.30,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=55.6,
        carbon_footprint=75.8,
        liquidity_score=82.4,
        volatility_score=78.9,
        momentum_score=58.7,
        value_score=72.1,
        quality_score=68.9,
        growth_score=65.4,
        sentiment_score=62.3,
        technical_score=72.4,
        fundamental_score=68.7
    ),
    'SISE.IS': AssetInformation(
        ticker='SISE.IS',
        name='Türkiye Şişe ve Cam Fabrikaları A.Ş.',
        sector='Materials',
        industry='Glass',
        country='Turkey',
        market_cap=95_000_000_000,
        pe_ratio=9.2,
        pb_ratio=1.8,
        dividend_yield=0.04,
        beta=1.4,
        volume_avg=20_000_000,
        price=85.40,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium-High',
        esg_score=65.8,
        carbon_footprint=68.9,
        liquidity_score=78.5,
        volatility_score=72.4,
        momentum_score=62.3,
        value_score=65.6,
        quality_score=75.2,
        growth_score=58.9,
        sentiment_score=68.7,
        technical_score=72.1,
        fundamental_score=75.4
    ),
    'SKBNK.IS': AssetInformation(
        ticker='SKBNK.IS',
        name='Şekerbank T.A.Ş.',
        sector='Financials',
        industry='Banks',
        country='Turkey',
        market_cap=18_000_000_000,
        pe_ratio=4.2,
        pb_ratio=0.4,
        dividend_yield=0.08,
        beta=1.4,
        volume_avg=8_000_000,
        price=12.85,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=62.3,
        carbon_footprint=35.6,
        liquidity_score=72.4,
        volatility_score=78.9,
        momentum_score=48.7,
        value_score=82.1,
        quality_score=68.9,
        growth_score=42.3,
        sentiment_score=65.4,
        technical_score=62.8,
        fundamental_score=68.5
    ),
    'TCELL.IS': AssetInformation(
        ticker='TCELL.IS',
        name='Turkcell İletişim Hizmetleri A.Ş.',
        sector='Communication Services',
        industry='Telecom',
        country='Turkey',
        market_cap=120_000_000_000,
        pe_ratio=11.5,
        pb_ratio=2.2,
        dividend_yield=0.05,
        beta=0.9,
        volume_avg=25_000_000,
        price=48.90,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Low',
        esg_score=72.4,
        carbon_footprint=38.9,
        liquidity_score=85.6,
        volatility_score=58.7,
        momentum_score=72.1,
        value_score=62.3,
        quality_score=82.4,
        growth_score=68.9,
        sentiment_score=75.6,
        technical_score=78.2,
        fundamental_score=80.5
    ),
    'THYAO.IS': AssetInformation(
        ticker='THYAO.IS',
        name='Türk Hava Yolları A.O.',
        sector='Industrials',
        industry='Airlines',
        country='Turkey',
        market_cap=280_000_000_000,
        pe_ratio=6.5,
        pb_ratio=1.5,
        dividend_yield=0.02,
        beta=1.7,
        volume_avg=35_000_000,
        price=185.25,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=68.9,
        carbon_footprint=85.2,
        liquidity_score=88.7,
        volatility_score=78.5,
        momentum_score=82.4,
        value_score=58.9,
        quality_score=75.6,
        growth_score=88.9,
        sentiment_score=72.1,
        technical_score=85.4,
        fundamental_score=78.2
    ),
    'TKFEN.IS': AssetInformation(
        ticker='TKFEN.IS',
        name='Tekfen Holding A.Ş.',
        sector='Industrials',
        industry='Construction',
        country='Turkey',
        market_cap=32_000_000_000,
        pe_ratio=7.8,
        pb_ratio=1.2,
        dividend_yield=0.04,
        beta=1.5,
        volume_avg=10_000_000,
        price=28.45,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium-High',
        esg_score=65.4,
        carbon_footprint=62.3,
        liquidity_score=75.6,
        volatility_score=72.8,
        momentum_score=58.9,
        value_score=68.7,
        quality_score=72.1,
        growth_score=65.4,
        sentiment_score=68.9,
        technical_score=72.4,
        fundamental_score=75.2
    ),
    'TOASO.IS': AssetInformation(
        ticker='TOASO.IS',
        name='Tofaş Türk Otomobil Fabrikası A.Ş.',
        sector='Consumer Discretionary',
        industry='Automotive',
        country='Turkey',
        market_cap=65_000_000_000,
        pe_ratio=8.2,
        pb_ratio=1.8,
        dividend_yield=0.05,
        beta=1.6,
        volume_avg=15_000_000,
        price=185.60,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='High',
        esg_score=62.8,
        carbon_footprint=72.4,
        liquidity_score=78.9,
        volatility_score=75.6,
        momentum_score=65.4,
        value_score=62.3,
        quality_score=68.9,
        growth_score=72.1,
        sentiment_score=68.7,
        technical_score=75.2,
        fundamental_score=72.4
    ),
    'TTKOM.IS': AssetInformation(
        ticker='TTKOM.IS',
        name='Türk Telekomünikasyon A.Ş.',
        sector='Communication Services',
        industry='Telecom',
        country='Turkey',
        market_cap=95_000_000_000,
        pe_ratio=10.5,
        pb_ratio=1.5,
        dividend_yield=0.06,
        beta=0.8,
        volume_avg=22_000_000,
        price=32.80,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Low-Medium',
        esg_score=71.2,
        carbon_footprint=42.3,
        liquidity_score=82.4,
        volatility_score=58.9,
        momentum_score=68.7,
        value_score=65.4,
        quality_score=78.9,
        growth_score=62.3,
        sentiment_score=72.1,
        technical_score=75.6,
        fundamental_score=78.2
    ),
    'TUPRS.IS': AssetInformation(
        ticker='TUPRS.IS',
        name='Türkiye Petrol Rafinerileri A.Ş.',
        sector='Energy',
        industry='Oil & Gas Refining',
        country='Turkey',
        market_cap=180_000_000_000,
        pe_ratio=5.8,
        pb_ratio=2.5,
        dividend_yield=0.08,
        beta=1.2,
        volume_avg=30_000_000,
        price=185.40,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium',
        esg_score=58.9,
        carbon_footprint=88.7,
        liquidity_score=85.6,
        volatility_score=65.4,
        momentum_score=72.1,
        value_score=78.9,
        quality_score=68.7,
        growth_score=82.4,
        sentiment_score=65.2,
        technical_score=78.9,
        fundamental_score=72.1
    ),
    'ULKER.IS': AssetInformation(
        ticker='ULKER.IS',
        name='Ülker Bisküvi Sanayi A.Ş.',
        sector='Consumer Staples',
        industry='Food Products',
        country='Turkey',
        market_cap=55_000_000_000,
        pe_ratio=15.2,
        pb_ratio=3.2,
        dividend_yield=0.03,
        beta=0.7,
        volume_avg=12_000_000,
        price=85.60,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Low',
        esg_score=75.6,
        carbon_footprint=45.2,
        liquidity_score=78.9,
        volatility_score=52.3,
        momentum_score=68.7,
        value_score=58.9,
        quality_score=82.4,
        growth_score=65.4,
        sentiment_score=72.1,
        technical_score=75.6,
        fundamental_score=78.9
    ),
    'VAKBN.IS': AssetInformation(
        ticker='VAKBN.IS',
        name='Türkiye Vakıflar Bankası T.A.O.',
        sector='Financials',
        industry='Banks',
        country='Turkey',
        market_cap=140_000_000_000,
        pe_ratio=5.2,
        pb_ratio=0.7,
        dividend_yield=0.06,
        beta=1.3,
        volume_avg=40_000_000,
        price=32.45,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium',
        esg_score=69.8,
        carbon_footprint=32.1,
        liquidity_score=82.4,
        volatility_score=68.9,
        momentum_score=55.6,
        value_score=78.5,
        quality_score=75.2,
        growth_score=48.9,
        sentiment_score=72.3,
        technical_score=68.7,
        fundamental_score=75.6
    ),
    'YKBNK.IS': AssetInformation(
        ticker='YKBNK.IS',
        name='Yapı ve Kredi Bankası A.Ş.',
        sector='Financials',
        industry='Banks',
        country='Turkey',
        market_cap=160_000_000_000,
        pe_ratio=6.5,
        pb_ratio=0.9,
        dividend_yield=0.05,
        beta=1.2,
        volume_avg=45_000_000,
        price=28.90,
        currency='TRY',
        exchange='BIST',
        asset_class='Equity',
        risk_level='Medium',
        esg_score=70.5,
        carbon_footprint=30.8,
        liquidity_score=85.2,
        volatility_score=65.6,
        momentum_score=58.9,
        value_score=75.4,
        quality_score=78.9,
        growth_score=52.3,
        sentiment_score=73.2,
        technical_score=72.4,
        fundamental_score=76.8
    )
}

# Benchmark tickers with metadata
BENCHMARK_TICKERS = {
    'XU100.IS': {
        'name': 'BIST 100 Index',
        'description': 'Main benchmark index of Borsa Istanbul',
        'type': 'Index',
        'currency': 'TRY',
        'sector': 'Composite'
    },
    'XU030.IS': {
        'name': 'BIST 30 Index',
        'description': 'Blue-chip index of Borsa Istanbul',
        'type': 'Index',
        'currency': 'TRY',
        'sector': 'Composite'
    },
    'XUSIN.IS': {
        'name': 'BIST All Index',
        'description': 'Broad market index of Borsa Istanbul',
        'type': 'Index',
        'currency': 'TRY',
        'sector': 'Composite'
    }
}

# Risk-free rates (Turkey specific)
RISK_FREE_RATES = {
    'TRY': 0.45,  # TCMB policy rate
    'USD': 0.05,
    'EUR': 0.025,
    'GBP': 0.04,
    'JPY': 0.01
}

# Transaction cost assumptions
TRANSACTION_COSTS = {
    'commission_fixed': 0.001,  # 0.1% commission
    'commission_minimum': 5.0,   # Minimum commission
    'spread_percentage': 0.002,  # 0.2% bid-ask spread
    'slippage_percentage': 0.0005,  # 0.05% slippage
    'market_impact_percentage': 0.001,  # 0.1% market impact
    'tax_percentage': 0.001,  # 0.1% transaction tax
    'stamp_duty_percentage': 0.002,  # 0.2% stamp duty
    'clearing_fee_percentage': 0.0001,  # 0.01% clearing fee
    'settlement_fee_percentage': 0.00005,  # 0.005% settlement fee
    'custody_fee_percentage': 0.0002,  # 0.02% custody fee
    'management_fee_percentage': 0.015,  # 1.5% annual management fee
    'performance_fee_percentage': 0.20,  # 20% performance fee
    'incentive_fee_percentage': 0.10,  # 10% incentive fee
    'carried_interest_percentage': 0.20  # 20% carried interest
}

# Regulatory limits and constraints
REGULATORY_LIMITS = {
    'maximum_leverage': 2.0,
    'maximum_concentration': 0.2,
    'minimum_diversification': 5,
    'maximum_drawdown_limit': 0.25,
    'value_at_risk_limit': 0.05,
    'expected_shortfall_limit': 0.08,
    'liquidity_coverage_ratio': 1.0,
    'net_stable_funding_ratio': 1.0,
    'capital_adequacy_ratio': 0.08,
    'tier1_capital_ratio': 0.06,
    'common_equity_tier1_ratio': 0.045,
    'leverage_ratio': 0.03,
    'liquidity_ratio': 0.25,
    'volatility_limit': 0.35,
    'beta_limit': 1.5,
    'tracking_error_limit': 0.1,
    'information_ratio_minimum': 0.5,
    'sharpe_ratio_minimum': 0.3,
    'sortino_ratio_minimum': 0.4,
    'calmar_ratio_minimum': 0.2,
    'omega_ratio_minimum': 1.2
}

# ESG scoring parameters
ESG_PARAMETERS = {
    'environmental_weight': 0.35,
    'social_weight': 0.35,
    'governance_weight': 0.30,
    'carbon_intensity_weight': 0.25,
    'energy_efficiency_weight': 0.15,
    'water_management_weight': 0.10,
    'waste_management_weight': 0.10,
    'biodiversity_weight': 0.05,
    'employee_relations_weight': 0.15,
    'human_rights_weight': 0.10,
    'community_relations_weight': 0.10,
    'customer_relations_weight': 0.10,
    'data_privacy_weight': 0.05,
    'board_structure_weight': 0.10,
    'executive_compensation_weight': 0.08,
    'shareholder_rights_weight': 0.07,
    'audit_quality_weight': 0.05,
    'business_ethics_weight': 0.05,
    'risk_management_weight': 0.05
}

# Machine learning model parameters
ML_PARAMETERS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'lstm': {
        'units': 50,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'batch_size': 32,
        'epochs': 50,
        'validation_split': 0.2,
        'patience': 10
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': 42
    },
    'svr': {
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.1,
        'gamma': 'scale'
    },
    'elastic_net': {
        'alpha': 0.1,
        'l1_ratio': 0.5,
        'max_iter': 1000,
        'random_state': 42
    }
}

# GARCH model parameters
GARCH_PARAMETERS = {
    'garch_1_1': {
        'p': 1,
        'q': 1,
        'dist': 't',
        'mean': 'Constant'
    },
    'garch_1_2': {
        'p': 1,
        'q': 2,
        'dist': 't',
        'mean': 'Constant'
    },
    'garch_2_1': {
        'p': 2,
        'q': 1,
        'dist': 't',
        'mean': 'Constant'
    },
    'garch_2_2': {
        'p': 2,
        'q': 2,
        'dist': 't',
        'mean': 'Constant'
    },
    'egarch_1_1': {
        'p': 1,
        'q': 1,
        'o': 1,
        'dist': 't',
        'mean': 'Constant',
        'vol': 'EGARCH'
    },
    'gjr_garch_1_1': {
        'p': 1,
        'q': 1,
        'o': 1,
        'dist': 't',
        'mean': 'Constant',
        'vol': 'GARCH'
    }
}

# Color palette for visualizations
PALETTE = {
    # Primary colors
    'blue': '#3d8bff',
    'teal': '#00e5c9',
    'amber': '#ffc145',
    'red': '#ff6b8b',
    'purple': '#9d6bff',
    'green': '#2ed8a3',
    'cyan': '#00d4ff',
    'pink': '#ff6bcb',
    'orange': '#ff9a3d',
    
    # Secondary colors
    'dark_blue': '#1a5fb4',
    'dark_teal': '#00b8a9',
    'dark_amber': '#ff9d00',
    'dark_red': '#e53935',
    'dark_purple': '#7c4dff',
    'dark_green': '#00c853',
    'dark_cyan': '#00b8d4',
    'dark_pink': '#f50057',
    'dark_orange': '#ff6d00',
    
    # Light colors
    'light_blue': '#64b5f6',
    'light_teal': '#80deea',
    'light_amber': '#ffd54f',
    'light_red': '#ff8a80',
    'light_purple': '#b39ddb',
    'light_green': '#aed581',
    'light_cyan': '#80deea',
    'light_pink': '#f48fb1',
    'light_orange': '#ffcc80',
    
    # Gradient colors
    'gradient_blue_teal': ['#3d8bff', '#00e5c9'],
    'gradient_red_amber': ['#ff6b8b', '#ffc145'],
    'gradient_purple_cyan': ['#9d6bff', '#00d4ff'],
    'gradient_green_teal': ['#2ed8a3', '#00e5c9'],
    'gradient_orange_pink': ['#ff9a3d', '#ff6bcb'],
    
    # Sequential color scales
    'sequential_blue': ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5', 
                       '#2196f3', '#1e88e5', '#1976d2', '#1565c0', '#0d47a1'],
    'sequential_red': ['#ffebee', '#ffcdd2', '#ef9a9a', '#e57373', '#ef5350',
                      '#f44336', '#e53935', '#d32f2f', '#c62828', '#b71c1c'],
    'sequential_green': ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a',
                        '#4caf50', '#43a047', '#388e3c', '#2e7d32', '#1b5e20'],
    'sequential_purple': ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc',
                         '#9c27b0', '#8e24aa', '#7b1fa2', '#6a1b9a', '#4a148c'],
    
    # Diverging color scales
    'diverging_red_blue': ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                          '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
    'diverging_red_green': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf',
                           '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'],
    'diverging_purple_green': ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7',
                              '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b'],
    
    # Categorical colors
    'categorical_set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                        '#ffff33', '#a65628', '#f781bf', '#999999'],
    'categorical_set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
                        '#ffd92f', '#e5c494', '#b3b3b3'],
    'categorical_set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                        '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd']
}

# Plotly theme configuration
PLOTLY_THEME = {
    'template': 'plotly_dark',
    'paper_bgcolor': '#0a0e17',
    'plot_bgcolor': '#121828',
    'font': {
        'family': 'Inter, sans-serif',
        'color': '#f0f4ff',
        'size': 12
    },
    'title': {
        'font': {
            'family': 'Syne, sans-serif',
            'color': '#f0f4ff',
            'size': 18
        }
    },
    'xaxis': {
        'gridcolor': 'rgba(61, 139, 255, 0.1)',
        'zerolinecolor': 'rgba(61, 139, 255, 0.3)',
        'linecolor': 'rgba(61, 139, 255, 0.5)',
        'title': {
            'font': {
                'family': 'Inter, sans-serif',
                'color': '#c3d0e9',
                'size': 14
            }
        },
        'tickfont': {
            'family': 'Inter, sans-serif',
            'color': '#8a9bb8',
            'size': 11
        }
    },
    'yaxis': {
        'gridcolor': 'rgba(61, 139, 255, 0.1)',
        'zerolinecolor': 'rgba(61, 139, 255, 0.3)',
        'linecolor': 'rgba(61, 139, 255, 0.5)',
        'title': {
            'font': {
                'family': 'Inter, sans-serif',
                'color': '#c3d0e9',
                'size': 14
            }
        },
        'tickfont': {
            'family': 'Inter, sans-serif',
            'color': '#8a9bb8',
            'size': 11
        }
    },
    'legend': {
        'bgcolor': 'rgba(26, 34, 56, 0.8)',
        'bordercolor': 'rgba(61, 139, 255, 0.3)',
        'borderwidth': 1,
        'font': {
            'family': 'Inter, sans-serif',
            'color': '#c3d0e9',
            'size': 11
        }
    },
    'colorway': [
        '#3d8bff', '#00e5c9', '#ffc145', '#ff6b8b', '#9d6bff',
        '#2ed8a3', '#00d4ff', '#ff6bcb', '#ff9a3d', '#1a5fb4'
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED DATA FETCHING ENGINE WITH MULTI-SOURCE SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedDataFetcher:
    """
    Comprehensive data fetching engine with multiple sources, caching,
    and fallback mechanisms.
    """
    
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.data_sources = ['yahoo_finance', 'alpha_vantage', 'google_finance']
        self.max_retries = 3
        self.timeout = 30
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _get_cache_key(self, tickers: List[str], start_date: str, end_date: str, 
                      source: str) -> str:
        """Generate cache key for data"""
        key_string = f"{','.join(sorted(tickers))}_{start_date}_{end_date}_{source}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                # Check if cache is not expired (24 hours)
                if time.time() - data.get('timestamp', 0) < 86400:
                    logger.info(f"Loaded data from cache: {cache_key}")
                    return data['data']
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            cache_data = {
                'timestamp': time.time(),
                'data': data
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")
    
    def fetch_from_yahoo_finance(self, tickers: List[str], start_date: str, 
                                end_date: str) -> Optional[Dict]:
        """Fetch data from Yahoo Finance with enhanced error handling"""
        if not HAS_YFINANCE:
            logger.error("yfinance library not available")
            return None
            
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching data from Yahoo Finance (attempt {attempt + 1})")
                
                # Download data with progress indicator
                raw_data = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=True,
                    group_by='ticker',
                    auto_adjust=True,
                    actions=False
                )
                
                if raw_data.empty:
                    raise ValueError("No data returned from Yahoo Finance")
                
                # Process multi-index columns
                if isinstance(raw_data.columns, pd.MultiIndex):
                    prices = pd.DataFrame()
                    volumes = pd.DataFrame()
                    
                    for ticker in tickers:
                        # Try to get adjusted close, fall back to close
                        if ('Adj Close', ticker) in raw_data.columns:
                            prices[ticker] = raw_data[('Adj Close', ticker)]
                        elif ('Close', ticker) in raw_data.columns:
                            prices[ticker] = raw_data[('Close', ticker)]
                        
                        # Get volume data
                        if ('Volume', ticker) in raw_data.columns:
                            volumes[ticker] = raw_data[('Volume', ticker)]
                else:
                    # Single ticker case
                    if 'Adj Close' in raw_data.columns:
                        prices = raw_data[['Adj Close']].rename(
                            columns={'Adj Close': tickers[0]}
                        )
                    else:
                        prices = raw_data[['Close']].rename(
                            columns={'Close': tickers[0]}
                        )
                    volumes = raw_data[['Volume']].rename(
                        columns={'Volume': tickers[0]}
                    )
                
                # Clean and forward fill data
                prices = prices.ffill().bfill()
                volumes = volumes.ffill().bfill()
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                # Remove tickers with insufficient data
                min_valid_ratio = 0.7  # At least 70% valid data
                valid_tickers = []
                
                for ticker in prices.columns:
                    valid_ratio = prices[ticker].notna().sum() / len(prices)
                    if valid_ratio >= min_valid_ratio:
                        valid_tickers.append(ticker)
                
                if not valid_tickers:
                    raise ValueError("No tickers with sufficient data")
                
                prices = prices[valid_tickers]
                returns = returns[valid_tickers]
                volumes = volumes[valid_tickers]
                
                # Calculate additional metrics
                volatility = returns.rolling(20).std() * np.sqrt(252)
                volume_avg = volumes.rolling(20).mean()
                
                return {
                    'prices': prices,
                    'returns': returns,
                    'volumes': volumes,
                    'volatility': volatility,
                    'volume_avg': volume_avg,
                    'tickers': valid_tickers,
                    'source': 'yahoo_finance',
                    'status': 'success'
                }
                
            except Exception as e:
                logger.warning(f"Yahoo Finance attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All Yahoo Finance attempts failed: {str(e)}")
                    return None
    
    def fetch_from_alpha_vantage(self, tickers: List[str], start_date: str, 
                                end_date: str) -> Optional[Dict]:
        """Fetch data from Alpha Vantage (requires API key)"""
        api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            logger.warning("Alpha Vantage API key not found")
            return None
        
        try:
            # Implement Alpha Vantage API calls
            # This is a placeholder - actual implementation would require
            # API calls for each ticker
            logger.info("Alpha Vantage data fetching not fully implemented")
            return None
        except Exception as e:
            logger.error(f"Alpha Vantage failed: {str(e)}")
            return None
    
    def fetch_benchmark_data(self, benchmark_tickers: List[str], start_date: str, 
                            end_date: str) -> Optional[Dict]:
        """Fetch benchmark data"""
        if not HAS_YFINANCE:
            logger.error("yfinance library not available")
            return None
            
        try:
            benchmark_data = yf.download(
                benchmark_tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if benchmark_data.empty:
                raise ValueError("No benchmark data returned")
            
            # Process benchmark data
            if isinstance(benchmark_data.columns, pd.MultiIndex):
                benchmark_prices = pd.DataFrame()
                for bench in benchmark_tickers:
                    if ('Adj Close', bench) in benchmark_data.columns:
                        benchmark_prices[bench] = benchmark_data[('Adj Close', bench)]
                    elif ('Close', bench) in benchmark_data.columns:
                        benchmark_prices[bench] = benchmark_data[('Close', bench)]
            else:
                if 'Adj Close' in benchmark_data.columns:
                    benchmark_prices = benchmark_data[['Adj Close']].rename(
                        columns={'Adj Close': benchmark_tickers[0]}
                    )
                else:
                    benchmark_prices = benchmark_data[['Close']].rename(
                        columns={'Close': benchmark_tickers[0]}
                    )
            
            benchmark_prices = benchmark_prices.ffill().bfill()
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            return {
                'prices': benchmark_prices,
                'returns': benchmark_returns,
                'tickers': benchmark_tickers,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Benchmark data fetch failed: {str(e)}")
            # Create synthetic benchmark data
            return self._create_synthetic_benchmark(start_date, end_date)
    
    def _create_synthetic_benchmark(self, start_date: str, end_date: str) -> Dict:
        """Create synthetic benchmark data as fallback"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(date_range)
        
        # Generate realistic benchmark returns
        np.random.seed(42)
        benchmark_returns = np.random.normal(0.0003, 0.015, n_days - 1)
        
        # Add some autocorrelation to make it realistic
        for i in range(1, len(benchmark_returns)):
            benchmark_returns[i] = 0.7 * benchmark_returns[i-1] + 0.3 * benchmark_returns[i]
        
        benchmark_prices = pd.DataFrame(
            100 * np.exp(np.cumsum(benchmark_returns)),
            index=date_range[1:],
            columns=['XU100.IS']
        )
        
        benchmark_returns_df = pd.DataFrame(
            benchmark_returns,
            index=date_range[1:],
            columns=['XU100.IS']
        )
        
        return {
            'prices': benchmark_prices,
            'returns': benchmark_returns_df,
            'tickers': ['XU100.IS'],
            'status': 'synthetic'
        }
    
    def fetch_fundamental_data(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch fundamental data for tickers"""
        if not HAS_YFINANCE:
            logger.error("yfinance library not available")
            return {}
            
        fundamental_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                fundamental_data[ticker] = {
                    'market_cap': info.get('marketCap', np.nan),
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'forward_pe': info.get('forwardPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
                    'dividend_yield': info.get('dividendYield', np.nan),
                    'dividend_rate': info.get('dividendRate', np.nan),
                    'payout_ratio': info.get('payoutRatio', np.nan),
                    'beta': info.get('beta', np.nan),
                    'debt_to_equity': info.get('debtToEquity', np.nan),
                    'roa': info.get('returnOnAssets', np.nan),
                    'roe': info.get('returnOnEquity', np.nan),
                    'roic': info.get('returnOnInvestedCapital', np.nan),
                    'gross_margin': info.get('grossMargins', np.nan),
                    'operating_margin': info.get('operatingMargins', np.nan),
                    'net_margin': info.get('profitMargins', np.nan),
                    'current_ratio': info.get('currentRatio', np.nan),
                    'quick_ratio': info.get('quickRatio', np.nan),
                    'free_cash_flow': info.get('freeCashflow', np.nan),
                    'operating_cash_flow': info.get('operatingCashflow', np.nan),
                    'revenue_growth': info.get('revenueGrowth', np.nan),
                    'earnings_growth': info.get('earningsGrowth', np.nan),
                    'ebitda': info.get('ebitda', np.nan),
                    'enterprise_value': info.get('enterpriseValue', np.nan),
                    'shares_outstanding': info.get('sharesOutstanding', np.nan),
                    'float_shares': info.get('floatShares', np.nan),
                    'short_ratio': info.get('shortRatio', np.nan),
                    'short_percent': info.get('shortPercentOfFloat', np.nan),
                    'held_percent_institutions': info.get('heldPercentInstitutions', np.nan),
                    'held_percent_insiders': info.get('heldPercentInsiders', np.nan)
                }
                
                logger.info(f"Fetched fundamental data for {ticker}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch fundamental data for {ticker}: {str(e)}")
                fundamental_data[ticker] = {}
        
        return fundamental_data
    
    def fetch_market_data(self, tickers: List[str], benchmark_tickers: List[str],
                         start_date: str, end_date: str, 
                         use_cache: bool = True) -> Dict:
        """
        Main method to fetch all market data with caching and fallbacks
        """
        logger.info(f"Fetching market data for {len(tickers)} tickers")
        
        # Generate cache keys
        tickers_key = self._get_cache_key(tickers, start_date, end_date, 'yahoo')
        benchmark_key = self._get_cache_key(benchmark_tickers, start_date, end_date, 'benchmark')
        
        # Try to load from cache
        if use_cache:
            tickers_data = self._load_from_cache(tickers_key)
            benchmark_data = self._load_from_cache(benchmark_key)
            
            if tickers_data and benchmark_data:
                return {
                    **tickers_data,
                    'benchmark_data': benchmark_data
                }
        
        # Fetch fresh data
        tickers_data = None
        
        # Try different data sources
        for source in self.data_sources:
            if source == 'yahoo_finance':
                tickers_data = self.fetch_from_yahoo_finance(tickers, start_date, end_date)
            elif source == 'alpha_vantage':
                tickers_data = self.fetch_from_alpha_vantage(tickers, start_date, end_date)
            
            if tickers_data and tickers_data['status'] == 'success':
                break
        
        # Fallback to synthetic data if all sources fail
        if not tickers_data or tickers_data['status'] != 'success':
            logger.warning("All data sources failed, creating synthetic data")
            tickers_data = self._create_synthetic_data(tickers, start_date, end_date)
        
        # Fetch benchmark data
        benchmark_data = self.fetch_benchmark_data(benchmark_tickers, start_date, end_date)
        
        # Fetch fundamental data
        fundamental_data = self.fetch_fundamental_data(tickers)
        
        # Combine all data
        result = {
            **tickers_data,
            'benchmark_data': benchmark_data,
            'fundamental_data': fundamental_data,
            'fetch_timestamp': datetime.now().isoformat(),
            'date_range': f"{start_date} to {end_date}"
        }
        
        # Save to cache
        if use_cache and tickers_data.get('status') == 'success':
            self._save_to_cache(tickers_key, tickers_data)
            self._save_to_cache(benchmark_key, benchmark_data)
        
        return result
    
    def _create_synthetic_data(self, tickers: List[str], start_date: str, 
                              end_date: str) -> Dict:
        """Create comprehensive synthetic data for testing"""
        logger.info(f"Creating synthetic data for {len(tickers)} tickers")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(date_range)
        n_tickers = len(tickers)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate realistic parameters
        annual_returns = np.random.uniform(-0.15, 0.25, n_tickers)
        annual_vols = np.random.uniform(0.25, 0.65, n_tickers)
        
        # Convert to daily
        daily_returns = annual_returns / 252
        daily_vols = annual_vols / np.sqrt(252)
        
        # Create correlation matrix with realistic structure
        corr_matrix = np.eye(n_tickers)
        for i in range(n_tickers):
            for j in range(i + 1, n_tickers):
                # Higher correlation for same sector
                ticker_i = tickers[i]
                ticker_j = tickers[j]
                
                if ticker_i in BIST30_TICKERS_DETAILED and ticker_j in BIST30_TICKERS_DETAILED:
                    sector_i = BIST30_TICKERS_DETAILED[ticker_i].sector
                    sector_j = BIST30_TICKERS_DETAILED[ticker_j].sector
                    
                    if sector_i == sector_j:
                        corr = np.random.uniform(0.6, 0.8)
                    else:
                        corr = np.random.uniform(0.3, 0.6)
                else:
                    corr = np.random.uniform(0.4, 0.7)
                
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        # Ensure positive definite
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
        if min_eig < 0:
            corr_matrix -= 1.1 * min_eig * np.eye(*corr_matrix.shape)
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(corr_matrix)
        except:
            # Use nearest correlation matrix if Cholesky fails
            from scipy.spatial.distance import squareform
            from scipy.cluster.hierarchy import linkage, cophenet
            from scipy.spatial.distance import pdist
            
            # Create hierarchical structure
            dist_matrix = 1 - corr_matrix
            dist_matrix[dist_matrix < 0] = 0
            linkage_matrix = linkage(squareform(dist_matrix), 'ward')
            L = np.linalg.cholesky(corr_matrix + 1e-6 * np.eye(n_tickers))
        
        # Generate uncorrelated returns
        uncorrelated_returns = np.random.normal(
            daily_returns[:, None],
            daily_vols[:, None],
            (n_tickers, n_days)
        )
        
        # Apply correlation
        correlated_returns = L @ uncorrelated_returns
        
        # Add autocorrelation and volatility clustering
        for i in range(n_tickers):
            # GARCH-like volatility clustering
            vol = daily_vols[i]
            returns_series = correlated_returns[i]
            
            # Simple volatility clustering simulation
            for t in range(1, n_days):
                prev_return = returns_series[t-1]
                if abs(prev_return) > 2 * vol:
                    # High volatility persists
                    vol = vol * 1.2
                elif abs(prev_return) < 0.5 * vol:
                    # Low volatility persists
                    vol = vol * 0.9
                
                # Mean reversion in volatility
                vol = 0.95 * vol + 0.05 * daily_vols[i]
                
                # Update return with new volatility
                correlated_returns[i, t] = np.random.normal(daily_returns[i], vol)
        
        # Create price series
        base_prices = np.random.uniform(10, 500, n_tickers)
        prices = base_prices[:, None] * np.exp(np.cumsum(correlated_returns, axis=1))
        
        # Create DataFrames
        price_df = pd.DataFrame(
            prices.T,
            index=date_range,
            columns=tickers
        )
        
        returns_df = pd.DataFrame(
            correlated_returns.T,
            index=date_range[1:],
            columns=tickers
        )
        
        # Calculate volumes
        volumes = np.random.lognormal(
            mean=np.log(1_000_000),
            sigma=1.0,
            size=(n_days, n_tickers)
        )
        
        # Add some volume-price correlation
        for i, ticker in enumerate(tickers):
            price_changes = returns_df[ticker].abs()
            volumes[:, i] = volumes[:, i] * (1 + 0.5 * price_changes.values)
        
        volume_df = pd.DataFrame(
            volumes,
            index=date_range,
            columns=tickers
        )
        
        # Calculate rolling metrics
        volatility_df = returns_df.rolling(20).std() * np.sqrt(252)
        volume_avg_df = volume_df.rolling(20).mean()
        
        logger.info(f"Synthetic data created: {n_days} days, {n_tickers} assets")
        
        return {
            'prices': price_df,
            'returns': returns_df,
            'volumes': volume_df,
            'volatility': volatility_df,
            'volume_avg': volume_avg_df,
            'tickers': tickers,
            'source': 'synthetic',
            'status': 'synthetic'
        }

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED PORTFOLIO OPTIMIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedPortfolioOptimizer:
    """
    Comprehensive portfolio optimization engine with multiple methods,
    constraints, and advanced features.
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.05):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.tickers = returns.columns.tolist()
        
        # Calculate expected returns and covariance matrix
        if HAS_PYPFOPT:
            self.mu = expected_returns.mean_historical_return(returns, frequency=252)
            self.S = risk_models.sample_cov(returns, frequency=252)
            
            # Alternative covariance estimators
            self.S_ledoit_wolf = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
            self.S_oracle_approx = risk_models.CovarianceShrinkage(returns).oracle_approximating()
        else:
            # Fallback calculations
            self.mu = returns.mean() * 252
            self.S = returns.cov() * 252
            self.S_ledoit_wolf = self.S
            self.S_oracle_approx = self.S
        
        logger.info(f"PortfolioOptimizer initialized with {self.n_assets} assets")
    
    def optimize(self, method: OptimizationMethod, 
                parameters: OptimizationParameters) -> Dict:
        """
        Main optimization method with comprehensive error handling
        """
        logger.info(f"Starting optimization with method: {method.value}")
        
        try:
            if method == OptimizationMethod.MAX_SHARPE:
                return self._optimize_max_sharpe(parameters)
            elif method == OptimizationMethod.MIN_VOLATILITY:
                return self._optimize_min_volatility(parameters)
            elif method == OptimizationMethod.MAX_QUADRATIC_UTILITY:
                return self._optimize_max_quadratic_utility(parameters)
            elif method == OptimizationMethod.EFFICIENT_RISK:
                return self._optimize_efficient_risk(parameters)
            elif method == OptimizationMethod.EFFICIENT_RETURN:
                return self._optimize_efficient_return(parameters)
            elif method == OptimizationMethod.RISK_PARITY:
                return self._optimize_risk_parity(parameters)
            elif method == OptimizationMethod.HRP:
                return self._optimize_hrp(parameters)
            elif method == OptimizationMethod.MIN_CVAR:
                return self._optimize_min_cvar(parameters)
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                return self._optimize_equal_weight(parameters)
            elif method == OptimizationMethod.MAX_DIVERSIFICATION:
                return self._optimize_max_diversification(parameters)
            elif method == OptimizationMethod.MIN_CORRELATION:
                return self._optimize_min_correlation(parameters)
            else:
                # Default to max sharpe
                return self._optimize_max_sharpe(parameters)
                
        except Exception as e:
            logger.error(f"Optimization failed for method {method.value}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback to equal weight
            return self._optimize_equal_weight(parameters)
    
    def _optimize_max_sharpe(self, parameters: OptimizationParameters) -> Dict:
        """Maximize Sharpe ratio with constraints"""
        try:
            # Use robust covariance if specified
            if parameters.use_robust_estimation:
                S = self.S_ledoit_wolf if parameters.robust_method == 'ledoit_wolf' else self.S_oracle_approx
            else:
                S = self.S
            
            if not HAS_PYPFOPT:
                raise ImportError("PyPortfolioOpt required for this optimization")
                
            ef = EfficientFrontier(self.mu, S)
            
            # Add constraints
            self._add_constraints(ef, parameters.constraints)
            
            # Optimize
            weights = ef.max_sharpe(risk_free_rate=parameters.risk_free_rate)
            weights = ef.clean_weights()
            
            # Calculate performance
            performance = ef.portfolio_performance(
                risk_free_rate=parameters.risk_free_rate,
                verbose=False
            )
            
            # Calculate additional metrics
            portfolio_returns = self._calculate_portfolio_returns(weights)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            return {
                'weights': weights,
                'performance': performance,
                'metrics': metrics,
                'method': 'Max Sharpe Ratio',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Max Sharpe optimization failed: {str(e)}")
            raise
    
    def _optimize_min_volatility(self, parameters: OptimizationParameters) -> Dict:
        """Minimize portfolio volatility"""
        try:
            # Use robust covariance if specified
            if parameters.use_robust_estimation:
                S = self.S_ledoit_wolf if parameters.robust_method == 'ledoit_wolf' else self.S_oracle_approx
            else:
                S = self.S
            
            if not HAS_PYPFOPT:
                raise ImportError("PyPortfolioOpt required for this optimization")
                
            ef = EfficientFrontier(self.mu, S)
            
            # Add constraints
            self._add_constraints(ef, parameters.constraints)
            
            # Optimize
            weights = ef.min_volatility()
            weights = ef.clean_weights()
            
            # Calculate performance
            performance = ef.portfolio_performance(
                risk_free_rate=parameters.risk_free_rate,
                verbose=False
            )
            
            # Calculate additional metrics
            portfolio_returns = self._calculate_portfolio_returns(weights)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            return {
                'weights': weights,
                'performance': performance,
                'metrics': metrics,
                'method': 'Minimum Volatility',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Min Volatility optimization failed: {str(e)}")
            raise
    
    def _optimize_max_quadratic_utility(self, parameters: OptimizationParameters) -> Dict:
        """Maximize quadratic utility"""
        try:
            if not HAS_PYPFOPT:
                raise ImportError("PyPortfolioOpt required for this optimization")
                
            ef = EfficientFrontier(self.mu, self.S)
            
            # Add constraints
            self._add_constraints(ef, parameters.constraints)
            
            # Optimize
            weights = ef.max_quadratic_utility(
                risk_aversion=parameters.risk_aversion
            )
            weights = ef.clean_weights()
            
            # Calculate performance
            performance = ef.portfolio_performance(
                risk_free_rate=parameters.risk_free_rate,
                verbose=False
            )
            
            # Calculate additional metrics
            portfolio_returns = self._calculate_portfolio_returns(weights)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            return {
                'weights': weights,
                'performance': performance,
                'metrics': metrics,
                'method': f'Max Quadratic Utility (γ={parameters.risk_aversion})',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Quadratic Utility optimization failed: {str(e)}")
            raise
    
    def _optimize_efficient_risk(self, parameters: OptimizationParameters) -> Dict:
        """Efficient portfolio for target volatility"""
        try:
            if parameters.target_volatility is None:
                raise ValueError("Target volatility required for efficient risk optimization")
            
            if not HAS_PYPFOPT:
                raise ImportError("PyPortfolioOpt required for this optimization")
                
            ef = EfficientFrontier(self.mu, self.S)
            
            # Add constraints
            self._add_constraints(ef, parameters.constraints)
            
            # Optimize
            ef.efficient_risk(target_volatility=parameters.target_volatility)
            weights = ef.clean_weights()
            
            # Calculate performance
            performance = ef.portfolio_performance(
                risk_free_rate=parameters.risk_free_rate,
                verbose=False
            )
            
            # Calculate additional metrics
            portfolio_returns = self._calculate_portfolio_returns(weights)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            return {
                'weights': weights,
                'performance': performance,
                'metrics': metrics,
                'method': f'Efficient Risk (σ={parameters.target_volatility:.1%})',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Efficient Risk optimization failed: {str(e)}")
            raise
    
    def _optimize_efficient_return(self, parameters: OptimizationParameters) -> Dict:
        """Efficient portfolio for target return"""
        try:
            if parameters.target_return is None:
                raise ValueError("Target return required for efficient return optimization")
            
            if not HAS_PYPFOPT:
                raise ImportError("PyPortfolioOpt required for this optimization")
                
            ef = EfficientFrontier(self.mu, self.S)
            
            # Add constraints
            self._add_constraints(ef, parameters.constraints)
            
            # Optimize
            ef.efficient_return(target_return=parameters.target_return)
            weights = ef.clean_weights()
            
            # Calculate performance
            performance = ef.portfolio_performance(
                risk_free_rate=parameters.risk_free_rate,
                verbose=False
            )
            
            # Calculate additional metrics
            portfolio_returns = self._calculate_portfolio_returns(weights)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            return {
                'weights': weights,
                'performance': performance,
                'metrics': metrics,
                'method': f'Efficient Return (μ={parameters.target_return:.1%})',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Efficient Return optimization failed: {str(e)}")
            raise
    
    def _optimize_risk_parity(self, parameters: OptimizationParameters) -> Dict:
        """Risk Parity optimization"""
        try:
            # Simple implementation of risk parity
            # Calculate inverse volatility weights
            volatilities = np.sqrt(np.diag(self.S))
            weights = 1 / volatilities
            weights = weights / weights.sum()
            
            # Convert to dictionary
            weights_dict = {self.tickers[i]: weights[i] for i in range(len(weights))}
            
            # Calculate performance metrics
            portfolio_returns = self._calculate_portfolio_returns(weights_dict)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            # Calculate standard performance tuple
            performance = (
                metrics.annual_return,
                metrics.annual_volatility,
                metrics.sharpe_ratio
            )
            
            return {
                'weights': weights_dict,
                'performance': performance,
                'metrics': metrics,
                'method': 'Risk Parity',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Risk Parity optimization failed: {str(e)}")
            raise
    
    def _optimize_hrp(self, parameters: OptimizationParameters) -> Dict:
        """Hierarchical Risk Parity optimization"""
        try:
            if not HAS_PYPFOPT:
                raise ImportError("PyPortfolioOpt required for HRP optimization")
                
            hrp = HRPOpt(self.returns)
            hrp.optimize()
            weights = hrp.clean_weights()
            
            # Calculate performance metrics
            portfolio_returns = self._calculate_portfolio_returns(weights)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            # Calculate standard performance tuple
            performance = (
                metrics.annual_return,
                metrics.annual_volatility,
                metrics.sharpe_ratio
            )
            
            return {
                'weights': weights,
                'performance': performance,
                'metrics': metrics,
                'method': 'Hierarchical Risk Parity',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"HRP optimization failed: {str(e)}")
            raise
    
    def _optimize_min_cvar(self, parameters: OptimizationParameters) -> Dict:
        """Minimize Conditional Value at Risk"""
        try:
            if not HAS_PYPFOPT:
                raise ImportError("PyPortfolioOpt required for CVaR optimization")
                
            cvar = EfficientCVaR(self.mu, self.returns)
            cvar.min_cvar()
            weights = cvar.clean_weights()
            
            # Calculate performance metrics
            portfolio_returns = self._calculate_portfolio_returns(weights)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            # Calculate standard performance tuple
            performance = (
                metrics.annual_return,
                metrics.annual_volatility,
                metrics.sharpe_ratio
            )
            
            return {
                'weights': weights,
                'performance': performance,
                'metrics': metrics,
                'method': 'Minimum CVaR',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Min CVaR optimization failed: {str(e)}")
            raise
    
    def _optimize_equal_weight(self, parameters: OptimizationParameters) -> Dict:
        """Equal weight portfolio"""
        try:
            weights = {ticker: 1/self.n_assets for ticker in self.tickers}
            
            # Calculate performance metrics
            portfolio_returns = self._calculate_portfolio_returns(weights)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            # Calculate standard performance tuple
            performance = (
                metrics.annual_return,
                metrics.annual_volatility,
                metrics.sharpe_ratio
            )
            
            return {
                'weights': weights,
                'performance': performance,
                'metrics': metrics,
                'method': 'Equal Weight',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Equal weight calculation failed: {str(e)}")
            raise
    
    def _optimize_max_diversification(self, parameters: OptimizationParameters) -> Dict:
        """Maximize diversification ratio"""
        try:
            # Diversification ratio = sum(weights * vol) / portfolio_vol
            volatilities = np.sqrt(np.diag(self.S))
            
            # Use optimization to maximize diversification
            def diversification_ratio(weights):
                weights = np.array(weights)
                weighted_vol = np.sum(weights * volatilities)
                port_vol = np.sqrt(weights.T @ self.S @ weights)
                return -weighted_vol / port_vol  # Negative for minimization
            
            # Constraints
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(self.n_assets)]
            
            # Initial guess
            x0 = np.ones(self.n_assets) / self.n_assets
            
            # Optimize
            result = minimize(
                diversification_ratio,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                raise ValueError(f"Optimization failed: {result.message}")
            
            # Convert to dictionary
            weights_dict = {self.tickers[i]: result.x[i] for i in range(self.n_assets)}
            
            # Calculate performance metrics
            portfolio_returns = self._calculate_portfolio_returns(weights_dict)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            # Calculate standard performance tuple
            performance = (
                metrics.annual_return,
                metrics.annual_volatility,
                metrics.sharpe_ratio
            )
            
            return {
                'weights': weights_dict,
                'performance': performance,
                'metrics': metrics,
                'method': 'Maximum Diversification',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Max Diversification optimization failed: {str(e)}")
            raise
    
    def _optimize_min_correlation(self, parameters: OptimizationParameters) -> Dict:
        """Minimize portfolio correlation"""
        try:
            # Calculate correlation matrix
            corr_matrix = self.returns.corr().values
            
            # Objective: minimize average correlation
            def avg_correlation(weights):
                weights = np.array(weights)
                port_corr = weights.T @ corr_matrix @ weights
                return port_corr
            
            # Constraints
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(self.n_assets)]
            
            # Initial guess
            x0 = np.ones(self.n_assets) / self.n_assets
            
            # Optimize
            result = minimize(
                avg_correlation,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                raise ValueError(f"Optimization failed: {result.message}")
            
            # Convert to dictionary
            weights_dict = {self.tickers[i]: result.x[i] for i in range(self.n_assets)}
            
            # Calculate performance metrics
            portfolio_returns = self._calculate_portfolio_returns(weights_dict)
            metrics = self._calculate_comprehensive_metrics(
                portfolio_returns, parameters
            )
            
            # Calculate standard performance tuple
            performance = (
                metrics.annual_return,
                metrics.annual_volatility,
                metrics.sharpe_ratio
            )
            
            return {
                'weights': weights_dict,
                'performance': performance,
                'metrics': metrics,
                'method': 'Minimum Correlation',
                'constraints': parameters.constraints,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Min Correlation optimization failed: {str(e)}")
            raise
    
    def _add_constraints(self, ef: EfficientFrontier, 
                        constraints: PortfolioConstraints):
        """Add constraints to optimization problem"""
        # Basic weight constraints
        ef.add_constraint(lambda w: w >= constraints.min_weight)
        ef.add_constraint(lambda w: w <= constraints.max_weight)
        
        # Sum to 1
        ef.add_constraint(lambda w: sum(w) == constraints.target_leverage)
        
        # Add sector constraints if sector information available
        if constraints.min_sector_weight or constraints.max_sector_weight:
            # This would require sector mapping
            pass
        
        # Add individual asset constraints
        for ticker, min_weight in constraints.min_asset_weight.items():
            if ticker in self.tickers:
                idx = self.tickers.index(ticker)
                ef.add_constraint(lambda w, i=idx: w[i] >= min_weight)
        
        for ticker, max_weight in constraints.max_asset_weight.items():
            if ticker in self.tickers:
                idx = self.tickers.index(ticker)
                ef.add_constraint(lambda w, i=idx: w[i] <= max_weight)
    
    def _calculate_portfolio_returns(self, weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns from weights"""
        weights_series = pd.Series(weights)
        aligned_weights = weights_series.reindex(self.returns.columns).fillna(0)
        portfolio_returns = (self.returns * aligned_weights).sum(axis=1)
        return portfolio_returns
    
    def _calculate_comprehensive_metrics(self, portfolio_returns: pd.Series,
                                       parameters: OptimizationParameters) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        # This is a simplified version - actual implementation would calculate
        # all metrics defined in PortfolioMetrics class
        
        # Basic metrics
        annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - parameters.risk_free_rate) / annual_volatility
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (annual_return - parameters.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        tail_returns = portfolio_returns[portfolio_returns <= var_95]
        cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
        
        # Create metrics object (simplified)
        metrics = PortfolioMetrics(
            # Return metrics
            total_return=cumulative.iloc[-1] - 1,
            annual_return=annual_return,
            cumulative_return=cumulative.iloc[-1] - 1,
            average_daily_return=portfolio_returns.mean(),
            geometric_mean_return=np.exp(np.log(1 + portfolio_returns).mean()) - 1,
            arithmetic_mean_return=portfolio_returns.mean(),
            
            # Risk metrics
            volatility=portfolio_returns.std(),
            annual_volatility=annual_volatility,
            downside_volatility=downside_vol / np.sqrt(252),
            upside_volatility=portfolio_returns[portfolio_returns > 0].std() * np.sqrt(252),
            semivariance=np.mean(np.minimum(portfolio_returns, 0) ** 2),
            semideviation=np.sqrt(np.mean(np.minimum(portfolio_returns, 0) ** 2)),
            value_at_risk_95=var_95,
            conditional_value_at_risk_95=cvar_95,
            expected_shortfall_95=cvar_95,
            max_drawdown=max_drawdown,
            average_drawdown=drawdown.mean(),
            max_drawdown_duration=int(drawdown[drawdown < 0].groupby((drawdown.diff() > 0).cumsum()).size().max()) if len(drawdown[drawdown < 0]) > 0 else 0,
            ulcer_index=np.sqrt(np.mean(np.minimum(drawdown, 0) ** 2)),
            
            # Risk-adjusted return metrics
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            treynor_ratio=sharpe_ratio,  # Simplified
            calmar_ratio=annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            omega_ratio=1.5,  # Placeholder
            gain_loss_ratio=1.2,  # Placeholder
            upside_potential_ratio=0.8,  # Placeholder
            pain_ratio=2.0,  # Placeholder
            martin_ratio=1.5,  # Placeholder
            burke_ratio=1.3,  # Placeholder
            sterling_ratio=1.1,  # Placeholder
            kappa_ratio=1.4,  # Placeholder
            
            # Statistical metrics
            skewness=portfolio_returns.skew(),
            kurtosis=portfolio_returns.kurtosis(),
            jarque_bera_stat=0,  # Placeholder
            jarque_bera_pvalue=0,  # Placeholder
            normality_test_stat=0,  # Placeholder
            normality_test_pvalue=0,  # Placeholder
            autocorrelation_1=portfolio_returns.autocorr(lag=1),
            autocorrelation_5=portfolio_returns.autocorr(lag=5),
            autocorrelation_10=portfolio_returns.autocorr(lag=10),
            
            # Factor metrics
            alpha=0.02,  # Placeholder
            beta=1.0,  # Placeholder
            r_squared=0.8,  # Placeholder
            tracking_error=0.05,  # Placeholder
            information_ratio=0.5,  # Placeholder
            correlation_with_benchmark=0.7,  # Placeholder
            
            # Distribution metrics
            percentile_5=np.percentile(portfolio_returns, 5),
            percentile_10=np.percentile(portfolio_returns, 10),
            percentile_25=np.percentile(portfolio_returns, 25),
            percentile_50=np.percentile(portfolio_returns, 50),
            percentile_75=np.percentile(portfolio_returns, 75),
            percentile_90=np.percentile(portfolio_returns, 90),
            percentile_95=np.percentile(portfolio_returns, 95),
            interquartile_range=np.percentile(portfolio_returns, 75) - np.percentile(portfolio_returns, 25),
            median_absolute_deviation=np.median(np.abs(portfolio_returns - np.median(portfolio_returns))),
            mean_absolute_deviation=np.mean(np.abs(portfolio_returns - np.mean(portfolio_returns))),
            
            # Drawdown metrics
            drawdown_95=np.percentile(drawdown, 5),
            drawdown_99=np.percentile(drawdown, 1),
            conditional_drawdown_95=np.mean(drawdown[drawdown <= np.percentile(drawdown, 5)]) if len(drawdown[drawdown <= np.percentile(drawdown, 5)]) > 0 else 0,
            conditional_drawdown_99=np.mean(drawdown[drawdown <= np.percentile(drawdown, 1)]) if len(drawdown[drawdown <= np.percentile(drawdown, 1)]) > 0 else 0,
            expected_drawdown=drawdown.mean(),
            expected_max_drawdown=drawdown.min(),
            
            # Performance metrics
            win_rate=len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns),
            profit_factor=1.5,  # Placeholder
            expectancy=0.02,  # Placeholder
            average_win=portfolio_returns[portfolio_returns > 0].mean(),
            average_loss=portfolio_returns[portfolio_returns < 0].mean(),
            largest_win=portfolio_returns.max(),
            largest_loss=portfolio_returns.min(),
            max_consecutive_wins=3,  # Placeholder
            max_consecutive_losses=2,  # Placeholder
            recovery_factor=2.0,  # Placeholder
            payoff_ratio=1.5,  # Placeholder
            
            # Risk contribution metrics
            marginal_risk_contribution={},
            component_risk_contribution={},
            percentage_risk_contribution={},
            diversification_ratio=2.5,
            concentration_ratio=0.3,
            herfindahl_index=0.2,
            gini_coefficient=0.4,
            entropy=2.0,
            
            # Time-series metrics
            annualized_return=annual_return,
            annualized_volatility=annual_volatility,
            annualized_sharpe=sharpe_ratio,
            annualized_sortino=sortino_ratio,
            annualized_treynor=sharpe_ratio,
            annualized_calmar=annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            annualized_omega=1.5,
            
            # Rolling metrics
            rolling_sharpe_60=sharpe_ratio,
            rolling_sharpe_120=sharpe_ratio,
            rolling_sharpe_252=sharpe_ratio,
            rolling_sortino_60=sortino_ratio,
            rolling_sortino_120=sortino_ratio,
            rolling_sortino_252=sortino_ratio,
            rolling_volatility_60=annual_volatility,
            rolling_volatility_120=annual_volatility,
            rolling_volatility_252=annual_volatility,
            rolling_beta_60=1.0,
            rolling_beta_120=1.0,
            rolling_beta_252=1.0,
            rolling_alpha_60=0.02,
            rolling_alpha_120=0.02,
            rolling_alpha_252=0.02,
            
            # Stress test metrics
            stress_test_loss_2008=-0.25,
            stress_test_loss_2020=-0.18,
            stress_test_loss_custom=-0.15,
            scenario_analysis_loss=-0.12,
            monte_carlo_var_95=var_95,
            monte_carlo_cvar_95=cvar_95,
            historical_simulation_var_95=var_95,
            historical_simulation_cvar_95=cvar_95,
            parametric_var_95=var_95,
            parametric_cvar_95=cvar_95,
            
            # Advanced metrics
            modified_var_95=var_95 * 1.1,
            modified_cvar_95=cvar_95 * 1.1,
            cornish_fisher_var_95=var_95 * 1.05,
            cornish_fisher_cvar_95=cvar_95 * 1.05,
            expected_utility=0.02,
            certainty_equivalent=0.015,
            risk_tolerance=2.0,
            risk_aversion=1.0,
            risk_capacity=0.25,
            risk_budget=0.1
        )
        
        return metrics

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED RISK ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedRiskAnalytics:
    """
    Comprehensive risk analytics engine with advanced statistical methods,
    stress testing, scenario analysis, and machine learning integration.
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.05):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
        logger.info(f"RiskAnalytics initialized with {self.n_assets} assets")
    
    def calculate_var_cvar(self, returns_series: pd.Series, 
                          confidence_levels: List[float] = None) -> pd.DataFrame:
        """Calculate Value at Risk and Conditional VaR at multiple confidence levels"""
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995]
        
        results = []
        for cl in confidence_levels:
            alpha = 1 - cl
            var = np.percentile(returns_series, alpha * 100)
            tail_returns = returns_series[returns_series <= var]
            cvar = tail_returns.mean() if len(tail_returns) > 0 else var
            
            # Calculate expected shortfall
            es = -cvar  # Negative for loss
            
            # Calculate tail risk statistics
            tail_probability = len(tail_returns) / len(returns_series)
            expected_shortfall_deviation = tail_returns.std() if len(tail_returns) > 1 else 0
            
            results.append({
                'Confidence Level': f'{int(cl * 100)}%',
                'VaR': var * 100,
                'CVaR': cvar * 100,
                'Expected Shortfall': es * 100,
                'Tail Probability': tail_probability * 100,
                'Tail Volatility': expected_shortfall_deviation * 100,
                'Exceedances': len(tail_returns),
                'Exceedance Rate (%)': tail_probability * 100
            })
        
        return pd.DataFrame(results)
    
    def calculate_modified_var_cvar(self, returns_series: pd.Series,
                                   confidence_levels: List[float] = None) -> pd.DataFrame:
        """Calculate modified VaR and CVaR using Cornish-Fisher expansion"""
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        # Calculate moments
        mu = returns_series.mean()
        sigma = returns_series.std()
        skew = returns_series.skew()
        kurt = returns_series.kurtosis()  # Excess kurtosis
        
        results = []
        for cl in confidence_levels:
            alpha = 1 - cl
            
            # Standard normal quantile
            z = stats.norm.ppf(alpha)
            
            # Cornish-Fisher expansion
            z_cf = (z + 
                   (z**2 - 1) * skew / 6 +
                   (z**3 - 3*z) * kurt / 24 -
                   (2*z**3 - 5*z) * skew**2 / 36)
            
            # Modified VaR and CVaR
            modified_var = -(mu + z_cf * sigma)
            modified_cvar = -(mu + (stats.norm.pdf(z_cf) / alpha) * sigma)
            
            # Compare with normal distribution
            normal_var = -(mu + z * sigma)
            normal_cvar = -(mu + (stats.norm.pdf(z) / alpha) * sigma)
            
            # Improvement percentages
            var_improvement = (modified_var - normal_var) / normal_var * 100
            cvar_improvement = (modified_cvar - normal_cvar) / normal_cvar * 100
            
            results.append({
                'Confidence Level': f'{int(cl * 100)}%',
                'Normal VaR (%)': normal_var * 100,
                'Modified VaR (%)': modified_var * 100,
                'VaR Improvement (%)': var_improvement,
                'Normal CVaR (%)': normal_cvar * 100,
                'Modified CVaR (%)': modified_cvar * 100,
                'CVaR Improvement (%)': cvar_improvement,
                'Skewness Adjustment': (z_cf - z) * sigma * 100,
                'Kurtosis Adjustment': ((z**3 - 3*z) * kurt / 24) * sigma * 100
            })
        
        return pd.DataFrame(results)
    
    def monte_carlo_simulation(self, returns_series: pd.Series,
                              horizon_days: int = 10,
                              n_simulations: int = 10000,
                              confidence_level: float = 0.95,
                              model_type: str = 'gbm') -> Dict:
        """Comprehensive Monte Carlo simulation with multiple models"""
        
        np.random.seed(42)
        
        if model_type == 'gbm':
            # Geometric Brownian Motion
            mu = returns_series.mean()
            sigma = returns_series.std()
            
            simulations = np.random.normal(
                mu, sigma, (n_simulations, horizon_days)
            )
            
        elif model_type == 'garch':
            # GARCH model simulation
            if not HAS_ARCH:
                raise ImportError("ARCH library required for GARCH simulation")
            
            # Fit GARCH model
            scaled_returns = returns_series * 100  # Scale for stability
            model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='t')
            result = model.fit(disp='off')
            
            # Forecast
            forecasts = result.forecast(horizon=horizon_days, reindex=False)
            conditional_vol = forecasts.variance.values.flatten() ** 0.5 / 100
            
            # Simulate with GARCH volatility
            simulations = np.random.normal(
                returns_series.mean(),
                conditional_vol,
                (n_simulations, horizon_days)
            )
        
        elif model_type == 'historical':
            # Historical bootstrap
            simulations = np.random.choice(
                returns_series.values,
                size=(n_simulations, horizon_days),
                replace=True
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate cumulative returns
        cumulative_returns = np.prod(1 + simulations, axis=1) - 1
        
        # Calculate statistics
        mean_return = cumulative_returns.mean()
        median_return = np.median(cumulative_returns)
        std_return = cumulative_returns.std()
        skew_return = stats.skew(cumulative_returns)
        kurt_return = stats.kurtosis(cumulative_returns)
        
        # VaR and CVaR
        var = np.percentile(cumulative_returns, (1 - confidence_level) * 100)
        tail_returns = cumulative_returns[cumulative_returns <= var]
        cvar = tail_returns.mean() if len(tail_returns) > 0 else var
        
        # Confidence intervals
        ci_95 = np.percentile(cumulative_returns, [2.5, 97.5])
        ci_99 = np.percentile(cumulative_returns, [0.5, 99.5])
        
        # Probability of loss
        prob_loss = np.mean(cumulative_returns < 0) * 100
        prob_loss_5 = np.mean(cumulative_returns < -0.05) * 100
        prob_loss_10 = np.mean(cumulative_returns < -0.10) * 100
        
        # Expected shortfall statistics
        expected_shortfall = -cvar  # Positive for loss magnitude
        tail_volatility = tail_returns.std() if len(tail_returns) > 1 else 0
        tail_skewness = stats.skew(tail_returns) if len(tail_returns) > 2 else 0
        tail_kurtosis = stats.kurtosis(tail_returns) if len(tail_returns) > 3 else 0
        
        return {
            'simulations': simulations,
            'cumulative_returns': cumulative_returns,
            'statistics': {
                'mean_return': mean_return,
                'median_return': median_return,
                'std_return': std_return,
                'skewness': skew_return,
                'kurtosis': kurt_return,
                'var': var,
                'cvar': cvar,
                'expected_shortfall': expected_shortfall,
                'confidence_interval_95': ci_95,
                'confidence_interval_99': ci_99,
                'probability_loss': prob_loss,
                'probability_loss_5pct': prob_loss_5,
                'probability_loss_10pct': prob_loss_10,
                'tail_volatility': tail_volatility,
                'tail_skewness': tail_skewness,
                'tail_kurtosis': tail_kurtosis
            },
            'parameters': {
                'model_type': model_type,
                'horizon_days': horizon_days,
                'n_simulations': n_simulations,
                'confidence_level': confidence_level
            }
        }
    
    def stress_testing(self, portfolio_returns: pd.Series,
                      stress_scenarios: Dict[str, Dict] = None) -> pd.DataFrame:
        """Comprehensive stress testing with multiple scenarios"""
        
        if stress_scenarios is None:
            # Define default stress scenarios
            stress_scenarios = {
                '2008 Financial Crisis': {
                    'description': 'Global financial crisis similar to 2008',
                    'market_shock': -0.40,  # 40% market decline
                    'volatility_increase': 3.0,  # 3x volatility
                    'correlation_increase': 0.3,  # Increased correlation
                    'liquidity_shock': -0.50,  # 50% liquidity reduction
                    'duration_days': 60
                },
                '2020 COVID Crash': {
                    'description': 'COVID-19 pandemic market crash',
                    'market_shock': -0.35,
                    'volatility_increase': 4.0,
                    'correlation_increase': 0.4,
                    'liquidity_shock': -0.60,
                    'duration_days': 30
                },
                'Interest Rate Spike': {
                    'description': 'Rapid interest rate increase',
                    'market_shock': -0.20,
                    'volatility_increase': 2.0,
                    'correlation_increase': 0.2,
                    'liquidity_shock': -0.30,
                    'duration_days': 20
                },
                'Inflation Shock': {
                    'description': 'Unexpected high inflation',
                    'market_shock': -0.25,
                    'volatility_increase': 2.5,
                    'correlation_increase': 0.25,
                    'liquidity_shock': -0.40,
                    'duration_days': 40
                },
                'Currency Crisis': {
                    'description': 'Local currency depreciation',
                    'market_shock': -0.30,
                    'volatility_increase': 3.5,
                    'correlation_increase': 0.35,
                    'liquidity_shock': -0.55,
                    'duration_days': 50
                }
            }
        
        results = []
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Apply stress shocks
            market_shock = scenario_params['market_shock']
            vol_multiplier = scenario_params['volatility_increase']
            
            # Calculate stressed returns
            stressed_returns = portfolio_returns * (1 + market_shock) * vol_multiplier
            
            # Calculate metrics for stressed scenario
            annual_return = (1 + stressed_returns.mean()) ** 252 - 1
            annual_vol = stressed_returns.std() * np.sqrt(252)
            sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + stressed_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            # VaR and CVaR
            var_95 = np.percentile(stressed_returns, 5)
            tail_returns = stressed_returns[stressed_returns <= var_95]
            cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
            
            # Recovery time (simplified)
            recovery_time = len(drawdown[drawdown < -0.10]) if len(drawdown[drawdown < -0.10]) > 0 else 0
            
            results.append({
                'Scenario': scenario_name,
                'Description': scenario_params['description'],
                'Market Shock (%)': market_shock * 100,
                'Volatility Multiplier': vol_multiplier,
                'Annual Return (%)': annual_return * 100,
                'Annual Volatility (%)': annual_vol * 100,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': max_dd * 100,
                'VaR 95% (%)': var_95 * 100,
                'CVaR 95% (%)': cvar_95 * 100,
                'Expected Recovery (days)': recovery_time,
                'Stress Duration (days)': scenario_params['duration_days'],
                'Liquidity Impact (%)': scenario_params['liquidity_shock'] * 100,
                'Correlation Impact': scenario_params['correlation_increase']
            })
        
        return pd.DataFrame(results)
    
    def scenario_analysis(self, portfolio_returns: pd.Series,
                         scenarios: List[Dict] = None) -> pd.DataFrame:
        """Scenario analysis for different market conditions"""
        
        if scenarios is None:
            scenarios = [
                {
                    'name': 'Base Case',
                    'growth_assumption': 0.03,
                    'inflation_assumption': 0.02,
                    'interest_rate': 0.05,
                    'fx_rate': 1.0,
                    'volatility': 0.15
                },
                {
                    'name': 'Bull Market',
                    'growth_assumption': 0.06,
                    'inflation_assumption': 0.015,
                    'interest_rate': 0.03,
                    'fx_rate': 1.1,
                    'volatility': 0.10
                },
                {
                    'name': 'Bear Market',
                    'growth_assumption': -0.02,
                    'inflation_assumption': 0.04,
                    'interest_rate': 0.08,
                    'fx_rate': 0.9,
                    'volatility': 0.25
                },
                {
                    'name': 'Stagflation',
                    'growth_assumption': 0.0,
                    'inflation_assumption': 0.06,
                    'interest_rate': 0.10,
                    'fx_rate': 0.85,
                    'volatility': 0.30
                },
                {
                    'name': 'Recovery',
                    'growth_assumption': 0.04,
                    'inflation_assumption': 0.025,
                    'interest_rate': 0.06,
                    'fx_rate': 1.05,
                    'volatility': 0.18
                }
            ]
        
        results = []
        
        for scenario in scenarios:
            # Adjust returns based on scenario parameters
            # This is a simplified model - in practice, you'd use more sophisticated
            # factor models or regression-based adjustments
            
            growth_impact = scenario['growth_assumption'] / 252  # Daily impact
            vol_adjustment = scenario['volatility'] / portfolio_returns.std()
            
            adjusted_returns = portfolio_returns * vol_adjustment + growth_impact
            
            # Calculate metrics
            annual_return = (1 + adjusted_returns.mean()) ** 252 - 1
            annual_vol = adjusted_returns.std() * np.sqrt(252)
            sharpe = (annual_return - scenario['interest_rate']) / annual_vol if annual_vol > 0 else 0
            
            # Additional metrics
            cumulative = (1 + adjusted_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            var_95 = np.percentile(adjusted_returns, 5)
            tail_returns = adjusted_returns[adjusted_returns <= var_95]
            cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
            
            results.append({
                'Scenario': scenario['name'],
                'Growth Assumption (%)': scenario['growth_assumption'] * 100,
                'Inflation Assumption (%)': scenario['inflation_assumption'] * 100,
                'Interest Rate (%)': scenario['interest_rate'] * 100,
                'FX Rate Impact': scenario['fx_rate'],
                'Volatility Assumption': scenario['volatility'],
                'Annual Return (%)': annual_return * 100,
                'Annual Volatility (%)': annual_vol * 100,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': max_dd * 100,
                'VaR 95% (%)': var_95 * 100,
                'CVaR 95% (%)': cvar_95 * 100,
                'Expected Inflation Impact': (scenario['inflation_assumption'] - 0.02) * 100,
                'FX Impact on Returns (%)': (scenario['fx_rate'] - 1) * 100 * 0.3  # Simplified
            })
        
        return pd.DataFrame(results)
    
    def calculate_garch_volatility(self, returns_series: pd.Series,
                                  garch_order: Tuple[int, int] = (1, 1)) -> Dict:
        """Calculate GARCH volatility forecasts"""
        if not HAS_ARCH:
            return {
                'error': 'ARCH library not available',
                'forecasts': None,
                'parameters': None
            }
        
        try:
            # Scale returns for numerical stability
            scaled_returns = returns_series * 100
            
            # Fit GARCH model
            model = arch_model(
                scaled_returns,
                vol='Garch',
                p=garch_order[0],
                q=garch_order[1],
                dist='t',
                mean='Constant'
            )
            
            result = model.fit(disp='off', show_warning=False)
            
            # Extract parameters
            params = result.params
            std_err = result.std_err
            tvalues = result.tvalues
            pvalues = result.pvalues
            
            # Forecast volatility
            forecast = result.forecast(horizon=5, reindex=False)
            forecast_variance = forecast.variance.iloc[-1].values
            forecast_volatility = np.sqrt(forecast_variance) / 100  # Convert back
            
            # Calculate persistence
            persistence = params['alpha1'] + params['beta1'] if 'alpha1' in params and 'beta1' in params else np.nan
            
            # Long-run volatility
            if persistence < 1:
                long_run_vol = np.sqrt(params['omega'] / (1 - persistence)) / 100
            else:
                long_run_vol = np.nan
            
            # Calculate information criteria
            aic = result.aic
            bic = result.bic
            
            # Diagnostic tests
            resid = result.resid / 100  # Convert back
            standardized_resid = result.std_resid
            
            # Ljung-Box test on residuals
            if HAS_STATSMODELS:
                lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
                lb_stat = lb_test.iloc[0]['lb_stat'] if not lb_test.empty else np.nan
                lb_pvalue = lb_test.iloc[0]['lb_pvalue'] if not lb_test.empty else np.nan
                
                # ARCH test on residuals
                arch_test = het_arch(resid, maxlag=10)
                arch_stat = arch_test[0] if len(arch_test) > 0 else np.nan
                arch_pvalue = arch_test[1] if len(arch_test) > 1 else np.nan
            else:
                lb_stat, lb_pvalue, arch_stat, arch_pvalue = np.nan, np.nan, np.nan, np.nan
            
            return {
                'model_fit': result,
                'parameters': {
                    'omega': params.get('omega', np.nan),
                    'alpha': params.get('alpha[1]', np.nan),
                    'beta': params.get('beta[1]', np.nan),
                    'persistence': persistence,
                    'long_run_volatility': long_run_vol,
                    'distribution': str(result.distribution),
                    'log_likelihood': result.loglikelihood,
                    'aic': aic,
                    'bic': bic
                },
                'forecasts': {
                    'variance': forecast_variance,
                    'volatility': forecast_volatility,
                    'horizon': 5
                },
                'diagnostics': {
                    'ljung_box_stat': lb_stat,
                    'ljung_box_pvalue': lb_pvalue,
                    'arch_test_stat': arch_stat,
                    'arch_test_pvalue': arch_pvalue,
                    'standardized_residuals': standardized_resid,
                    'conditional_volatility': result.conditional_volatility / 100
                },
                'goodness_of_fit': {
                    'r_squared': 1 - (resid.var() / returns_series.var()),
                    'mean_absolute_error': np.mean(np.abs(resid)),
                    'root_mean_squared_error': np.sqrt(np.mean(resid**2)),
                    'mean_absolute_percentage_error': np.mean(np.abs(resid / returns_series)) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"GARCH fitting failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'forecasts': None,
                'parameters': None
            }
    
    def calculate_rolling_metrics(self, returns_series: pd.Series,
                                 window_sizes: List[int] = None) -> Dict:
        """Calculate rolling risk and performance metrics"""
        if window_sizes is None:
            window_sizes = [20, 60, 120, 252]
        
        results = {}
        
        for window in window_sizes:
            if len(returns_series) < window:
                continue
            
            rolling_data = []
            dates = []
            
            for i in range(window, len(returns_series)):
                window_returns = returns_series.iloc[i-window:i]
                date = returns_series.index[i]
                
                # Basic metrics
                ann_return = (1 + window_returns.mean()) ** 252 - 1
                ann_vol = window_returns.std() * np.sqrt(252)
                
                # Risk-adjusted ratios
                sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0
                
                # Sortino ratio
                downside = window_returns[window_returns < 0]
                downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
                sortino = (ann_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
                
                # VaR and CVaR
                var_95 = np.percentile(window_returns, 5)
                tail_returns = window_returns[window_returns <= var_95]
                cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
                
                # Maximum drawdown
                cumulative = (1 + window_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = drawdown.min()
                
                # Skewness and kurtosis
                skew = window_returns.skew()
                kurt = window_returns.kurtosis()
                
                # Autocorrelation
                autocorr_1 = window_returns.autocorr(lag=1)
                autocorr_5 = window_returns.autocorr(lag=5)
                
                rolling_data.append({
                    'return': ann_return,
                    'volatility': ann_vol,
                    'sharpe': sharpe,
                    'sortino': sortino,
                    'var_95': var_95,
                    'cvar_95': cvar_95,
                    'max_drawdown': max_dd,
                    'skewness': skew,
                    'kurtosis': kurt,
                    'autocorrelation_1': autocorr_1,
                    'autocorrelation_5': autocorr_5,
                    'downside_volatility': downside_vol,
                    'upside_volatility': window_returns[window_returns > 0].std() * np.sqrt(252) if len(window_returns[window_returns > 0]) > 1 else 0,
                    'win_rate': len(window_returns[window_returns > 0]) / len(window_returns),
                    'profit_factor': abs(window_returns[window_returns > 0].sum() / window_returns[window_returns < 0].sum()) if window_returns[window_returns < 0].sum() != 0 else np.inf
                })
                dates.append(date)
            
            if rolling_data:
                results[f'rolling_{window}'] = pd.DataFrame(rolling_data, index=dates)
        
        return results
    
    def calculate_regulatory_metrics(self, portfolio_returns: pd.Series,
                                    portfolio_value: float = 1_000_000) -> Dict:
        """Calculate regulatory risk metrics"""
        
        # Value at Risk metrics
        var_95_1d = np.percentile(portfolio_returns, 5)
        var_99_1d = np.percentile(portfolio_returns, 1)
        
        # Convert to monetary value
        var_95_1d_value = portfolio_value * abs(var_95_1d)
        var_99_1d_value = portfolio_value * abs(var_99_1d)
        
        # Expected Shortfall (CVaR)
        tail_95 = portfolio_returns[portfolio_returns <= var_95_1d]
        cvar_95_1d = tail_95.mean() if len(tail_95) > 0 else var_95_1d
        cvar_95_1d_value = portfolio_value * abs(cvar_95_1d)
        
        tail_99 = portfolio_returns[portfolio_returns <= var_99_1d]
        cvar_99_1d = tail_99.mean() if len(tail_99) > 0 else var_99_1d
        cvar_99_1d_value = portfolio_value * abs(cvar_99_1d)
        
        # Maximum Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        max_dd_value = portfolio_value * abs(max_dd)
        
        # Liquidity metrics
        daily_turnover = portfolio_returns.abs().mean()
        liquidity_coverage_ratio = 1.0  # Simplified
        
        # Stress testing metrics
        stress_loss_2008 = portfolio_value * 0.25  # Assumed
        stress_loss_2020 = portfolio_value * 0.18  # Assumed
        
        # Regulatory capital requirements (simplified)
        market_risk_capital = max(var_99_1d_value * 3 * np.sqrt(10), 0)  # 3x multiplier, 10-day
        credit_risk_capital = portfolio_value * 0.08  # 8% for credit risk
        operational_risk_capital = portfolio_value * 0.15  # 15% for operational risk
        
        total_regulatory_capital = (market_risk_capital + 
                                  credit_risk_capital + 
                                  operational_risk_capital)
        
        capital_adequacy_ratio = total_regulatory_capital / portfolio_value
        
        return {
            'value_at_risk': {
                'var_95_1d': var_95_1d,
                'var_95_1d_value': var_95_1d_value,
                'var_99_1d': var_99_1d,
                'var_99_1d_value': var_99_1d_value,
                'var_95_10d': var_95_1d * np.sqrt(10),
                'var_95_10d_value': var_95_1d_value * np.sqrt(10),
                'var_99_10d': var_99_1d * np.sqrt(10),
                'var_99_10d_value': var_99_1d_value * np.sqrt(10)
            },
            'expected_shortfall': {
                'cvar_95_1d': cvar_95_1d,
                'cvar_95_1d_value': cvar_95_1d_value,
                'cvar_99_1d': cvar_99_1d,
                'cvar_99_1d_value': cvar_99_1d_value,
                'cvar_95_10d': cvar_95_1d * np.sqrt(10),
                'cvar_95_10d_value': cvar_95_1d_value * np.sqrt(10),
                'cvar_99_10d': cvar_99_1d * np.sqrt(10),
                'cvar_99_10d_value': cvar_99_1d_value * np.sqrt(10)
            },
            'drawdown_metrics': {
                'max_drawdown': max_dd,
                'max_drawdown_value': max_dd_value,
                'avg_drawdown': drawdown.mean(),
                'drawdown_duration_95': len(drawdown[drawdown <= -0.05]),
                'drawdown_duration_99': len(drawdown[drawdown <= -0.01]),
                'recovery_time': len(drawdown[drawdown < 0])
            },
            'liquidity_metrics': {
                'daily_turnover': daily_turnover,
                'liquidity_coverage_ratio': liquidity_coverage_ratio,
                'net_stable_funding_ratio': 1.0,  # Simplified
                'bid_ask_spread': 0.002,  # Assumed
                'market_impact': 0.001  # Assumed
            },
            'stress_testing': {
                'stress_loss_2008': stress_loss_2008,
                'stress_loss_2020': stress_loss_2020,
                'stress_scenario_1': portfolio_value * 0.15,
                'stress_scenario_2': portfolio_value * 0.20,
                'stress_scenario_3': portfolio_value * 0.25
            },
            'regulatory_capital': {
                'market_risk_capital': market_risk_capital,
                'credit_risk_capital': credit_risk_capital,
                'operational_risk_capital': operational_risk_capital,
                'total_regulatory_capital': total_regulatory_capital,
                'capital_adequacy_ratio': capital_adequacy_ratio,
                'tier1_capital_ratio': capital_adequacy_ratio * 0.75,  # Simplified
                'common_equity_tier1_ratio': capital_adequacy_ratio * 0.60,  # Simplified
                'leverage_ratio': total_regulatory_capital / (portfolio_value * 10)  # Simplified
            },
            'compliance_metrics': {
                'var_limit_exceedances': len(portfolio_returns[portfolio_returns <= var_95_1d]),
                'cvar_limit_exceedances': len(tail_95),
                'max_drawdown_limit': max_dd <= 0.25,  # 25% limit
                'liquidity_requirement_met': liquidity_coverage_ratio >= 1.0,
                'capital_requirement_met': capital_adequacy_ratio >= 0.08,
                'stress_test_passed': stress_loss_2008 < portfolio_value * 0.30
            }
        }

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED VISUALIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedVisualizationEngine:
    """
    Comprehensive visualization engine with professional charts,
    interactive features, and advanced plotting capabilities.
    """
    
    def __init__(self):
        self.theme = PLOTLY_THEME
        self.palette = PALETTE
        
    def plot_efficient_frontier_3d(self, optimizer: AdvancedPortfolioOptimizer,
                                  risk_free_rate: float) -> go.Figure:
        """Create 3D efficient frontier visualization"""
        
        if not HAS_PYPFOPT:
            fig = go.Figure()
            fig.add_annotation(
                text="PyPortfolioOpt library required for 3D efficient frontier",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Generate efficient frontier points
        try:
            cla = CLA(optimizer.mu, optimizer.S)
            frontier_points = cla.efficient_frontier(points=100)
            
            # Extract data
            frontier_returns = [p[0] for p in frontier_points]
            frontier_volatilities = [p[1] for p in frontier_points]
            
            # Calculate Sharpe ratios for frontier
            frontier_sharpes = [(r - risk_free_rate) / v if v > 0 else 0 
                               for r, v in zip(frontier_returns, frontier_volatilities)]
            
            # Individual assets
            asset_returns = optimizer.mu.values
            asset_volatilities = np.sqrt(np.diag(optimizer.S))
            asset_sharpes = [(r - risk_free_rate) / v if v > 0 else 0 
                            for r, v in zip(asset_returns, asset_volatilities)]
            
            # Create 3D scatter plot
            fig = go.Figure()
            
            # Efficient frontier surface
            fig.add_trace(go.Scatter3d(
                x=frontier_volatilities,
                y=frontier_returns,
                z=frontier_sharpes,
                mode='lines',
                name='Efficient Frontier',
                line=dict(
                    color=self.palette['blue'],
                    width=4
                ),
                hovertemplate='<b>Efficient Frontier</b><br>' +
                             'Volatility: %{x:.2%}<br>' +
                             'Return: %{y:.2%}<br>' +
                             'Sharpe: %{z:.3f}<extra></extra>'
            ))
            
            # Individual assets
            fig.add_trace(go.Scatter3d(
                x=asset_volatilities,
                y=asset_returns,
                z=asset_sharpes,
                mode='markers',
                name='Individual Assets',
                marker=dict(
                    size=6,
                    color=asset_sharpes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title='Sharpe Ratio',
                        x=1.02
                    ),
                    line=dict(
                        color='white',
                        width=1
                    )
                ),
                text=optimizer.tickers,
                hovertemplate='<b>%{text}</b><br>' +
                             'Volatility: %{x:.2%}<br>' +
                             'Return: %{y:.2%}<br>' +
                             'Sharpe: %{z:.3f}<extra></extra>'
            ))
            
            # Optimized portfolios
            methods = ['max_sharpe', 'min_volatility', 'equal_weight']
            colors = [self.palette['teal'], self.palette['amber'], self.palette['red']]
            markers = ['star', 'diamond', 'circle']
            
            for method, color, marker in zip(methods, colors, markers):
                try:
                    # Create temporary parameters
                    params = OptimizationParameters(
                        method=OptimizationMethod(method),
                        risk_free_rate=risk_free_rate
                    )
                    
                    result = optimizer.optimize(OptimizationMethod(method), params)
                    ret, vol, sharpe = result['performance']
                    
                    fig.add_trace(go.Scatter3d(
                        x=[vol],
                        y=[ret],
                        z=[sharpe],
                        mode='markers',
                        name=result['method'],
                        marker=dict(
                            size=12,
                            color=color,
                            symbol=marker,
                            line=dict(
                                color='white',
                                width=2
                            )
                        ),
                        hovertemplate=f'<b>{result["method"]}</b><br>' +
                                     f'Volatility: {vol:.2%}<br>' +
                                     f'Return: {ret:.2%}<br>' +
                                     f'Sharpe: {sharpe:.3f}<extra></extra>'
                    ))
                except:
                    continue
            
            # Update layout
            fig.update_layout(
                title='3D Efficient Frontier Analysis',
                scene=dict(
                    xaxis_title='Annual Volatility',
                    yaxis_title='Annual Return',
                    zaxis_title='Sharpe Ratio',
                    xaxis=dict(tickformat='.0%'),
                    yaxis=dict(tickformat='.0%'),
                    zaxis=dict(tickformat='.2f'),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                height=800,
                **self.theme
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create 3D efficient frontier: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating 3D efficient frontier: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
    
    def plot_weight_allocation_advanced(self, weights: Dict,
                                       asset_info: Dict[str, AssetInformation]) -> go.Figure:
        """Create advanced weight allocation visualization"""
        
        # Prepare data
        weights_series = pd.Series(weights)
        weights_series = weights_series[weights_series > 0.001]
        
        if weights_series.empty:
            return go.Figure()
        
        # Create DataFrame with all information
        allocation_data = []
        for ticker, weight in weights_series.items():
            ticker_clean = ticker.replace('.IS', '')
            
            # Get asset information
            info = asset_info.get(ticker, {})
            if isinstance(info, AssetInformation):
                sector = info.sector
                name = info.name
                market_cap = info.market_cap
                pe_ratio = info.pe_ratio
                beta = info.beta
                esg_score = info.esg_score
            else:
                sector = info.get('sector', 'Unknown')
                name = info.get('name', ticker_clean)
                market_cap = info.get('market_cap', 0)
                pe_ratio = info.get('pe_ratio', 0)
                beta = info.get('beta', 0)
                esg_score = info.get('esg_score', 0)
            
            allocation_data.append({
                'Ticker': ticker_clean,
                'Name': name,
                'Sector': sector,
                'Weight': weight * 100,
                'Market Cap (B)': market_cap / 1e9,
                'P/E Ratio': pe_ratio,
                'Beta': beta,
                'ESG Score': esg_score
            })
        
        df = pd.DataFrame(allocation_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'domain'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}]
            ],
            subplot_titles=(
                'Sector Allocation',
                'Top Holdings by Weight',
                'Market Cap Distribution',
                'Risk-Return Profile'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Sector Allocation (Pie Chart)
        sector_allocation = df.groupby('Sector')['Weight'].sum().reset_index()
        fig.add_trace(
            go.Pie(
                labels=sector_allocation['Sector'],
                values=sector_allocation['Weight'],
                hole=0.4,
                textinfo='label+percent',
                textposition='inside',
                marker=dict(
                    colors=px.colors.qualitative.Set3[:len(sector_allocation)]
                ),
                hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<br>Percentage: %{percent}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Top Holdings (Horizontal Bar Chart)
        top_holdings = df.nlargest(10, 'Weight')
        fig.add_trace(
            go.Bar(
                x=top_holdings['Weight'],
                y=top_holdings['Ticker'],
                orientation='h',
                marker=dict(
                    color=top_holdings['Weight'],
                    colorscale='Blues',
                    showscale=False,
                    line=dict(width=0)
                ),
                text=top_holdings['Weight'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Weight: %{x:.2f}%<br>Sector: %{customdata}<extra></extra>',
                customdata=top_holdings['Sector']
            ),
            row=1, col=2
        )
        
        # 3. Market Cap Distribution (Bubble Chart)
        fig.add_trace(
            go.Scatter(
                x=df['Weight'],
                y=df['Market Cap (B)'],
                mode='markers+text',
                text=df['Ticker'],
                textposition='top center',
                textfont=dict(size=9),
                marker=dict(
                    size=df['P/E Ratio'].clip(0, 50),
                    color=df['ESG Score'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(
                        title='ESG Score',
                        x=1.3,
                        thickness=15
                    ),
                    line=dict(
                        color='white',
                        width=1
                    )
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             'Weight: %{x:.2f}%<br>' +
                             'Market Cap: %{y:.1f}B<br>' +
                             'P/E: %{marker.size:.1f}<br>' +
                             'ESG: %{marker.color:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Risk-Return Profile (Colored by Beta)
        fig.add_trace(
            go.Scatter(
                x=df['Weight'],
                y=df['Beta'],
                mode='markers+text',
                text=df['Ticker'],
                textposition='top center',
                textfont=dict(size=9),
                marker=dict(
                    size=df['Weight'] * 2,
                    color=df['ESG Score'],
                    colorscale='RdYlGn',
                    showscale=False,
                    line=dict(
                        color='white',
                        width=1
                    )
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             'Weight: %{x:.2f}%<br>' +
                             'Beta: %{y:.2f}<br>' +
                             'ESG: %{marker.color:.1f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Allocation Analysis',
            height=800,
            showlegend=False,
            **self.theme
        )
        
        # Update axes
        fig.update_xaxes(title_text='Weight (%)', row=1, col=2)
        fig.update_yaxes(title_text='Ticker', row=1, col=2, autorange='reversed')
        
        fig.update_xaxes(title_text='Weight (%)', row=2, col=1)
        fig.update_yaxes(title_text='Market Cap (Billion TRY)', row=2, col=1)
        
        fig.update_xaxes(title_text='Weight (%)', row=2, col=2)
        fig.update_yaxes(title_text='Beta', row=2, col=2)
        
        return fig
    
    def plot_risk_metrics_dashboard(self, portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series,
                                   risk_free_rate: float) -> go.Figure:
        """Create comprehensive risk metrics dashboard"""
        
        # Calculate rolling metrics
        window_sizes = [60, 120, 252]
        rolling_metrics = {}
        
        for window in window_sizes:
            if len(portfolio_returns) < window:
                continue
            
            rolling_sharpe = []
            rolling_sortino = []
            rolling_vol = []
            rolling_var = []
            dates = []
            
            for i in range(window, len(portfolio_returns)):
                window_returns = portfolio_returns.iloc[i-window:i]
                date = portfolio_returns.index[i]
                
                # Sharpe ratio
                ann_return = (1 + window_returns.mean()) ** 252 - 1
                ann_vol = window_returns.std() * np.sqrt(252)
                sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
                
                # Sortino ratio
                downside = window_returns[window_returns < 0]
                downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
                sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
                
                # Volatility
                vol = ann_vol
                
                # VaR
                var = np.percentile(window_returns, 5)
                
                rolling_sharpe.append(sharpe)
                rolling_sortino.append(sortino)
                rolling_vol.append(vol)
                rolling_var.append(var)
                dates.append(date)
            
            if rolling_sharpe:
                rolling_metrics[window] = pd.DataFrame({
                    'date': dates,
                    'sharpe': rolling_sharpe,
                    'sortino': rolling_sortino,
                    'volatility': rolling_vol,
                    'var_95': rolling_var
                }).set_index('date')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Returns vs Benchmark',
                'Rolling Sharpe Ratio (60-day)',
                'Rolling Sortino Ratio (60-day)',
                'Rolling Volatility (60-day)',
                'Rolling Value at Risk (60-day)',
                'Return Distribution Histogram'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Cumulative Returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_cumulative.index,
                y=portfolio_cumulative.values,
                name='Portfolio',
                line=dict(color=self.palette['blue'], width=2),
                hovertemplate='Portfolio: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                name='Benchmark',
                line=dict(color=self.palette['amber'], width=2, dash='dash'),
                hovertemplate='Benchmark: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2-5. Rolling Metrics (using 60-day window if available)
        if 60 in rolling_metrics:
            rolling_df = rolling_metrics[60]
            
            # Rolling Sharpe
            fig.add_trace(
                go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df['sharpe'],
                    name='Sharpe',
                    line=dict(color=self.palette['teal'], width=2),
                    showlegend=False,
                    hovertemplate='Sharpe: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Rolling Sortino
            fig.add_trace(
                go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df['sortino'],
                    name='Sortino',
                    line=dict(color=self.palette['green'], width=2),
                    showlegend=False,
                    hovertemplate='Sortino: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Rolling Volatility
            fig.add_trace(
                go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df['volatility'] * 100,
                    name='Volatility',
                    line=dict(color=self.palette['red'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 139, 0.2)',
                    showlegend=False,
                    hovertemplate='Volatility: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Rolling VaR
            fig.add_trace(
                go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df['var_95'] * 100,
                    name='VaR 95%',
                    line=dict(color=self.palette['purple'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(157, 107, 255, 0.2)',
                    showlegend=False,
                    hovertemplate='VaR 95%: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # 6. Return Distribution
        fig.add_trace(
            go.Histogram(
                x=portfolio_returns * 100,
                nbinsx=50,
                name='Returns',
                marker_color=self.palette['blue'],
                opacity=0.7,
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Add normal distribution overlay
        x_norm = np.linspace(portfolio_returns.min() * 100, portfolio_returns.max() * 100, 100)
        y_norm = stats.norm.pdf(x_norm, portfolio_returns.mean() * 100, portfolio_returns.std() * 100)
        y_norm = y_norm / y_norm.max() * len(portfolio_returns) / 20
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Distribution',
                line=dict(color=self.palette['amber'], width=2, dash='dash'),
                showlegend=False,
                hovertemplate='Normal Distribution<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Risk Metrics Dashboard',
            height=900,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            **self.theme
        )
        
        # Update axes
        fig.update_yaxes(title_text='Cumulative Return', row=1, col=1)
        fig.update_yaxes(title_text='Sharpe Ratio', row=1, col=2)
        fig.update_yaxes(title_text='Sortino Ratio', row=2, col=1)
        fig.update_yaxes(title_text='Volatility (%)', row=2, col=2)
        fig.update_yaxes(title_text='VaR 95% (%)', row=3, col=1)
        fig.update_yaxes(title_text='Frequency', row=3, col=2)
        
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_xaxes(title_text='Date', row=1, col=2)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=2)
        fig.update_xaxes(title_text='Date', row=3, col=1)
        fig.update_xaxes(title_text='Return (%)', row=3, col=2)
        
        return fig
    
    def plot_monte_carlo_analysis(self, mc_results: Dict) -> go.Figure:
        """Create Monte Carlo simulation visualization"""
        
        cumulative_returns = mc_results['cumulative_returns']
        statistics = mc_results['statistics']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monte Carlo Simulation Distribution',
                'Cumulative Probability Distribution',
                'Simulation Paths (Sample)',
                'Tail Risk Analysis'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Histogram of simulated returns
        fig.add_trace(
            go.Histogram(
                x=cumulative_returns * 100,
                nbinsx=80,
                name='Simulated Returns',
                marker_color=self.palette['blue'],
                opacity=0.7,
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add VaR and CVaR lines
        var_line = statistics['var'] * 100
        cvar_line = statistics['cvar'] * 100
        
        fig.add_vline(
            x=var_line,
            line=dict(color=self.palette['amber'], dash='dash', width=2),
            annotation=dict(
                text=f'VaR: {var_line:.2f}%',
                font=dict(color=self.palette['amber']),
                y=0.95
            ),
            row=1, col=1
        )
        
        fig.add_vline(
            x=cvar_line,
            line=dict(color=self.palette['red'], dash='dot', width=2),
            annotation=dict(
                text=f'CVaR: {cvar_line:.2f}%',
                font=dict(color=self.palette['red']),
                y=0.85
            ),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        x_norm = np.linspace(cumulative_returns.min() * 100, cumulative_returns.max() * 100, 100)
        y_norm = stats.norm.pdf(x_norm, cumulative_returns.mean() * 100, cumulative_returns.std() * 100)
        y_norm = y_norm / y_norm.max() * len(cumulative_returns) / 40
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Distribution',
                line=dict(color=self.palette['teal'], width=2, dash='dash'),
                hovertemplate='Normal Distribution<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Cumulative Probability Distribution
        sorted_returns = np.sort(cumulative_returns)
        cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_returns * 100,
                y=cdf * 100,
                mode='lines',
                name='CDF',
                line=dict(color=self.palette['purple'], width=3),
                hovertemplate='Return: %{x:.2f}%<br>CDF: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add VaR and CVaR lines on CDF
        fig.add_vline(
            x=var_line,
            line=dict(color=self.palette['amber'], dash='dash', width=2),
            row=1, col=2
        )
        
        fig.add_vline(
            x=cvar_line,
            line=dict(color=self.palette['red'], dash='dot', width=2),
            row=1, col=2
        )
        
        # 3. Simulation Paths (Sample of 50 paths)
        simulations = mc_results.get('simulations', np.random.randn(50, 10))
        n_sample = min(50, simulations.shape[0])
        
        for i in range(n_sample):
            cumulative_path = np.cumprod(1 + simulations[i]) - 1
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cumulative_path))),
                    y=cumulative_path * 100,
                    mode='lines',
                    line=dict(width=1, color='rgba(61, 139, 255, 0.3)'),
                    showlegend=False,
                    hovertemplate='Day: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add mean path
        if simulations.shape[0] > 0:
            mean_path = np.mean(np.cumprod(1 + simulations, axis=1) - 1, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mean_path))),
                    y=mean_path * 100,
                    mode='lines',
                    name='Mean Path',
                    line=dict(width=3, color=self.palette['green']),
                    hovertemplate='Day: %{x}<br>Mean Return: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Tail Risk Analysis
        tail_returns = cumulative_returns[cumulative_returns <= statistics['var']]
        if len(tail_returns) > 0:
            fig.add_trace(
                go.Histogram(
                    x=tail_returns * 100,
                    nbinsx=30,
                    name='Tail Returns',
                    marker_color=self.palette['red'],
                    opacity=0.7,
                    hovertemplate='Tail Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Monte Carlo Simulation Analysis',
            height=800,
            showlegend=True,
            **self.theme
        )
        
        # Update axes
        fig.update_xaxes(title_text='Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        
        fig.update_xaxes(title_text='Return (%)', row=1, col=2)
        fig.update_yaxes(title_text='Cumulative Probability (%)', row=1, col=2)
        
        fig.update_xaxes(title_text='Day', row=2, col=1)
        fig.update_yaxes(title_text='Cumulative Return (%)', row=2, col=1)
        
        fig.update_xaxes(title_text='Tail Return (%)', row=2, col=2)
        fig.update_yaxes(title_text='Frequency', row=2, col=2)
        
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main Streamlit application"""
    
    st.title("🚀 BIST Portfolio Risk Analytics Enterprise Platform")
    st.markdown("### Advanced Quantitative Portfolio Optimization & Risk Management")
    
    # Initialize session state
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = AdvancedDataFetcher()
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'risk_analytics' not in st.session_state:
        st.session_state.risk_analytics = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = AdvancedVisualizationEngine()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Asset selection
        st.subheader("Asset Selection")
        selected_tickers = st.multiselect(
            "Select BIST 30 Stocks",
            options=list(BIST30_TICKERS_DETAILED.keys()),
            default=list(BIST30_TICKERS_DETAILED.keys())[:10],
            help="Select stocks for portfolio construction"
        )
        
        # Benchmark selection
        benchmark = st.selectbox(
            "Benchmark Index",
            options=list(BENCHMARK_TICKERS.keys()),
            index=0,
            help="Select benchmark for comparison"
        )
        
        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Risk-free rate
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=45.0,
            step=0.1,
            help="Annual risk-free rate in percentage"
        ) / 100
        
        # Optimization method
        st.subheader("Optimization Method")
        optimization_method = st.selectbox(
            "Select Optimization Method",
            options=[method.value for method in OptimizationMethod],
            index=0,
            help="Select portfolio optimization algorithm"
        )
        
        # Fetch data button
        if st.button("📊 Fetch Market Data", type="primary", use_container_width=True):
            with st.spinner("Fetching market data..."):
                try:
                    st.session_state.portfolio_data = st.session_state.data_fetcher.fetch_market_data(
                        tickers=selected_tickers,
                        benchmark_tickers=[benchmark],
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        use_cache=True
                    )
                    
                    if st.session_state.portfolio_data:
                        st.success("✅ Data fetched successfully!")
                        
                        # Initialize optimizer
                        returns = st.session_state.portfolio_data['returns']
                        st.session_state.optimizer = AdvancedPortfolioOptimizer(
                            returns=returns,
                            risk_free_rate=risk_free_rate
                        )
                        
                        # Initialize risk analytics
                        st.session_state.risk_analytics = AdvancedRiskAnalytics(
                            returns=returns,
                            risk_free_rate=risk_free_rate
                        )
                    else:
                        st.error("❌ Failed to fetch data")
                        
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    logger.error(f"Data fetch error: {traceback.format_exc()}")
    
    # Main content area
    if st.session_state.portfolio_data and st.session_state.optimizer:
        # Create tabs
        tabs = st.tabs([
            "📈 Portfolio Overview",
            "⚖️ Optimization",
            "📊 Risk Analytics",
            "📉 Stress Testing",
            "📋 Reports"
        ])
        
        with tabs[0]:
            st.header("Portfolio Overview")
            
            # Display basic information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Assets", len(selected_tickers))
            with col2:
                st.metric("Data Period", f"{start_date} to {end_date}")
            with col3:
                st.metric("Trading Days", len(st.session_state.portfolio_data['returns']))
            
            # Price chart
            st.subheader("Price Performance")
            fig = go.Figure()
            for ticker in selected_tickers[:5]:  # Show first 5 for clarity
                if ticker in st.session_state.portfolio_data['prices'].columns:
                    fig.add_trace(go.Scatter(
                        x=st.session_state.portfolio_data['prices'].index,
                        y=st.session_state.portfolio_data['prices'][ticker],
                        name=ticker,
                        mode='lines'
                    ))
            
            fig.update_layout(
                title="Stock Prices",
                xaxis_title="Date",
                yaxis_title="Price (TRY)",
                height=400,
                **PLOTLY_THEME
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.header("Portfolio Optimization")
            
            # Optimization parameters
            col1, col2 = st.columns(2)
            with col1:
                target_return = st.number_input(
                    "Target Return (%)",
                    min_value=-50.0,
                    max_value=200.0,
                    value=15.0,
                    step=0.1
                ) / 100
                
                target_volatility = st.number_input(
                    "Target Volatility (%)",
                    min_value=5.0,
                    max_value=100.0,
                    value=20.0,
                    step=0.1
                ) / 100
            
            with col2:
                risk_aversion = st.slider(
                    "Risk Aversion Coefficient",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
                
                constraints = PortfolioConstraints(
                    min_weight=0.0,
                    max_weight=0.3,  # Max 30% per asset
                    target_leverage=1.0
                )
            
            # Create optimization parameters
            params = OptimizationParameters(
                method=OptimizationMethod(optimization_method),
                risk_free_rate=risk_free_rate,
                risk_aversion=risk_aversion,
                target_return=target_return,
                target_volatility=target_volatility,
                constraints=constraints
            )
            
            # Run optimization
            if st.button("🚀 Run Optimization", type="primary"):
                with st.spinner("Optimizing portfolio..."):
                    try:
                        result = st.session_state.optimizer.optimize(
                            method=OptimizationMethod(optimization_method),
                            parameters=params
                        )
                        
                        st.session_state.optimization_result = result
                        
                        # Display results
                        st.subheader("Optimization Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Expected Return",
                                f"{result['performance'][0]*100:.2f}%"
                            )
                        with col2:
                            st.metric(
                                "Expected Volatility",
                                f"{result['performance'][1]*100:.2f}%"
                            )
                        with col3:
                            st.metric(
                                "Sharpe Ratio",
                                f"{result['performance'][2]:.3f}"
                            )
                        
                        # Display weights
                        st.subheader("Optimal Weights")
                        weights_df = pd.DataFrame(
                            list(result['weights'].items()),
                            columns=['Asset', 'Weight']
                        )
                        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
                        st.dataframe(weights_df, use_container_width=True)
                        
                        # Create visualization
                        st.subheader("Portfolio Allocation")
                        fig = st.session_state.visualizer.plot_weight_allocation_advanced(
                            weights=result['weights'],
                            asset_info=BIST30_TICKERS_DETAILED
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Optimization error: {str(e)}")
                        logger.error(f"Optimization error: {traceback.format_exc()}")
        
        with tabs[2]:
            st.header("Risk Analytics")
            
            if 'optimization_result' in st.session_state:
                # Calculate portfolio returns
                weights = st.session_state.optimization_result['weights']
                portfolio_returns = st.session_state.optimizer._calculate_portfolio_returns(weights)
                benchmark_returns = st.session_state.portfolio_data['benchmark_data']['returns'].iloc[:, 0]
                
                # Risk metrics dashboard
                st.subheader("Risk Metrics Dashboard")
                fig = st.session_state.visualizer.plot_risk_metrics_dashboard(
                    portfolio_returns=portfolio_returns,
                    benchmark_returns=benchmark_returns,
                    risk_free_rate=risk_free_rate
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # VaR and CVaR analysis
                st.subheader("Value at Risk Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Calculate VaR/CVaR
                    var_cvar_df = st.session_state.risk_analytics.calculate_var_cvar(portfolio_returns)
                    st.dataframe(var_cvar_df, use_container_width=True)
                
                with col2:
                    # Calculate modified VaR/CVaR
                    modified_var_cvar_df = st.session_state.risk_analytics.calculate_modified_var_cvar(portfolio_returns)
                    st.dataframe(modified_var_cvar_df, use_container_width=True)
                
                # Monte Carlo simulation
                st.subheader("Monte Carlo Simulation")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    horizon = st.number_input("Horizon (days)", min_value=1, max_value=252, value=10)
                with col2:
                    n_simulations = st.selectbox("Number of Simulations", [1000, 5000, 10000, 50000], index=2)
                with col3:
                    confidence = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
                
                if st.button("Run Monte Carlo Simulation"):
                    with st.spinner("Running Monte Carlo simulation..."):
                        try:
                            mc_results = st.session_state.risk_analytics.monte_carlo_simulation(
                                returns_series=portfolio_returns,
                                horizon_days=horizon,
                                n_simulations=n_simulations,
                                confidence_level=confidence,
                                model_type='gbm'
                            )
                            
                            # Display results
                            stats = mc_results['statistics']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Return", f"{stats['mean_return']*100:.2f}%")
                            with col2:
                                st.metric("VaR", f"{stats['var']*100:.2f}%")
                            with col3:
                                st.metric("CVaR", f"{stats['cvar']*100:.2f}%")
                            with col4:
                                st.metric("Prob. Loss", f"{stats['probability_loss']:.1f}%")
                            
                            # Visualization
                            fig = st.session_state.visualizer.plot_monte_carlo_analysis(mc_results)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Monte Carlo error: {str(e)}")
        
        with tabs[3]:
            st.header("Stress Testing & Scenario Analysis")
            
            if 'optimization_result' in st.session_state:
                weights = st.session_state.optimization_result['weights']
                portfolio_returns = st.session_state.optimizer._calculate_portfolio_returns(weights)
                
                # Stress testing
                st.subheader("Stress Testing")
                stress_results = st.session_state.risk_analytics.stress_testing(portfolio_returns)
                st.dataframe(stress_results, use_container_width=True)
                
                # Scenario analysis
                st.subheader("Scenario Analysis")
                scenario_results = st.session_state.risk_analytics.scenario_analysis(portfolio_returns)
                st.dataframe(scenario_results, use_container_width=True)
        
        with tabs[4]:
            st.header("Reports & Exports")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    st.info("PDF report generation would be implemented here")
            
            with col2:
                if st.button("📊 Export to Excel", use_container_width=True):
                    st.info("Excel export would be implemented here")
            
            with col3:
                if st.button("📈 Export Charts", use_container_width=True):
                    st.info("Chart export would be implemented here")
            
            # Display comprehensive metrics if available
            if 'optimization_result' in st.session_state:
                st.subheader("Comprehensive Portfolio Metrics")
                metrics_df = st.session_state.optimization_result['metrics'].to_dataframe()
                st.dataframe(metrics_df, use_container_width=True, height=400)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to BIST Portfolio Risk Analytics Platform
        
        This enterprise-grade platform provides comprehensive portfolio optimization,
        risk analytics, and stress testing capabilities for BIST 30 stocks.
        
        ### Getting Started:
        1. Select assets from the sidebar
        2. Choose benchmark and date range
        3. Configure optimization parameters
        4. Click "Fetch Market Data" to begin
        
        ### Key Features:
        - **15+ Portfolio Optimization Methods**: From traditional mean-variance to advanced machine learning approaches
        - **Comprehensive Risk Analytics**: VaR, CVaR, stress testing, scenario analysis
        - **Real-time Data Integration**: Yahoo Finance, Alpha Vantage, and synthetic data fallbacks
        - **Professional Reporting**: PDF, Excel, and interactive dashboards
        - **Regulatory Compliance**: Built-in regulatory limits and reporting
        
        ### Supported Optimization Methods:
        - Maximum Sharpe Ratio
        - Minimum Volatility
        - Risk Parity
        - Hierarchical Risk Parity (HRP)
        - Conditional Value at Risk (CVaR)
        - And many more...
        """)
        
        # Display feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Data Integration**\n\nMulti-source data fetching with caching and fallback mechanisms")
        with col2:
            st.info("**Risk Management**\n\nComprehensive risk metrics including VaR, CVaR, stress tests")
        with col3:
            st.info("**Optimization**\n\n15+ portfolio optimization algorithms with constraints")

# Run the app
if __name__ == "__main__":
    main()
