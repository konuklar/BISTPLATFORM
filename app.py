import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import PyPortfolioOpt components
from pypfopt import expected_returns, risk_models, efficient_frontier
from pypfopt import objective_functions
from pypfopt import plotting
from pypfopt import CLA
from pypfopt import DiscreteAllocation
from pypfopt import EfficientCVaR, EfficientSemivariance

# Advanced risk metrics and visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import scipy.stats as stats

# Set professional styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("tab10")
professional_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                      '#bcbd22', '#17becf']

class ProfessionalPortfolioAnalytics:
    """
    Institutional-grade portfolio analytics with advanced optimization techniques
    and comprehensive risk metrics including VaR, CVaR, and Monte Carlo simulations.
    """
    
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        """
        Initialize portfolio analytics engine.
        
        Parameters:
        -----------
        tickers : list
            List of asset tickers
        start_date : str
            Start date for historical data
        end_date : str
            End date for historical data
        risk_free_rate : float
            Annual risk-free rate
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.returns = None
        self.prices = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess institutional-grade financial data."""
        print("📊 Loading institutional financial data...")
        
        # Download data
        self.data = yf.download(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date,
            progress=False
        )['Adj Close']
        
        # Handle missing data
        self.data = self.data.ffill().bfill()
        
        # Calculate returns
        self.returns = self.data.pct_change().dropna()
        self.prices = self.data
        
        print(f"✅ Data loaded: {len(self.returns)} days of returns data")
        print(f"📈 Assets: {', '.join(self.tickers)}")
        
    def calculate_advanced_metrics(self, weights=None):
        """
        Calculate comprehensive portfolio performance and risk metrics.
        
        Parameters:
        -----------
        weights : array
            Portfolio weights (if None, uses equal weighting)
            
        Returns:
        --------
        metrics_df : DataFrame
            Comprehensive metrics dataframe
        """
        if weights is None:
            weights = np.array([1/len(self.tickers)] * len(self.tickers))
        
        # Portfolio returns
        portfolio_returns = self.returns.dot(weights)
        
        # Basic statistics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Advanced risk metrics
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        # Sortino Ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else np.nan
        
        # Calmar Ratio
        max_dd = self.calculate_max_drawdown(portfolio_returns)
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else np.nan
        
        # Omega Ratio
        omega_ratio = self.calculate_omega_ratio(portfolio_returns, self.risk_free_rate/252)
        
        # Information Ratio (vs equal weighted benchmark)
        benchmark_returns = self.returns.mean(axis=1)
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = (annual_return - benchmark_returns.mean()*252) / tracking_error if tracking_error > 0 else np.nan
        
        # Value at Risk (VaR) metrics
        var_95 = self.calculate_var(portfolio_returns, alpha=0.05)
        var_99 = self.calculate_var(portfolio_returns, alpha=0.01)
        
        # Conditional VaR (CVaR/Expected Shortfall)
        cvar_95 = self.calculate_cvar(portfolio_returns, alpha=0.05)
        cvar_99 = self.calculate_cvar(portfolio_returns, alpha=0.01)
        
        # Relative CVaR
        relative_cvar = self.calculate_relative_cvar(portfolio_returns, benchmark_returns)
        
        # Create comprehensive metrics dataframe
        metrics_dict = {
            'Annual Return (%)': annual_return * 100,
            'Annual Volatility (%)': annual_volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Omega Ratio': omega_ratio,
            'Information Ratio': information_ratio,
            'Max Drawdown (%)': max_dd * 100,
            'VaR 95% (%)': var_95 * 100,
            'VaR 99% (%)': var_99 * 100,
            'CVaR 95% (%)': cvar_95 * 100,
            'CVaR 99% (%)': cvar_99 * 100,
            'Relative CVaR': relative_cvar,
            'Skewness': portfolio_returns.skew(),
            'Kurtosis': portfolio_returns.kurtosis(),
            'Positive Periods (%)': (portfolio_returns > 0).mean() * 100
        }
        
        return pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])
    
    def calculate_var(self, returns, alpha=0.05, method='historical'):
        """
        Calculate Value at Risk using different methods.
        
        Parameters:
        -----------
        returns : Series
            Return series
        alpha : float
            Confidence level
        method : str
            'historical', 'parametric', or 'monte_carlo'
            
        Returns:
        --------
        var : float
            Value at Risk
        """
        if method == 'historical':
            return np.percentile(returns, alpha * 100)
        
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            return stats.norm.ppf(alpha, mu, sigma)
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mu = returns.mean()
            sigma = returns.std()
            skew = returns.skew()
            kurt = returns.kurtosis()
            
            # Fit t-distribution
            df, loc, scale = stats.t.fit(returns)
            mc_returns = stats.t.rvs(df, loc, scale, size=10000)
            return np.percentile(mc_returns, alpha * 100)
        
        return None
    
    def calculate_cvar(self, returns, alpha=0.05):
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = self.calculate_var(returns, alpha, 'historical')
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else var
    
    def calculate_relative_cvar(self, portfolio_returns, benchmark_returns):
        """Calculate Relative CVaR."""
        excess_returns = portfolio_returns - benchmark_returns
        return self.calculate_cvar(excess_returns, alpha=0.05)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_omega_ratio(self, returns, threshold=0):
        """Calculate Omega Ratio."""
        returns_threshold = returns - threshold
        positive_sum = returns_threshold[returns_threshold > 0].sum()
        negative_sum = abs(returns_threshold[returns_threshold < 0].sum())
        return positive_sum / negative_sum if negative_sum != 0 else np.nan
    
    def optimize_portfolio_mean_variance(self):
        """Optimize portfolio using Mean-Variance optimization."""
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.data)
        S = risk_models.sample_cov(self.data)
        
        # Optimize for maximum Sharpe ratio
        ef = efficient_frontier.EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = ef.clean_weights()
        
        # Get performance
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        
        return weights, perf
    
    def optimize_portfolio_cvar(self):
        """Optimize portfolio using CVaR (Conditional Value at Risk) minimization."""
        ef_cvar = EfficientCVaR(self.mu, self.S)
        ef_cvar.min_cvar()
        weights_cvar = ef_cvar.clean_weights()
        perf_cvar = ef_cvar.portfolio_performance(verbose=False)
        
        return weights_cvar, perf_cvar
    
    def optimize_portfolio_semivariance(self):
        """Optimize portfolio using Semivariance minimization."""
        historical_returns = self.returns
        
        ef_semivariance = EfficientSemivariance(self.mu, historical_returns)
        ef_semivariance.efficient_return(target_return=self.mu.mean())
        weights_semi = ef_semivariance.clean_weights()
        
        return weights_semi
    
    def monte_carlo_simulation(self, weights, n_simulations=10000, days=252):
        """
        Perform Monte Carlo simulation for portfolio returns.
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        n_simulations : int
            Number of simulations
        days : int
            Time horizon in days
            
        Returns:
        --------
        simulation_results : dict
            Monte Carlo simulation results
        """
        portfolio_returns = self.returns.dot(weights)
        
        # Fit distribution parameters
        mu_daily = portfolio_returns.mean()
        sigma_daily = portfolio_returns.std()
        
        # Generate simulations
        simulations = np.random.normal(
            mu_daily, 
            sigma_daily, 
            (days, n_simulations)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + simulations, axis=0)
        
        # Calculate statistics
        final_returns = cumulative_returns[-1, :] - 1
        var_95 = np.percentile(final_returns, 5)
        var_99 = np.percentile(final_returns, 1)
        cvar_95 = final_returns[final_returns <= var_95].mean()
        cvar_99 = final_returns[final_returns <= var_99].mean()
        
        return {
            'simulations': simulations,
            'cumulative_returns': cumulative_returns,
            'final_returns': final_returns,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'mean_return': final_returns.mean(),
            'std_return': final_returns.std()
        }
    
    def plot_efficient_frontier(self, mu=None, S=None):
        """Plot professional efficient frontier."""
        if mu is None:
            mu = expected_returns.mean_historical_return(self.data)
        if S is None:
            S = risk_models.sample_cov(self.data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot efficient frontier
        ef = efficient_frontier.EfficientFrontier(mu, S)
        
        # Generate efficient frontier
        rets = []
        vols = []
        sharpe_ratios = []
        
        for target_return in np.linspace(mu.min(), mu.max(), 50):
            try:
                ef.efficient_return(target_return)
                weights = ef.clean_weights()
                port_return = np.dot(weights, mu)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
                sharpe = (port_return - self.risk_free_rate) / port_vol
                
                rets.append(port_return * 100)
                vols.append(port_vol * 100)
                sharpe_ratios.append(sharpe)
            except:
                continue
        
        # Create scatter plot with color by Sharpe ratio
        scatter = ax.scatter(vols, rets, c=sharpe_ratios, cmap='viridis', 
                           alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=12)
        
        # Find and mark optimal portfolios
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        max_sharpe_weights = ef.clean_weights()
        max_sharpe_ret, max_sharpe_vol, _ = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate
        )
        
        ef.min_volatility()
        min_vol_weights = ef.clean_weights()
        min_vol_ret, min_vol_vol, _ = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate
        )
        
        # Mark optimal points
        ax.scatter(max_sharpe_vol*100, max_sharpe_ret*100, 
                  color='red', s=200, marker='*', label='Max Sharpe', 
                  edgecolors='black', linewidth=1.5)
        ax.scatter(min_vol_vol*100, min_vol_ret*100, 
                  color='green', s=200, marker='*', label='Min Volatility',
                  edgecolors='black', linewidth=1.5)
        
        # Formatting
        ax.set_xlabel('Annual Volatility (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Annual Return (%)', fontsize=14, fontweight='bold')
        ax.set_title('Efficient Frontier with Optimal Portfolios', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_var_analytics(self, portfolio_returns):
        """Comprehensive VaR analytics visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Historical Returns Distribution
        axes[0].hist(portfolio_returns * 100, bins=50, density=True, 
                    alpha=0.7, color=professional_colors[0], edgecolor='black')
        axes[0].axvline(portfolio_returns.mean() * 100, color='red', 
                       linestyle='--', linewidth=2, label='Mean')
        
        # Add VaR lines
        var_95 = self.calculate_var(portfolio_returns, 0.05, 'historical') * 100
        var_99 = self.calculate_var(portfolio_returns, 0.01, 'historical') * 100
        
        axes[0].axvline(var_95, color='orange', linestyle='--', 
                       linewidth=2, label='VaR 95%')
        axes[0].axvline(var_99, color='darkred', linestyle='--', 
                       linewidth=2, label='VaR 99%')
        
        axes[0].set_xlabel('Daily Return (%)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Returns Distribution with VaR', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. VaR Comparison (Different Methods)
        methods = ['historical', 'parametric', 'monte_carlo']
        var_values = []
        
        for method in methods:
            var_95 = self.calculate_var(portfolio_returns, 0.05, method)
            var_99 = self.calculate_var(portfolio_returns, 0.01, method)
            var_values.append([var_95 * 100, var_99 * 100])
        
        var_df = pd.DataFrame(var_values, 
                             index=methods, 
                             columns=['VaR 95%', 'VaR 99%'])
        
        var_df.plot(kind='bar', ax=axes[1], color=[professional_colors[1], professional_colors[2]])
        axes[1].set_ylabel('VaR (%)', fontsize=12)
        axes[1].set_title('VaR Comparison by Method', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 3. CVaR/Expected Shortfall
        alpha_levels = np.arange(0.01, 0.11, 0.01)
        var_values = []
        cvar_values = []
        
        for alpha in alpha_levels:
            var_values.append(self.calculate_var(portfolio_returns, alpha, 'historical') * 100)
            cvar_values.append(self.calculate_cvar(portfolio_returns, alpha) * 100)
        
        axes[2].plot(alpha_levels * 100, var_values, marker='o', 
                    label='VaR', color=professional_colors[0], linewidth=2)
        axes[2].plot(alpha_levels * 100, cvar_values, marker='s', 
                    label='CVaR', color=professional_colors[3], linewidth=2)
        axes[2].set_xlabel('Confidence Level (%)', fontsize=12)
        axes[2].set_ylabel('Risk Measure (%)', fontsize=12)
        axes[2].set_title('VaR vs CVaR Across Confidence Levels', 
                         fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Rolling VaR
        window = 63  # 3 months
        rolling_var = portfolio_returns.rolling(window).apply(
            lambda x: self.calculate_var(x, 0.05, 'historical')
        ) * 100
        
        axes[3].plot(rolling_var.index, rolling_var, 
                    color=professional_colors[4], linewidth=2)
        axes[3].axhline(rolling_var.mean(), color='red', 
                       linestyle='--', label=f'Mean: {rolling_var.mean():.2f}%')
        axes[3].fill_between(rolling_var.index, rolling_var, 
                            rolling_var.mean(), alpha=0.2, color=professional_colors[4])
        axes[3].set_xlabel('Date', fontsize=12)
        axes[3].set_ylabel('VaR 95% (%)', fontsize=12)
        axes[3].set_title(f'Rolling {window}-Day VaR (95%)', fontsize=14, fontweight='bold')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. Monte Carlo Simulation Results
        weights = np.array([1/len(self.tickers)] * len(self.tickers))
        mc_results = self.monte_carlo_simulation(weights, n_simulations=5000)
        
        axes[4].hist(mc_results['final_returns'] * 100, bins=50, 
                    alpha=0.7, color=professional_colors[5], edgecolor='black')
        axes[4].axvline(mc_results['var_95'] * 100, color='orange', 
                       linestyle='--', linewidth=2, label='VaR 95%')
        axes[4].axvline(mc_results['var_99'] * 100, color='darkred', 
                       linestyle='--', linewidth=2, label='VaR 99%')
        axes[4].axvline(mc_results['cvar_95'] * 100, color='red', 
                       linestyle=':', linewidth=2, label='CVaR 95%')
        axes[4].set_xlabel('Final Return (%)', fontsize=12)
        axes[4].set_ylabel('Frequency', fontsize=12)
        axes[4].set_title('Monte Carlo Simulation Distribution', 
                         fontsize=14, fontweight='bold')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # 6. CVaR Decomposition
        asset_cvars = []
        for i, ticker in enumerate(self.tickers):
            asset_returns = self.returns[ticker]
            asset_cvar = self.calculate_cvar(asset_returns, 0.05) * 100
            asset_cvars.append(asset_cvar)
        
        axes[5].bar(self.tickers, asset_cvars, color=professional_colors[6:6+len(self.tickers)])
        axes[5].axhline(np.mean(asset_cvars), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(asset_cvars):.2f}%')
        axes[5].set_xlabel('Assets', fontsize=12)
        axes[5].set_ylabel('CVaR 95% (%)', fontsize=12)
        axes[5].set_title('Individual Asset CVaR', fontsize=14, fontweight='bold')
        axes[5].tick_params(axis='x', rotation=45)
        axes[5].legend()
        axes[5].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def create_performance_dashboard(self, weights):
        """
        Create comprehensive institutional performance dashboard.
        
        Parameters:
        -----------
        weights : dict
            Portfolio weights
            
        Returns:
        --------
        fig : matplotlib Figure
            Dashboard figure
        """
        # Convert weights to array
        weights_array = np.array([weights[ticker] for ticker in self.tickers])
        portfolio_returns = self.returns.dot(weights_array)
        
        # Create dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Cumulative Returns (Top Left)
        ax1 = plt.subplot(3, 3, 1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        ax1.plot(cumulative_returns.index, cumulative_returns, 
                linewidth=2, color=professional_colors[0])
        ax1.set_title('Portfolio Cumulative Returns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown (Top Middle)
        ax2 = plt.subplot(3, 3, 2)
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        ax2.fill_between(drawdown.index, drawdown * 100, 0, 
                        color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown * 100, color='darkred', linewidth=1.5)
        ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio (Top Right)
        ax3 = plt.subplot(3, 3, 3)
        rolling_window = 63
        rolling_sharpe = portfolio_returns.rolling(window=rolling_window).apply(
            lambda x: (x.mean() * 252 - self.risk_free_rate) / (x.std() * np.sqrt(252))
        )
        ax3.plot(rolling_sharpe.index, rolling_sharpe, 
                linewidth=2, color=professional_colors[1])
        ax3.axhline(rolling_sharpe.mean(), color='red', 
                   linestyle='--', label=f'Mean: {rolling_sharpe.mean():.2f}')
        ax3.set_title(f'Rolling {rolling_window}-Day Sharpe Ratio', 
                     fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Portfolio Weights (Middle Left)
        ax4 = plt.subplot(3, 3, 4)
        wedges, texts, autotexts = ax4.pie(
            weights_array, 
            labels=self.tickers, 
            autopct='%1.1f%%',
            colors=professional_colors[:len(self.tickers)]
        )
        ax4.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
        
        # 5. Risk-Return Scatter (Middle)
        ax5 = plt.subplot(3, 3, 5)
        
        # Calculate individual asset metrics
        asset_returns = self.returns.mean() * 252
        asset_vols = self.returns.std() * np.sqrt(252)
        
        # Plot individual assets
        ax5.scatter(asset_vols * 100, asset_returns * 100, 
                   s=150, alpha=0.7, color=professional_colors[:len(self.tickers)], 
                   edgecolors='black')
        
        # Plot portfolio
        port_return = portfolio_returns.mean() * 252
        port_vol = portfolio_returns.std() * np.sqrt(252)
        ax5.scatter(port_vol * 100, port_return * 100, 
                   s=300, color='red', marker='*', 
                   edgecolors='black', linewidth=2, label='Portfolio')
        
        # Add labels
        for i, ticker in enumerate(self.tickers):
            ax5.annotate(ticker, (asset_vols[i]*100, asset_returns[i]*100), 
                        fontsize=9, ha='center', va='center')
        
        ax5.set_xlabel('Annual Volatility (%)', fontsize=12)
        ax5.set_ylabel('Annual Return (%)', fontsize=12)
        ax5.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Correlation Heatmap (Middle Right)
        ax6 = plt.subplot(3, 3, 6)
        corr_matrix = self.returns.corr()
        im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", 
                               color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                               fontsize=8)
        
        ax6.set_xticks(range(len(self.tickers)))
        ax6.set_yticks(range(len(self.tickers)))
        ax6.set_xticklabels(self.tickers, rotation=45)
        ax6.set_yticklabels(self.tickers)
        ax6.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax6)
        
        # 7. Rolling Volatility (Bottom Left)
        ax7 = plt.subplot(3, 3, 7)
        rolling_vol = portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252) * 100
        ax7.plot(rolling_vol.index, rolling_vol, 
                linewidth=2, color=professional_colors[2])
        ax7.axhline(rolling_vol.mean(), color='red', 
                   linestyle='--', label=f'Mean: {rolling_vol.mean():.2f}%')
        ax7.set_xlabel('Date', fontsize=12)
        ax7.set_ylabel('Annual Volatility (%)', fontsize=12)
        ax7.set_title(f'Rolling {rolling_window}-Day Volatility', 
                     fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Return Distribution (Bottom Middle)
        ax8 = plt.subplot(3, 3, 8)
        ax8.hist(portfolio_returns * 100, bins=50, density=True, 
                alpha=0.7, color=professional_colors[3], edgecolor='black')
        
        # Add normal distribution for comparison
        x = np.linspace(portfolio_returns.min() * 100, portfolio_returns.max() * 100, 100)
        normal_pdf = stats.norm.pdf(x, portfolio_returns.mean() * 100, portfolio_returns.std() * 100)
        ax8.plot(x, normal_pdf, 'r--', linewidth=2, label='Normal Distribution')
        
        ax8.set_xlabel('Daily Return (%)', fontsize=12)
        ax8.set_ylabel('Density', fontsize=12)
        ax8.set_title('Return Distribution', fontsize=14, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Metrics Summary Table (Bottom Right)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        # Calculate key metrics
        metrics = self.calculate_advanced_metrics(weights_array)
        
        # Create table
        table_data = []
        for metric, value in zip(metrics['Metric'], metrics['Value']):
            if 'Ratio' in metric or 'Return' in metric or 'Volatility' in metric:
                table_data.append([metric, f'{value:.2f}' if isinstance(value, (int, float)) else value])
        
        table = ax9.table(cellText=table_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style table cells
        for i, key in enumerate(table_data):
            cell = table[(i+1, 0)]
            cell.set_text_props(fontweight='bold')
        
        ax9.set_title('Key Performance Metrics', fontsize=14, fontweight='bold', y=1.08)
        
        plt.suptitle('INSTITUTIONAL PORTFOLIO ANALYTICS DASHBOARD', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def generate_institutional_report(self):
        """
        Generate comprehensive institutional report with all analytics.
        """
        print("=" * 80)
        print("INSTITUTIONAL PORTFOLIO ANALYTICS REPORT")
        print("=" * 80)
        
        # 1. Optimize portfolio
        print("\n1. PORTFOLIO OPTIMIZATION RESULTS")
        print("-" * 40)
        
        # Mean-Variance Optimization
        weights_mv, perf_mv = self.optimize_portfolio_mean_variance()
        print("\nMean-Variance Optimization (Max Sharpe):")
        print(f"Expected Annual Return: {perf_mv[0]*100:.2f}%")
        print(f"Annual Volatility: {perf_mv[1]*100:.2f}%")
        print(f"Sharpe Ratio: {perf_mv[2]:.3f}")
        
        print("\nOptimal Weights:")
        for ticker, weight in weights_mv.items():
            if weight > 0.001:
                print(f"  {ticker}: {weight*100:.2f}%")
        
        # 2. Calculate advanced metrics
        print("\n\n2. ADVANCED RISK METRICS")
        print("-" * 40)
        
        # Convert weights to array
        weights_array = np.array([weights_mv[ticker] for ticker in self.tickers])
        metrics_df = self.calculate_advanced_metrics(weights_array)
        
        pd.set_option('display.float_format', '{:.3f}'.format)
        print("\n" + metrics_df.to_string(index=False))
        
        # 3. VaR Analytics
        print("\n\n3. VALUE AT RISK ANALYTICS")
        print("-" * 40)
        
        portfolio_returns = self.returns.dot(weights_array)
        
        # Calculate VaR using different methods
        print("\nVaR 95% (1-day):")
        print(f"  Historical: {self.calculate_var(portfolio_returns, 0.05, 'historical')*100:.2f}%")
        print(f"  Parametric: {self.calculate_var(portfolio_returns, 0.05, 'parametric')*100:.2f}%")
        print(f"  Monte Carlo: {self.calculate_var(portfolio_returns, 0.05, 'monte_carlo')*100:.2f}%")
        
        print("\nVaR 99% (1-day):")
        print(f"  Historical: {self.calculate_var(portfolio_returns, 0.01, 'historical')*100:.2f}%")
        print(f"  Parametric: {self.calculate_var(portfolio_returns, 0.01, 'parametric')*100:.2f}%")
        print(f"  Monte Carlo: {self.calculate_var(portfolio_returns, 0.01, 'monte_carlo')*100:.2f}%")
        
        print("\nExpected Shortfall (CVaR):")
        print(f"  CVaR 95%: {self.calculate_cvar(portfolio_returns, 0.05)*100:.2f}%")
        print(f"  CVaR 99%: {self.calculate_cvar(portfolio_returns, 0.01)*100:.2f}%")
        
        # 4. Monte Carlo Simulation
        print("\n\n4. MONTE CARLO SIMULATION")
        print("-" * 40)
        
        mc_results = self.monte_carlo_simulation(weights_array)
        print(f"\nSimulation Results (1-year horizon, 10,000 simulations):")
        print(f"  Mean Return: {mc_results['mean_return']*100:.2f}%")
        print(f"  Std Deviation: {mc_results['std_return']*100:.2f}%")
        print(f"  VaR 95%: {mc_results['var_95']*100:.2f}%")
        print(f"  VaR 99%: {mc_results['var_99']*100:.2f}%")
        print(f"  CVaR 95%: {mc_results['cvar_95']*100:.2f}%")
        print(f"  CVaR 99%: {mc_results['cvar_99']*100:.2f}%")
        
        # 5. Risk Decomposition
        print("\n\n5. RISK DECOMPOSITION")
        print("-" * 40)
        
        # Calculate marginal contributions to risk
        cov_matrix = self.returns.cov()
        portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        
        print(f"\nPortfolio Volatility: {portfolio_vol*np.sqrt(252)*100:.2f}%")
        print("\nMarginal Contribution to Risk (% of total):")
        
        mctr = np.dot(cov_matrix, weights_array) / portfolio_vol
        for i, ticker in enumerate(self.tickers):
            contribution = weights_array[i] * mctr[i] / portfolio_vol * 100
            print(f"  {ticker}: {contribution:.2f}%")
        
        print("\n" + "=" * 80)
        print("REPORT COMPLETE")
        print("=" * 80)
        
        return weights_mv, metrics_df


# Example usage with real data
if __name__ == "__main__":
    # Define portfolio assets
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'DIS']
    
    # Initialize analytics engine
    print("🚀 Initializing Professional Portfolio Analytics Engine...")
    analytics = ProfessionalPortfolioAnalytics(
        tickers=tickers,
        start_date='2020-01-01',
        end_date='2024-01-01',
        risk_free_rate=0.02
    )
    
    # Generate comprehensive report
    optimal_weights, metrics = analytics.generate_institutional_report()
    
    # Create visualizations
    print("\n📊 Generating professional visualizations...")
    
    # 1. Efficient Frontier
    fig1, ax1 = analytics.plot_efficient_frontier()
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    
    # 2. VaR Analytics Dashboard
    weights_array = np.array([optimal_weights[ticker] for ticker in tickers])
    portfolio_returns = analytics.returns.dot(weights_array)
    
    fig2 = analytics.plot_var_analytics(portfolio_returns)
    plt.savefig('var_analytics.png', dpi=300, bbox_inches='tight')
    
    # 3. Performance Dashboard
    fig3 = analytics.create_performance_dashboard(optimal_weights)
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
    
    print("\n✅ Analysis complete! Check saved visualizations:")
    print("   - efficient_frontier.png")
    print("   - var_analytics.png")
    print("   - performance_dashboard.png")
    
    # Show one plot
    plt.show()
