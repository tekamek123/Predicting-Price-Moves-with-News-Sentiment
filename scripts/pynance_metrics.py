"""
Calculate financial metrics using PyNance.

This module provides functions to calculate various financial metrics
such as returns, volatility, Sharpe ratio, and other risk metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Try to import pynance, fallback to manual calculations if not available
try:
    import pynance
    PYANCE_AVAILABLE = True
except ImportError:
    try:
        # Try alternative package name
        import pynance_tools as pynance
        PYANCE_AVAILABLE = True
    except ImportError:
        PYANCE_AVAILABLE = False
        # Note: We'll use manual calculations for all metrics
        # PyNance is optional - all calculations are implemented manually


def calculate_returns(
    df: pd.DataFrame,
    price_column: str = 'Close',
    method: str = 'simple'
) -> pd.DataFrame:
    """
    Calculate returns (simple or logarithmic).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    price_column : str
        Column name for price data (default: 'Close')
    method : str
        Return calculation method: 'simple' or 'log' (default: 'simple')

    Returns:
    --------
    pd.DataFrame
        DataFrame with added returns columns
    """
    df = df.copy()
    prices = df[price_column]
    
    if method == 'simple':
        df['returns'] = prices.pct_change()
        df['returns_pct'] = df['returns'] * 100
    elif method == 'log':
        df['log_returns'] = np.log(prices / prices.shift(1))
        df['log_returns_pct'] = df['log_returns'] * 100
    else:
        raise ValueError("Method must be 'simple' or 'log'")
    
    return df


def calculate_volatility(
    df: pd.DataFrame,
    returns_column: str = 'returns',
    window: int = 30,
    annualized: bool = True
) -> pd.DataFrame:
    """
    Calculate volatility (standard deviation of returns).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with returns data
    returns_column : str
        Column name for returns (default: 'returns')
    window : int
        Rolling window for volatility calculation (default: 30)
    annualized : bool
        Whether to annualize volatility (default: True)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added volatility column
    """
    df = df.copy()
    
    if returns_column not in df.columns:
        df = calculate_returns(df)
        returns_column = 'returns'
    
    # Calculate rolling volatility
    df[f'volatility_{window}'] = df[returns_column].rolling(window=window).std()
    
    if annualized:
        # Annualize by multiplying by sqrt(252) for daily data
        df[f'volatility_{window}_annualized'] = df[f'volatility_{window}'] * np.sqrt(252)
    
    return df


def calculate_sharpe_ratio(
    df: pd.DataFrame,
    returns_column: str = 'returns',
    risk_free_rate: float = 0.02,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate Sharpe Ratio.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with returns data
    returns_column : str
        Column name for returns (default: 'returns')
    risk_free_rate : float
        Annual risk-free rate (default: 0.02 for 2%)
    window : int
        Rolling window for calculation (default: 252 for 1 year of daily data)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added Sharpe ratio column
    """
    df = df.copy()
    
    if returns_column not in df.columns:
        df = calculate_returns(df)
        returns_column = 'returns'
    
    # Calculate excess returns
    daily_rf_rate = risk_free_rate / 252
    df['excess_returns'] = df[returns_column] - daily_rf_rate
    
    # Calculate rolling Sharpe ratio
    rolling_mean = df['excess_returns'].rolling(window=window).mean()
    rolling_std = df['excess_returns'].rolling(window=window).std()
    
    df[f'sharpe_ratio_{window}'] = (rolling_mean / rolling_std) * np.sqrt(252)
    
    return df


def calculate_max_drawdown(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> pd.DataFrame:
    """
    Calculate Maximum Drawdown.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    price_column : str
        Column name for price data (default: 'Close')

    Returns:
    --------
    pd.DataFrame
        DataFrame with added drawdown columns
    """
    df = df.copy()
    prices = df[price_column]
    
    # Calculate running maximum
    running_max = prices.expanding().max()
    
    # Calculate drawdown
    df['drawdown'] = (prices - running_max) / running_max
    df['drawdown_pct'] = df['drawdown'] * 100
    
    # Calculate maximum drawdown
    df['max_drawdown'] = df['drawdown'].expanding().min()
    df['max_drawdown_pct'] = df['max_drawdown'] * 100
    
    return df


def calculate_beta(
    df: pd.DataFrame,
    market_returns: pd.Series,
    returns_column: str = 'returns',
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate Beta (sensitivity to market movements).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock returns data
    market_returns : pd.Series
        Market returns (e.g., S&P 500 returns)
    returns_column : str
        Column name for stock returns (default: 'returns')
    window : int
        Rolling window for calculation (default: 252)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added beta column
    """
    df = df.copy()
    
    if returns_column not in df.columns:
        df = calculate_returns(df)
        returns_column = 'returns'
    
    # Align market returns with stock returns
    aligned_returns = df[returns_column]
    aligned_market = market_returns.reindex(aligned_returns.index).fillna(method='ffill')
    
    # Calculate rolling beta using covariance and variance
    rolling_cov = aligned_returns.rolling(window=window).cov(aligned_market)
    rolling_var = aligned_market.rolling(window=window).var()
    
    df[f'beta_{window}'] = rolling_cov / rolling_var
    
    return df


def calculate_alpha(
    df: pd.DataFrame,
    market_returns: pd.Series,
    returns_column: str = 'returns',
    risk_free_rate: float = 0.02,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate Alpha (excess return over market).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock returns data
    market_returns : pd.Series
        Market returns
    returns_column : str
        Column name for stock returns (default: 'returns')
    risk_free_rate : float
        Annual risk-free rate (default: 0.02)
    window : int
        Rolling window for calculation (default: 252)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added alpha column
    """
    df = df.copy()
    
    if returns_column not in df.columns:
        df = calculate_returns(df)
        returns_column = 'returns'
    
    # Calculate excess returns
    daily_rf_rate = risk_free_rate / 252
    stock_excess = df[returns_column] - daily_rf_rate
    
    # Align market returns
    aligned_market = market_returns.reindex(df.index).fillna(method='ffill')
    market_excess = aligned_market - daily_rf_rate
    
    # Calculate rolling alpha
    stock_mean = stock_excess.rolling(window=window).mean()
    market_mean = market_excess.rolling(window=window).mean()
    
    # Alpha = stock_return - (risk_free_rate + beta * (market_return - risk_free_rate))
    # Simplified: alpha = stock_excess - beta * market_excess
    if f'beta_{window}' in df.columns:
        df[f'alpha_{window}'] = (stock_mean - df[f'beta_{window}'] * market_mean) * 252
    else:
        # Calculate beta first if not available
        df = calculate_beta(df, market_returns, returns_column, window)
        stock_mean = stock_excess.rolling(window=window).mean()
        market_mean = market_excess.rolling(window=window).mean()
        df[f'alpha_{window}'] = (stock_mean - df[f'beta_{window}'] * market_mean) * 252
    
    return df


def calculate_all_metrics(
    df: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Calculate all financial metrics at once.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    market_returns : pd.Series, optional
        Market returns for beta and alpha calculation
    risk_free_rate : float
        Annual risk-free rate (default: 0.02)

    Returns:
    --------
    pd.DataFrame
        DataFrame with all calculated metrics
    """
    df = df.copy()
    
    # Calculate returns
    df = calculate_returns(df)
    
    # Calculate volatility
    df = calculate_volatility(df)
    
    # Calculate Sharpe ratio
    df = calculate_sharpe_ratio(df, risk_free_rate=risk_free_rate)
    
    # Calculate drawdown
    df = calculate_max_drawdown(df)
    
    # Calculate beta and alpha if market returns provided
    if market_returns is not None:
        df = calculate_beta(df, market_returns)
        df = calculate_alpha(df, market_returns, risk_free_rate=risk_free_rate)
    
    return df


def get_metrics_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for all calculated metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with calculated metrics

    Returns:
    --------
    dict
        Dictionary with summary statistics
    """
    summary = {}
    
    # Returns summary
    if 'returns' in df.columns:
        returns = df['returns'].dropna()
        if len(returns) > 0:
            summary['returns'] = {
                'mean_daily': float(returns.mean()),
                'std_daily': float(returns.std()),
                'total_return': float((1 + returns).prod() - 1),
                'annualized_return': float((1 + returns.mean()) ** 252 - 1)
            }
    
    # Volatility summary
    vol_columns = [col for col in df.columns if 'volatility' in col and 'annualized' in col]
    for col in vol_columns:
        if col in df.columns:
            vol = df[col].dropna()
            if len(vol) > 0:
                summary[col] = {
                    'mean': float(vol.mean()),
                    'current': float(vol.iloc[-1]) if len(vol) > 0 else None
                }
    
    # Sharpe ratio summary
    sharpe_columns = [col for col in df.columns if 'sharpe_ratio' in col]
    for col in sharpe_columns:
        if col in df.columns:
            sharpe = df[col].dropna()
            if len(sharpe) > 0:
                summary[col] = {
                    'mean': float(sharpe.mean()),
                    'current': float(sharpe.iloc[-1]) if len(sharpe) > 0 else None
                }
    
    # Max drawdown summary
    if 'max_drawdown_pct' in df.columns:
        mdd = df['max_drawdown_pct'].dropna()
        if len(mdd) > 0:
            summary['max_drawdown'] = {
                'worst': float(mdd.min()),
                'current': float(mdd.iloc[-1]) if len(mdd) > 0 else None
            }
    
    return summary

