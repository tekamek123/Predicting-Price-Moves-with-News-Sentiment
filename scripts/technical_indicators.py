"""
Calculate technical indicators using TA-Lib.

This module provides functions to calculate various technical indicators
such as moving averages, RSI, MACD, and more.
"""

import pandas as pd
import numpy as np
import talib
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def calculate_moving_averages(
    df: pd.DataFrame,
    price_column: str = 'Close',
    periods: list = [20, 50, 200]
) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    price_column : str
        Column name for price data (default: 'Close')
    periods : list
        List of periods for moving averages (default: [20, 50, 200])

    Returns:
    --------
    pd.DataFrame
        DataFrame with added moving average columns
    """
    df = df.copy()
    prices = df[price_column].values
    
    for period in periods:
        # Simple Moving Average
        sma = talib.SMA(prices, timeperiod=period)
        df[f'SMA_{period}'] = sma
        
        # Exponential Moving Average
        ema = talib.EMA(prices, timeperiod=period)
        df[f'EMA_{period}'] = ema
    
    return df


def calculate_rsi(
    df: pd.DataFrame,
    price_column: str = 'Close',
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    price_column : str
        Column name for price data (default: 'Close')
    period : int
        RSI period (default: 14)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added RSI column
    """
    df = df.copy()
    prices = df[price_column].values
    
    rsi = talib.RSI(prices, timeperiod=period)
    df[f'RSI_{period}'] = rsi
    
    # Add RSI signals
    df[f'RSI_{period}_overbought'] = df[f'RSI_{period}'] > 70
    df[f'RSI_{period}_oversold'] = df[f'RSI_{period}'] < 30
    
    return df


def calculate_macd(
    df: pd.DataFrame,
    price_column: str = 'Close',
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    price_column : str
        Column name for price data (default: 'Close')
    fastperiod : int
        Fast EMA period (default: 12)
    slowperiod : int
        Slow EMA period (default: 26)
    signalperiod : int
        Signal line EMA period (default: 9)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added MACD columns
    """
    df = df.copy()
    prices = df[price_column].values
    
    macd, signal, histogram = talib.MACD(
        prices,
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod
    )
    
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_histogram'] = histogram
    
    # Add MACD signals
    df['MACD_bullish'] = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    df['MACD_bearish'] = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    
    return df


def calculate_bollinger_bands(
    df: pd.DataFrame,
    price_column: str = 'Close',
    period: int = 20,
    nbdevup: int = 2,
    nbdevdn: int = 2
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    price_column : str
        Column name for price data (default: 'Close')
    period : int
        Period for moving average (default: 20)
    nbdevup : int
        Number of standard deviations for upper band (default: 2)
    nbdevdn : int
        Number of standard deviations for lower band (default: 2)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added Bollinger Bands columns
    """
    df = df.copy()
    prices = df[price_column].values
    
    upper, middle, lower = talib.BBANDS(
        prices,
        timeperiod=period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
        matype=0
    )
    
    df[f'BB_upper_{period}'] = upper
    df[f'BB_middle_{period}'] = middle
    df[f'BB_lower_{period}'] = lower
    
    # Calculate %B (position within bands)
    df[f'BB_percentB_{period}'] = (prices - lower) / (upper - lower)
    
    # Add signals
    df[f'BB_squeeze_{period}'] = (upper - lower) < (upper - lower).rolling(20).mean()
    
    return df


def calculate_stochastic(
    df: pd.DataFrame,
    high_column: str = 'High',
    low_column: str = 'Low',
    close_column: str = 'Close',
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    high_column : str
        Column name for high prices
    low_column : str
        Column name for low prices
    close_column : str
        Column name for close prices
    fastk_period : int
        Fast %K period (default: 14)
    slowk_period : int
        Slow %K period (default: 3)
    slowd_period : int
        Slow %D period (default: 3)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added Stochastic columns
    """
    df = df.copy()
    high = df[high_column].values
    low = df[low_column].values
    close = df[close_column].values
    
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=0,
        slowd_period=slowd_period,
        slowd_matype=0
    )
    
    df['Stoch_K'] = slowk
    df['Stoch_D'] = slowd
    
    # Add signals
    df['Stoch_overbought'] = (slowk > 80) & (slowd > 80)
    df['Stoch_oversold'] = (slowk < 20) & (slowd < 20)
    
    return df


def calculate_all_indicators(
    df: pd.DataFrame,
    include_moving_averages: bool = True,
    include_rsi: bool = True,
    include_macd: bool = True,
    include_bollinger: bool = True,
    include_stochastic: bool = False
) -> pd.DataFrame:
    """
    Calculate all technical indicators at once.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    include_moving_averages : bool
        Whether to calculate moving averages
    include_rsi : bool
        Whether to calculate RSI
    include_macd : bool
        Whether to calculate MACD
    include_bollinger : bool
        Whether to calculate Bollinger Bands
    include_stochastic : bool
        Whether to calculate Stochastic Oscillator

    Returns:
    --------
    pd.DataFrame
        DataFrame with all calculated indicators
    """
    df = df.copy()
    
    if include_moving_averages:
        df = calculate_moving_averages(df)
    
    if include_rsi:
        df = calculate_rsi(df)
    
    if include_macd:
        df = calculate_macd(df)
    
    if include_bollinger:
        df = calculate_bollinger_bands(df)
    
    if include_stochastic:
        df = calculate_stochastic(df)
    
    return df


def get_indicator_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for all calculated indicators.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with calculated indicators

    Returns:
    --------
    dict
        Dictionary with summary statistics for each indicator
    """
    summary = {}
    
    # RSI summary
    rsi_columns = [col for col in df.columns if col.startswith('RSI_') and not col.endswith('_overbought') and not col.endswith('_oversold')]
    for col in rsi_columns:
        if col in df.columns:
            summary[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'current': float(df[col].iloc[-1]) if len(df) > 0 else None
            }
    
    # MACD summary
    if 'MACD' in df.columns:
        summary['MACD'] = {
            'mean': float(df['MACD'].mean()),
            'std': float(df['MACD'].std()),
            'current': float(df['MACD'].iloc[-1]) if len(df) > 0 else None
        }
    
    # Moving averages summary
    ma_columns = [col for col in df.columns if col.startswith('SMA_') or col.startswith('EMA_')]
    for col in ma_columns:
        if col in df.columns:
            summary[col] = {
                'mean': float(df[col].mean()),
                'current': float(df[col].iloc[-1]) if len(df) > 0 else None
            }
    
    return summary

