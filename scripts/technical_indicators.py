"""
Calculate technical indicators using TA-Lib or fallback methods.

This module provides functions to calculate various technical indicators
such as moving averages, RSI, MACD, and more.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Try to import TA-Lib, fallback to pandas-ta or manual calculations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    try:
        import pandas_ta as ta
        PANDAS_TA_AVAILABLE = True
        print("Warning: TA-Lib not available. Using pandas-ta instead.")
    except ImportError:
        PANDAS_TA_AVAILABLE = False
        print("Warning: Neither TA-Lib nor pandas-ta available. Using manual calculations.")


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
    
    for period in periods:
        if TALIB_AVAILABLE:
            # Use TA-Lib
            prices = df[price_column].values
            sma = talib.SMA(prices, timeperiod=period)
            ema = talib.EMA(prices, timeperiod=period)
            df[f'SMA_{period}'] = sma
            df[f'EMA_{period}'] = ema
        elif PANDAS_TA_AVAILABLE:
            # Use pandas-ta
            df.ta.sma(length=period, append=True)
            df.ta.ema(length=period, append=True)
            # Rename columns to match expected format
            if f'SMA_{period}' in df.columns:
                df.rename(columns={f'SMA_{period}': f'SMA_{period}'}, inplace=True)
            if f'EMA_{period}' in df.columns:
                df.rename(columns={f'EMA_{period}': f'EMA_{period}'}, inplace=True)
        else:
            # Manual calculation
            df[f'SMA_{period}'] = df[price_column].rolling(window=period).mean()
            df[f'EMA_{period}'] = df[price_column].ewm(span=period, adjust=False).mean()
    
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
    
    if TALIB_AVAILABLE:
        prices = df[price_column].values
        rsi = talib.RSI(prices, timeperiod=period)
        df[f'RSI_{period}'] = rsi
    elif PANDAS_TA_AVAILABLE:
        df.ta.rsi(length=period, append=True)
        if f'RSI_{period}' not in df.columns:
            # pandas-ta might use different naming
            rsi_cols = [col for col in df.columns if 'RSI' in col.upper()]
            if rsi_cols:
                df[f'RSI_{period}'] = df[rsi_cols[0]]
    else:
        # Manual RSI calculation
        delta = df[price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
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
    
    if TALIB_AVAILABLE:
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
    elif PANDAS_TA_AVAILABLE:
        df.ta.macd(fast=fastperiod, slow=slowperiod, signal=signalperiod, append=True)
        # pandas-ta uses MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        macd_cols = [col for col in df.columns if 'MACD' in col.upper() and not col.endswith('_histogram')]
        if len(macd_cols) >= 2:
            df['MACD'] = df[macd_cols[0]]
            df['MACD_signal'] = df[macd_cols[1]]
            if len(macd_cols) >= 3:
                df['MACD_histogram'] = df[macd_cols[2]]
    else:
        # Manual MACD calculation
        ema_fast = df[price_column].ewm(span=fastperiod, adjust=False).mean()
        ema_slow = df[price_column].ewm(span=slowperiod, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=signalperiod, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Add MACD signals
    df['MACD_bullish'] = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    df['MACD_bearish'] = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    
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
    
    if TALIB_AVAILABLE:
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
    elif PANDAS_TA_AVAILABLE:
        df.ta.bbands(length=period, std=nbdevup, append=True)
        # pandas-ta uses BBU_20_2.0, BBM_20_2.0, BBL_20_2.0
        bb_cols = [col for col in df.columns if 'BB' in col.upper()]
        if len(bb_cols) >= 3:
            df[f'BB_upper_{period}'] = df[bb_cols[0]]
            df[f'BB_middle_{period}'] = df[bb_cols[1]]
            df[f'BB_lower_{period}'] = df[bb_cols[2]]
    else:
        # Manual Bollinger Bands calculation
        sma = df[price_column].rolling(window=period).mean()
        std = df[price_column].rolling(window=period).std()
        df[f'BB_upper_{period}'] = sma + (std * nbdevup)
        df[f'BB_middle_{period}'] = sma
        df[f'BB_lower_{period}'] = sma - (std * nbdevdn)
    
    # Calculate %B (position within bands)
    prices = df[price_column].values
    df[f'BB_percentB_{period}'] = (prices - df[f'BB_lower_{period}']) / (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'])
    
    # Add signals
    df[f'BB_squeeze_{period}'] = (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']) < (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']).rolling(20).mean()
    
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
    
    if TALIB_AVAILABLE:
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
    elif PANDAS_TA_AVAILABLE:
        df.ta.stoch(append=True)
        stoch_cols = [col for col in df.columns if 'STOCH' in col.upper()]
        if len(stoch_cols) >= 2:
            df['Stoch_K'] = df[stoch_cols[0]]
            df['Stoch_D'] = df[stoch_cols[1]]
    else:
        # Manual Stochastic calculation
        low_min = df[low_column].rolling(window=fastk_period).min()
        high_max = df[high_column].rolling(window=fastk_period).max()
        k_percent = 100 * ((df[close_column] - low_min) / (high_max - low_min))
        df['Stoch_K'] = k_percent.rolling(window=slowk_period).mean()
        df['Stoch_D'] = df['Stoch_K'].rolling(window=slowd_period).mean()
    
    # Add signals
    df['Stoch_overbought'] = (df['Stoch_K'] > 80) & (df['Stoch_D'] > 80)
    df['Stoch_oversold'] = (df['Stoch_K'] < 20) & (df['Stoch_D'] < 20)
    
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

