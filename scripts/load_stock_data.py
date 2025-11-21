"""
Load and prepare stock price data for quantitative analysis.

This module provides functions to load stock price data from various sources
and prepare it for technical analysis.
"""

import pandas as pd
import yfinance as yf
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


def load_stock_data_from_csv(
    ticker: str,
    data_dir: str = '../data/Data'
) -> pd.DataFrame:
    """
    Load stock price data from local CSV file.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    data_dir : str
        Directory containing CSV files (default: '../data/Data')

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    csv_path = os.path.join(data_dir, f'{ticker}.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Standardize column names (handle case variations)
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['date', 'datetime']:
                column_mapping[col] = 'date'
            elif col_lower == 'close':
                column_mapping[col] = 'Close'
            elif col_lower == 'open':
                column_mapping[col] = 'Open'
            elif col_lower == 'high':
                column_mapping[col] = 'High'
            elif col_lower == 'low':
                column_mapping[col] = 'Low'
            elif col_lower == 'volume':
                column_mapping[col] = 'Volume'
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        elif df.index.name in ['Date', 'date', 'DATE'] or isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            elif 'date' in df.columns:
                pass  # Already has date column
            elif isinstance(df.index, pd.DatetimeIndex):
                df['date'] = df.index
                df.reset_index(drop=True, inplace=True)
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add ticker symbol as a column
        df['ticker'] = ticker
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data from CSV for {ticker}: {str(e)}")


def load_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = "1y",
    use_local_data: bool = True,
    data_dir: str = '../data/Data'
) -> pd.DataFrame:
    """
    Load stock price data from local CSV file or yfinance.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, uses period.
        Only used if use_local_data=False
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses today.
        Only used if use_local_data=False
    period : str, optional
        Period to download data. Only used if use_local_data=False
    use_local_data : bool
        If True, load from local CSV files in data_dir. If False, download from yfinance.
        Default: True
    data_dir : str
        Directory containing CSV files (default: '../data/Data')

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
    """
    # Try to load from local CSV first if use_local_data is True
    if use_local_data:
        try:
            df = load_stock_data_from_csv(ticker, data_dir)
            print(f"✓ Loaded {ticker} data from local CSV file")
            return df
        except FileNotFoundError:
            print(f"⚠ Local CSV file not found for {ticker}, falling back to yfinance...")
        except Exception as e:
            print(f"⚠ Error loading local data for {ticker}: {e}")
            print("Falling back to yfinance...")
    
    # Fall back to yfinance if local data not available or use_local_data=False
    try:
        stock = yf.Ticker(ticker)
        
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date)
        else:
            df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data retrieved for ticker {ticker}")
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Rename Date column if it exists
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        
        # Add ticker symbol as a column
        df['ticker'] = ticker
        
        print(f"✓ Loaded {ticker} data from yfinance")
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data for {ticker}: {str(e)}")


def load_multiple_stocks(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = "1y",
    use_local_data: bool = True,
    data_dir: str = '../data/Data'
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple stock tickers.

    Parameters:
    -----------
    tickers : List[str]
        List of stock ticker symbols
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, optional
        Period to download data

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with ticker symbols as keys and DataFrames as values
    """
    stock_data = {}
    
    for ticker in tickers:
        try:
            df = load_stock_data(ticker, start_date, end_date, period, use_local_data, data_dir)
            stock_data[ticker] = df
            print(f"✓ Loaded data for {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Error loading {ticker}: {str(e)}")
            continue
    
    return stock_data


def prepare_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare stock data for analysis by ensuring proper data types and adding derived columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data

    Returns:
    --------
    pd.DataFrame
        Prepared DataFrame with additional columns
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
    
    # Ensure numeric columns are numeric
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate price change
    if 'Close' in df.columns:
        df['price_change'] = df['Close'].diff()
        df['price_change_pct'] = df['Close'].pct_change() * 100
    
    # Calculate typical price (High + Low + Close) / 3
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Sort by date
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    return df


def validate_stock_data(df: pd.DataFrame) -> Dict:
    """
    Validate stock data and return validation report.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate

    Returns:
    --------
    dict
        Validation report with statistics
    """
    report = {
        'total_rows': len(df),
        'date_range': None,
        'missing_values': {},
        'data_quality': {}
    }
    
    # Check date range
    if 'date' in df.columns:
        report['date_range'] = {
            'min': df['date'].min(),
            'max': df['date'].max()
        }
    
    # Check missing values
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float((missing_count / len(df)) * 100) if len(df) > 0 else 0
            }
    
    # Data quality checks
    if 'Close' in df.columns:
        report['data_quality']['price_range'] = {
            'min': float(df['Close'].min()),
            'max': float(df['Close'].max()),
            'mean': float(df['Close'].mean())
        }
    
    if 'Volume' in df.columns:
        report['data_quality']['volume_stats'] = {
            'min': float(df['Volume'].min()),
            'max': float(df['Volume'].max()),
            'mean': float(df['Volume'].mean())
        }
    
    return report

