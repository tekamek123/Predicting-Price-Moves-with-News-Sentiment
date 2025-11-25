"""
Date alignment utilities for aligning news and stock price data
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def normalize_dates_to_trading_days(
    news_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    news_date_col: str = 'date',
    stock_date_col: str = 'date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize dates in both datasets to align news with trading days.
    
    For news published on non-trading days (weekends, holidays), align to the next trading day.
    This ensures each news item matches the corresponding stock trading day.
    
    Parameters:
    -----------
    news_df : pd.DataFrame
        DataFrame with news data containing date column
    stock_df : pd.DataFrame
        DataFrame with stock price data containing date column
    news_date_col : str
        Name of date column in news_df (default: 'date')
    stock_date_col : str
        Name of date column in stock_df (default: 'date')
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (aligned_news_df, aligned_stock_df) with normalized dates
    """
    # Make copies to avoid modifying original dataframes
    news_df = news_df.copy()
    stock_df = stock_df.copy()
    
    # Ensure date columns are datetime
    news_df[news_date_col] = pd.to_datetime(news_df[news_date_col], errors='coerce')
    stock_df[stock_date_col] = pd.to_datetime(stock_df[stock_date_col], errors='coerce')
    
    # Remove rows with invalid dates
    news_df = news_df[news_df[news_date_col].notna()].copy()
    stock_df = stock_df[stock_df[stock_date_col].notna()].copy()
    
    # Extract date only (remove time component) for stock data
    stock_df['trading_date'] = stock_df[stock_date_col].dt.date
    stock_trading_dates = set(stock_df['trading_date'].unique())
    
    # Function to align news date to next trading day
    def align_to_trading_day(news_date):
        """Align news date to the next available trading day"""
        news_date_only = news_date.date()
        
        # If news date is already a trading day, use it
        if news_date_only in stock_trading_dates:
            return news_date_only
        
        # Otherwise, find the next trading day
        # Get all trading dates after the news date
        future_trading_dates = [td for td in stock_trading_dates if td >= news_date_only]
        
        if future_trading_dates:
            return min(future_trading_dates)
        else:
            # If no future trading date, use the latest available trading date
            return max(stock_trading_dates) if stock_trading_dates else None
    
    # Apply alignment to news data
    news_df['aligned_date'] = news_df[news_date_col].apply(align_to_trading_day)
    
    # Remove news items that couldn't be aligned
    news_df = news_df[news_df['aligned_date'].notna()].copy()
    
    # Convert aligned_date back to datetime for consistency
    news_df['aligned_date'] = pd.to_datetime(news_df['aligned_date'])
    
    # Create aligned date column in stock_df for merging
    stock_df['aligned_date'] = pd.to_datetime(stock_df['trading_date'])
    
    # Remove the temporary trading_date column
    stock_df = stock_df.drop(columns=['trading_date'])
    
    return news_df, stock_df


def align_news_and_stock_data(
    news_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    news_date_col: str = 'date',
    stock_date_col: str = 'date',
    stock_ticker_col: str = 'stock',
    filter_by_ticker: bool = True
) -> pd.DataFrame:
    """
    Align news and stock data by dates and optionally by stock ticker.
    
    Parameters:
    -----------
    news_df : pd.DataFrame
        DataFrame with news data (columns: headline, date, stock, etc.)
    stock_df : pd.DataFrame
        DataFrame with stock price data (columns: date, Close, etc.)
    news_date_col : str
        Name of date column in news_df (default: 'date')
    stock_date_col : str
        Name of date column in stock_df (default: 'date')
    stock_ticker_col : str
        Name of stock ticker column in news_df (default: 'stock')
    filter_by_ticker : bool
        If True, only align news with stock data for matching tickers
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with aligned news and stock data
    """
    # Normalize dates
    news_aligned, stock_aligned = normalize_dates_to_trading_days(
        news_df, stock_df, news_date_col, stock_date_col
    )
    
    # Prepare for merging
    if filter_by_ticker and stock_ticker_col in news_aligned.columns:
        # Get unique tickers from news data
        news_tickers = set(news_aligned[stock_ticker_col].dropna().unique())
        
        # Filter stock data to only include tickers present in news
        if 'ticker' in stock_aligned.columns:
            stock_aligned = stock_aligned[stock_aligned['ticker'].isin(news_tickers)].copy()
        
        # Merge on aligned_date and ticker
        merged_df = pd.merge(
            news_aligned,
            stock_aligned,
            left_on=['aligned_date', stock_ticker_col],
            right_on=['aligned_date', 'ticker'],
            how='inner',
            suffixes=('_news', '_stock')
        )
    else:
        # Merge only on aligned_date
        merged_df = pd.merge(
            news_aligned,
            stock_aligned,
            on='aligned_date',
            how='inner',
            suffixes=('_news', '_stock')
        )
    
    # Sort by date
    merged_df = merged_df.sort_values('aligned_date').reset_index(drop=True)
    
    return merged_df


def get_date_alignment_summary(
    news_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    news_date_col: str = 'date'
) -> dict:
    """
    Generate summary statistics for date alignment process.
    
    Parameters:
    -----------
    news_df : pd.DataFrame
        Original news DataFrame
    stock_df : pd.DataFrame
        Original stock DataFrame
    aligned_df : pd.DataFrame
        Aligned/merged DataFrame
    news_date_col : str
        Name of date column in news_df
    
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'original_news_count': len(news_df),
        'original_stock_count': len(stock_df),
        'aligned_records': len(aligned_df),
        'alignment_rate': len(aligned_df) / len(news_df) if len(news_df) > 0 else 0,
        'date_range_news': {
            'min': news_df[news_date_col].min() if news_date_col in news_df.columns else None,
            'max': news_df[news_date_col].max() if news_date_col in news_df.columns else None
        },
        'date_range_stock': {
            'min': stock_df['date'].min() if 'date' in stock_df.columns else None,
            'max': stock_df['date'].max() if 'date' in stock_df.columns else None
        },
        'date_range_aligned': {
            'min': aligned_df['aligned_date'].min() if 'aligned_date' in aligned_df.columns else None,
            'max': aligned_df['aligned_date'].max() if 'aligned_date' in aligned_df.columns else None
        }
    }
    
    return summary

