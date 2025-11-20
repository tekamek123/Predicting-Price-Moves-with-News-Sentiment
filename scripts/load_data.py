"""
Data loading utilities for Financial News Sentiment Analysis
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional


def load_financial_news_data(file_path: str) -> pd.DataFrame:
    """
    Load financial news data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing financial news data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: headline, url, publisher, date, stock
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the loaded dataset and return validation report.
    
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
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'date_range': None,
        'unique_stocks': df['stock'].nunique() if 'stock' in df.columns else 0,
        'unique_publishers': df['publisher'].nunique() if 'publisher' in df.columns else 0,
    }
    
    if 'date' in df.columns and df['date'].notna().any():
        report['date_range'] = {
            'min': df['date'].min(),
            'max': df['date'].max()
        }
    
    return report


