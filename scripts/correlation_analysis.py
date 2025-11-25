"""
Correlation analysis between news sentiment and stock price movements
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_daily_returns(
    df: pd.DataFrame,
    price_column: str = 'Close',
    date_column: str = 'date',
    method: str = 'pct_change'
) -> pd.DataFrame:
    """
    Calculate daily percentage change in stock prices (daily returns).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock price data
    price_column : str
        Name of price column (default: 'Close')
    date_column : str
        Name of date column (default: 'date')
    method : str
        Calculation method: 'pct_change' (default) or 'log'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'daily_returns' column
    """
    df = df.copy()
    
    if price_column not in df.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame")
    
    # Sort by date to ensure proper calculation
    if date_column in df.columns:
        df = df.sort_values(date_column).reset_index(drop=True)
    
    # Calculate daily returns
    if method == 'pct_change':
        df['daily_returns'] = df[price_column].pct_change() * 100  # Convert to percentage
    elif method == 'log':
        df['daily_returns'] = np.log(df[price_column] / df[price_column].shift(1)) * 100
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pct_change' or 'log'")
    
    return df


def calculate_correlation(
    sentiment_scores: pd.Series,
    stock_returns: pd.Series,
    method: str = 'pearson'
) -> Dict[str, float]:
    """
    Calculate correlation between sentiment scores and stock returns.
    
    Parameters:
    -----------
    sentiment_scores : pd.Series
        Series of sentiment scores
    stock_returns : pd.Series
        Series of stock returns (daily percentage changes)
    method : str
        Correlation method: 'pearson' (default), 'spearman', or 'kendall'
    
    Returns:
    --------
    dict
        Dictionary with correlation coefficient, p-value, and statistics
    """
    # Align series by index (date)
    aligned_data = pd.DataFrame({
        'sentiment': sentiment_scores,
        'returns': stock_returns
    }).dropna()
    
    if len(aligned_data) < 2:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'sample_size': len(aligned_data),
            'method': method,
            'error': 'Insufficient data for correlation calculation'
        }
    
    sentiment = aligned_data['sentiment']
    returns = aligned_data['returns']
    
    # Calculate correlation
    if method == 'pearson':
        corr_coef, p_value = stats.pearsonr(sentiment, returns)
    elif method == 'spearman':
        corr_coef, p_value = stats.spearmanr(sentiment, returns)
    elif method == 'kendall':
        corr_coef, p_value = stats.kendalltau(sentiment, returns)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson', 'spearman', or 'kendall'")
    
    # Calculate additional statistics
    n = len(aligned_data)
    mean_sentiment = sentiment.mean()
    mean_returns = returns.mean()
    std_sentiment = sentiment.std()
    std_returns = returns.std()
    
    return {
        'correlation': corr_coef,
        'p_value': p_value,
        'sample_size': n,
        'method': method,
        'mean_sentiment': mean_sentiment,
        'mean_returns': mean_returns,
        'std_sentiment': std_sentiment,
        'std_returns': std_returns,
        'is_significant': p_value < 0.05 if not np.isnan(p_value) else False
    }


def analyze_sentiment_returns_correlation(
    df: pd.DataFrame,
    sentiment_column: str = 'daily_sentiment_score',
    returns_column: str = 'daily_returns',
    date_column: str = 'aligned_date',
    method: str = 'pearson'
) -> Dict[str, any]:
    """
    Perform comprehensive correlation analysis between sentiment and stock returns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with aligned sentiment scores and stock returns
    sentiment_column : str
        Name of sentiment score column (default: 'daily_sentiment_score')
    returns_column : str
        Name of returns column (default: 'daily_returns')
    date_column : str
        Name of date column (default: 'aligned_date')
    method : str
        Correlation method: 'pearson' (default), 'spearman', or 'kendall'
    
    Returns:
    --------
    dict
        Comprehensive correlation analysis results
    """
    if sentiment_column not in df.columns:
        raise ValueError(f"Column '{sentiment_column}' not found in DataFrame")
    if returns_column not in df.columns:
        raise ValueError(f"Column '{returns_column}' not found in DataFrame")
    
    # Prepare data
    analysis_df = df[[date_column, sentiment_column, returns_column]].copy()
    analysis_df = analysis_df.dropna()
    
    if len(analysis_df) < 2:
        return {
            'error': 'Insufficient data for correlation analysis',
            'sample_size': len(analysis_df)
        }
    
    sentiment = analysis_df[sentiment_column]
    returns = analysis_df[returns_column]
    
    # Calculate correlation
    correlation_result = calculate_correlation(sentiment, returns, method)
    
    # Calculate lagged correlations (sentiment today vs returns tomorrow)
    lagged_correlations = {}
    for lag in [1, 2, 3, 5]:
        if len(analysis_df) > lag:
            lagged_returns = returns.shift(-lag)
            lagged_corr = calculate_correlation(sentiment, lagged_returns, method)
            lagged_correlations[f'lag_{lag}'] = lagged_corr
    
    # Calculate lead correlations (sentiment today vs returns yesterday)
    lead_correlations = {}
    for lead in [1, 2, 3, 5]:
        if len(analysis_df) > lead:
            lead_returns = returns.shift(lead)
            lead_corr = calculate_correlation(sentiment, lead_returns, method)
            lead_correlations[f'lead_{lead}'] = lead_corr
    
    # Additional statistics
    positive_sentiment_days = analysis_df[sentiment > 0.1]
    negative_sentiment_days = analysis_df[sentiment < -0.1]
    neutral_sentiment_days = analysis_df[(sentiment >= -0.1) & (sentiment <= 0.1)]
    
    result = {
        'correlation': correlation_result,
        'lagged_correlations': lagged_correlations,
        'lead_correlations': lead_correlations,
        'date_range': {
            'start': analysis_df[date_column].min(),
            'end': analysis_df[date_column].max()
        },
        'sentiment_statistics': {
            'positive_days': len(positive_sentiment_days),
            'negative_days': len(negative_sentiment_days),
            'neutral_days': len(neutral_sentiment_days),
            'avg_returns_positive_sentiment': positive_sentiment_days[returns_column].mean() if len(positive_sentiment_days) > 0 else np.nan,
            'avg_returns_negative_sentiment': negative_sentiment_days[returns_column].mean() if len(negative_sentiment_days) > 0 else np.nan,
            'avg_returns_neutral_sentiment': neutral_sentiment_days[returns_column].mean() if len(neutral_sentiment_days) > 0 else np.nan
        }
    }
    
    return result


def get_correlation_summary(correlation_result: Dict) -> str:
    """
    Generate a human-readable summary of correlation results.
    
    Parameters:
    -----------
    correlation_result : dict
        Result from analyze_sentiment_returns_correlation
    
    Returns:
    --------
    str
        Formatted summary string
    """
    if 'error' in correlation_result:
        return f"Error: {correlation_result['error']}"
    
    corr = correlation_result['correlation']
    summary_lines = [
        "=" * 60,
        "CORRELATION ANALYSIS SUMMARY",
        "=" * 60,
        f"\nMethod: {corr['method'].upper()}",
        f"Sample Size: {corr['sample_size']} days",
        f"\nCorrelation Coefficient: {corr['correlation']:.4f}",
        f"P-value: {corr['p_value']:.6f}",
        f"Statistically Significant: {'Yes' if corr['is_significant'] else 'No'} (α=0.05)",
    ]
    
    # Interpret correlation strength
    abs_corr = abs(corr['correlation'])
    if abs_corr < 0.1:
        strength = "negligible"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.5:
        strength = "moderate"
    elif abs_corr < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if corr['correlation'] > 0 else "negative"
    summary_lines.append(f"\nCorrelation Strength: {strength} ({direction})")
    
    # Sentiment statistics
    if 'sentiment_statistics' in correlation_result:
        stats = correlation_result['sentiment_statistics']
        summary_lines.extend([
            "\n" + "-" * 60,
            "SENTIMENT-BASED RETURNS ANALYSIS",
            "-" * 60,
            f"\nPositive Sentiment Days: {stats['positive_days']}",
            f"  Average Returns: {stats['avg_returns_positive_sentiment']:.4f}%",
            f"\nNegative Sentiment Days: {stats['negative_days']}",
            f"  Average Returns: {stats['avg_returns_negative_sentiment']:.4f}%",
            f"\nNeutral Sentiment Days: {stats['neutral_days']}",
            f"  Average Returns: {stats['avg_returns_neutral_sentiment']:.4f}%",
        ])
    
    # Lagged correlations
    if correlation_result.get('lagged_correlations'):
        summary_lines.extend([
            "\n" + "-" * 60,
            "LAGGED CORRELATIONS (Sentiment → Future Returns)",
            "-" * 60,
        ])
        for lag, lag_corr in correlation_result['lagged_correlations'].items():
            summary_lines.append(
                f"{lag.replace('_', ' ').title()}: {lag_corr['correlation']:.4f} "
                f"(p={lag_corr['p_value']:.4f})"
            )
    
    summary_lines.append("\n" + "=" * 60)
    
    return "\n".join(summary_lines)

