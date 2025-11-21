"""
Visualize technical indicators and financial metrics.

This module provides functions to create visualizations for
technical analysis and financial metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_price_with_indicators(
    df: pd.DataFrame,
    price_column: str = 'Close',
    indicators: Optional[List[str]] = None,
    title: str = "Stock Price with Technical Indicators",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10)
):
    """
    Plot stock price with technical indicators.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price and indicator data
    price_column : str
        Column name for price data
    indicators : List[str], optional
        List of indicator column names to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Price with Moving Averages
    ax1 = axes[0]
    ax1.plot(df.index, df[price_column], label='Close Price', linewidth=2, color='black')
    
    # Add moving averages
    ma_columns = [col for col in df.columns if col.startswith('SMA_') or col.startswith('EMA_')]
    for col in ma_columns[:3]:  # Limit to 3 to avoid clutter
        ax1.plot(df.index, df[col], label=col, alpha=0.7, linestyle='--')
    
    # Add Bollinger Bands if available
    bb_upper = [col for col in df.columns if 'BB_upper' in col]
    bb_lower = [col for col in df.columns if 'BB_lower' in col]
    if bb_upper and bb_lower:
        ax1.fill_between(df.index, df[bb_upper[0]], df[bb_lower[0]], 
                        alpha=0.2, label='Bollinger Bands', color='gray')
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Price and Moving Averages', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MACD
    ax2 = axes[1]
    if 'MACD' in df.columns:
        ax2.plot(df.index, df['MACD'], label='MACD', linewidth=2, color='blue')
        ax2.plot(df.index, df['MACD_signal'], label='Signal', linewidth=2, color='red')
        ax2.bar(df.index, df['MACD_histogram'], label='Histogram', alpha=0.3, color='green')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('MACD', fontsize=12)
        ax2.set_title('MACD Indicator', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: RSI
    ax3 = axes[2]
    rsi_columns = [col for col in df.columns if col.startswith('RSI_') and 
                   not col.endswith('_overbought') and not col.endswith('_oversold')]
    if rsi_columns:
        rsi_col = rsi_columns[0]
        ax3.plot(df.index, df[rsi_col], label=rsi_col, linewidth=2, color='purple')
        ax3.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
        ax3.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
        ax3.fill_between(df.index, 30, 70, alpha=0.1, color='yellow')
        ax3.set_ylabel('RSI', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title('RSI Indicator', fontsize=14)
        ax3.set_ylim(0, 100)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_returns_analysis(
    df: pd.DataFrame,
    returns_column: str = 'returns',
    title: str = "Returns Analysis",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10)
):
    """
    Plot returns analysis including distribution and cumulative returns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with returns data
    returns_column : str
        Column name for returns
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    returns = df[returns_column].dropna()
    
    # Plot 1: Returns over time
    ax1 = axes[0, 0]
    ax1.plot(df.index, returns * 100, alpha=0.7, linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Returns (%)', fontsize=12)
    ax1.set_title('Daily Returns', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Returns distribution
    ax2 = axes[0, 1]
    ax2.hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=returns.mean() * 100, color='red', linestyle='--', 
                label=f'Mean: {returns.mean()*100:.2f}%')
    ax2.set_xlabel('Returns (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Returns Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative returns
    ax3 = axes[1, 0]
    cumulative_returns = (1 + returns).cumprod() - 1
    ax3.plot(df.index, cumulative_returns * 100, linewidth=2, color='green')
    ax3.set_ylabel('Cumulative Returns (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Cumulative Returns', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drawdown
    ax4 = axes[1, 1]
    if 'drawdown_pct' in df.columns:
        ax4.fill_between(df.index, df['drawdown_pct'], 0, alpha=0.3, color='red')
        ax4.set_ylabel('Drawdown (%)', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_title('Drawdown', fontsize=14)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_risk_metrics(
    df: pd.DataFrame,
    title: str = "Risk Metrics Analysis",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10)
):
    """
    Plot risk metrics including volatility and Sharpe ratio.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with risk metrics
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Volatility
    ax1 = axes[0, 0]
    vol_columns = [col for col in df.columns if 'volatility' in col and 'annualized' in col]
    if vol_columns:
        ax1.plot(df.index, df[vol_columns[0]] * 100, linewidth=2, color='blue')
        ax1.set_ylabel('Volatility (%)', fontsize=12)
        ax1.set_title('Annualized Volatility', fontsize=14)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sharpe Ratio
    ax2 = axes[0, 1]
    sharpe_columns = [col for col in df.columns if 'sharpe_ratio' in col]
    if sharpe_columns:
        ax2.plot(df.index, df[sharpe_columns[0]], linewidth=2, color='green')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.set_title('Sharpe Ratio', fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Beta (if available)
    ax3 = axes[1, 0]
    beta_columns = [col for col in df.columns if 'beta' in col]
    if beta_columns:
        ax3.plot(df.index, df[beta_columns[0]], linewidth=2, color='orange')
        ax3.axhline(y=1, color='red', linestyle='--', label='Market Beta (1.0)')
        ax3.set_ylabel('Beta', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title('Beta (Market Sensitivity)', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Alpha (if available)
    ax4 = axes[1, 1]
    alpha_columns = [col for col in df.columns if 'alpha' in col]
    if alpha_columns:
        ax4.plot(df.index, df[alpha_columns[0]] * 100, linewidth=2, color='purple')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_ylabel('Alpha (%)', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_title('Alpha (Excess Return)', fontsize=14)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_indicator_comparison(
    df: pd.DataFrame,
    indicators: List[str],
    title: str = "Indicator Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 8)
):
    """
    Compare multiple indicators on the same plot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with indicator data
    indicators : List[str]
        List of indicator column names to compare
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for indicator in indicators:
        if indicator in df.columns:
            ax.plot(df.index, df[indicator], label=indicator, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Indicator Correlation Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10)
):
    """
    Plot correlation heatmap for indicators.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with indicator data
    columns : List[str], optional
        List of columns to include in correlation
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    if columns is None:
        # Select numeric columns (indicators)
        columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        # Exclude date and index columns
        columns = [col for col in columns if col not in ['date', 'index']]
    
    # Calculate correlation
    corr_data = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

