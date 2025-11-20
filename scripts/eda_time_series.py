"""
Time Series Analysis for Financial News Publication Frequency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def analyze_publication_frequency(df: pd.DataFrame) -> Dict:
    """
    Analyze how publication frequency varies over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'date' column
        
    Returns:
    --------
    dict
        Dictionary containing various frequency analyses
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    
    df = df.copy()
    df = df[df['date'].notna()].copy()
    df = df.sort_values('date')
    
    # Set date as index for time series operations
    df['date_only'] = df['date'].dt.date
    df['datetime'] = pd.to_datetime(df['date'])
    
    # Daily frequency
    daily_counts = df.groupby(df['datetime'].dt.date).size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['datetime'])
    
    # Weekly frequency
    weekly_counts = df.groupby(df['datetime'].dt.to_period('W')).size().reset_index(name='count')
    weekly_counts['week'] = weekly_counts['datetime'].astype(str)
    
    # Monthly frequency
    monthly_counts = df.groupby(df['datetime'].dt.to_period('M')).size().reset_index(name='count')
    monthly_counts['month'] = monthly_counts['datetime'].astype(str)
    
    # Calculate statistics
    stats = {
        'daily_mean': daily_counts['count'].mean(),
        'daily_std': daily_counts['count'].std(),
        'daily_median': daily_counts['count'].median(),
        'max_daily': daily_counts['count'].max(),
        'min_daily': daily_counts['count'].min(),
        'total_articles': len(df),
        'total_days': daily_counts['count'].shape[0],
        'avg_articles_per_day': len(df) / daily_counts['count'].shape[0] if daily_counts['count'].shape[0] > 0 else 0
    }
    
    # Identify spikes (days with significantly more articles)
    mean = daily_counts['count'].mean()
    std = daily_counts['count'].std()
    threshold = mean + 2 * std
    spikes = daily_counts[daily_counts['count'] > threshold].copy()
    
    analysis = {
        'daily_counts': daily_counts,
        'weekly_counts': weekly_counts,
        'monthly_counts': monthly_counts,
        'statistics': stats,
        'spikes': spikes,
        'date_range': {
            'start': df['datetime'].min(),
            'end': df['datetime'].max()
        }
    }
    
    return analysis


def analyze_publishing_times(df: pd.DataFrame) -> Dict:
    """
    Analyze publishing times to identify peak hours.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'date' column containing datetime information
        
    Returns:
    --------
    dict
        Dictionary containing hourly and time-based analyses
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    
    df = df.copy()
    df = df[df['date'].notna()].copy()
    
    # Extract time components
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['time_of_day'] = pd.cut(df['hour'], 
                               bins=[0, 6, 12, 18, 24],
                               labels=['Night (0-6)', 'Morning (6-12)', 
                                      'Afternoon (12-18)', 'Evening (18-24)'],
                               include_lowest=True)
    
    # Hourly distribution
    hourly_counts = df['hour'].value_counts().sort_index().reset_index()
    hourly_counts.columns = ['hour', 'count']
    
    # Time of day distribution
    time_of_day_counts = df['time_of_day'].value_counts().reset_index()
    time_of_day_counts.columns = ['time_of_day', 'count']
    
    # Peak hours (top 3 hours with most publications)
    peak_hours = hourly_counts.nlargest(3, 'count')
    
    analysis = {
        'hourly_distribution': hourly_counts,
        'time_of_day_distribution': time_of_day_counts,
        'peak_hours': peak_hours,
        'most_active_hour': hourly_counts.loc[hourly_counts['count'].idxmax(), 'hour'],
        'least_active_hour': hourly_counts.loc[hourly_counts['count'].idxmin(), 'hour']
    }
    
    return analysis


def identify_market_events(df: pd.DataFrame, spike_threshold: float = 2.0) -> pd.DataFrame:
    """
    Identify potential market events based on publication spikes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'date' column
    spike_threshold : float
        Number of standard deviations above mean to consider a spike
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with identified spike dates and potential events
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    
    frequency_analysis = analyze_publication_frequency(df)
    daily_counts = frequency_analysis['daily_counts']
    
    mean = daily_counts['count'].mean()
    std = daily_counts['count'].std()
    threshold = mean + spike_threshold * std
    
    spikes = daily_counts[daily_counts['count'] > threshold].copy()
    spikes = spikes.sort_values('count', ascending=False)
    
    # Add spike intensity (how many std above mean)
    spikes['spike_intensity'] = (spikes['count'] - mean) / std
    
    return spikes


def plot_publication_frequency(analysis: Dict, save_path: Optional[str] = None):
    """
    Plot publication frequency over time.
    
    Parameters:
    -----------
    analysis : dict
        Dictionary from analyze_publication_frequency
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Daily trend
    daily = analysis['daily_counts']
    axes[0].plot(daily['date'], daily['count'], linewidth=1, alpha=0.7, color='steelblue')
    axes[0].axhline(analysis['statistics']['daily_mean'], color='red', 
                   linestyle='--', label=f"Mean: {analysis['statistics']['daily_mean']:.1f}")
    axes[0].fill_between(daily['date'], 
                        analysis['statistics']['daily_mean'] - analysis['statistics']['daily_std'],
                        analysis['statistics']['daily_mean'] + analysis['statistics']['daily_std'],
                        alpha=0.2, color='gray', label='±1 Std Dev')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Number of Articles')
    axes[0].set_title('Daily Publication Frequency Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Weekly trend
    weekly = analysis['weekly_counts']
    axes[1].bar(range(len(weekly)), weekly['count'], color='coral', edgecolor='black')
    axes[1].set_xticks(range(0, len(weekly), max(1, len(weekly)//10)))
    axes[1].set_xticklabels(weekly['week'][::max(1, len(weekly)//10)], rotation=45, ha='right')
    axes[1].set_xlabel('Week')
    axes[1].set_ylabel('Number of Articles')
    axes[1].set_title('Weekly Publication Frequency')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Monthly trend
    monthly = analysis['monthly_counts']
    axes[2].bar(range(len(monthly)), monthly['count'], color='lightgreen', edgecolor='black')
    axes[2].set_xticks(range(len(monthly)))
    axes[2].set_xticklabels(monthly['month'], rotation=45, ha='right')
    axes[2].set_xlabel('Month')
    axes[2].set_ylabel('Number of Articles')
    axes[2].set_title('Monthly Publication Frequency')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_publishing_times(analysis: Dict, save_path: Optional[str] = None):
    """
    Plot publishing time distributions.
    
    Parameters:
    -----------
    analysis : dict
        Dictionary from analyze_publishing_times
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Hourly distribution
    hourly = analysis['hourly_distribution']
    axes[0].bar(hourly['hour'], hourly['count'], color='steelblue', edgecolor='black')
    axes[0].axvline(analysis['most_active_hour'], color='red', linestyle='--', 
                   label=f"Peak Hour: {analysis['most_active_hour']}:00")
    axes[0].set_xlabel('Hour of Day (UTC-4)')
    axes[0].set_ylabel('Number of Articles')
    axes[0].set_title('Articles Published by Hour of Day')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Time of day distribution
    time_of_day = analysis['time_of_day_distribution']
    axes[1].bar(time_of_day['time_of_day'], time_of_day['count'], 
               color='gold', edgecolor='black')
    axes[1].set_xlabel('Time of Day')
    axes[1].set_ylabel('Number of Articles')
    axes[1].set_title('Articles Published by Time of Day')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_spikes(spikes: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot identified publication spikes.
    
    Parameters:
    -----------
    spikes : pd.DataFrame
        DataFrame with spike dates and intensities
    save_path : str, optional
        Path to save the plot
    """
    if len(spikes) == 0:
        print("No spikes identified.")
        return
    
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(spikes)), spikes['count'], 
           color='red', edgecolor='black', alpha=0.7)
    plt.xticks(range(len(spikes)), 
              [pd.to_datetime(date).strftime('%Y-%m-%d') for date in spikes['date']],
              rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.title('Publication Spikes (Potential Market Events)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(spikes['count']):
        plt.text(i, v + max(spikes['count']) * 0.01, str(int(v)), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Time Series Analysis Module")
    print("Import this module and use the functions with your data.")


