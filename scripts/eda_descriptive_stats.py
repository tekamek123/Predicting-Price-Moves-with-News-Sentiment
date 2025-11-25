"""
Descriptive Statistics Analysis for Financial News Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic statistics for textual lengths (headline length).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'headline' column
        
    Returns:
    --------
    pd.DataFrame
        Statistics summary
    """
    if 'headline' not in df.columns:
        raise ValueError("DataFrame must contain 'headline' column")
    
    df = df.copy()
    df['headline_length'] = df['headline'].astype(str).str.len()
    df['headline_word_count'] = df['headline'].astype(str).str.split().str.len()
    
    stats = {
        'headline_length': {
            'mean': df['headline_length'].mean(),
            'median': df['headline_length'].median(),
            'std': df['headline_length'].std(),
            'min': df['headline_length'].min(),
            'max': df['headline_length'].max(),
            'q25': df['headline_length'].quantile(0.25),
            'q75': df['headline_length'].quantile(0.75),
        },
        'headline_word_count': {
            'mean': df['headline_word_count'].mean(),
            'median': df['headline_word_count'].median(),
            'std': df['headline_word_count'].std(),
            'min': df['headline_word_count'].min(),
            'max': df['headline_word_count'].max(),
            'q25': df['headline_word_count'].quantile(0.25),
            'q75': df['headline_word_count'].quantile(0.75),
        }
    }
    
    return pd.DataFrame(stats), df


def count_articles_per_publisher(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of articles per publisher.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'publisher' column
        
    Returns:
    --------
    pd.DataFrame
        Count of articles per publisher, sorted by count
    """
    if 'publisher' not in df.columns:
        raise ValueError("DataFrame must contain 'publisher' column")
    
    publisher_counts = df['publisher'].value_counts().reset_index()
    publisher_counts.columns = ['publisher', 'article_count']
    publisher_counts = publisher_counts.sort_values('article_count', ascending=False)
    
    return publisher_counts


def analyze_publication_dates(df: pd.DataFrame) -> Dict:
    """
    Analyze publication dates to see trends over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'date' column
        
    Returns:
    --------
    dict
        Analysis results including daily, weekly, monthly trends
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    
    df = df.copy()
    df = df[df['date'].notna()].copy()
    
    # Extract time components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['date_only'] = df['date'].dt.date
    
    analysis = {
        'articles_per_day': df.groupby('date_only').size().reset_index(name='count'),
        'articles_per_weekday': df['day_of_week'].value_counts().to_dict(),
        'articles_per_month': df.groupby(['year', 'month']).size().reset_index(name='count'),
        'articles_per_hour': df['hour'].value_counts().sort_index().to_dict(),
        'total_days': df['date_only'].nunique(),
        'date_range': {
            'start': df['date'].min(),
            'end': df['date'].max()
        }
    }
    
    return analysis, df


def plot_text_statistics(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot distributions of headline lengths and word counts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with headline_length and headline_word_count columns
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Headline length distribution
    axes[0].hist(df['headline_length'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['headline_length'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["headline_length"].mean():.1f}')
    axes[0].axvline(df['headline_length'].median(), color='green', linestyle='--', 
                    label=f'Median: {df["headline_length"].median():.1f}')
    axes[0].set_xlabel('Headline Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Headline Lengths')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Word count distribution
    axes[1].hist(df['headline_word_count'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(df['headline_word_count'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["headline_word_count"].mean():.1f}')
    axes[1].axvline(df['headline_word_count'].median(), color='green', linestyle='--', 
                    label=f'Median: {df["headline_word_count"].median():.1f}')
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Headline Word Counts')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_publisher_analysis(publisher_counts: pd.DataFrame, top_n: int = 20, 
                           save_path: Optional[str] = None):
    """
    Plot top publishers by article count.
    
    Parameters:
    -----------
    publisher_counts : pd.DataFrame
        DataFrame with publisher and article_count columns
    top_n : int
        Number of top publishers to display
    save_path : str, optional
        Path to save the plot
    """
    top_publishers = publisher_counts.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_publishers)), top_publishers['article_count'], 
             color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_publishers)), top_publishers['publisher'])
    plt.xlabel('Number of Articles')
    plt.ylabel('Publisher')
    plt.title(f'Top {top_n} Publishers by Article Count')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(top_publishers['article_count']):
        plt.text(v + max(top_publishers['article_count']) * 0.01, i, 
                str(v), va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_publication_trends(analysis: Dict, save_path: Optional[str] = None):
    """
    Plot publication trends over time.
    
    Parameters:
    -----------
    analysis : dict
        Dictionary containing publication trend analysis
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Daily trend
    daily = analysis['articles_per_day']
    axes[0, 0].plot(pd.to_datetime(daily['date_only']), daily['count'], 
                    linewidth=1, alpha=0.7)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Articles')
    axes[0, 0].set_title('Daily Publication Trend')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Weekly trend (day of week)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = {day: analysis['articles_per_weekday'].get(day, 0) 
                     for day in weekday_order}
    axes[0, 1].bar(weekday_order, [weekday_counts[day] for day in weekday_order], 
                   color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Number of Articles')
    axes[0, 1].set_title('Articles Published by Day of Week')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Monthly trend
    monthly = analysis['articles_per_month']
    monthly['year_month'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
    axes[1, 0].bar(range(len(monthly)), monthly['count'], color='lightgreen', edgecolor='black')
    axes[1, 0].set_xticks(range(len(monthly)))
    axes[1, 0].set_xticklabels(monthly['year_month'], rotation=45, ha='right')
    axes[1, 0].set_xlabel('Year-Month')
    axes[1, 0].set_ylabel('Number of Articles')
    axes[1, 0].set_title('Monthly Publication Trend')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Hourly trend
    hours = sorted(analysis['articles_per_hour'].keys())
    hour_counts = [analysis['articles_per_hour'][h] for h in hours]
    axes[1, 1].bar(hours, hour_counts, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('Hour of Day (UTC-4)')
    axes[1, 1].set_ylabel('Number of Articles')
    axes[1, 1].set_title('Articles Published by Hour of Day')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Descriptive Statistics Analysis Module")
    print("Import this module and use the functions with your data.")

