"""
Publisher Analysis for Financial News Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import re
import warnings
warnings.filterwarnings('ignore')


def analyze_publishers(df: pd.DataFrame) -> Dict:
    """
    Comprehensive analysis of publishers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'publisher' column
        
    Returns:
    --------
    dict
        Dictionary containing various publisher analyses
    """
    if 'publisher' not in df.columns:
        raise ValueError("DataFrame must contain 'publisher' column")
    
    # Basic counts
    publisher_counts = df['publisher'].value_counts().reset_index()
    publisher_counts.columns = ['publisher', 'article_count']
    publisher_counts['percentage'] = (publisher_counts['article_count'] / len(df)) * 100
    
    # Statistics
    stats = {
        'total_publishers': df['publisher'].nunique(),
        'total_articles': len(df),
        'avg_articles_per_publisher': len(df) / df['publisher'].nunique(),
        'top_publisher': publisher_counts.iloc[0]['publisher'],
        'top_publisher_count': publisher_counts.iloc[0]['article_count'],
        'top_publisher_percentage': publisher_counts.iloc[0]['percentage']
    }
    
    # Publisher diversity (concentration)
    # Calculate Herfindahl-Hirschman Index (HHI) for market concentration
    market_shares = publisher_counts['percentage'] / 100
    hhi = (market_shares ** 2).sum() * 10000  # HHI is typically scaled to 0-10000
    
    stats['hhi'] = hhi
    stats['concentration_level'] = 'Highly Concentrated' if hhi > 2500 else \
                                   'Moderately Concentrated' if hhi > 1500 else \
                                   'Low Concentration'
    
    analysis = {
        'publisher_counts': publisher_counts,
        'statistics': stats
    }
    
    return analysis


def identify_publisher_domains(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify unique domains from publisher email addresses.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'publisher' column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with domain counts
    """
    if 'publisher' not in df.columns:
        raise ValueError("DataFrame must contain 'publisher' column")
    
    domains = []
    
    for publisher in df['publisher'].astype(str):
        # Check if it's an email address
        if '@' in publisher:
            domain = publisher.split('@')[1] if '@' in publisher else publisher
            domains.append(domain)
        else:
            domains.append('N/A')
    
    domain_counts = pd.Series(domains).value_counts().reset_index()
    domain_counts.columns = ['domain', 'count']
    domain_counts['percentage'] = (domain_counts['count'] / len(df)) * 100
    
    return domain_counts.sort_values('count', ascending=False)


def analyze_publisher_content(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """
    Analyze content differences between publishers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'publisher' and 'headline' columns
    top_n : int
        Number of top publishers to analyze
        
    Returns:
    --------
    dict
        Dictionary with content analysis per publisher
    """
    if 'publisher' not in df.columns or 'headline' not in df.columns:
        raise ValueError("DataFrame must contain 'publisher' and 'headline' columns")
    
    top_publishers = df['publisher'].value_counts().head(top_n).index.tolist()
    
    publisher_content = {}
    
    for publisher in top_publishers:
        publisher_df = df[df['publisher'] == publisher]
        
        # Average headline length
        avg_length = publisher_df['headline'].astype(str).str.len().mean()
        
        # Average word count
        avg_words = publisher_df['headline'].astype(str).str.split().str.len().mean()
        
        # Number of unique stocks covered
        unique_stocks = publisher_df['stock'].nunique() if 'stock' in df.columns else 0
        
        publisher_content[publisher] = {
            'article_count': len(publisher_df),
            'avg_headline_length': avg_length,
            'avg_word_count': avg_words,
            'unique_stocks_covered': unique_stocks,
            'percentage_of_total': (len(publisher_df) / len(df)) * 100
        }
    
    return publisher_content


def analyze_publisher_timing(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """
    Analyze publishing patterns by publisher.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'publisher' and 'date' columns
    top_n : int
        Number of top publishers to analyze
        
    Returns:
    --------
    dict
        Dictionary with timing analysis per publisher
    """
    if 'publisher' not in df.columns or 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'publisher' and 'date' columns")
    
    df = df[df['date'].notna()].copy()
    top_publishers = df['publisher'].value_counts().head(top_n).index.tolist()
    
    publisher_timing = {}
    
    for publisher in top_publishers:
        publisher_df = df[df['publisher'] == publisher]
        
        # Peak hour
        peak_hour = publisher_df['date'].dt.hour.mode()[0] if len(publisher_df) > 0 else None
        
        # Most active day of week
        peak_day = publisher_df['date'].dt.day_name().mode()[0] if len(publisher_df) > 0 else None
        
        # Publishing frequency (articles per day on average)
        date_range = (publisher_df['date'].max() - publisher_df['date'].min()).days
        freq_per_day = len(publisher_df) / max(date_range, 1) if date_range > 0 else 0
        
        publisher_timing[publisher] = {
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'articles_per_day': freq_per_day,
            'total_articles': len(publisher_df)
        }
    
    return publisher_timing


def plot_publisher_distribution(analysis: Dict, top_n: int = 20, 
                               save_path: Optional[str] = None):
    """
    Plot publisher distribution.
    
    Parameters:
    -----------
    analysis : dict
        Dictionary from analyze_publishers
    top_n : int
        Number of top publishers to display
    save_path : str, optional
        Path to save the plot
    """
    publisher_counts = analysis['publisher_counts'].head(top_n)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart
    axes[0].barh(range(len(publisher_counts)), publisher_counts['article_count'], 
                color='steelblue', edgecolor='black')
    axes[0].set_yticks(range(len(publisher_counts)))
    axes[0].set_yticklabels(publisher_counts['publisher'])
    axes[0].set_xlabel('Number of Articles')
    axes[0].set_ylabel('Publisher')
    axes[0].set_title(f'Top {top_n} Publishers by Article Count')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Pie chart for top publishers
    top_publishers = publisher_counts.head(10)
    other_count = analysis['publisher_counts']['article_count'].iloc[10:].sum()
    
    if other_count > 0:
        plot_data = pd.concat([
            top_publishers[['publisher', 'article_count']],
            pd.DataFrame([{'publisher': 'Others', 'article_count': other_count}])
        ])
    else:
        plot_data = top_publishers[['publisher', 'article_count']]
    
    axes[1].pie(plot_data['article_count'], labels=plot_data['publisher'], 
               autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Publisher Distribution (Top 10)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_publisher_content_analysis(content_analysis: Dict, save_path: Optional[str] = None):
    """
    Plot content analysis by publisher.
    
    Parameters:
    -----------
    content_analysis : dict
        Dictionary from analyze_publisher_content
    save_path : str, optional
        Path to save the plot
    """
    df = pd.DataFrame(content_analysis).T.reset_index()
    df.columns = ['publisher'] + list(df.columns[1:])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Average headline length
    axes[0, 0].barh(range(len(df)), df['avg_headline_length'], 
                   color='steelblue', edgecolor='black')
    axes[0, 0].set_yticks(range(len(df)))
    axes[0, 0].set_yticklabels(df['publisher'])
    axes[0, 0].set_xlabel('Average Headline Length (characters)')
    axes[0, 0].set_title('Average Headline Length by Publisher')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Average word count
    axes[0, 1].barh(range(len(df)), df['avg_word_count'], 
                   color='coral', edgecolor='black')
    axes[0, 1].set_yticks(range(len(df)))
    axes[0, 1].set_yticklabels(df['publisher'])
    axes[0, 1].set_xlabel('Average Word Count')
    axes[0, 1].set_title('Average Word Count by Publisher')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Unique stocks covered
    if 'unique_stocks_covered' in df.columns:
        axes[1, 0].barh(range(len(df)), df['unique_stocks_covered'], 
                       color='lightgreen', edgecolor='black')
        axes[1, 0].set_yticks(range(len(df)))
        axes[1, 0].set_yticklabels(df['publisher'])
        axes[1, 0].set_xlabel('Number of Unique Stocks Covered')
        axes[1, 0].set_title('Stock Coverage by Publisher')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Article count
    axes[1, 1].barh(range(len(df)), df['article_count'], 
                   color='gold', edgecolor='black')
    axes[1, 1].set_yticks(range(len(df)))
    axes[1, 1].set_yticklabels(df['publisher'])
    axes[1, 1].set_xlabel('Number of Articles')
    axes[1, 1].set_title('Total Articles by Publisher')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_domain_analysis(domain_counts: pd.DataFrame, top_n: int = 15, 
                        save_path: Optional[str] = None):
    """
    Plot domain analysis.
    
    Parameters:
    -----------
    domain_counts : pd.DataFrame
        DataFrame from identify_publisher_domains
    top_n : int
        Number of top domains to display
    save_path : str, optional
        Path to save the plot
    """
    top_domains = domain_counts.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_domains)), top_domains['count'], 
             color='purple', edgecolor='black')
    plt.yticks(range(len(top_domains)), top_domains['domain'])
    plt.xlabel('Number of Articles')
    plt.ylabel('Domain')
    plt.title(f'Top {top_n} Publisher Domains')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(top_domains['count']):
        plt.text(v + max(top_domains['count']) * 0.01, i, 
                str(v), va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Publisher Analysis Module")
    print("Import this module and use the functions with your data.")


