"""
Sentiment analysis utilities for financial news headlines
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Try to import sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Using NLTK VADER instead.")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    # Download required NLTK data if not already downloaded
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    NLTK_VADER_AVAILABLE = True
except ImportError:
    NLTK_VADER_AVAILABLE = False
    print("Warning: NLTK VADER not available. Using simple rule-based sentiment.")

if not TEXTBLOB_AVAILABLE and not NLTK_VADER_AVAILABLE:
    print("Note: Using simple rule-based sentiment analysis (TextBlob and NLTK not available).")


def analyze_sentiment_textblob(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using TextBlob.
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    dict
        Dictionary with 'polarity' and 'subjectivity' scores
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'polarity': 0.0, 'subjectivity': 0.0}
    
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,  # Range: -1 (negative) to 1 (positive)
        'subjectivity': blob.sentiment.subjectivity  # Range: 0 (objective) to 1 (subjective)
    }


def analyze_sentiment_vader(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner).
    VADER is optimized for social media text and works well with financial news.
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    dict
        Dictionary with 'compound', 'pos', 'neu', 'neg' scores
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
    
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    return {
        'compound': scores['compound'],  # Range: -1 (negative) to 1 (positive)
        'pos': scores['pos'],  # Positive score
        'neu': scores['neu'],  # Neutral score
        'neg': scores['neg']   # Negative score
    }


def analyze_sentiment_simple(text: str) -> Dict[str, float]:
    """
    Simple rule-based sentiment analysis (fallback when libraries unavailable).
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    dict
        Dictionary with 'polarity' score
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'polarity': 0.0}
    
    text_lower = text.lower()
    
    # Simple positive/negative word lists
    positive_words = ['up', 'rise', 'gain', 'surge', 'rally', 'growth', 'profit', 
                     'success', 'positive', 'strong', 'beat', 'exceed', 'bullish']
    negative_words = ['down', 'fall', 'drop', 'decline', 'loss', 'crash', 'fail',
                     'negative', 'weak', 'miss', 'below', 'bearish', 'worry']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    # Calculate simple polarity
    total_words = len(text_lower.split())
    if total_words == 0:
        polarity = 0.0
    else:
        polarity = (pos_count - neg_count) / max(total_words, 1)
        polarity = max(-1.0, min(1.0, polarity))  # Clamp to [-1, 1]
    
    return {'polarity': polarity}


def calculate_sentiment_score(
    text: str,
    method: str = 'auto'
) -> float:
    """
    Calculate a single sentiment score from text.
    
    Parameters:
    -----------
    text : str
        Text to analyze
    method : str
        Method to use: 'textblob', 'vader', 'simple', or 'auto' (default: 'auto')
        'auto' will use the best available method
    
    Returns:
    --------
    float
        Sentiment score: -1 (very negative) to 1 (very positive)
    """
    if method == 'auto':
        if TEXTBLOB_AVAILABLE:
            result = analyze_sentiment_textblob(text)
            return result['polarity']
        elif NLTK_VADER_AVAILABLE:
            result = analyze_sentiment_vader(text)
            return result['compound']
        else:
            result = analyze_sentiment_simple(text)
            return result['polarity']
    elif method == 'textblob' and TEXTBLOB_AVAILABLE:
        result = analyze_sentiment_textblob(text)
        return result['polarity']
    elif method == 'vader' and NLTK_VADER_AVAILABLE:
        result = analyze_sentiment_vader(text)
        return result['compound']
    else:
        result = analyze_sentiment_simple(text)
        return result['polarity']


def classify_sentiment(score: float) -> str:
    """
    Classify sentiment score into category.
    
    Parameters:
    -----------
    score : float
        Sentiment score (-1 to 1)
    
    Returns:
    --------
    str
        Sentiment category: 'positive', 'negative', or 'neutral'
    """
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'


def analyze_sentiment_batch(
    df: pd.DataFrame,
    text_column: str = 'headline',
    method: str = 'auto',
    add_classification: bool = True
) -> pd.DataFrame:
    """
    Perform sentiment analysis on a batch of texts in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing text to analyze
    text_column : str
        Name of column containing text (default: 'headline')
    method : str
        Sentiment analysis method: 'textblob', 'vader', 'simple', or 'auto'
    add_classification : bool
        If True, add sentiment classification column (default: True)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added sentiment columns
    """
    df = df.copy()
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    print(f"Analyzing sentiment for {len(df)} headlines using {method if method != 'auto' else 'best available'} method...")
    
    # Calculate sentiment scores
    df['sentiment_score'] = df[text_column].apply(
        lambda x: calculate_sentiment_score(str(x) if pd.notna(x) else '', method)
    )
    
    # Add classification if requested
    if add_classification:
        df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)
    
    # Add detailed scores if using VADER
    if (method == 'vader' or (method == 'auto' and NLTK_VADER_AVAILABLE and not TEXTBLOB_AVAILABLE)):
        vader_scores = df[text_column].apply(
            lambda x: analyze_sentiment_vader(str(x) if pd.notna(x) else '')
        )
        df['sentiment_pos'] = vader_scores.apply(lambda x: x['pos'])
        df['sentiment_neu'] = vader_scores.apply(lambda x: x['neu'])
        df['sentiment_neg'] = vader_scores.apply(lambda x: x['neg'])
    
    print(f"✓ Sentiment analysis complete")
    
    return df


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_column: str = 'aligned_date',
    sentiment_column: str = 'sentiment_score',
    aggregation_method: str = 'mean'
) -> pd.DataFrame:
    """
    Aggregate sentiment scores by date.
    
    If multiple articles appear on the same day, compute average (or other aggregation)
    daily sentiment scores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment scores and dates
    date_column : str
        Name of date column (default: 'aligned_date')
    sentiment_column : str
        Name of sentiment score column (default: 'sentiment_score')
    aggregation_method : str
        Aggregation method: 'mean', 'median', or 'weighted' (default: 'mean')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily aggregated sentiment scores
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame")
    if sentiment_column not in df.columns:
        raise ValueError(f"Column '{sentiment_column}' not found in DataFrame")
    
    # Group by date and aggregate
    if aggregation_method == 'mean':
        daily_sentiment = df.groupby(date_column)[sentiment_column].mean().reset_index()
    elif aggregation_method == 'median':
        daily_sentiment = df.groupby(date_column)[sentiment_column].median().reset_index()
    elif aggregation_method == 'weighted':
        # Weight by article length (if available)
        if 'headline' in df.columns:
            df['headline_length'] = df['headline'].str.len()
            daily_sentiment = (
                df.groupby(date_column)
                .apply(lambda x: np.average(x[sentiment_column], weights=x['headline_length']))
                .reset_index()
            )
            daily_sentiment.columns = [date_column, sentiment_column]
        else:
            daily_sentiment = df.groupby(date_column)[sentiment_column].mean().reset_index()
    else:
        daily_sentiment = df.groupby(date_column)[sentiment_column].mean().reset_index()
    
    # Add count of articles per day
    article_counts = df.groupby(date_column).size().reset_index(name='article_count')
    daily_sentiment = daily_sentiment.merge(article_counts, on=date_column, how='left')
    
    # Rename sentiment column to indicate it's daily aggregated
    daily_sentiment.rename(columns={sentiment_column: 'daily_sentiment_score'}, inplace=True)
    
    return daily_sentiment


def get_sentiment_summary(df: pd.DataFrame, sentiment_column: str = 'sentiment_score') -> Dict:
    """
    Generate summary statistics for sentiment analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment scores
    sentiment_column : str
        Name of sentiment score column
    
    Returns:
    --------
    dict
        Summary statistics
    """
    if sentiment_column not in df.columns:
        return {'error': f"Column '{sentiment_column}' not found"}
    
    scores = df[sentiment_column].dropna()
    
    summary = {
        'total_articles': len(df),
        'articles_with_sentiment': len(scores),
        'mean_sentiment': scores.mean(),
        'median_sentiment': scores.median(),
        'std_sentiment': scores.std(),
        'min_sentiment': scores.min(),
        'max_sentiment': scores.max(),
        'positive_count': len(scores[scores > 0.1]) if 'sentiment' in df.columns else None,
        'negative_count': len(scores[scores < -0.1]) if 'sentiment' in df.columns else None,
        'neutral_count': len(scores[(scores >= -0.1) & (scores <= 0.1)]) if 'sentiment' in df.columns else None
    }
    
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        summary['sentiment_distribution'] = sentiment_counts
    
    return summary

