"""
Unit tests for descriptive statistics EDA functions
"""

import pytest
import pandas as pd
import numpy as np
from scripts.eda_descriptive_stats import (
    calculate_text_statistics,
    count_articles_per_publisher,
    analyze_publication_dates
)


def test_calculate_text_statistics():
    """Test text statistics calculation"""
    df = pd.DataFrame({
        'headline': ['Short', 'This is a longer headline with more words', 'Medium length headline']
    })
    
    stats, df_with_stats = calculate_text_statistics(df)
    
    assert isinstance(stats, pd.DataFrame)
    assert 'headline_length' in df_with_stats.columns
    assert 'headline_word_count' in df_with_stats.columns
    assert len(df_with_stats) == 3


def test_calculate_text_statistics_missing_column():
    """Test that ValueError is raised when headline column is missing"""
    df = pd.DataFrame({'other_column': ['test']})
    
    with pytest.raises(ValueError):
        calculate_text_statistics(df)


def test_count_articles_per_publisher():
    """Test counting articles per publisher"""
    df = pd.DataFrame({
        'publisher': ['Publisher A', 'Publisher B', 'Publisher A', 'Publisher C']
    })
    
    result = count_articles_per_publisher(df)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert result.iloc[0]['publisher'] == 'Publisher A'
    assert result.iloc[0]['article_count'] == 2


def test_analyze_publication_dates():
    """Test publication date analysis"""
    df = pd.DataFrame({
        'date': pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-02 11:00:00',
            '2024-01-03 12:00:00'
        ])
    })
    
    analysis, df_with_dates = analyze_publication_dates(df)
    
    assert isinstance(analysis, dict)
    assert 'total_days' in analysis
    assert 'date_range' in analysis
    assert 'year' in df_with_dates.columns
    assert 'month' in df_with_dates.columns

