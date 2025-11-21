"""
Unit tests for data loading utilities
"""

import pytest
import pandas as pd
import os
import tempfile
from scripts.load_data import load_financial_news_data, validate_data


def test_load_financial_news_data():
    """Test loading financial news data from CSV"""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("headline,url,publisher,date,stock\n")
        f.write("Test Headline 1,http://test.com/1,Publisher A,2024-01-01 10:00:00,AAPL\n")
        f.write("Test Headline 2,http://test.com/2,Publisher B,2024-01-02 11:00:00,MSFT\n")
        temp_path = f.name
    
    try:
        df = load_financial_news_data(temp_path)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'headline' in df.columns
        assert 'date' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
    finally:
        os.unlink(temp_path)


def test_load_financial_news_data_file_not_found():
    """Test that FileNotFoundError is raised for non-existent file"""
    with pytest.raises(FileNotFoundError):
        load_financial_news_data("nonexistent_file.csv")


def test_validate_data():
    """Test data validation function"""
    df = pd.DataFrame({
        'headline': ['Test 1', 'Test 2'],
        'url': ['http://test.com/1', 'http://test.com/2'],
        'publisher': ['Publisher A', 'Publisher B'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'stock': ['AAPL', 'MSFT']
    })
    
    report = validate_data(df)
    
    assert isinstance(report, dict)
    assert 'total_rows' in report
    assert report['total_rows'] == 2
    assert 'unique_stocks' in report
    assert report['unique_stocks'] == 2
    assert 'unique_publishers' in report
    assert report['unique_publishers'] == 2

