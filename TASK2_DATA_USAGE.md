# Task 2: Using Local Data from /data/Data Folder

## Overview
Task 2 has been updated to support loading stock price data from local CSV files in the `/data/Data` folder.

## Available Stock Data
The following stock CSV files are available in `data/Data/`:
- `AAPL.csv` - Apple Inc.
- `MSFT.csv` - Microsoft Corporation
- `GOOG.csv` - Alphabet Inc. (Google)
- `META.csv` - Meta Platforms Inc.
- `AMZN.csv` - Amazon.com Inc.
- `NVDA.csv` - NVIDIA Corporation

## Data Format
Each CSV file contains the following columns:
- `Date` - Trading date
- `Open` - Opening price
- `High` - Highest price
- `Low` - Lowest price
- `Close` - Closing price
- `Volume` - Trading volume

## Usage in Notebook

### Option 1: Use Local CSV Data (Recommended)
```python
# Configuration
TICKER = 'AAPL'  # Options: AAPL, MSFT, GOOG, META, AMZN, NVDA
USE_LOCAL_DATA = True  # Use local CSV files
DATA_DIR = '../data/Data'  # Path to CSV files

# Load stock data
stock_df = load_stock_data(
    ticker=TICKER,
    use_local_data=USE_LOCAL_DATA,
    data_dir=DATA_DIR
)
```

### Option 2: Download from yfinance
```python
# Configuration
TICKER = 'AAPL'
USE_LOCAL_DATA = False  # Download from yfinance
START_DATE = '2020-01-01'
END_DATE = None
PERIOD = '2y'

# Load stock data
stock_df = load_stock_data(
    ticker=TICKER,
    start_date=START_DATE,
    end_date=END_DATE,
    period=PERIOD,
    use_local_data=USE_LOCAL_DATA
)
```

## How It Works

1. **If `use_local_data=True`**:
   - First tries to load from `data/Data/{TICKER}.csv`
   - If file not found, falls back to yfinance
   - Prints status message indicating data source

2. **If `use_local_data=False`**:
   - Downloads data directly from yfinance
   - Uses start_date/end_date or period parameters

## Benefits of Using Local Data

1. **Faster Loading**: No internet connection required
2. **Consistent Data**: Same dataset for all analyses
3. **Historical Data**: Local files may contain more historical data
4. **Offline Analysis**: Work without internet connection

## Example: Analyzing Multiple Stocks

```python
# Load multiple stocks from local data
tickers = ['AAPL', 'MSFT', 'GOOG', 'META', 'AMZN', 'NVDA']
stock_data = load_multiple_stocks(
    tickers=tickers,
    use_local_data=True,
    data_dir='../data/Data'
)

# Each stock's data is in stock_data dictionary
for ticker, df in stock_data.items():
    print(f"{ticker}: {len(df)} rows")
    # Perform analysis on df
```

## Updating the Notebook

To use local data in the notebook, update Cell 4 (data loading cell) with:

```python
# Configuration
TICKER = 'AAPL'  # Change to any available ticker
USE_LOCAL_DATA = True  # Set to True to use local CSV files
DATA_DIR = '../data/Data'  # Directory containing CSV files

# Load stock data
print(f"Loading data for {TICKER}...")
if USE_LOCAL_DATA:
    print(f"Using local data from {DATA_DIR} folder...")
stock_df = load_stock_data(
    ticker=TICKER,
    use_local_data=USE_LOCAL_DATA,
    data_dir=DATA_DIR
)
```

## Notes

- Local CSV files are automatically detected and used when available
- If a ticker's CSV file doesn't exist, the code automatically falls back to yfinance
- All date filtering can be done after loading the data if needed
- The local CSV files contain all available historical data

