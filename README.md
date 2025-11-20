# Financial News Sentiment Analysis Project

## Project Overview

This project focuses on analyzing financial news data to discover correlations between news sentiment and stock market movements. The analysis combines Data Engineering, Financial Analytics, and Machine Learning Engineering techniques.

## Business Objective

Nova Financial Solutions aims to enhance its predictive analytics capabilities to boost financial forecasting accuracy through advanced data analysis. The primary objectives are:

1. **Sentiment Analysis**: Perform sentiment analysis on news headlines to quantify tone and sentiment
2. **Correlation Analysis**: Establish statistical correlations between news sentiment and stock price movements

## Dataset

**Financial News and Stock Price Integration Dataset (FNSPID)**

The dataset contains:
- `headline`: Article release headline
- `url`: Direct link to the full news article
- `publisher`: Author/creator of article
- `date`: Publication date and time (UTC-4 timezone)
- `stock`: Stock ticker symbol

## Project Structure

```
├── .vscode/          # VS Code settings
├── .github/          # GitHub workflows and CI/CD
├── src/              # Source code modules
├── notebooks/        # Jupyter notebooks for analysis
├── tests/            # Unit tests
└── scripts/          # Utility scripts
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd week2
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## Tasks

- **Task 1**: Git and GitHub setup, EDA
- **Task 2**: (To be added)
- **Task 3**: (To be added)

## Key Dates

- Challenge Introduction: 10:30 AM UTC, Wednesday, 19 Nov 2025
- Interim Submission: 8:00 PM UTC, Sunday, 23 Nov 2025
- Final Submission: 8:00 PM UTC, Tuesday, 25 Nov 2025

## Team

- Facilitator: Kerod, Mahbubah, Filimon

## Learning Objectives

- Configure reproducible Python data-science environment with GitHub integration
- Perform EDA on text and time series data
- Compute technical indicators (MA, RSI, MACD)
- Run sentiment analysis on news headlines
- Measure correlation between news sentiment and daily stock returns
- Document findings and write publication-style reports

