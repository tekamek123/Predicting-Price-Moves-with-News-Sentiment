# Setup Instructions for Jupyter Notebook

## Issue Fixed: ModuleNotFoundError

The error occurred because Jupyter was using the system Python instead of the virtual environment.

## Solution Applied

1. ✅ Installed all required packages in the virtual environment
2. ✅ Registered the venv as a Jupyter kernel

## How to Use the Notebook

### Option 1: Select the Correct Kernel in Jupyter

1. Open the notebook in Jupyter
2. Click on the kernel name in the top right (it might say "Python 3" or similar)
3. Select **"Python (week2-venv)"** from the kernel list
4. Run your cells again

### Option 2: Start Jupyter from the Virtual Environment

1. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Start Jupyter from the activated venv:
   ```powershell
   jupyter notebook notebooks/task1_eda_analysis.ipynb
   ```

### Option 3: Use VS Code

1. Open the notebook in VS Code
2. VS Code should automatically detect the virtual environment
3. If not, select the Python interpreter:
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose `.\venv\Scripts\python.exe`

## Verify Installation

Run this in a notebook cell to verify:
```python
import sys
print(sys.executable)  # Should show path to venv\Scripts\python.exe
import pandas as pd
print(f"Pandas version: {pd.__version__}")  # Should work without error
```

## Installed Packages

All required packages are now installed in the virtual environment:
- pandas, numpy
- matplotlib, seaborn, plotly
- nltk, textblob, wordcloud, gensim
- yfinance
- jupyter, ipykernel
- And all dependencies

## Next Steps

1. Select the correct kernel in your notebook
2. Run the first cell to load the data
3. Continue with the EDA analysis

