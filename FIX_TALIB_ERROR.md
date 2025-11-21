# Fix TA-Lib Import Error

## Problem
You're seeing: `ModuleNotFoundError: No module named 'talib'`

## Solution
The code already has fallback logic that uses manual calculations when TA-Lib is not available. The error you're seeing is likely due to:

1. **Cached Python bytecode** (.pyc files)
2. **Jupyter kernel needs restart**

## Quick Fix Steps

### Option 1: Restart Jupyter Kernel (Recommended)
1. In your Jupyter notebook, go to **Kernel → Restart Kernel**
2. Re-run all cells from the beginning

### Option 2: Clear Python Cache
Run this in your terminal:
```powershell
cd "D:\My Projects\Kifiya AI Mastery Training\week2"
Remove-Item -Recurse -Force scripts\__pycache__
Remove-Item -Recurse -Force scripts\*.pyc
```

### Option 3: Verify the Import Works
Test the import in a new Python session:
```python
import sys
sys.path.append('scripts')
from technical_indicators import calculate_moving_averages
print("Import successful!")
```

You should see: `Note: Using manual calculations for technical indicators (TA-Lib and pandas-ta not available).`

## How It Works

The code automatically falls back to manual calculations when TA-Lib is not available:

1. **First tries**: TA-Lib (if installed)
2. **Then tries**: pandas-ta (if installed)
3. **Finally uses**: Manual calculations using pandas/numpy (always works)

All indicators are calculated manually using pandas operations:
- **Moving Averages**: `df.rolling().mean()` and `df.ewm()`
- **RSI**: Manual calculation using gain/loss ratios
- **MACD**: Using exponential moving averages
- **Bollinger Bands**: Using rolling mean and std

## Verify It's Working

After restarting the kernel, you should see this message when importing:
```
Note: Using manual calculations for technical indicators (TA-Lib and pandas-ta not available).
```

This means the fallback is working correctly and all indicators will be calculated manually.

## Optional: Install TA-Lib (Advanced)

If you want to use TA-Lib instead of manual calculations:

### Windows Installation:
1. Download TA-Lib wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Install using pip:
   ```powershell
   pip install TA_Lib-0.4.28-cp311-win_amd64.whl
   ```
   (Replace with the correct version for your Python version)

### Note:
Manual calculations work perfectly fine and produce the same results. TA-Lib is optional.

