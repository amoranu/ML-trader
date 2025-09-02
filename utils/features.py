import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, average_true_range
from ta.volume import on_balance_volume
from ta.momentum import RSIIndicator
from ta.trend import MACD

def get_chop_index(high, low, close, window=14):
    """Calculates the Choppiness Index (CHOP)."""
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    highh = high.rolling(window).max()
    lowl = low.rolling(window).min()

    chop = 100 * np.log10(atr.rolling(window).sum() / (highh - lowl)) / np.log10(window)
    return chop

def add_features(df):
    data = df.copy()
    data['original_close'] = data['Close'].copy()
    data['returns'] = data['Close'].pct_change()
    data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
    data['MACDh_12_26_9'] = MACD(close=data['Close'], window_fast=12, window_slow=26, window_sign=9).macd_diff()
    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['bb_percent'] = bb.bollinger_pband()
    data['atr'] = average_true_range(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['obv'] = on_balance_volume(close=data['Close'], volume=data['Volume'])
    data['chop_index'] = get_chop_index(data['High'], data['Low'], data['Close'])
    data['overnight_gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    return data