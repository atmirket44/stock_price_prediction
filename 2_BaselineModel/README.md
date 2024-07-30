# Install the necessary libraries
!pip install ta
!pip install mplfinance
!pip install optuna

# Importing necessary libraries
import optuna
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import ta  # Technical analysis library
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from keras.optimizers import Adam

plt.style.use('fivethirtyeight')
