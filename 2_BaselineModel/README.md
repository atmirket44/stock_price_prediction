{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c8a8e9d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install the necessary libraries\n",
    "!pip install ta\n",
    "!pip install mplfinance\n",
    "!pip install optuna\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "80b446b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing necessary libraries\n",
    "import optuna\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import yfinance as yf\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import RobustScaler\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, LSTM, Dropout\n",
    "from keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
    "import ta  # Technical analysis library\n",
    "from mplfinance.original_flavor import candlestick_ohlc\n",
    "import matplotlib.dates as mdates\n",
    "from keras.optimizers import Adam\n",
    "\n",
    "plt.style.use('fivethirtyeight')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a4dcdb7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to download stock data from Yahoo Finance\n",
    "def download_stock_data(ticker, start_date, end_date):\n",
    "    print(\"Downloading stock data...\")\n",
    "    data = yf.download(ticker, start=start_date, end=end_date)\n",
    "    print(\"Data downloaded successfully!\")\n",
    "    return data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "423b1d3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to plot closing price history\n",
    "def plot_close_price_history(data):\n",
    "    print(\"Plotting close price history...\")\n",
    "    plt.figure(figsize=(16,8))\n",
    "    plt.title('Close Price History')\n",
    "    plt.plot(data['Close'])\n",
    "    plt.xlabel('Date', fontsize=18)\n",
    "    plt.ylabel('Close Price USD ($)', fontsize=18)\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9731a3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to plot candlestick chart\n",
    "def plot_candlestick_chart(data):\n",
    "    print(\"Plotting candlestick chart...\")\n",
    "    fig, ax = plt.subplots(figsize=(16,8))\n",
    "    candlestick_ohlc(ax, zip(mdates.date2num(data.index.to_pydatetime()), data['Open'], data['High'], data['Low'], data['Close']), width=0.6)\n",
    "    ax.set_title('Candlestick Chart')\n",
    "    ax.xaxis_date()\n",
    "    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))\n",
    "    ax.set_xlabel('Date', fontsize=18)\n",
    "    ax.set_ylabel('Price USD ($)', fontsize=18)\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3d83b8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to add technical indicators\n",
    "def add_technical_indicators(data):\n",
    "    print(\"Adding technical indicators...\")\n",
    "    data['SMA'] = ta.trend.sma_indicator(data['Close'], window=15)\n",
    "    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=15)\n",
    "    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)\n",
    "    data['MACD'] = ta.trend.macd_diff(data['Close'])\n",
    "    data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])\n",
    "    data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])\n",
    "    print(\"Technical indicators added successfully!\")\n",
    "    return data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "540bcd93",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to prepare training data with additional features\n",
    "def prepare_training_data(data, window_size):\n",
    "    print(\"Preparing training data...\")\n",
    "    scaler = RobustScaler()\n",
    "    close_scaler = RobustScaler()\n",
    "    close_prices = data[:, 0].reshape(-1, 1)\n",
    "    close_scaler.fit(close_prices)\n",
    "    scaled_data = scaler.fit_transform(data)\n",
    "    x_train, y_train = [], []\n",
    "\n",
    "    for i in range(window_size, len(scaled_data)):\n",
    "        x_train.append(scaled_data[i-window_size:i, :])\n",
    "        y_train.append(scaled_data[i, 0])\n",
    "\n",
    "    print(\"Training data prepared!\")\n",
    "    return np.array(x_train), np.array(y_train), scaler, close_scaler\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84a4d358",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to build LSTM model\n",
    "def build_lstm_model(window_size, num_features, best_params):\n",
    "    units = best_params['units']\n",
    "    dropout_rate = best_params['dropout_rate']\n",
    "    learning_rate = best_params['learning_rate']\n",
    "    batch_size = best_params['batch_size']\n",
    "\n",
    "    print(\"Building LSTM model...\")\n",
    "    model = Sequential()\n",
    "    model.add(LSTM(units=units, return_sequences=True, input_shape=(window_size, num_features)))\n",
    "    model.add(Dropout(dropout_rate))\n",
    "    model.add(LSTM(units=units, return_sequences=True))\n",
    "    model.add(Dropout(dropout_rate))\n",
    "    model.add(LSTM(units=units))\n",
    "    model.add(Dropout(dropout_rate))\n",
    "    model.add(Dense(units=1))\n",
    "    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')\n",
    "    print(\"Model built successfully!\")\n",
    "    return model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "230a45db",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to train LSTM model\n",
    "def train_lstm_model(model, x_train, y_train, epochs, batch_size):\n",
    "    print(\"Training LSTM model...\")\n",
    "    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)\n",
    "    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)\n",
    "    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping, reduce_lr])\n",
    "    print(\"Model trained successfully!\")\n",
    "    return history\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc365c6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to prepare and reshape test data\n",
    "def prepare_test_data(data, scaler, window_size):\n",
    "    print(\"Preparing test data...\")\n",
    "    scaled_data = scaler.transform(data)\n",
    "    x_test = [scaled_data[i-window_size:i, :] for i in range(window_size
