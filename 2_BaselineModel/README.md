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
   "id": "d1dcaee4",
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
   "id": "b22bffcb",
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
   "id": "8cbf5d76",
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
   "id": "eb2e17c4",
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
   "id": "9e2b3a93",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to prepare and reshape test data\n",
    "def prepare_test_data(data, scaler, window_size):\n",
    "    print(\"Preparing test data...\")\n",
    "    scaled_data = scaler.transform(data)\n",
    "    x_test = [scaled_data[i-window_size:i, :] for i in range(window_size, len(scaled_data))]\n",
    "    x_test = np.array(x_test)\n",
    "    print(\"Test data prepared!\")\n",
    "    return x_test\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21b5a99a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to make predictions\n",
    "def make_predictions(model, x_test, close_scaler):\n",
    "    print(\"Making predictions...\")\n",
    "    predictions = [close_scaler.inverse_transform([[model.predict(np.reshape(x_input, (1, x_input.shape[0], x_input.shape[1]))[0, 0]]])[0, 0] for x_input in x_test]\n",
    "    print(\"Predictions made successfully!\")\n",
    "    return np.array(predictions)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ac6ff9c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to calculate Directional Accuracy (DA)\n",
    "def calculate_da(predictions, true_values):\n",
    "    min_len = min(len(predictions), len(true_values))\n",
    "    correct_directions = sum(1 for i in range(1, min_len) if np.sign(predictions[i] - predictions[i-1]) == np.sign(true_values[i] - true_values[i-1]))\n",
    "    directional_accuracy = correct_directions / (min_len - 1)\n",
    "    return directional_accuracy\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b230d702",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to calculate Mean Absolute Error (MAE)\n",
    "def calculate_mae(predictions, true_values):\n",
    "    min_len = min(len(predictions), len(true_values))\n",
    "    mae = np.mean(np.abs(predictions[:min_len] - true_values[:min_len]))\n",
    "    return mae\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98bce4d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Objective function for Optuna\n",
    "def objective(trial, window_size, num_features, x_train, y_train, close_values, scaler, close_scaler, close_data):\n",
    "    units = trial.suggest_int('units', 50, 200)\n",
    "    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)\n",
    "    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)\n",
    "    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])\n",
    "\n",
    "    model = Sequential()\n",
    "    model.add(LSTM(units=units, return_sequences=True, input_shape=(window_size, num_features)))\n",
    "    model.add(Dropout(dropout_rate))\n",
    "    model.add(LSTM(units=units, return_sequences=True))\n",
    "    model.add(Dropout(dropout_rate))\n",
    "    model.add(LSTM(units=units))\n",
    "    model.add(Dropout(dropout_rate))\n",
    "    model.add(Dense(units=1))\n",
    "    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')\n",
    "\n",
    "    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)\n",
    "    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)\n",
    "    history = model.fit(x_train, y_train, epochs=100, batch_size=batch_size, verbose=0, callbacks=[early_stopping, reduce_lr])\n",
    "\n",
    "    x_test = prepare_test_data(close_values, scaler, window_size)\n",
    "    predictions = make_predictions(model, x_test, close_scaler)\n",
    "\n",
    "    mae = calculate_mae(predictions, close_data['Close'].values[window_size:])\n",
    "    return mae\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b045d8db",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Main function\n",
    "def main():\n",
    "    try:\n",
    "        ticker = input(\"Enter the stock symbol (e.g., AAPL): \").upper()\n",
    "\n",
    "        start_date = '2021-01-01'\n",
    "        end_date = '2024-01-01'\n",
    "        window_size = 60\n",
    "        epochs = 100\n",
    "\n",
    "        data = download_stock_data(ticker, start_date, end_date)\n",
    "        plot_close_price_history(data)\n",
    "        plot_candlestick_chart(data)\n",
    "\n",
    "        close_data = data.filter(['Close', 'Volume'])\n",
    "        close_data = add_technical_indicators(close_data)\n",
    "        close_data = close_data.dropna()\n",
    "\n",
    "        close_values = close_data.values\n",
    "        x_train, y_train, scaler, close_scaler = prepare_training_data(close_values, window_size)\n",
    "        num_features = close_values.shape[1]\n",
    "\n",
    "        study = optuna.create_study(direction='minimize')\n",
    "        study.optimize(lambda trial: objective(trial, window_size, num_features, x_train, y_train, close_values, scaler, close_scaler, close_data), n_trials=10)\n",
    "\n",
    "        best_params = study.best_params\n",
    "        model = build_lstm_model(window_size, num_features, best_params)\n",
    "        history = train_lstm_model(model, x_train, y_train, epochs, best_params['batch_size'])\n",
    "\n",
    "        x_test = prepare_test_data(close_values, scaler, window_size)\n",
    "        predictions = make_predictions(model, x_test, close_scaler)\n",
    "\n",
    "        da = calculate_da(predictions, close_data['Close'].values[window_size:])\n",
    "        print(f\"Directional Accuracy (DA): {da}\")\n",
    "\n",
    "        mae = calculate_mae(predictions, close_data['Close'].values[window_size:])\n",
    "        print(f\"Mean Absolute Error (MAE): {mae}\")\n",
    "\n",
    "        print(\"Final predicted price:\", predictions[-1])\n",
    "\n",
    "    except Exception as e:\n",
    "        print(\"An error occurred:\", str(e))\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 5,
 "nbformat_minor": 10
}
