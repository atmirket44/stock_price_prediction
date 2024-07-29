# Baseline Model

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Stock Price Prediction with LSTM\n",
    "This notebook demonstrates the process of predicting stock prices using an LSTM model. The workflow includes data downloading, technical indicator calculation, model training, and evaluation. Hyperparameter tuning is also performed using Optuna."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
     "version": "3.8.5"
    }
   },
   "outputs": [],
   "source": [
    "# Install the necessary libraries\n",
    "!pip install ta\n",
    "!pip install mplfinance\n",
    "!pip install optuna"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "plt.style.use('fivethirtyeight')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Download Stock Data\n",
    "The following function downloads historical stock data from Yahoo Finance for a given ticker and date range."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to download stock data from Yahoo Finance\n",
    "def download_stock_data(ticker, start_date, end_date):\n",
    "    print(\"Downloading stock data...\")\n",
    "    data = yf.download(ticker, start=start_date, end=end_date)\n",
    "    print(\"Data downloaded successfully!\")\n",
    "    return data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Plotting Functions\n",
    "Here we define functions to plot the historical close price and candlestick charts of the stock data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Adding Technical Indicators\n",
    "This function adds various technical indicators to the stock data, such as SMA, EMA, RSI, MACD, and Bollinger Bands."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "    return data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Preparing Training Data\n",
    "This function scales the data and prepares it for training by creating sequences of the specified window size."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "    return np.array(x_train), np.array(y_train), scaler, close_scaler"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Building and Training the LSTM Model\n",
    "Here, we define functions to build and train the LSTM model for stock price prediction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "    return history"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Preparing Test Data\n",
    "This function prepares test data by scaling and reshaping it for prediction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "    return x_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Making Predictions\n",
    "The following function uses the trained LSTM model to make predictions on the test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to make predictions\n",
    "def make_predictions(model, x_test, close_scaler):\n",
    "    print(\"Making predictions...\")\n",
    "    predictions = [close_scaler.inverse_transform([[model.predict(np.reshape(x_input, (1, x_input.shape[0], x_input.shape[1])))[0, 0]]])[0, 0] for x_input in x_test]\n",
    "    print(\"Predictions made successfully!\")\n",
    "    return np.array(predictions)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluation Metrics\n",
    "Functions to calculate evaluation metrics such as Directional Accuracy (DA) and Mean Absolute Error (MAE)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to calculate Directional Accuracy (DA)\n",
    "def calculate_da(predictions, true_values):\n",
    "    min_len = min(len(predictions), len(true_values))\n",
    "    correct_directions = sum(1 for i in range(1, min_len) if np.sign(predictions[i] - predictions[i-1]) == np.sign(true_values[i] - true_values[i-1]))\n",
    "    directional_accuracy = correct_directions / (min_len - 1)\n",
    "    return directional_accuracy\n",
    "\n",
    "# Function to calculate Mean Absolute Error (MAE)\n",
    "def calculate_mae(predictions, true_values):\n",
    "    min_len = min(len(predictions), len(true_values))\n",
    "    mae = np.mean(np.abs(predictions[:min_len] - true_values[:min_len]))\n",
    "    return mae"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Hyperparameter Optimization with Optuna\n",
    "This section defines the objective function for Optuna and performs hyperparameter optimization to find the best model parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
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
    "    return mae"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Main Function\n",
    "The main function coordinates the entire process, including downloading data, training the model, performing hyperparameter optimization, and evaluating the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
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
    "    main()"
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
   "name": "python",
   "version": "3.8.5",
   "mimetype": "text/x-python",
   "file_extension": ".py",
   "pygments_lexer": "ipython3",
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   }
  }
 },
 "nbformat": 5,
 "nbformat_minor": 10
}

