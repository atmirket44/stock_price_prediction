# Dataset Characteristics

## Overview

This project aims to predict the closing prices of a stock using an LSTM neural network model. The model is trained on historical stock data obtained from Yahoo Finance, enriched with various technical indicators such as SMA, EMA, RSI, MACD, and Bollinger Bands. The project evaluates the model's performance using metrics like Directional Accuracy (DA) and Mean Absolute Error (MAE).

## Guidelines

- **Dataset Overview** :
      
  - Stock Symbol: The dataset focuses on a specific stock symbol (e.g., AAPL).

  - Time Period: Data spans from January 1, 2021, to January 1, 2024.

  - Frequency: Daily stock prices.

  - Features: Date, Open, High, Low, Close, Adj Close, Volume, and various technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands).

- **Missing Values** :

  - Check for any missing values in the dataset.
    
  - Handle missing values by either filling them with appropriate statistics (e.g., mean, median) or using forward/backward fill techniques.
    
  - Drop any rows with missing values if they are few and do not significantly impact the dataset.

- **Feature Distributions** :
  
  - Analyze and visualize the distribution of key features such as Close price, Volume, and technical indicators.
    
  - Use histograms, box plots, and density plots to understand the distributions.
    
  - Identify any outliers that may need special handling or transformation.

- **Possible Biases** :
  
  - Assess the dataset for any inherent biases, such as:
    - Time-based biases (e.g., data predominantly from a specific period with unusual market conditions).
    - Feature biases (e.g., certain features being overrepresented or underrepresented).
      
  - Ensure the dataset represents a wide range of market conditions to improve the model's generalizability.

- **Correlations**

Follow the instructions provided in brackets in the [Jupyter/Colab notebook](exploratory_data_analysis.ipynb) in this folder.

## Submission

Complete the [Jupyter/Colab notebook](exploratory_data_analysis.ipynb) to conduct your analysis of the dataset characteristics.
