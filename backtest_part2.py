# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from yfinance
def load_data(ticker, period='1y', interval='1h'):
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

# Mathematical Functions for trading signals
def add_math_features(df):
    # Price difference between consecutive days
    df['Price_Diff'] = df['Close'].diff(1)
    
    # Price ratio between consecutive days
    df['Price_Ratio'] = df['Close'] / df['Close'].shift(1)
    
    # Geometric progression factor estimation
    df['Geo_Progression'] = df['Close'] / df['Close'].shift(2) ** (1/2)
    return df

# Generate signals based on mathematical calculations
def generate_signals(df):
    df['Signal'] = 0
    df.loc[df['Price_Diff'] > 0, 'Signal'] = 1  # Buy when price increases
    df.loc[df['Price_Diff'] < 0, 'Signal'] = -1  # Sell when price decreases
    return df

# Backtesting engine to calculate returns
def backtest_strategy(df, initial_capital=10000):
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Market_Returns'] * df['Position']
    
    df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    final_value = df['Portfolio_Value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    return total_return, df

# Example of running the entire model
if __name__ == "__main__":
    # Load data
    data = load_data('BTC-USD', period='1y', interval='1d')
    
    # Add mathematical features
    data = add_math_features(data)
    
    # Generate signals
    data = generate_signals(data)
    
    # Backtest strategy
    total_return, backtest_results = backtest_strategy(data)
    
    # Output results
    print(f"Total Return: {total_return * 100:.2f}%")
    backtest_results[['Close', 'Price_Diff', 'Price_Ratio', 'Geo_Progression', 'Portfolio_Value']].plot(subplots=True, figsize=(10, 8))
    plt.show()
