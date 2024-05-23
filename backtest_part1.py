# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from yfinance (you can specify multiple tickers if needed)
def load_data(ticker, period='1y', interval='1h'):
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

# Technical Indicators (examples)
def add_indicators(df):
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    
    return df

# Strategy logic for signals (example: RSI & SMA crossover)
def generate_signals(df):
    df['Signal'] = 0
    df.loc[(df['RSI'] < 30) & (df['SMA_20'] > df['SMA_50']), 'Signal'] = 1  # Buy Signal
    df.loc[(df['RSI'] > 70) & (df['SMA_20'] < df['SMA_50']), 'Signal'] = -1  # Sell Signal
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
    
    # Add indicators
    data = add_indicators(data)
    
    # Generate signals
    data = generate_signals(data)
    
    # Backtest strategy
    total_return, backtest_results = backtest_strategy(data)
    
    # Output results
    print(f"Total Return: {total_return * 100:.2f}%")
    backtest_results[['Close', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower', 'RSI', 'Portfolio_Value']].plot(subplots=True, figsize=(10, 8))
    plt.show()
