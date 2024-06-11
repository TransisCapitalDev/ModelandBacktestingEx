import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from yfinance
def load_data(ticker, period='1y', interval='1h'):
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

# Compute Exponential Moving Averages (EMAs)
def compute_emas(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    return df

# Calculate Fibonacci retracement levels
def calculate_fibonacci_retracements(df):
    max_price = df['Close'].max()
    min_price = df['Close'].min()
    diff = max_price - min_price
    df['Fib_0.236'] = max_price - diff * 0.236
    df['Fib_0.382'] = max_price - diff * 0.382
    df['Fib_0.618'] = max_price - diff * 0.618
    return df

# Numerical differentiation to calculate price momentum
def calculate_derivatives(df):
    df['Price_Diff'] = df['Close'].diff()
    df['Momentum'] = df['Price_Diff'].rolling(window=5).sum()
    df['Momentum_Sign'] = np.sign(df['Momentum'])
    return df

# Generate trading signals
def generate_signals(df):
    df['Signal'] = 0
    df.loc[(df['EMA_20'] > df['EMA_50']) & (df['Momentum_Sign'] > 0), 'Signal'] = 1  # Buy
    df.loc[(df['EMA_20'] < df['EMA_50']) & (df['Momentum_Sign'] < 0), 'Signal'] = -1  # Sell
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
    print(data)
    # Compute EMAs
    data = compute_emas(data)
    print(data)
    # Calculate Fibonacci retracement levels
    data = calculate_fibonacci_retracements(data)
    print(data)
    # Calculate numerical derivatives
    data = calculate_derivatives(data)
    print(data)
    # Generate trading signals
    data = generate_signals(data)
    print(data)
    # Backtest strategy
    total_return, backtest_results = backtest_strategy(data)
    print(backtest_results)
    # Output results
    print(f"Total Return: {total_return * 100:.2f}%")
    backtest_results[['Close', 'EMA_20', 'EMA_50', 'EMA_100', 'Momentum', 'Portfolio_Value']].plot(subplots=True, figsize=(12, 10))
    plt.show()
