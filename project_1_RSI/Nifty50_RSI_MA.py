# Quant Project: Moving Average Crossover Strategy on Nifty 50 

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Download Nifty 50 data
ticker = "^NSEI"
data = yf.download(ticker, start="2020-01-01", end="2024-12-31")[['Close']].dropna()

# Moving Averages
short_window = 50
long_window = 200
data['SMA50'] = data['Close'].rolling(window=short_window).mean()
data['SMA200'] = data['Close'].rolling(window=long_window).mean()

# Signals
data['Signal'] = 0
data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1
data['Position'] = data['Signal'].shift(1)
data['Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Returns'] * data['Position']

# Calculate rolling volatility (30-day window)
data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)  # annualized

# Plot
plt.figure(figsize=(12, 6))
(1 + data[['Returns', 'Strategy_Returns']]).cumprod().plot()
plt.title("Nifty 50 Strategy vs Buy-and-Hold")
plt.ylabel("Cumulative Returns")
plt.grid(True)
plt.show()

# Plot Volatility
plt.figure(figsize=(12, 5))
plt.plot(data['Volatility'], label='30-day Rolling Volatility', color='purple')
plt.title('Nifty 50 Volatility Over Time')
plt.ylabel('Annualized Volatility')
plt.grid(True)
plt.legend()
plt.show()

# Metrics
total_return = (1 + data['Strategy_Returns']).prod() - 1
market_return = (1 + data['Returns']).prod() - 1
sharpe_ratio = data['Strategy_Returns'].mean() / data['Strategy_Returns'].std() * np.sqrt(252)
print(f"Total Strategy Return: {total_return:.2%}")
print(f"Buy & Hold Return: {market_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

