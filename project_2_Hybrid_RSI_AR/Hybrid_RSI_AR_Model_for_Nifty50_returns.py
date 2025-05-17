# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

# Step 1: Download Nifty 50 data
nifty = yf.download("^NSEI", start="2020-01-01", end="2024-12-31")
close_prices = nifty['Close'].dropna()

# Step 2: Calculate Daily Returns
returns = close_prices.pct_change().dropna()

# Step 3: Calculate RSI (14-day)
def compute_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=window).mean()
    loss = down.rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi = compute_rsi(close_prices)

# Step 4: Seasonal Decomposition
decomp = seasonal_decompose(close_prices, model='multiplicative', period=30)
trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid

# Visualize decomposition
decomp.plot()
plt.suptitle("Seasonal Decomposition of Nifty 50", fontsize=14)
plt.tight_layout()
plt.show()

# Step 5: Clean residual and perform train-test split
residual_clean = residual.dropna()
train_size = int(len(residual_clean) * 0.9)
train_resid = residual_clean[:train_size]
test_resid = residual_clean[train_size:]

# Step 6: ADF test for stationarity
adf_result = adfuller(train_resid)
print("ADF Statistic (train residual):", adf_result[0])
print("p-value:", adf_result[1])

# Step 7: ACF and PACF plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(train_resid, lags=40, ax=plt.gca())
plt.title("ACF (Train Residuals)")

plt.subplot(1, 2, 2)
plot_pacf(train_resid, lags=40, ax=plt.gca(), method='ywm')
plt.title("PACF (Train Residuals)")
plt.tight_layout()
plt.show()

# Step 8: Fit AR model
ar_model = AutoReg(train_resid, lags=5).fit()
print(ar_model.summary())

# Step 9: Predict test residuals
pred_resid = ar_model.predict(start=len(train_resid), end=len(train_resid)+len(test_resid)-1)
pred_resid.index = test_resid.index

# Step 10: Visualize train/test actual vs predicted residuals
plt.figure(figsize=(12, 5))
plt.plot(train_resid.index, train_resid, label='Training Residuals', color='blue')
plt.plot(test_resid.index, test_resid, label='Actual Test Residuals', color='black')
plt.plot(pred_resid.index, pred_resid, label='Predicted Residuals', color='red', linestyle='--')
plt.axvline(test_resid.index[0], color='gray', linestyle=':', label='Train/Test Split')
plt.title("Train vs Test Residual Forecast (AR Model)")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Reconstruct forecasted price using last trend + seasonal + predicted residuals
valid_trend = trend.dropna()
last_trend_value = valid_trend.iloc[-1]

# Use the last 30 seasonal values and tile them
seasonal_pattern = seasonal[-30:].values
seasonal_future = np.tile(seasonal_pattern, int(np.ceil(len(pred_resid)/30)))[:len(pred_resid)]

# Combine components
reconstructed_price = last_trend_value * seasonal_future * (1 + pred_resid.values)

# Step 12: Plot reconstructed forecasted prices
plt.figure(figsize=(10, 5))
plt.plot(pred_resid.index, reconstructed_price, label='Forecasted Nifty 50 Price', color='green')
plt.title("Forecasted Price (Hybrid: Decomposed + AR)")
plt.xlabel("Date")
plt.ylabel("Forecasted Price")
plt.grid(True)
plt.legend(bbox_to_anchor = (1,-0.1))
plt.tight_layout()
plt.show()

# Step 13: Fetch actual prices for forecast period
actual_prices = close_prices[pred_resid.index.min():pred_resid.index.max()]

# Align lengths if needed
actual_prices = actual_prices.loc[pred_resid.index]

# Calculate reconstructed forecasted prices (already done before)
# reconstructed_price is forecasted price array aligned with pred_resid.index

# Plot forecasted vs actual prices
plt.figure(figsize=(12,6))
plt.plot(actual_prices.index, actual_prices.values, label='Actual Price', color='blue')
plt.plot(pred_resid.index, reconstructed_price, label='Forecasted Price', color='red', linestyle='--')
plt.title("Forecasted vs Actual Nifty 50 Prices (2024)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


