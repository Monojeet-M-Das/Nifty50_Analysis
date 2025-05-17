# 📊 Hybrid Time Series Forecasting using AR Model

This project forecasts Nifty 50 index prices using a hybrid time series approach. The model decomposes the time series into trend, seasonality, and residuals, then applies an AutoRegressive (AR) model on the residuals.

## Features

- Download historical Nifty 50 data
- Calculate daily returns and RSI
- Decompose time series into trend/seasonal/residual
- Fit AR model on residuals
- Forecast future values by recombining components
- Visualize results with matplotlib

## Project Structure

- `hybrid_ar_forecast_project.ipynb`: Main Jupyter Notebook
- `README.md`: Project overview
- `.gitignore`: Git ignored files
- `requirements.txt`: Dependencies for the project

## Technologies

- Python
- Jupyter Notebook
- yfinance
- statsmodels
- pandas
- numpy
- matplotlib

## Tags

`time-series` `forecasting` `stock-market` `AR-model` `RSI` `quant-finance` `jupyter-notebook`
