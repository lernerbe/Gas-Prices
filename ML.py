import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# dataframe
df = pd.read_csv("daily.csv")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# visualize the data
plt.figure(figsize=(10, 6))
plt.plot(df['Price'], label='Gas Price')
plt.title('Daily Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# fir the arima model
model = ARIMA(df['Price'], order=(5, 1, 2))
model_fit = model.fit()

forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# plot the prices
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Price'], label='Historical Gas Prices')
plt.plot(pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='B')[1:], 
         forecast, label='Forecasted Gas Prices', linestyle='--')
plt.title('Gas Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# print
future_dates = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Price': forecast})
print(forecast_df)
