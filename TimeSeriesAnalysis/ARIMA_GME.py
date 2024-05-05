# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

from pandas_datareader import data as pdr
import yfinance as yf
from scipy.signal import periodogram

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

from pmdarima.arima.utils import ndiffs

import pmdarima as pm

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error



# import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    #returns = stockData.pct_change()
    #meanReturns = returns.mean()
    #covMatrix = returns.cov()
    return stockData  #, covMatrix

# Set training and test data
#google_ser = get_data("GOOG","2020-01-01","2022-12-31")
gme_ser = get_data("GME","2020-01-01","2023-07-01")  #training data
#google_ser_oot = get_data("GOOG","2023-01-01","2023-09-26")
gme_ser_oot = get_data("GME","2023-07-01","2023-09-26")  #test data
gme_ser_tot = get_data("GME","2020-01-01","2023-09-26")  #all data

print(gme_ser)

#  Check if time series is stationary with ADF test(it is not)
result = adfuller(gme_ser)
print(result)
print(f"ADF Statistic training data: {result [0]}")
print(f"p-value training data: {result [1]}")

#  Plot training data and auto correlation function
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

ax1.plot(gme_ser)
ax1.set_title("0 diff")
plot_acf(gme_ser, ax=ax2)
ax2.set_title("ACF 0 diff")


#  difference data and check if stationary with ADF and eyeball test (it is)
#  check acf and pacf to get p and q in ARIMA model
diff = gme_ser.diff().dropna()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,10))

ax1.plot(diff)
ax1.set_title("1 diff")
plot_acf(diff, ax=ax2, label='ACF')
ax2.set_title("ACF 1 diff")
plot_pacf(diff, ax=ax3, label='PACF')
ax3.set_title("PACF 1 diff")

result = adfuller(diff)
print(result)
print(f"ADF Statistic diff1 data: {result [0]}")
print(f"p-value diff1 data: {result [1]}")

#  difference data a second time and check if stationary with ADF and eyeball test (it is)
#  check acf and pacf to get p and q in ARIMA model
diff2 = diff.diff().dropna()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,10))

ax1.plot(diff2)
ax1.set_title("2 diff")
plot_acf(diff2, ax=ax2)
ax2.set_title("ACF 2 diff")
plot_pacf(diff2, ax=ax3)
ax3.set_title("PACF 2 diff")

result = adfuller(diff2)
print(result)
print(f"ADF Statistic diff2 data: {result [0]}")
print(f"p-value diff2 data: {result [1]}")


# automatic routine to test with ADF how many times differencing is needed (1)
nrdiffs = ndiffs(gme_ser.values, test="adf")
print(f"Number of diffs needed: {nrdiffs}")


#  FFT to check for seasonality in 1 diff data(no seasonality)
fhat = np.fft.fft(diff,len(diff))  # Compute FFT
PSD = fhat * np.conj(fhat)/len(diff)  # Compute power spectrum
freq = np.arange(len(diff))  # Create x-axis of frequencies
L = np.arange(1, np.floor(len(diff)/2), dtype='int')

fig,axs = plt.subplots(2,1)

plt.sca(axs[0])
plt.plot(gme_ser.values)
axs[0].set_title("0 diff data")
plt.sca(axs[1])
plt.plot(freq[L], PSD[L])
axs[1].set_title("Frequencies in 1 diff data")


#  Function to create manual ARIMA model with p,d,q (4,1,4 here)
model = ARIMA(gme_ser.values, order=(4,1,4))
result = model.fit()
print(result.summary())

#  Calc residuals and analyse them
residuals = pd.DataFrame(result.resid[2:])

residuals_adftest = adfuller(residuals)
print(f"ADF Statistic residuals manual: {residuals_adftest [0]}")
print(f"p-value residuals manual: {residuals_adftest [1]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

#ax1.plot(residuals[2:-1])
ax1.plot(residuals)
ax1.set_title("Residuals manual ARIMA")
#ax2.hist(residuals[2:-1], density=True)
ax2.hist(residuals, density=True)
ax2.set_title("Residuals hist manual ARIMA")
residuals.plot(kind='kde', ax=ax2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

plot_acf(residuals, ax=ax1)
ax1.set_title("acf residuals manual ARIMA")

plot_pacf(residuals, ax=ax2)
ax2.set_title("pacf residuals manual ARIMA")

plot_predict(result, start=2, end=-1, dynamic=False)
plt.plot(gme_ser.values[1:-1])
plt.title("Prediction manual ARIMA vs actual values of training data")



#  Function to create ARIMA model with automatically found p,d,q (0,1,0 here)
auto_arima = pm.auto_arima(gme_ser.values, stepwise=False, seasonal=False)
print(auto_arima.summary())
#  Plot diagnostics of ARIMA model
auto_arima.plot_diagnostics(figsize=(16,10))


#  create forecast for manual and auto model
forecast_test = result.forecast(len(gme_ser_oot))
forecast_test_auto = auto_arima.predict(len(gme_ser_oot))


#  plot forecasts and oot data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
ax1.plot(forecast_test)
ax1.plot(forecast_test_auto)
ax1.plot(gme_ser_oot.values)
ax1.set_title("Manual forecast (b) vs auto forecast (o) vs oot data (g)")
ax2.plot(gme_ser_tot)
ax2.set_title("full data")

mae = mean_absolute_error(gme_ser_oot.values, forecast_test)
mape = mean_absolute_percentage_error(gme_ser_oot.values, forecast_test)
rmse = mean_squared_error(gme_ser_oot.values, forecast_test)

print(f'mae - manual: {mae}')
print(f'mape - manual: {mape}')
print(f'rmse - manual: {rmse}')

mae_auto = mean_absolute_error(gme_ser_oot.values, forecast_test_auto)
mape_auto = mean_absolute_percentage_error(gme_ser_oot.values, forecast_test_auto)
rmse_auto = mean_squared_error(gme_ser_oot.values, forecast_test_auto)

print(f'mae - auto: {mae_auto}')
print(f'mape - auto: {mape_auto}')
print(f'rmse - auto: {rmse_auto}')

plt.show()
#print(type(google_ser))
#plt.plot(google_ser)
#plt.show()