import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
google_ser = get_data("GOOG","2020-01-01","2023-07-01")  #training data
#google_ser_oot = get_data("GOOG","2023-01-01","2023-09-26")
google_ser_oot = get_data("GOOG","2023-07-01","2023-09-26")  #test data
google_ser_tot = get_data("GOOG","2020-01-01","2023-09-26")  #all data

#print(google_ser)

#  Check if time series is stationary with ADF test(it is not)
result = adfuller(google_ser)
print(result)
print(f"ADF Statistic: {result [0]}")
print(f"p-value: {result [1]}")

#  Plot training data and auto correlation function
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

ax1.plot(google_ser)
ax1.set_title("training data")
plot_acf(google_ser, ax=ax2)
ax2.set_title("ACF training")





#decompose seasonality (no seasonality => double smooth)
#decompose_result = seasonal_decompose(google_ser,model='add',period=1)
decompose_result = seasonal_decompose(google_ser,model='mult',period=1)
decompose_result.plot()

HWES2_ADD = ExponentialSmoothing(google_ser_tot.values,trend='add').fit().fittedvalues
HWES2_MUL = ExponentialSmoothing(google_ser_tot.values,trend='mul').fit().fittedvalues

idx_oot = pd.date_range(start='2023-07-01',end='2023-09-26',freq='D')
print(idx_oot)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
ax1.plot(HWES2_ADD)
ax1.set_title("Holt Winters: Additive Trend")
ax2.plot(HWES2_MUL)
ax2.set_title("Holt Winters: Multiplicative Trend")

# Fit the model
model_mul = ExponentialSmoothing(endog=google_ser.values,trend='mul')
result_mul = model_mul.fit()
prediction_mul = result_mul.forecast(len(idx_oot))

model_add = ExponentialSmoothing(endog=google_ser.values,trend='add')
result_add = model_add.fit()
prediction_add = result_add.forecast(len(idx_oot))

fig, (ax1) = plt.subplots(1, 1, figsize=(8,4))
ax1.plot(google_ser_tot)
ax1.plot(idx_oot,prediction_mul)
ax1.plot(idx_oot,prediction_add)

#ax = google_ser.plot()
#result.fittedvalues.plot(ax=ax)
#train_visitors['no_of_visits'].plot(legend=True,label='TRAIN')
#test_visitors['no_of_visits'].plot(legend=True,label='TEST',figsize=(6,4))
#test_predictions.plot(legend=True,label='PREDICTION')
#plt.title('Train, Test and Predicted data points using Holt Winters Exponential Smoothing')

plt.show()


