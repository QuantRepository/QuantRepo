import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    #returns = stockData.pct_change()
    #meanReturns = returns.mean()
    #covMatrix = returns.cov()
    return stockData  #, covMatrix

# Get data and split in test and training
google_ser = get_data("GOOG","2020-01-01","2023-07-01")  #training data
google_ser_oot = get_data("GOOG","2023-07-01","2023-09-26")  #test data
google_ser_tot = get_data("GOOG","2020-01-01","2023-09-26")  #all data

google_ser_tot_arr = google_ser_tot.values
nrdata_tot = len(google_ser_tot)

google_ser_oot_arr = google_ser_oot.values
nrdata_oot = len(google_ser_oot)

google_ser_arr = google_ser.values
nrdata_train = len(google_ser)

#Scale the data and make sure there is no information leak
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data_tot = scaler.fit_transform(google_ser_tot_arr.reshape(-1,1))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data_test = scaler.fit_transform(google_ser_oot_arr.reshape(-1,1))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data_training = scaler.fit_transform(google_ser_arr.reshape(-1,1))


x_train = []
y_train = []

nrmemory = 20

for i in range(nrmemory, len(scaled_data_training)):
    x_train.append(scaled_data_training[i-nrmemory:i,0])
    y_train.append(scaled_data_training[i,0])

#Change feature and label into the right format
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#Initialize LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile LSTM
model.compile(optimizer='adam', loss='mean_squared_error')

#Fit LSTM
model.fit(x_train, y_train, batch_size=1, epochs=1)

##Get the predicted price and compare
data_oot = scaled_data_tot[nrdata_train - nrmemory: , :]

x_test = []
y_test = google_ser_tot_arr.reshape(-1,1)[nrdata_train:, :]
for i in range(nrmemory, len(data_oot)):
    x_test.append(data_oot[i-nrmemory:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(len(x_test))


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#errors of model
mae = mean_absolute_error(y_test,predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions)

print(f'mae - manual: {mae}')
print(f'mape - manual: {mape}')
print(f'rmse - manual: {rmse}')

#Plot predictions vs actual closing prize

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

ax1.plot(google_ser_tot)
ax1.set_title("Prediction vs actual closing prize")

keylist = list(google_ser_oot.keys())
ax1.plot(google_ser_tot)
ax1.plot(keylist,predictions)

ax2.plot(google_ser_tot.values)
ax2.plot(np.arange(nrdata_tot-nrdata_oot, nrdata_tot, 1),predictions)

plt.show()

print(type(google_ser_tot))
