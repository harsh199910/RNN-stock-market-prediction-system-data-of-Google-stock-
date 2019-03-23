# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:33:09 2019

@author: HARSH
"""
#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset only training dataset

dataset_train = pd.read_csv('Google_Stock_Price_train.csv')
#creating numpy array
training_set = dataset_train.iloc[:, 1:2].values

#feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc =MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#creating data struture with 60 timestep with 1 output

x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train),  np.array(y_train)  

#reshaping 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#building RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM    

regressor = Sequential()

#adding some LSTM layers and dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#adding extra LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding extra LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

#compling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

#getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#making the prediction
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test) 

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)












