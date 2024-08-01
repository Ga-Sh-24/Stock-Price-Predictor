import numpy as np
import pandas as pd
from keras.models import load_model
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


model = load_model('C:\\Users\\Garima Shrivastava\\OneDrive\\Desktop\\stock price prediction\\Stock Price Prediction Model.keras')

st.header('Stock Price Predictor')

stock = st.text_input('Enter the stock symbol', 'MSFT')     #by default, the stock symbol of Google is considered

start = '2010-01-01'
end = '2023-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

#splitting into 80% training data and 20% test data
train_data = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
test_data =pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = train_data.tail(100)
test_data = pd.concat([past_100_days, test_data], ignore_index=True)
test_data_scale = scaler.fit_transform(test_data)

st.subheader('Original Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
figure1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(figure1)

st.subheader('Original Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
figure2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(figure2)

st.subheader('Original Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
figure3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(figure3)

x = []
y = []

for i in range(100, test_data_scale.shape[0]):
    x.append(test_data_scale[i-100:i])
    y.append(test_data_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
figure4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(figure4)