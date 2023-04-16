#!/usr/bin/env python
# coding: utf-8

# will be updating this more...................


import datetime
from fbprophet import Prophet
import yfinance as yf
import streamlit as st
from plotly import graph_objs as go
from datetime import date
from fbprophet.plot import plot_plotly

#importing the required packages.

#-fbprophet is an open ai python library devloped by the data science team of facebook. It is known for its easy to use functions and it is also more accurate than the other neural networks used for time series data forecasting such as LSTM(Long hsort term memory) or ARIMA(Autoregressive Integrated Moving Average).
#-yfinance-It helps us to download the required datasets using yahoo finance.
#-streamlit library makes it easier to create a webpage and provides various in built function that makes webapp devlopment so easy.
#-plotly library is known for data visualization other than matplotlib or seaborn.

# In[11]:



#defining the time period to be overlooked.
# In[12]:


st.title("Stock Prediction App")


stocks1=st.text_input(label="Enter here if you are searching for a stock that is not available in  the list.")

stocks=["AAPL","GOOG","MSFT","GME","MS","RELIANCE.NS","ADANIENT.NS",stocks1]
selected_stocks=st.selectbox("Select dataset for prediction",stocks)


nYEARS=st.slider("Years of prediction.",1,4)
date1 = st.date_input('Select the timeframe of data based on which you need the predictions.', datetime.date(1990,1,1),max_value= date.today())
st.write(date1)

"Will recommend you to keep the date time to analyse the data is from 2022-01-01"

START=date1
TODAY=date.today().strftime("%Y-%m-%d")


period=nYEARS*365

#st.cache will save the dataset in the cache memory and every time we need the same dataset it will load that dataset from the cache memory instead of downloading it again and again.

def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data;
#this function will download and load the required dataset according to the specified time period.

def plotRawData():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Stock opening price'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='Stock closing price'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

#this function will simply plot the raw dataset that we have loaded.

data_load_state=st.text("LOADING DATA....")
data=load_data(selected_stocks)
data_load_state.text("The data is loaded.")

st.subheader("Raw data")
st.write(data.tail())

plotRawData()
#calling the function to plot the raw data.



#For forecasting, now we using the fbprophet library for which we will have to rename the date column to ds and the other value on whose basis we are going to predict the future dataframes as y.This is basically a norm to use the library.

#The mathematical equation behind the Prophet model is defined as:
#y(t) = g(t) + s(t) + h(t) + e(t)
# g(t) representing the trend. Prophet uses a piecewise linear model for trend forecasting.
#s(t) represents periodic changes (weekly, monthly, yearly).
#h(t) represents the effects of holidays (recall: Holidays impact businesses).
#e(t) is the error term.


df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

model=Prophet(daily_seasonality=True,growth="linear")



model.fit(df_train)
future=model.make_future_dataframe(periods=period)
forecast=model.predict(future)

st.subheader("Forecast data")
st.write(forecast)

fig1=plot_plotly(model,forecast)
st.plotly_chart(fig1)

#plotting the forecasted data

st.write("Forecast Components")
fig2=model.plot_components(forecast)
st.write(fig2)


st.write(forecast.describe())


'''
ds — Datestamp or timestamp which values in that row pertain to

trend — Value of the trend component alone

yhat_lower — Lower bound of the uncertainty interval around the final prediction

yhat_upper — Upper bound of the uncertainty interval around the final prediction

trend_lower — Lower bound of the uncertainty interval around the trend component

trend_upper— Upper bound of the uncertainty interval around the trend component

additive_terms— Combined value of all additive seasonalities

additive_terms_lower — Lower bound of the uncertainty interval around the additive seasonalities

additive_terms_upper — Upper bound of the uncertainty interval around the additive seasonalities

weekly— Value of the weekly seasonality component

weekly_lower — Lower bound of the uncertainty interval around the weekly component

weekly_upper — Upper bound of the uncertainty interval around the weekly component

yearly — Value of the yearly seasonality component

yearly_lower — Lower bound of the uncertainty interval around the yearly component

yearly_upper — Upper bound of the uncertainty interval around the yearly component

multiplicative_terms — Combined value of all multiplicative seasonalities

multiplicative_terms_lower — Lower bound of the uncertainty interval around the multiplicative seasonalities

multiplicative_terms_upper — Upper bound of the uncertainty interval around the multiplicative seasonalities

yhat — Final predicted value; a combination of trend, multiplicative_terms and additive_terms'''



# In[ ]:




