# SimpleStockPredictingModel
It is a basic stock predicting webapp created using python with the known Facebook forecasting model known as fbprophet and is represented using Streamlit.

Used libraries:
  #-fbprophet is an open ai python library devloped by the data science team of facebook. It is known for its easy to use functions and it is also more accurate than the other neural networks used for time series data forecasting such as
  ----LSTM(Long hsort term memory)
  ----ARIMA(Autoregressive Integrated Moving Average).
  
  #-yfinance-It helps us to download the required datasets using yahoo finance.
  #-streamlit library makes it easier to create a webpage and provides various in built function that makes webapp devlopment so easy.
  #-plotly library is known for data visualization other than matplotlib or seaborn.
  #-bs4(BeautifulSoup4 or BeautifulSoup) is a well known data scrapping library in python, it provides various keywords that can perform different scrapping tasks.
  #-requests library is used to manage http requests in python such as parsing the data,decoding etc. 

On the sidebar we could see the two bifurcations of the predicting web-application:
  -Time Frame Stock Price Predictor
  
  -->This will fetch and predict the upcoming uncertainities and rise of the selected stocks.
  -->It can fetch as old data as possible.
  -->It can be used as a dynamic predictor.
  -->Also there are accurate graphs plotted on basis of the predicted data.
  
  -Live Data Stock Price Predictor
  
  -->This will fetch and predict the upcoming hour to hour uncertainities and rise of the selected stocks.
  -->It can fetch fetch the current day data only.
  -->Also there are accurate graphs plotted on basis of the predicted data of the current day.
  -->It is more benificial for real time trading as it could provide the predictions the next minutes and few even hours



####For ease the components of the predicted stock price chart are also plotted on the basis of different timeframes.####
####A news api is also integrated to this model that provides latest news straight from the well known stock markets of the world.####


  -->Also there are accurate graphs plotted on basis of the predicted dat
