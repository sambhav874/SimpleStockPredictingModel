#!/usr/bin/env python
# coding: utf-8

# In[10]:


from fbprophet import Prophet
import yfinance as yf
import streamlit as st
from plotly import graph_objs as go
from datetime import date
from fbprophet.plot import plot_plotly
import tensorflow


# In[11]:


START="2023-01-01"
TODAY=date.today().strftime("%Y-%m-%d")


# In[12]:


st.title("Stock Prediction App")

stock_types=[]
selected_stock_type=st.selectbox("Select the type of stock")


stocks=["AAPL","GOOG","MSFT","GME","MS","RELIANCE.NS","ADANIENT.NS"]
selected_stocks=st.selectbox("Select dataset for prediction",stocks)



nYEARS=st.slider("Years of prediction.",1,4)
period=nYEARS*365
@st.cache
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data;

def plotRawData():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Stock opening price'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='Stock closing price'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

#For forecasting




# In[13]:

data_load_state=st.text("LOADING DATA....")
data=load_data(selected_stocks)
data_load_state.text("The data is loaded.")

st.subheader("Raw data")
st.write(data.tail())

plotRawData()

df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

model=Prophet(daily_seasonality=True)
model.fit(df_train)
future=model.make_future_dataframe(periods=period)
forecast=model.predict(future)

st.subheader("Forecast data")
st.write(forecast)

fig1=plot_plotly(model,forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2=model.plot_components(forecast)
st.write(fig2)


st.write(forecast.describe())


'''
ds — Datestamp or timestamp which values in that row pertain to
trend — Value of the trend component alone
yhat_lower — Lower bound of the uncertainty interval around the final prediction
yhat_upper — Upper bound of the uncertainty interval around the final prediction
trend_lower — Lower bound of the uncertainty interval around the trend component‘
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




