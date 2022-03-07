from pyrsistent import v
import streamlit as st
import pandas as pd
import pandas_datareader as web
import numpy as np
import seaborn as sns
from datetime import date, timedelta
import plotly.express as px
import plotly.io as pio

# Functions
@st.cache(allow_output_mutation=True)
def get_data_today(tickers):
    """Solve yahoo error with different dates"""
    data = pd.DataFrame()
    for tic in tickers:
        try:
            dt = web.DataReader(tic, "yahoo", date.today()-timedelta(days=250), date.today())["Adj Close"].dropna()           
            dt = dt.rename(tic)
            data = data.join(dt, how="outer")
        except KeyError:
            print("The market is closed today")
    data = data.fillna(method='ffill')
    return data

@st.cache(allow_output_mutation=True)
def price_ret_summary(tickers):
    data = get_data_today(tickers)
    p = pd.DataFrame()
    r = pd.DataFrame()
    for t in tickers:
        p.loc[t, "1d"] = data[t][-1]
        p.loc[t, "10d"] = data[t][-8]
        p.loc[t, "1m"] = data[t][-22]
        p.loc[t, "2m"] = data[t][-44]
        p.loc[t, "6m"] = data[t][-132]
        r.loc[t, "1d"] = ((data[t].pct_change())[-1])*100
        r.loc[t, "10d"] = ((data[t].pct_change(8))[-1])*100
        r.loc[t, "1m"] = ((data[t].pct_change(22))[-1])*100
        r.loc[t, "2m"] = ((data[t].pct_change(44))[-1])*100
        r.loc[t, "6m"] = ((data[t].pct_change(132))[-1])*100
    return p, r

####### Sidebar
select_page = st.sidebar.selectbox("Select a page:", options=("Dashboard", "Charts"))

####### Dashboard
if select_page == "Dashboard":
    header = st.container()
    box1 = st.container()
    box2 = st.container()
    # Equity market indexes
    tickers1 = ["^GSPC", "^IXIC", "000001.SS", "^STOXX50E", "EEM", "^VIX"]
    price1, retsumm1 = price_ret_summary(tickers1) 
    price1.rename(index={"^GSPC":"S&P 500", "^IXIC":"NASDAQ", "000001.SS":"SSE Comp.", "^STOXX50E":"Eurostoxx50", "EEM":"Emerging Mkts", "^VIX":"VIX"}, inplace=True)
    retsumm1.rename(index={"^GSPC":"S&P 500", "^IXIC":"NASDAQ", "000001.SS":"SSE Comp.", "^STOXX50E":"Eurostoxx50", "EEM":"Emerging Mkts", "^VIX":"VIX"}, inplace=True)
    # Currencies
    tickers2 = ["EURUSD=X", "GBPUSD=X", "USDCNY=X", "GBPEUR=X", "BTC-USD", "ETH-USD"] 
    price2, retsumm2 = price_ret_summary(tickers2) 
    price2.rename(index={"EURUSD=X":"EUR/USD", "GBPUSD=X":"GBP/USD", "USDCNY=X":"Renminbi/USD", "GBPEUR=X":"GBP/EUR", "BTC-USD":"BTC/USD", "ETH-USD":"ETH/USD"}, inplace=True)
    retsumm2.rename(index={"EURUSD=X":"EUR/USD", "GBPUSD=X":"GBP/USD", "USDCNY=X":"Renminbi/USD", "GBPEUR=X":"GBP/EUR", "BTC-USD":"BTC/USD", "ETH-USD":"ETH/USD"}, inplace=True)
    # Commodities (futures)
    tickers3 = ["^BCOM", "GC=F", "CL=F", "SI=F", "PA=F", "ZW=F"]
    price3, retsumm3 = price_ret_summary(tickers3) 
    price3.rename(index={"^BCOM":"Comm.Index", "GC=F":"Gold", "CL=F":"Crude", "SI=F":"Silver", "PA=F":"Palladium", "ZW=F":"Wheat"}, inplace=True)
    retsumm3.rename(index={"^BCOM":"Comm.Index", "GC=F":"Gold", "CL=F":"Crude", "SI=F":"Silver", "PA=F":"Palladium", "ZW=F":"Wheat"}, inplace=True)
    # Interest rates and Macro (monthly)
    # tickers4 = ["DGS10", "SOFR", "ECBESTRVOLWGTTRMDMNRT"] 
    # price4, retsumm4 = price_ret_summary(tickers4) 
    # price4.rename(index={"DGS10":"10y US Yield", "SOFR":"SOFR", "ECBESTRVOLWGTTRMDMNRT":"ESTR"}, inplace=True)
    # retsumm4.rename(index={"DGS10":"10y US Yield", "SOFR":"SOFR", "ECBESTRVOLWGTTRMDMNRT":"ESTR"}, inplace=True)
    # tickers5 = ["T10YIE", "UNRATE", "LRHUTTTTEZM156S"]
    # price5, retsumm5 = price_ret_summary(tickers5) 
    # price5.rename(index={"T10YIE":"10y US Inflation", "UNRATE":"US Unempl.", "LRHUTTTTEZM156S":"EU Unempl."}, inplace=True)
    # retsumm5.rename(index={"T10YIE":"10y US Inflation", "UNRATE":"US Unempl.", "LRHUTTTTEZM156S":"EU Unempl."}, inplace=True)
    
    with header:
        st.title("Financial Dashboard")
        st.write("Daily and monthly returns for indexes; Value and monthly return for currencies and commodities")
    
    with box1:
        # Equity market indexes
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("S&P 500", f"{retsumm1.loc['S&P 500', '1d']:.2f}%", f"{retsumm1.loc['S&P 500', '1m']:.2f}%", delta_color="off")
        col2.metric("NASDAQ", f"{retsumm1.loc['NASDAQ', '1d']:.2f}%", f"{retsumm1.loc['NASDAQ', '1m']:.2f}%", delta_color="off")
        col3.metric("SSE Comp.", f"{retsumm1.loc['SSE Comp.', '1d']:.2f}%", f"{retsumm1.loc['SSE Comp.', '1m']:.2f}%", delta_color="off")
        col4.metric("Eurostoxx50", f"{retsumm1.loc['Eurostoxx50', '1d']:.2f}%", f"{retsumm1.loc['Eurostoxx50', '1m']:.2f}%", delta_color="off")
        col5.metric("Emerging Mkts", f"{retsumm1.loc['Emerging Mkts', '1d']:.2f}%", f"{retsumm1.loc['Emerging Mkts', '1m']:.2f}%", delta_color="off")
        col6.metric("VIX", f"{price1.loc['VIX', '1d']:.2f}", f"{retsumm1.loc['VIX', '1d']:.2f}%")
        # Currencies
        col1.metric("EUR/USD", f"{price2.loc['EUR/USD', '1d']:.2f}", f"{retsumm2.loc['EUR/USD', '1m']:.2f}%")
        col2.metric("GBP/USD", f"{price2.loc['GBP/USD', '1d']:.2f}", f"{retsumm2.loc['GBP/USD', '1m']:.2f}%")
        col3.metric("Renminbi/USD", f"{price2.loc['Renminbi/USD', '1d']:.2f}", f"{retsumm2.loc['Renminbi/USD', '1m']:.2f}%")
        col4.metric("GBP/EUR", f"{price2.loc['GBP/EUR', '1d']:.2f}", f"{retsumm2.loc['GBP/EUR', '1m']:.2f}%")
        col5.metric("BTC/USD", f"{price2.loc['BTC/USD', '1d']:.2f}", f"{retsumm2.loc['BTC/USD', '1m']:.2f}%")
        col6.metric("ETH/USD", f"{price2.loc['ETH/USD', '1d']:.2f}", f"{retsumm2.loc['ETH/USD', '1m']:.2f}%")
        # Commodities
        col1.metric("Comm.Index", f"{retsumm3.loc['Comm.Index', '1d']:.2f}%", f"{retsumm3.loc['Comm.Index', '1m']:.2f}%", delta_color="off")
        col2.metric("Gold", f"{price3.loc['Gold', '1d']:.2f}", f"{retsumm3.loc['Gold', '1m']:.2f}%")
        col3.metric("Crude", f"{price3.loc['Crude', '1d']:.2f}", f"{retsumm3.loc['Crude', '1m']:.2f}%")
        col4.metric("Silver", f"{price3.loc['Silver', '1d']:.2f}", f"{retsumm3.loc['Silver', '1m']:.2f}%")
        col5.metric("Palladium", f"{price3.loc['Palladium', '1d']:.2f}", f"{retsumm3.loc['Palladium', '1m']:.2f}%")
        col6.metric("Wheat", f"{price3.loc['Wheat', '1d']:.2f}", f"{retsumm3.loc['Wheat', '1m']:.2f}%")
        # Interest rates and Macro
        # col1.metric("10y US Yield", f"{price4.loc['10y US Yield', '1d']:.2f}%", f"{retsumm4.loc['10y US Yield', '1m']:.2f}%", delta_color="off")
        # col2.metric("SOFR", f"{price4.loc['SOFR', '1d']:.2f}%", f"{retsumm4.loc['SOFR', '1m']:.2f}%", delta_color="off")
        # col3.metric("ESTR", f"{price4.loc['ESTR', '1d']:.2f}%", f"{retsumm4.loc['ESTR', '1m']:.2f}%", delta_color="off")
        # col4.metric("10y US Inflation", f"{price5.loc['10y US Inflation', '1d']:.2f}%", f"{retsumm5.loc['10y US Inflation', '1d']:.2f}%", delta_color="off")
        # col5.metric("US Unempl.", f"{price5.loc['US Unempl.', '1d']:.2f}%", f"{retsumm5.loc['US Unempl.', '1d']:.2f}%", delta_color="off")
        # col6.metric("EU Unempl.", f"{price5.loc['EU Unempl.', '1d']:.2f}%", f"{retsumm5.loc['EU Unempl.', '1d']:.2f}%", delta_color="off")
        
    with box2:
        st.subheader("Equity market indexes")
        st.dataframe(price1)
        st.dataframe(retsumm1.style.background_gradient(axis=1))
        st.subheader("Currencies")
        st.dataframe(price2)
        st.dataframe(retsumm2.style.background_gradient(axis=1))
        st.subheader("Commodities")
        st.dataframe(price3)
        st.dataframe(retsumm3.style.background_gradient(axis=1))
        # st.subheader("Interest Rates")
        # st.dataframe(retsumm1.style.background_gradient(axis=1))

####### Charts
if select_page == "Charts":
    header = st.container()
    box1 = st.container()

    with header:
        st.title("Charts")
    
    with box1:
        sns.set()
        start = st.date_input("Start Date", date.today()-timedelta(days=250))
        end = st.date_input("End Date", date.today())
        tic = st.text_input('Ticker', 'AAPL')
        data = web.DataReader(tic, "yahoo", start, end)["Adj Close"].dropna()
        st.write(f"Total log. return: {(data[-1]/data[0]-1)*100:.2f}%")
        st.write(f"Total return: {(np.log(data[-1]/data[0]))*100:.2f}%")
        # Price chart
        fig = px.line(data)
        st.plotly_chart(fig, use_container_width=True, template="plotly_dark", line=dict(color='firebrick', width=4))



