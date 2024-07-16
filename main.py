from time import sleep
import uuid
import pandas as pd
from sklearn.metrics import mean_absolute_error
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import wikipedia
import plotly.graph_objects as go
import yfinance as yf

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Forcastify", page_icon="📈")
# Change background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: lightblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.markdown("<h1 style='text-align: center; font-size: 30px;'><b>Group :</b><b style='color: orange'>1</b></h1>", unsafe_allow_html=True)
st.sidebar.title("Options")
start_date_key = str(uuid.uuid4())
start_date = st.sidebar.date_input("Start date", date(2018, 1, 1), key=start_date_key)
end_date = st.sidebar.date_input("End date", date.today())

# Header
st.markdown("<h1 style='text-align: center;'>Stock Forecast App 📈</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>Forcasti.</b><b style='color: orange'>fy</b> is a simple web app for stock price prediction using the <a href='https://facebook.github.io/prophet/'>Prophet</a> library.</p>", unsafe_allow_html=True)

selected_tab = option_menu(
    menu_title=None,
    options=["Dataframes", "Plots", "Statistics", "Information", "Forecasting", "Comparison"],
    icons=["table", "bar-chart", "calculator", "info-circle", "graph-up-arrow", "arrow-down-up"],
    menu_icon="📊",
    default_index=0,
    orientation="horizontal",
)

# Stock selection
stocks = (
    "AAPL", "GOOG", "MSFT", "GME", "AMC", "TSLA", "AMZN", "NFLX", "NVDA", "AMD", "PYPL",
    "FB", "BABA", "INTC", "ORCL", "IBM", "CSCO", "V", "MA", "DIS", "NKE", "PFE", "JNJ",
    "WMT", "T", "XOM", "CVX", "KO", "PEP", "MCD", "BA", "MMM", "GE", "CAT", "HD",
    "LOW", "SBUX", "SPY", "QQQ", "DIA", "GLD", "SLV", "VTI", "IWM", "ARKK", "PLTR",
    "SNAP", "TWTR", "SHOP", "ZM", "ROKU", "UBER", "LYFT", "SQ", "BYND", "ZM", "DKNG"
)

# Stocks abreviations
selected_stock = st.sidebar.selectbox("Select stock for prediction", stocks)
selected_stocks = st.sidebar.multiselect("Select stocks for comparison", stocks)

years_to_predict = st.sidebar.slider("Years of prediction:", 1, 5)
period = years_to_predict * 365

@st.cache_data
def load_data(ticker, start, end):
    """
    Load historical stock price data from Yahoo Finance.

    Parameters:
    - ticker (str): Stock symbol (e.g., AAPL).
    - start (str): Start date in the format 'YYYY-MM-DD'.
    - end (str): End date in the format 'YYYY-MM-DD'.

    Returns:
    - data (pd.DataFrame): DataFrame containing historical stock price data.
    """
    try:
        data = yf.download(ticker, start, end)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def plot_data(data):
    """
    Plot historical stock price data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing historical stock price data.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", line=dict(color='blue')))
    fig.update_layout(title_text="Stock Prices Over Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_multiple_data(data, stock_names):
    """
    Plot forecasted stock prices for multiple stocks.

    Parameters:
    - data (list): List of DataFrames containing forecasted stock price data.
    - stock_names (list): List of stock names corresponding to the forecasted data.
    """
    fig = go.Figure()
    for i, stock_data in enumerate(data):
        fig.add_trace(go.Scatter(x=stock_data['ds'], y=stock_data['yhat'], name=f"yhat - {stock_names[i]}"))
    fig.update_layout(title_text="Stock Prices Over Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_volume(data):
    """
    Plot historical stock volume data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing historical stock volume data.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name="stock_volume"))
    fig.update_layout(title_text="Stock Volume Over Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

# Display a loading spinner while loading data
with st.spinner("Loading data..."):
    data = load_data(selected_stock, start_date, end_date)
    sleep(1)

# Display the success message
success_message = st.success("Data loaded successfully!")

# Introduce a delay before clearing the success message
sleep(1)

# Clear the success message
success_message.empty()

# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Convert end_date to datetime
end_date_datetime = pd.to_datetime(end_date)

# Filter forecast based on end_date
forecast = forecast[forecast['ds'] >= end_date_datetime]

# Information Tab
if selected_tab == "Information":
    st.markdown("<h2><span style='color: orange;'>Information</span> about {}</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section provides information about {} from Wikipedia.".format(selected_stock))

    try:
        page = wikipedia.page(selected_stock)
        content = page.content

        card_html = f"""
        <div style="background-color: white; color: black; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);">
            <p>{content}</p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error fetching information for {selected_stock}: {str(e)}")


# Dataframes Tab
if selected_tab == "Dataframes":
    # Display historical data
    st.markdown("<h2><span style='color: orange;'>{}</span> Historical Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays historical stock price data for {} from {} to {}.".format(selected_stock, start_date, end_date))
    
    # Copy data
    new_data = data.copy()

    # Drop Adj Close and Volume columns
    new_data = data.drop(columns=['Adj Close', 'Volume'])
    st.dataframe(new_data, use_container_width=True)

    # Display forecast data
    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays the forecasted stock price data for {} using the Prophet model from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    
    # Copy forecast dataframe
    new_forecast = forecast.copy()

    # Drop unwanted columns
    new_forecast = new_forecast.drop(columns=[
        'additive_terms', 
        'additive_terms_lower', 
        'additive_terms_upper', 
        'weekly', 
        'weekly_lower', 
        'weekly_upper', 
        'yearly', 
        'yearly_lower', 
        'yearly_upper', 
        'multiplicative_terms', 
        'multiplicative_terms_lower', 
        'multiplicative_terms_upper'
    ])
    
    # Rename columns
    new_forecast = new_forecast.rename(columns={
        "ds": "Date", 
        "yhat": "Close", 
        "yhat_lower": "Close Lower",
        "yhat_upper": "Close Upper",
        "trend": "Trend", 
        "trend_lower": "Trend Lower", 
        "trend_upper": "Trend Upper"
    })

    st.dataframe(new_forecast, use_container_width=True)

# Plots Tab
if selected_tab == "Plots":
    # Raw data plot
    plot_data(data)

    # Data Volume plot
    plot_volume(data)

# Statistics Tab
if selected_tab == "Statistics":
    st.markdown("<h2><span style='color: orange;'>Descriptive </span>Statistics</h2>", unsafe_allow_html=True)
    st.write("This section provides descriptive statistics for the selected stock.")

    # Descriptive Statistics Table
    # drop the date column
    #data = data.drop(columns=['Date', 'Adj Close', 'Volume'])
    st.table(data.describe())
     # Define statistics to show
    statistics = {
        "Mean Open": data['Open'].mean(),
        "Mean Close": data['Close'].mean(),
        "Mean Volume": data['Volume'].mean(),
        "Mean High": data['High'].mean(),
        "Mean Low": data['Low'].mean(),
        "MAE": mean_absolute_error(data['Open'], data['Close'])
    }

    st.write(statistics)
# Forecasting Tab    
if selected_tab == "Forecasting":
    # Plotting forecast
    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Plot</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))

    # Merge actual and forecasted data
    combined = pd.concat([df_train.set_index('ds'), forecast.set_index('ds')], axis=1).reset_index()

    # Custom plot for forecast
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=combined['ds'], y=combined['y'], mode='lines', name='Actual', line=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=combined['ds'], y=combined['yhat'], mode='lines', name='Predicted', line=dict(color='red')))
    fig_forecast.add_trace(go.Scatter(x=combined['ds'], y=combined['yhat_upper'], fill='tonexty', mode='none', name='Upper Bound', line=dict(color='rgba(255,0,0,0)', width=0), fillcolor='rgba(255,0,0,0.1)'))
    fig_forecast.add_trace(go.Scatter(x=combined['ds'], y=combined['yhat_lower'], fill='tonexty', mode='none', name='Lower Bound', line=dict(color='rgba(255,0,0,0)', width=0), fillcolor='rgba(255,0,0,0.1)'))

    fig_forecast.update_layout(title=f'Forecasted Stock Price for {selected_stock}', xaxis_title='Date', yaxis_title='Stock Price', showlegend=True)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Plotting forecast components
    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Components</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section breaks down the forecast components, including trends and seasonality, for {} from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    components = model.plot_components(forecast)
    st.write(components)

# Comparison Tab
if selected_tab == "Comparison":
    if selected_stocks:
        # Forecast multiple stocks
        stocks_data = []
        forcasted_data = []
        for stock in selected_stocks:
            stocks_data.append(load_data(stock, start_date, end_date))

        st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Comparison Plot</h2>".format(', '.join(selected_stocks)), unsafe_allow_html=True)
        st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(', '.join(selected_stocks), end_date, end_date + pd.Timedelta(days=period)))

        for i, data in enumerate(stocks_data):
            if data is not None:
                df_train = data[["Date", "Close"]]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                model = Prophet()
                model.fit(df_train)
                future = model.make_future_dataframe(periods=period)
                forecast = model.predict(future)
                forecast = forecast[forecast['ds'] >= end_date_datetime]
                st.markdown("<h3><span style='color: orange;'>{}</span> Forecast DataFrame</h3>".format(selected_stocks[i]), unsafe_allow_html=True)

                # Copy forecast dataframe
                new_forecast = forecast.copy()

                # Drop unwanted columns
                new_forecast = new_forecast.drop(columns=[
                    'additive_terms', 
                    'additive_terms_lower', 
                    'additive_terms_upper', 
                    'weekly', 
                    'weekly_lower', 
                    'weekly_upper', 
                    'yearly', 
                    'yearly_lower', 
                    'yearly_upper', 
                    'multiplicative_terms', 
                    'multiplicative_terms_lower', 
                    'multiplicative_terms_upper'
                ])

                # Rename columns
                new_forecast = new_forecast.rename(columns={
                    "ds": "Date", 
                    "yhat": "Close", 
                    "yhat_lower": "Close Lower",
                    "yhat_upper": "Close Upper",
                    "trend": "Trend", 
                    "trend_lower": "Trend Lower", 
                    "trend_upper": "Trend Upper"
                })

                st.dataframe(new_forecast, use_container_width=True)

                forcasted_data.append(forecast)

        plot_multiple_data(forcasted_data, selected_stocks)
    else:
        st.warning("Please select at least one stock if you want to compare them.")

# Display balloons at the end
# st.balloons()v
