import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- App Configuration ---
st.set_page_config(
    page_title="Marketstack Stock Data Viewer",
    page_icon="📈",
    layout="wide",
)

# --- CHARTING FUNCTION ---
def create_combo_chart(df: pd.DataFrame):
    """Creates a combination line and bar chart for close price and volume."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['close'], name="Close Price", line=dict(color='royalblue')),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name="Volume", marker=dict(color='lightgray')),
        secondary_y=True,
    )
    fig.update_layout(
        title_text=f"<b>Price and Volume for {df['symbol'].iloc[0]}</b>",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Close Price (USD)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True)
    return fig

# --- Sidebar for User Input ---
st.sidebar.title("Settings")
st.sidebar.write("Configure your data request below.")

# Symbol Input
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()

# Date Interval Selection
st.sidebar.subheader("Select Date Range")
date_to = st.sidebar.date_input("To", value=datetime.date.today())
date_from = st.sidebar.date_input("From", value=datetime.date.today() - datetime.timedelta(days=10))

# Date Validation
valid_date_range = True
if date_from > date_to:
    st.sidebar.error("Error: 'From' date cannot be after 'To' date.")
    valid_date_range = False

# --- Main Application ---
st.title(f"📈 Marketstack Stock Data Viewer for {symbol}")
st.write(
    f"""
    This app fetches and displays historical end-of-day stock data for **{symbol}** from **{date_from.strftime('%Y-%m-%d')}** to **{date_to.strftime('%Y-%m-%d')}**.
    """
)

# --- API Request ---
API_ACCESS_KEY = "api-key"
API_URL = "http://api.marketstack.com/v1/eod"

params = {
    "access_key": API_ACCESS_KEY,
    "symbols": symbol,
    "date_from": date_from.strftime('%Y-%m-%d'),
    "date_to": date_to.strftime('%Y-%m-%d'),
    "limit": 1000 # Max limit to ensure all data in range is fetched
}

# --- Fetch and Display Data ---
# Disable button if date range is invalid
if st.button(f"Fetch Data for {symbol}", disabled=not valid_date_range):
    if valid_date_range:
        try:
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date', ascending=True)

                st.subheader(f"Price and Volume Chart")
                combo_chart = create_combo_chart(df)
                st.plotly_chart(combo_chart, use_container_width=True)

                st.subheader(f"Historical Data Table")
                df_display = df.copy()
                df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
                df_display = df_display[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                st.dataframe(df_display)

                with st.expander("Show Raw JSON Data"):
                    st.json(data)
            else:
                st.warning(f"No data returned for '{symbol}' in the selected date range. Please check the symbol and dates.")
                st.json(data)

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from Marketstack API: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Instructions ---
st.info(
    """
    **How to use this app:**

    1.  Enter a stock symbol in the sidebar.
    2.  Select a valid date range.
    3.  Click the "Fetch Data" button to retrieve and display the data.
    """
)