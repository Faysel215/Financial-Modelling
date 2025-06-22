import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- App Configuration ---
st.set_page_config(
    page_title="Marketstack Stock Data Viewer",
    page_icon="📈",
    layout="wide",
)

# --- CHARTING FUNCTION ---
def create_combo_chart(df: pd.DataFrame):
    """Creates a combination line and bar chart for close price and volume."""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    # Add Close Price Line
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['close'], name="Close Price", line=dict(color='royalblue')),
        secondary_y=False,
    )

    # Add Volume Histogram (as a Bar chart)
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name="Volume", marker=dict(color='lightgray')),
        secondary_y=True,
    )

    # Update layout and titles
    fig.update_layout(
        title_text=f"<b>Price and Volume for {df['symbol'].iloc[0]}</b>",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Close Price (USD)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True)
    
    return fig

# --- Sidebar for User Input ---
st.sidebar.title("Settings")
st.sidebar.write("Enter a stock symbol to display its data.")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()

# --- Main Application ---
st.title(f"📈 Marketstack Stock Data Viewer for {symbol}")

st.write(
    f"This app fetches and displays the latest end-of-day stock data for **{symbol}** from the Marketstack API."
)

# --- API Request ---
API_ACCESS_KEY = "509f8011f18787726d82ebeda521e579"
API_URL = "http://api.marketstack.com/v1/eod"

params = {
    "access_key": API_ACCESS_KEY,
    "symbols": symbol,
    "limit": 100 # Fetch more data points for a better chart
}

# --- Fetch and Display Data ---
if st.button(f"Fetch Data for {symbol}"):
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "data" in data and data["data"]:
            # Convert to DataFrame
            df = pd.DataFrame(data["data"])
            
            # --- Data Cleaning and Formatting ---
            df['date'] = pd.to_datetime(df['date'])
            # Sort data by date to ensure the chart plots correctly
            df = df.sort_values(by='date', ascending=True)

            st.subheader(f"Price and Volume Chart")
            # --- Display Chart ---
            combo_chart = create_combo_chart(df)
            st.plotly_chart(combo_chart, use_container_width=True)

            st.subheader(f"End-of-Day Data for {symbol}")
            # --- Display Data Table ---
            # Format date for display in table
            df_display = df.copy()
            df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
            df_display = df_display[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            st.dataframe(df_display)

            # --- Display Raw JSON ---
            with st.expander("Show Raw JSON Data"):
                st.json(data)
        else:
            st.warning(f"No data returned for the symbol '{symbol}'. Please check if the symbol is correct.")
            st.json(data)

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from Marketstack API: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- Instructions ---
st.info(
    """
    **How to use this app:**

    1.  Enter a stock symbol (e.g., GOOGL, MSFT, TSLA) in the sidebar on the left.
    2.  Click the "Fetch Data" button to retrieve the latest 100 days of data.
    3.  View the interactive chart and the detailed data table below.
    """
)