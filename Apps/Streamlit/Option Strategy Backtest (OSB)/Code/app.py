import streamlit as st
import pandas as pd
import psycopg2
import numpy as np
import plotly.express as px

# Import the code for the other tabs
import optimization_tab
import neural_network_tab

# --- Configuration & Data Loading ---
@st.cache_data(ttl=3600)
def get_strategies():
    """Connects to DB and fetches the list of strategy definitions."""
    try:
        # Assumes you have secrets configured in .streamlit/secrets.toml
        conn = psycopg2.connect(**st.secrets["database"])
        df = pd.read_sql_query("SELECT * FROM strategies ORDER BY name;", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()

# --- Payoff Calculation Engine ---
def calculate_payoff(strategy_name, params, price_range):
    """Calculates P/L for a given strategy and price range."""
    payoffs = []
    
    for price in price_range:
        pnl = 0
        # This logic should match your previous implementation
        if strategy_name == "Long Call":
            pnl = max(0, price - params["Strike Price"]) - params["Premium Paid"]
        elif strategy_name == "Long Put":
            pnl = max(0, params["Strike Price"] - price) - params["Premium Paid"]
        elif strategy_name == "Covered Call":
            pnl = (min(params["Short Call Strike"], price) - params["Stock Purchase Price"]) + params["Premium Received"]
        elif strategy_name == "Bull Call Spread":
            pnl = max(0, price - params["Long Call Strike"]) - max(0, price - params["Short Call Strike"]) - params["Net Debit"]
        elif strategy_name == "Bear Put Spread":
            pnl = max(0, params["Long Put Strike"] - price) - max(0, params["Short Put Strike"] - price) - params["Net Debit"]
        elif strategy_name == "Bull Put Spread":
             pnl = -(max(0, params["Short Put Strike"] - price) - max(0, params["Long Put Strike"] - price)) + params["Net Credit"]
        elif strategy_name == "Bear Call Spread":
            pnl = -(max(0, price - params["Short Call Strike"]) - max(0, price - params["Long Call Strike"])) + params["Net Credit"]
        elif strategy_name == "Long Straddle":
            pnl = (max(0, price - params["Strike Price"]) + max(0, params["Strike Price"] - price)) - params["Total Premium Paid"]
        elif strategy_name == "Long Strangle":
            pnl = (max(0, price - params["Call Strike"]) + max(0, params["Put Strike"] - price)) - params["Total Premium Paid"]
        elif strategy_name == "Short Strangle":
            pnl = -(max(0, price - params["Call Strike"]) + max(0, params["Put Strike"] - price)) + params["Total Premium Received"]
        elif strategy_name == "Iron Condor":
            bull_put = -(max(0, params["Short Put Strike"] - price) - max(0, params["Long Put Strike"] - price))
            bear_call = -(max(0, price - params["Short Call Strike"]) - max(0, price - params["Long Call Strike"]))
            pnl = bull_put + bear_call + params["Net Credit"]
        elif strategy_name == "Iron Butterfly":
            pnl = -(max(0, price - params["Short Strike"]) + max(0, params["Short Strike"] - price)) + \
                  (max(0, price - params["Protective Call Strike"]) + max(0, params["Protective Put Strike"] - price)) + \
                  params["Net Credit"]
        elif strategy_name == "Collar":
            stock_pnl = price - params["Stock Purchase Price"]
            put_pnl = max(0, params["Long Put Strike"] - price)
            call_pnl = -max(0, price - params["Short Call Strike"])
            pnl = stock_pnl + put_pnl + call_pnl + params["Net Cost/Credit"]
        
        payoffs.append({"price": price, "pnl": pnl})
        
    return pd.DataFrame(payoffs)


# --- Main Application with Tabs ---

st.set_page_config(page_title="Options Strategy Toolkit", layout="wide")
st.title("Options Strategy Toolkit")

tab1, tab2, tab3 = st.tabs(["Payoff Analyzer", "Strategy Optimizer", "Neural Network Predictor"])

# --- Tab 1: Payoff Analyzer ---
with tab1:
    st.header("Static Payoff Analysis")
    strategies_df = get_strategies()

    if not strategies_df.empty:
        # Create a sidebar specific to this tab's controls
        with st.sidebar:
            st.header("Analyzer Setup")
            selected_strategy_name = st.selectbox("Choose a Strategy", strategies_df['name'], key="analyzer_select")
            strategy_details = strategies_df[strategies_df['name'] == selected_strategy_name].iloc[0]
            st.info(f"**Outlook:** {strategy_details['outlook']} | **Volatility:** {strategy_details['volatility_view']}")
            st.markdown(strategy_details['description'])
            st.markdown("---")
            
            params = {}
            st.subheader("Parameters")
            for param in strategy_details['parameters']:
                param_key = "analyzer_" + param.lower().replace(" ", "_")
                params[param] = st.number_input(param, value=100.0, step=0.1, key=param_key)
            
            st.markdown("---")
            st.subheader("Analysis Range")
            min_price = st.number_input("Min Price", value=50.0, key="analyzer_min_price")
            max_price = st.number_input("Max Price", value=150.0, key="analyzer_max_price")
        
        # Main panel content for Tab 1
        st.subheader(f"Payoff Profile: {selected_strategy_name}")
        
        price_range = np.linspace(min_price, max_price, 200)
        payoff_df = calculate_payoff(selected_strategy_name, params, price_range)
        
        max_profit = payoff_df['pnl'].max()
        max_loss = payoff_df['pnl'].min()
        is_unlimited_profit = payoff_df.iloc[-1]['pnl'] > payoff_df.iloc[-2]['pnl'] or payoff_df.iloc[0]['pnl'] > payoff_df.iloc[1]['pnl']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Maximum Profit", "Unlimited" if is_unlimited_profit else f"${max_profit:,.2f}")
        with col2:
            st.metric("Maximum Loss", f"${max_loss:,.2f}")
        
        fig = px.line(payoff_df, x="price", y="pnl", title=f"{selected_strategy_name} P/L at Expiration")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Could not load strategies from the database.")


# --- Tab 2: Strategy Optimizer ---
with tab2:
    optimization_tab.render()

# --- Tab 3: Neural Network Predictor ---
with tab3:
    neural_network_tab.render()
