import streamlit as st
import pandas as pd
import numpy as np
import itertools
from scipy.stats import norm

# --- Synthetic Data & Simplified Pricing for Demonstration ---

def generate_synthetic_prices(days=365):
    """Generates a simple synthetic stock price path."""
    np.random.seed(42) # for reproducibility
    initial_price = 150.0
    mu = 0.0002
    sigma = 0.015
    returns = np.random.normal(mu, sigma, days)
    prices = initial_price * (1 + returns).cumprod()
    return pd.Series(prices, name="price")

def simple_option_pricer(underlying_price, strike, days_to_expiry, volatility, option_type='put'):
    """A highly simplified Black-Scholes approximation."""
    if days_to_expiry == 0: return 0.0
    # Use a small time value to avoid division by zero in the model
    time_to_expiry_years = max(days_to_expiry / 365, 1e-6)
    
    d1 = (np.log(underlying_price / strike) + (0.5 * volatility ** 2) * time_to_expiry_years) / (volatility * np.sqrt(time_to_expiry_years))
    d2 = d1 - volatility * np.sqrt(time_to_expiry_years)
    
    if option_type == 'put':
        price = norm.cdf(-d2) * strike - norm.cdf(-d1) * underlying_price
        return price
    return 0

# --- Backtesting Engine ---

def run_single_backtest(params, historical_prices):
    """Simulates trading a single parameter set over historical data."""
    # This is a highly simplified backtest loop for a short put strategy.
    trade_log = []
    cash = 10000
    is_in_trade = False
    
    # Ensure there's enough data for the longest DTE
    if len(historical_prices) <= params['DTE']:
        return {'Total P/L': 0, 'Sharpe Ratio': 0, 'Max Drawdown': 0, 'Trades': 0}

    for i in range(len(historical_prices) - params['DTE']):
        if not is_in_trade:
            # Entry condition
            entry_price = historical_prices.iloc[i]
            strike_price = entry_price * (1 - params['Strike % OTM'])
            credit = simple_option_pricer(entry_price, strike_price, params['DTE'], 0.2, 'put')
            
            if credit > 0:
                is_in_trade = True
                trade_entry_date = i
                trade_exit_date = i + params['DTE']
    
        if is_in_trade and i == trade_exit_date:
            # Exit condition: Option expires
            exit_price = historical_prices.iloc[i]
            
            # Calculate P/L
            pnl_at_expiry = credit - max(0, strike_price - exit_price)
            cash += pnl_at_expiry
            trade_log.append(pnl_at_expiry)
            is_in_trade = False
            
    # --- Calculate Performance Metrics ---
    if not trade_log:
        return {'Total P/L': 0, 'Sharpe Ratio': 0, 'Max Drawdown': 0, 'Trades': 0}

    returns = pd.Series(trade_log)
    total_pnl = returns.sum()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    cumulative = returns.cumsum()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak)
    max_drawdown = drawdown.min()
    
    return {
        'Total P/L': total_pnl,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Trades': len(trade_log)
    }

# --- Main UI Function ---

def render():
    """Renders the entire UI for the Optimization tab."""
    st.header("⚙️ Strategy Parameter Optimization (Grid Search)")
    st.warning(
        """
        **Disclaimer:** This optimizer is for educational purposes. It uses **synthetic** stock data and a **simplified** option pricing model. 
        Results are not indicative of real-world performance.
        """, 
        icon="⚠️"
    )

    st.info("For this demonstration, we will find the optimal parameters for a simple **Short Put** strategy.")
    
    # --- UI Elements ---
    with st.sidebar:
        st.header("Grid Search Optimizer")
        
        st.subheader("Parameter Ranges to Test")
        param_ranges = {}
        param_ranges['DTE'] = st.multiselect("Days to Expiration (DTE)", [15, 30, 45, 60], default=[30, 45])
        param_ranges['Strike % OTM'] = st.multiselect("Strike Price (% Out of the Money)", [0.02, 0.03, 0.05, 0.07], default=[0.03, 0.05])
        
        st.subheader("Backtest Configuration")
        objective_function = st.selectbox(
            "Optimize For",
            ["Sharpe Ratio", "Total P/L", "Max Drawdown"]
        )
        
        run_button = st.button("🚀 Run Optimization", type="primary")

    # --- Main Panel Content ---
    if run_button:
        historical_prices = generate_synthetic_prices(days=730)
        
        keys, values = zip(*param_ranges.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        st.subheader("Optimization Results")
        
        results = []
        progress_bar = st.progress(0, text="Initializing...")
        
        for i, params in enumerate(param_combinations):
            status_text = f"Testing DTE={params['DTE']}, Strike % OTM={params['Strike % OTM']}..."
            progress_bar.progress((i + 1) / len(param_combinations), text=status_text)
            
            performance = run_single_backtest(params, historical_prices)
            performance.update(params)
            results.append(performance)
        
        progress_bar.empty()
        
        if not results:
            st.error("No trades were executed. Please check your parameters.")
        else:
            results_df = pd.DataFrame(results)
            
            is_ascending = True if objective_function == "Max Drawdown" else False
            sorted_df = results_df.sort_values(by=objective_function, ascending=is_ascending).reset_index(drop=True)
            
            st.dataframe(sorted_df.style.format({
                "Total P/L": "${:,.2f}",
                "Sharpe Ratio": "{:.2f}",
                "Max Drawdown": "${:,.2f}",
                "Strike % OTM": "{:.2%}"
            }))
            
            st.success(f"Optimization complete! Best results are sorted by **{objective_function}**.")
    else:
        st.info("Select parameter ranges in the sidebar and click 'Run Optimization'.")
