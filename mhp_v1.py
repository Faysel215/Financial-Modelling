import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- App Configuration & Title ---
st.set_page_config(layout="wide", page_title="Hawkes Process Trading Simulator")
st.title('📈 Hawkes Process Trading Strategy Simulator')
st.write("""
This interactive dashboard simulates a trading strategy based on a self-exciting Hawkes process. 
Adjust the parameters in the sidebar to see how market conditions and strategy rules affect performance.
""")

# --- Core Simulation Logic (Our functions from before) ---
# Using st.cache_data to speed up the app by not re-generating data on every widget change
@st.cache_data
def generate_hypothetical_data(days=90):
    dates = pd.to_datetime(pd.date_range(start='2024-10-01', periods=days, freq='D'))
    iv_baseline = np.zeros(days); iv_baseline[0] = 0.25
    for i in range(1, days): iv_baseline[i] = max(0.05, iv_baseline[i-1] + np.random.normal(0, 0.015))
    sentiment_scores = np.zeros(days)
    event_days = np.random.choice(np.arange(days), 15, replace=False)
    event_scores = np.random.uniform(-1.0, 1.0, 15); event_scores = np.where(np.abs(event_scores) < 0.4, np.sign(event_scores) * 0.8, event_scores)
    sentiment_scores[event_days] = event_scores
    return pd.DataFrame({'IV_Baseline': iv_baseline, 'Sentiment': sentiment_scores}, index=dates)

def calculate_intensity(t, history, baseline_func, sentiment_events, beta, delta, alpha):
    total_intensity = baseline_func(t)
    for day, score in sentiment_events.items():
        if day < t: total_intensity += np.exp(alpha * score) * beta * np.exp(-delta * (t - day))
    for event_time in history:
        if event_time < t: total_intensity += beta * np.exp(-delta * (t - event_time))
    return total_intensity

def simulate_hawkes_from_scratch(baseline_func, sentiment_events, params, end_time):
    alpha, beta, delta = params['alpha'], params['beta'], params['delta']
    T_high_res = np.linspace(0, end_time, end_time * 100)
    precomputed_intensity = [calculate_intensity(t, [], baseline_func, sentiment_events, beta, delta, alpha) for t in T_high_res]
    lambda_max = np.max(precomputed_intensity) * 1.5
    history, t = [], 0
    while t < end_time:
        dt = np.random.exponential(scale=1.0 / lambda_max)
        t += dt
        if t >= end_time: break
        lambda_true = calculate_intensity(t, history, baseline_func, sentiment_events, beta, delta, alpha)
        if np.random.uniform(0, 1) <= lambda_true / lambda_max: history.append(t)
    return history

def simulate_price_and_strategy(hawkes_events, baseline_func, sentiment_events, hawkes_params, trade_params, end_time):
    mu, sigma_base, sigma_shock_addon = trade_params['mu'], trade_params['sigma_base'], trade_params['sigma_shock_addon']
    trade_threshold, holding_period, cost = trade_params['trade_threshold'], trade_params['holding_period'], trade_params['cost']
    steps_per_day, n_steps, dt = 4, end_time * 4, 1/4
    time_grid = np.linspace(0, end_time, n_steps)
    price_path = np.zeros(n_steps); price_path[0] = 100
    trade_log, pnl, position, trade_exit_time = [], 0, 'FLAT', -1
    for i in range(1, n_steps):
        t = time_grid[i]
        is_in_shock_period = any(0 < (t - event_time) <= 2 for event_time in hawkes_events)
        current_sigma = sigma_base + sigma_shock_addon if is_in_shock_period else sigma_base
        random_shock = np.random.normal(0, np.sqrt(dt))
        price_path[i] = price_path[i-1] * np.exp((mu - 0.5 * current_sigma**2) * dt + current_sigma * random_shock)
        if position == 'IN_TRADE' and t >= trade_exit_time:
            exit_price = price_path[i]
            entry_price = trade_log[-1]['entry_price']
            trade_pnl = abs(exit_price - entry_price) - cost
            pnl += trade_pnl
            trade_log[-1].update({'exit_time': t, 'exit_price': exit_price, 'pnl': trade_pnl})
            position = 'FLAT'
        todays_events = [ev for ev in hawkes_events if time_grid[i-1] < ev <= t]
        if position == 'FLAT' and todays_events:
            event_time = todays_events[0]
            event_intensity = calculate_intensity(event_time, hawkes_events, baseline_func, sentiment_events, **hawkes_params)
            if event_intensity >= trade_threshold:
                position = 'IN_TRADE'
                entry_price = price_path[i]
                trade_exit_time = t + holding_period
                trade_log.append({'entry_time': t, 'entry_price': entry_price, 'intensity': event_intensity})
    return price_path, time_grid, trade_log, pnl

# --- Streamlit User Interface Sidebar ---
st.sidebar.header('⚙️ Model Parameters')

# Hawkes Parameters
st.sidebar.subheader('Hawkes Process Engine')
alpha = st.sidebar.slider('Alpha (Sentiment Impact)', 0.1, 3.0, 1.5, 0.1)
beta = st.sidebar.slider('Beta (Initial Excitement)', 0.1, 2.0, 0.8, 0.1)
delta = st.sidebar.slider('Delta (Decay Speed)', 0.1, 2.0, 0.5, 0.1)

# Asset Price Parameters
st.sidebar.subheader('Asset Price Simulation')
mu = st.sidebar.slider('Mu (Annualized Drift)', -0.2, 0.2, 0.05, 0.01)
sigma_base = st.sidebar.slider('Sigma (Baseline Volatility)', 0.1, 0.5, 0.2, 0.01)
sigma_shock_addon = st.sidebar.slider('Sigma Shock (Event Volatility)', 0.1, 1.0, 0.5, 0.05)

# Trading Strategy Parameters
st.sidebar.subheader('Trading Strategy')
trade_threshold = st.sidebar.number_input('Trade Trigger Intensity', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
holding_period = st.sidebar.number_input('Holding Period (Days)', min_value=1, max_value=20, value=5, step=1)
cost = st.sidebar.number_input('Transaction Cost ($)', min_value=0.0, max_value=5.0, value=0.5, step=0.05)

# --- Simulation Execution ---
if st.sidebar.button('🚀 Run Simulation'):
    
    with st.spinner('Running simulation... this may take a moment.'):
        # Pack params into dictionaries
        HAWKES_PARAMS = {'alpha': alpha, 'beta': beta, 'delta': delta}
        TRADE_PARAMS = {
            'mu': mu, 'sigma_base': sigma_base, 'sigma_shock_addon': sigma_shock_addon,
            'trade_threshold': trade_threshold, 'holding_period': holding_period, 'cost': cost
        }
        SIMULATION_END_TIME = 90
        
        # 1. Generate Data & Prep
        df = generate_hypothetical_data(days=SIMULATION_END_TIME)
        baseline_func = interp1d(np.arange(SIMULATION_END_TIME), df['IV_Baseline'], kind='linear', fill_value="extrapolate")
        sentiment_events = {(d - df.index[0]).days: s for d, s in df['Sentiment'].items() if s != 0}

        # 2. Run Hawkes Simulation
        hawkes_events = simulate_hawkes_from_scratch(baseline_func, sentiment_events, HAWKES_PARAMS, SIMULATION_END_TIME)

        # 3. Run Price & Strategy Simulation
        price_path, time_grid, trade_log, total_pnl = simulate_price_and_strategy(
            hawkes_events, baseline_func, sentiment_events, HAWKES_PARAMS, TRADE_PARAMS, SIMULATION_END_TIME
        )
        
        # --- Display Results ---
        st.header('📊 Simulation Results')

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total P&L ($)", f"{total_pnl:.2f}")
        col2.metric("Number of Trades", len(trade_log))
        col3.metric("Number of Hawkes Events", len(hawkes_events))

        # Create Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        
        # Top Panel: Hawkes
        ax1.set_title('Hawkes Process Events & Trading Threshold')
        ax1.set_ylabel('Intensity', color='b')
        intensity_curve = [calculate_intensity(t, hawkes_events, baseline_func, sentiment_events, **HAWKES_PARAMS) for t in time_grid]
        ax1.plot(time_grid, intensity_curve, 'b-', label='Actual Simulated Intensity λ(t)')
        ax1.axhline(TRADE_PARAMS['trade_threshold'], color='r', linestyle='--', lw=2, label=f"Trade Threshold")
        for event in hawkes_events: ax1.axvline(event, color='k', linestyle=':', alpha=0.5)
        ax1.plot([], [], color='k', linestyle=':', label='Hawkes Events')
        ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.4)

        # Bottom Panel: Price & Trades
        ax2.set_title('Simulated Asset Price & Trades')
        ax2.set_xlabel('Days'); ax2.set_ylabel('Asset Price ($)', color='g')
        ax2.plot(time_grid, price_path, color='g', label='Asset Price')
        entry_times = [trade['entry_time'] for trade in trade_log]
        entry_prices = [trade['entry_price'] for trade in trade_log]
        exit_times = [trade.get('exit_time') for trade in trade_log if trade.get('exit_time')]
        exit_prices = [trade.get('exit_price') for trade in trade_log if trade.get('exit_price')]
        ax2.scatter(entry_times, entry_prices, marker='^', s=150, color='blue', zorder=5, label='Enter Trade')
        ax2.scatter(exit_times, exit_prices, marker='v', s=150, color='purple', zorder=5, label='Exit Trade')
        ax2.legend(loc='upper left'); ax2.grid(True, alpha=0.4)
        
        # Display plot in Streamlit
        st.pyplot(fig)
        
        # Display trade log
        if trade_log:
            st.subheader("Trade Log")
            st.dataframe(pd.DataFrame(trade_log).set_index('entry_time'))

else:
    st.info('Adjust the parameters in the sidebar and click "Run Simulation" to start.')