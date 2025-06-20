import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
import json

# --- App Configuration & Title ---
st.set_page_config(layout="wide", page_title="Agentic Trading Simulator")
st.title('🤖 Agentic AI Trading Simulator')
st.write("""
This application simulates a live trading environment where autonomous AI agents collaborate to make decisions. 
Adjust the initial parameters for the agents and the market, then run the simulation to observe their performance.
""")

# --- Agent and System Classes (Translating the ERD to Code) ---

class LLMSentimentAgent:
    """
    Simulates an LLM agent that analyzes news and provides structured sentiment.
    In a real system, this would call a model like Gemini.
    """
    def __init__(self):
        st.sidebar.write("✅ LLM Sentiment Agent Initialized")
        self.event_types = ["Earnings Beat", "Product Launch", "Regulatory Scrutiny", "Market Downturn"]
        self.durations = ["short-term", "medium-term", "long-term"]

    def analyze(self, news_headline: str, sentiment_score: float) -> dict:
        """Analyzes a news item and returns a structured analysis."""
        # Simulate LLM analysis without making a real API call
        return {
            "sentiment_score": round(sentiment_score, 2),
            "event_type": random.choice(self.event_types),
            "expected_impact_duration": random.choice(self.durations),
            "confidence": round(random.uniform(0.75, 0.98), 2)
        }

class HawkesForecastingEngine:
    """Calculates and forecasts event intensity based on a Hawkes model."""
    def __init__(self, params):
        st.sidebar.write("✅ Hawkes Forecasting Engine Initialized")
        self.params = params
        self.history = []

    def update_with_history(self, history):
        self.history = history

    def predict_intensity(self, t, baseline_func, sentiment_events):
        """Calculates the intensity at a given time t."""
        alpha, beta, delta = self.params['alpha'], self.params['beta'], self.params['delta']
        intensity = baseline_func(t)
        # Add sentiment influence
        for day, score in sentiment_events.items():
            if day < t: intensity += np.exp(alpha * score) * beta * np.exp(-delta * (t - day))
        # Add self-exciting influence
        for event_time in self.history:
            if event_time < t: intensity += beta * np.exp(-delta * (t - event_time))
        return intensity

class DynamicStrategyAgent:
    """
    An agent that makes trading decisions based on market state.
    This is a stand-in for a more complex Reinforcement Learning agent.
    """
    def __init__(self, trade_threshold, holding_period):
        st.sidebar.write("✅ Dynamic Strategy Agent Initialized")
        self.trade_threshold = trade_threshold
        self.holding_period = holding_period
        self.position = 'FLAT'
        self.trade_exit_time = -1
        self.trade_log = []
        self.pnl = 0

    def get_state(self, t, intensity, price):
        """Represents the agent's view of the market."""
        return {"time": t, "intensity": intensity, "price": price, "position": self.position}

    def decide_action(self, state, event_this_step, cost):
        """The core logic of the agent's policy."""
        # 1. Check if we need to exit a trade
        if self.position == 'IN_TRADE' and state['time'] >= self.trade_exit_time:
            exit_price = state['price']
            entry_price = self.trade_log[-1]['entry_price']
            trade_pnl = abs(exit_price - entry_price) - cost
            self.pnl += trade_pnl
            self.trade_log[-1].update({'exit_time': state['time'], 'exit_price': exit_price, 'pnl': trade_pnl})
            self.position = 'FLAT'
            return "EXIT", trade_pnl

        # 2. Check if we should enter a trade
        if self.position == 'FLAT' and event_this_step:
            if state['intensity'] >= self.trade_threshold:
                self.position = 'IN_TRADE'
                self.trade_exit_time = state['time'] + self.holding_period
                self.trade_log.append({'entry_time': state['time'], 'entry_price': state['price'], 'intensity': state['intensity']})
                return "ENTER", None
        
        return "HOLD", None

class TradingSystem:
    """Orchestrates the entire live trading simulation."""
    def __init__(self, agents, market_params, end_time):
        self.sentiment_agent = agents['sentiment']
        self.hawkes_engine = agents['hawkes']
        self.strategy_agent = agents['strategy']
        self.market_params = market_params
        self.end_time = end_time

    def run_simulation(self, data):
        """The main event loop that simulates the live environment."""
        # Prep for simulation
        steps_per_day = 4
        n_steps = self.end_time * steps_per_day
        dt = 1 / steps_per_day
        time_grid = np.linspace(0, self.end_time, n_steps)
        
        # Create simulated data feed
        baseline_func = interp1d(np.arange(self.end_time), data['IV_Baseline'], kind='linear', fill_value="extrapolate")
        sentiment_events = { (d - data.index[0]).days: s for d, s in data['Sentiment'].items() if s != 0 }
        
        # Simulate Hawkes events ahead of time to represent the "real world"
        real_world_events = simulate_hawkes_from_scratch(baseline_func, sentiment_events, self.hawkes_engine.params, self.end_time)
        self.hawkes_engine.update_with_history(real_world_events)

        price_path = np.zeros(n_steps); price_path[0] = 100
        
        st.write("---")
        st.write("### 🟢 Live Trading Log")
        log_placeholder = st.empty()
        log_text = ""

        for i in range(1, n_steps):
            t = time_grid[i]
            
            # 1. DATA FEED: Check for news and events in this timestep
            event_this_step = any(time_grid[i-1] < ev <= t for ev in real_world_events)
            news_today = sentiment_events.get(int(t), 0)

            # 2. SENTIMENT AGENT: Analyze news if it exists
            if news_today != 0:
                sentiment_analysis = self.sentiment_agent.analyze("A news headline...", news_today)
                # In a real system, this analysis would affect the Hawkes forecast
            
            # 3. HAWKES ENGINE: Forecast intensity
            predicted_intensity = self.hawkes_engine.predict_intensity(t, baseline_func, sentiment_events)
            
            # 4. PRICE PROCESS: Update asset price
            is_in_shock_period = any(0 < (t - ev) <= 2 for ev in real_world_events)
            current_sigma = self.market_params['sigma_base'] + self.market_params['sigma_shock_addon'] if is_in_shock_period else self.market_params['sigma_base']
            random_shock = np.random.normal(0, np.sqrt(dt))
            price_path[i] = price_path[i-1] * np.exp((self.market_params['mu'] - 0.5 * current_sigma**2) * dt + current_sigma * random_shock)
            
            # 5. STRATEGY AGENT: Observe and decide
            state = self.strategy_agent.get_state(t, predicted_intensity, price_path[i])
            action, pnl_update = self.strategy_agent.decide_action(state, event_this_step, self.market_params['cost'])
            
            if action == "ENTER":
                log_text += f"`Day {t:.1f}`: **ENTERED TRADE** at ${price_path[i]:.2f} (Intensity: {predicted_intensity:.2f})\n"
            elif action == "EXIT":
                log_text += f"`Day {t:.1f}`: **EXITED TRADE** at ${price_path[i]:.2f} -> P&L: ${pnl_update:.2f}\n"
            log_placeholder.markdown(log_text)

        return price_path, time_grid, self.strategy_agent.trade_log, self.strategy_agent.pnl, real_world_events


# Helper for generating data and Hawkes events (reused from previous steps)
@st.cache_data
def generate_base_data(days=90):
    dates = pd.to_datetime(pd.date_range(start='2024-10-01', periods=days, freq='D'))
    iv_baseline = np.zeros(days); iv_baseline[0] = 0.25
    for i in range(1, days): iv_baseline[i] = max(0.05, iv_baseline[i-1] + np.random.normal(0, 0.015))
    sentiment_scores = np.zeros(days)
    event_days = np.random.choice(np.arange(days), 15, replace=False)
    event_scores = np.random.uniform(-1.0, 1.0, 15); event_scores = np.where(np.abs(event_scores) < 0.4, np.sign(event_scores) * 0.8, event_scores)
    sentiment_scores[event_days] = event_scores
    return pd.DataFrame({'IV_Baseline': iv_baseline, 'Sentiment': sentiment_scores}, index=dates)

def simulate_hawkes_from_scratch(baseline_func, sentiment_events, params, end_time):
    alpha, beta, delta = params['alpha'], params['beta'], params['delta']
    T_high_res = np.linspace(0, end_time, end_time * 100)
    precomputed_intensity = [HawkesForecastingEngine(params).predict_intensity(t, baseline_func, sentiment_events) for t in T_high_res]
    lambda_max = np.max(precomputed_intensity) * 1.5
    history, t = [], 0
    while t < end_time:
        dt = np.random.exponential(scale=1.0 / lambda_max)
        t += dt
        if t >= end_time: break
        lambda_true = HawkesForecastingEngine(params).predict_intensity(t, baseline_func, sentiment_events)
        if np.random.uniform(0, 1) <= lambda_true / lambda_max: history.append(t)
    return history


# --- Streamlit UI Sidebar ---
st.sidebar.header('⚙️ Agent & Market Parameters')

# Hawkes Parameters
with st.sidebar.expander("Hawkes Forecasting Engine Params", expanded=True):
    alpha = st.slider('Alpha (Sentiment Impact)', 0.1, 3.0, 1.5, 0.1, key='h1')
    beta = st.slider('Beta (Initial Excitement)', 0.1, 2.0, 0.8, 0.1, key='h2')
    delta = st.slider('Delta (Decay Speed)', 0.1, 2.0, 0.5, 0.1, key='h3')

# Asset Price Parameters
with st.sidebar.expander("Market Simulation Params", expanded=True):
    mu = st.slider('Mu (Annualized Drift)', -0.2, 0.2, 0.05, 0.01, key='m1')
    sigma_base = st.slider('Sigma (Baseline Volatility)', 0.1, 0.5, 0.2, 0.01, key='m2')
    sigma_shock_addon = st.slider('Sigma Shock (Event Volatility)', 0.1, 1.0, 0.5, 0.05, key='m3')
    cost = st.number_input('Transaction Cost ($)', 0.0, 5.0, 0.5, 0.05, key='m4')

# Trading Strategy Parameters
with st.sidebar.expander("Dynamic Strategy Agent Params", expanded=True):
    trade_threshold = st.number_input('Trade Trigger Intensity', 0.5, 5.0, 1.5, 0.1, key='s1')
    holding_period = st.number_input('Holding Period (Days)', 1, 20, 5, 1, key='s2')

# --- Simulation Execution ---
if st.sidebar.button('🚀 Run Agentic Simulation'):
    with st.spinner('Agents are analyzing and trading...'):
        # 1. Initialize Agents & System
        agents = {
            'sentiment': LLMSentimentAgent(),
            'hawkes': HawkesForecastingEngine({'alpha': alpha, 'beta': beta, 'delta': delta}),
            'strategy': DynamicStrategyAgent(trade_threshold, holding_period)
        }
        market_params = {'mu': mu, 'sigma_base': sigma_base, 'sigma_shock_addon': sigma_shock_addon, 'cost': cost}
        trading_system = TradingSystem(agents, market_params, end_time=90)

        # 2. Run the full simulation
        base_data = generate_base_data(days=90)
        price_path, time_grid, trade_log, total_pnl, real_world_events = trading_system.run_simulation(base_data)

        # 3. Display Results
        st.header('📊 Simulation Results')
        col1, col2, col3 = st.columns(3)
        col1.metric("Total P&L ($)", f"{total_pnl:.2f}")
        col2.metric("Number of Trades", len(trade_log))
        col3.metric("Number of Hawkes Events", len(real_world_events))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        ax1.set_title('Hawkes Process Events & Trading Threshold'); ax1.set_ylabel('Intensity', color='b')
        intensity_curve = [agents['hawkes'].predict_intensity(t, interp1d(np.arange(90), base_data['IV_Baseline'], kind='linear', fill_value="extrapolate"), {(d - base_data.index[0]).days: s for d, s in base_data['Sentiment'].items() if s != 0}) for t in time_grid]
        ax1.plot(time_grid, intensity_curve, 'b-'); ax1.axhline(trade_threshold, color='r', linestyle='--', lw=2, label="Trade Threshold")
        for event in real_world_events: ax1.axvline(event, color='k', linestyle=':', alpha=0.5)
        ax1.legend(); ax1.grid(True, alpha=0.4)

        ax2.set_title('Simulated Asset Price & Trades'); ax2.set_xlabel('Days'); ax2.set_ylabel('Asset Price ($)', color='g')
        ax2.plot(time_grid, price_path, color='g', label='Asset Price')
        ax2.scatter([t['entry_time'] for t in trade_log], [t['entry_price'] for t in trade_log], marker='^', s=150, color='blue', zorder=5, label='Enter Trade')
        ax2.scatter([t.get('exit_time') for t in trade_log if t.get('exit_time')], [t.get('exit_price') for t in trade_log if t.get('exit_price')], marker='v', s=150, color='purple', zorder=5, label='Exit Trade')
        ax2.legend(); ax2.grid(True, alpha=0.4)
        
        st.pyplot(fig)
        
        if trade_log:
            st.subheader("Final Trade Log")
            st.dataframe(pd.DataFrame(trade_log).set_index('entry_time'))
else:
    st.info('Adjust agent and market parameters in the sidebar and click "Run Agentic Simulation" to begin.')


