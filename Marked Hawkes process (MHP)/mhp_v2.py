import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Marked Hawkes Process Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title ---
st.title("Interactive Marked Hawkes Process Explorer")
st.markdown("Visually test how model parameters affect the intensity of volatility events using a Streamlit UI.")

# --- Functions ---

@st.cache_data
def synthesize_data(n_observations=365 * 5, seed=42):
    """Generates and caches a DataFrame of synthetic time series data."""
    np.random.seed(seed)
    time = np.arange(n_observations)
    
    # Implied Volatility (VIX-like data)
    iv = np.random.randn(n_observations) * 2 + 15
    spike_times = np.random.randint(50, n_observations, 15)
    iv[spike_times] += np.random.uniform(10, 20, len(spike_times))
    for i in range(len(spike_times)):
        if spike_times[i] + 1 < n_observations: iv[spike_times[i] + 1] += np.random.uniform(5, 10, 1)
        if spike_times[i] + 2 < n_observations: iv[spike_times[i] + 2] += np.random.uniform(1, 5, 1)

    # Financial News Sentiment
    sentiment = -0.05 * iv + np.random.randn(n_observations) * 0.2 + 0.1
    sentiment = np.clip(sentiment, -1, 1)

    # Polymarket Odds Change
    odds_change = np.random.randn(n_observations) * 0.02
    odds_change[spike_times] += np.random.uniform(-0.1, 0.1, len(spike_times))
    odds_change = np.clip(odds_change, -0.5, 0.5)

    return pd.DataFrame({
        'time': time,
        'implied_vol': iv,
        'sentiment': sentiment,
        'odds_change': odds_change
    })

def get_events_and_marks(df, threshold_std):
    """Identifies events and normalizes their marks based on a volatility threshold."""
    iv_mean = df['implied_vol'].mean()
    iv_std = df['implied_vol'].std()
    threshold_val = iv_mean + threshold_std * iv_std

    event_df = df[df['implied_vol'] > threshold_val].copy()
    if event_df.empty:
        return event_df, np.array([]), np.array([]), threshold_val

    event_times = event_df['time'].values
    marks = event_df[['sentiment', 'odds_change']].values

    # Normalize marks
    scaler = StandardScaler()
    marks_normalized = scaler.fit_transform(marks)
    
    event_df['sentiment_norm'] = marks_normalized[:, 0]
    event_df['odds_change_norm'] = marks_normalized[:, 1]
    
    return event_df, event_times, marks_normalized, threshold_val

def plot_results(df, event_df, threshold_val, mu, beta1, beta2, alpha):
    """Generates and displays the main plots."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Synthesized Data and Estimated Intensity', fontsize=16)

    # Plot 1: Implied Volatility and Events
    axs[0].plot(df['time'], df['implied_vol'], label='Implied Volatility', color='cornflowerblue', zorder=1)
    axs[0].axhline(threshold_val, color='red', linestyle='--', label=f'Event Threshold ({threshold_val:.2f})', zorder=2)
    if not event_df.empty:
        axs[0].plot(event_df['time'], event_df['implied_vol'], 'ro', label='Volatility Events', markersize=6, zorder=3)
    axs[0].set_ylabel('Implied Volatility')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Marks (Features)
    axs[1].plot(df['time'], df['sentiment'], label='Sentiment Score', color='seagreen')
    axs[1].plot(df['time'], df['odds_change'], label='Polymarket Odds Change', color='darkorange', alpha=0.8)
    axs[1].set_ylabel('Feature Value')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Estimated Intensity Function
    if not event_df.empty:
        event_times = event_df['time'].values
        marks_normalized = event_df[['sentiment_norm', 'odds_change_norm']].values
        plot_time = np.linspace(0, df['time'].max(), 500)
        intensity_values = []
        for t_now in plot_time:
            intensity = mu
            for i in range(len(event_times)):
                if event_times[i] < t_now:
                    t_i = event_times[i]
                    mark_i = marks_normalized[i]
                    kappa = beta1 * mark_i[0] + beta2 * mark_i[1]
                    intensity += kappa * alpha * np.exp(-alpha * (t_now - t_i))
            intensity_values.append(max(0, intensity)) # Intensity cannot be negative
        
        axs[2].plot(plot_time, intensity_values, label='Estimated Intensity λ(t)', color='purple')
        for t in event_times:
            axs[2].axvline(t, color='grey', linestyle=':', alpha=0.5)

    axs[2].set_xlabel('Time (Days)')
    axs[2].set_ylabel('Intensity λ(t)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


# --- Sidebar for Controls ---
st.sidebar.header("Model Controls")

if st.sidebar.button("Synthesize New Data"):
    # Clear the cache to generate new data
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.markdown("### Event Detection")
threshold_slider = st.sidebar.slider(
    'Volatility Threshold (Std Devs)', 0.5, 3.0, 1.5, 0.1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Hawkes Parameters")
mu_slider = st.sidebar.slider('μ (Baseline Intensity)', 0.0, 1.0, 0.1, 0.01)
beta1_slider = st.sidebar.slider('β₁ (Sentiment Impact)', -2.0, 2.0, 0.5, 0.05)
beta2_slider = st.sidebar.slider('β₂ (Odds Change Impact)', -2.0, 2.0, 0.5, 0.05)
alpha_slider = st.sidebar.slider('α (Decay Speed)', 0.1, 5.0, 1.0, 0.1)


# --- Main Application Logic ---

# 1. Load data
df_data = synthesize_data()

# 2. Identify events based on slider
event_df, event_times, marks_normalized, threshold_val = get_events_and_marks(df_data, threshold_slider)

# 3. Display data table and plots
st.header("Visualizations & Data")
if event_df.empty:
    st.warning("No events detected with the current threshold. Please lower the threshold.")
else:
    st.success(f"{len(event_df)} events detected above threshold {threshold_val:.2f}.")
    
    # Display the table of event data
    st.subheader("Detected Event Data")
    st.markdown("This table shows the data points that were classified as events.")
    st.dataframe(event_df[['time', 'implied_vol', 'sentiment', 'odds_change']].reset_index(drop=True))

# Generate and display the plots
fig = plot_results(df_data, event_df, threshold_val, mu_slider, beta1_slider, beta2_slider, alpha_slider)
st.pyplot(fig)


# --- Interpretation Section ---
st.header("Interpretation")
st.markdown(f"""
Based on the current parameters:
- **`β₁` (Sentiment Impact) = {beta1_slider:.2f}**: A positive value means positive sentiment *increases* the intensity of future volatility events (and vice-versa for negative sentiment).
- **`β₂` (Odds Impact) = {beta2_slider:.2f}**: A positive value means a large change in odds *increases* future volatility intensity.
- **`α` (Decay Speed) = {alpha_slider:.2f}**: This implies a characteristic memory time of **{1/alpha_slider:.2f} days**. This is the average time it takes for an event's influence to decay significantly.

Comparing the absolute values of `β₁` and `β₂` suggests which feature has a more dominant impact on exciting future events.
""")
if abs(beta1_slider) > abs(beta2_slider):
    st.info("**Conclusion:** Financial news sentiment currently has a stronger impact than Polymarket odds change.")
else:
    st.info("**Conclusion:** Polymarket odds change currently has a stronger impact than financial news sentiment.")
