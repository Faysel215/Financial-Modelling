import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from matplotlib.colors import Normalize

# --- Page Configuration ---
st.set_page_config(
    page_title="Non-Parametric Hawkes Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title ---
st.title("Non-Parametric Marked Hawkes Process Explorer")
st.markdown("Using a Neural Network to learn the mark influence function `κ(m)`.")

# --- Functions ---

@st.cache_data
def synthesize_data(n_observations=365 * 5, seed=42):
    """Generates and caches a DataFrame of synthetic time series data."""
    np.random.seed(seed)
    time = np.arange(n_observations)
    iv = np.random.randn(n_observations) * 2 + 15
    spike_times = np.random.randint(50, n_observations, 25)
    iv[spike_times] += np.random.uniform(10, 20, len(spike_times))
    for i in range(len(spike_times)):
        if spike_times[i] + 1 < n_observations: iv[spike_times[i] + 1] += np.random.uniform(5, 10, 1)
        if spike_times[i] + 2 < n_observations: iv[spike_times[i] + 2] += np.random.uniform(1, 5, 1)
    sentiment = -0.05 * iv + np.random.randn(n_observations) * 0.2 + 0.1
    sentiment = np.clip(sentiment, -1, 1)
    odds_change = np.random.randn(n_observations) * 0.02
    odds_change[spike_times] += np.random.uniform(-0.1, 0.1, len(spike_times))
    odds_change = np.clip(odds_change, -0.5, 0.5)
    return pd.DataFrame({
        'time': time, 'implied_vol': iv, 'sentiment': sentiment, 'odds_change': odds_change
    })

def get_events_and_marks(df, threshold_std):
    """Identifies events and normalizes their marks based on a volatility threshold."""
    iv_mean, iv_std = df['implied_vol'].mean(), df['implied_vol'].std()
    threshold_val = iv_mean + threshold_std * iv_std
    event_df = df[df['implied_vol'] > threshold_val].copy()
    if event_df.empty: return event_df, np.array([]), np.array([]), threshold_val
    event_times = event_df['time'].values
    marks = event_df[['sentiment', 'odds_change']].values
    scaler = StandardScaler()
    marks_normalized = scaler.fit_transform(marks)
    event_df['sentiment_norm'] = marks_normalized[:, 0]
    event_df['odds_change_norm'] = marks_normalized[:, 1]
    return event_df, event_times, marks_normalized, threshold_val

@st.cache_data
def train_nn_influence_model(_event_df, n_neurons, learning_window):
    """Trains an MLP to predict the influence of a mark."""
    if _event_df.empty or len(_event_df) < 5: return None
    event_times = _event_df['time'].values
    marks = _event_df[['sentiment_norm', 'odds_change_norm']].values
    
    # Create a target variable: count of subsequent events in the learning window
    y = np.zeros(len(event_times))
    for i in range(len(event_times)):
        t_i = event_times[i]
        # Count events in [t_i, t_i + window]
        y[i] = np.sum((event_times > t_i) & (event_times <= t_i + learning_window))

    # Train the MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(n_neurons,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        alpha=0.001,
        learning_rate_init=0.01
    )
    mlp.fit(marks, y)
    return mlp

def plot_results(df, event_df, threshold_val, mu, alpha, mlp_model):
    """Generates and displays the main plots including the learned influence surface."""
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(4, 2)

    # Plot 1: Implied Volatility
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['time'], df['implied_vol'], label='Implied Volatility', color='cornflowerblue', zorder=1)
    ax1.axhline(threshold_val, color='red', linestyle='--', label=f'Event Threshold ({threshold_val:.2f})', zorder=2)
    if not event_df.empty: ax1.plot(event_df['time'], event_df['implied_vol'], 'ro', label='Volatility Events', markersize=6, zorder=3)
    ax1.set_ylabel('Implied Volatility'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_title("Time Series Data & Volatility Events", fontsize=14)

    # Plot 2: Marks
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.plot(df['time'], df['sentiment'], label='Sentiment Score', color='seagreen')
    ax2.plot(df['time'], df['odds_change'], label='Polymarket Odds Change', color='darkorange', alpha=0.8)
    ax2.set_ylabel('Feature Value'); ax2.legend(); ax2.grid(True, alpha=0.3)

    # Plot 3: Estimated Intensity
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
    if not event_df.empty and mlp_model:
        event_times = event_df['time'].values
        marks_normalized = event_df[['sentiment_norm', 'odds_change_norm']].values
        plot_time = np.linspace(0, df['time'].max(), 500)
        intensity_values = []
        for t_now in plot_time:
            intensity = mu
            for i in range(len(event_times)):
                if event_times[i] < t_now:
                    kappa = mlp_model.predict([marks_normalized[i]])[0]
                    intensity += max(0, kappa) * alpha * np.exp(-alpha * (t_now - event_times[i]))
            intensity_values.append(max(0, intensity))
        ax3.plot(plot_time, intensity_values, label='Estimated Intensity λ(t)', color='purple')
        for t in event_times: ax3.axvline(t, color='grey', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (Days)'); ax3.set_ylabel('Intensity λ(t)'); ax3.legend(); ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learned Influence Surface
    ax4 = fig.add_subplot(gs[3, :])
    if mlp_model:
        x_min, x_max = event_df['sentiment_norm'].min() - 0.5, event_df['sentiment_norm'].max() + 0.5
        y_min, y_max = event_df['odds_change_norm'].min() - 0.5, event_df['odds_change_norm'].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        zz = mlp_model.predict(grid_points).reshape(xx.shape)
        
        contour = ax4.contourf(xx, yy, zz, levels=15, cmap='viridis')
        fig.colorbar(contour, ax=ax4, label='Influence κ(m)')
        ax4.scatter(event_df['sentiment_norm'], event_df['odds_change_norm'], c='red', s=20, edgecolor='white', label='Event Marks')
        ax4.set_xlabel('Normalized Sentiment Score'); ax4.set_ylabel('Normalized Odds Change'); ax4.legend()
        ax4.set_title("Learned Influence Function κ(m)", fontsize=14)

    plt.tight_layout()
    return fig

# --- Sidebar Controls ---
st.sidebar.header("Model Controls")
if st.sidebar.button("Synthesize New Data"): st.cache_data.clear()
st.sidebar.markdown("---")
st.sidebar.markdown("### Event Detection")
threshold_slider = st.sidebar.slider('Volatility Threshold (Std Devs)', 0.5, 3.0, 1.5, 0.1)
st.sidebar.markdown("---")
st.sidebar.markdown("### Hawkes Parameters")
mu_slider = st.sidebar.slider('μ (Baseline Intensity)', 0.0, 1.0, 0.1, 0.01)
alpha_slider = st.sidebar.slider('α (Decay Speed)', 0.1, 5.0, 1.0, 0.1)
st.sidebar.markdown("---")
st.sidebar.markdown("### Neural Network")
n_neurons_slider = st.sidebar.slider('Number of Hidden Neurons', 2, 50, 10, 1)
learning_window = 1.0 / alpha_slider
st.sidebar.info(f"Learning Window (1/α): {learning_window:.2f} days")

# --- Main Application Logic ---
df_data = synthesize_data()
event_df, event_times, marks_normalized, threshold_val = get_events_and_marks(df_data, threshold_slider)

st.header("Visualizations & Data")
if event_df.empty:
    st.warning("No events detected with the current threshold. Please lower the threshold.")
else:
    st.success(f"{len(event_df)} events detected above threshold {threshold_val:.2f}.")
    st.subheader("Detected Event Data")
    st.dataframe(event_df[['time', 'implied_vol', 'sentiment', 'odds_change']].reset_index(drop=True))

    # Train NN Model
    mlp_model = train_nn_influence_model(event_df, n_neurons_slider, learning_window)

    # Display plots
    if mlp_model:
        fig = plot_results(df_data, event_df, threshold_val, mu_slider, alpha_slider, mlp_model)
        st.pyplot(fig)
    else:
        st.error("Could not train the neural network model. Not enough event data.")

# --- Interpretation Section ---
st.header("Interpretation")
st.markdown("""
In this non-parametric model, we no longer estimate simple linear coefficients. Instead, the neural network learns a potentially complex surface representing the influence function `κ(m)`.
- **The Contour Plot is Key:** Look at the "Learned Influence Function" plot above. The bright yellow areas represent combinations of sentiment and odds change that are highly excitatory (i.e., they strongly trigger future volatility events). Dark purple areas have a low or even dampening effect.
- **Analyze the Shape:**
    - Is there a **gradient**? Does the influence consistently increase along the sentiment axis or the odds axis?
    - Are there **"hotspots"**? The model might learn that only events with, for example, *very negative sentiment* and *large odds changes* are truly destabilizing. This is a non-linear effect a simple linear model would miss.
    - The shape of this surface provides a rich, data-driven picture of what *kind* of shock truly drives market instability.
""")
