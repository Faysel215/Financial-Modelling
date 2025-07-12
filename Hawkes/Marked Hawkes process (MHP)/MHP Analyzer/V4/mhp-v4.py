import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Hawkes Process Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title ---
st.title("Advanced Marked Hawkes Process Analyzer")
st.markdown("Compare parametric and non-parametric estimators and explain model predictions with SHAP.")

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

# --- Parametric Model Functions ---
def neg_log_likelihood_parametric(params, event_times, marks):
    mu, beta1, beta2, alpha = params
    if mu <= 0 or alpha <= 0: return np.inf
    log_likelihood = 0
    T = event_times[-1]
    for i in range(len(event_times)):
        t_i = event_times[i]
        intensity_at_ti = mu
        for j in range(i):
            t_j, mark_j = event_times[j], marks[j]
            kappa = beta1 * mark_j[0] + beta2 * mark_j[1]
            intensity_at_ti += max(0, kappa) * alpha * np.exp(-alpha * (t_i - t_j))
        if intensity_at_ti <= 0: return np.inf
        log_likelihood += np.log(intensity_at_ti)
    integral_term = mu * T
    for i in range(len(event_times)):
        t_i, mark_i = event_times[i], marks[i]
        kappa = beta1 * mark_i[0] + beta2 * mark_i[1]
        integral_term += max(0, kappa) * (1 - np.exp(-alpha * (T - t_i)))
    return -(log_likelihood - integral_term)

# --- Non-Parametric (NN) Model Functions ---
@st.cache_data
def train_nn_model(_event_df, n_neurons):
    if len(_event_df) < 5: return None, 0
    event_times = _event_df['time'].values
    marks = _event_df[['sentiment_norm', 'odds_change_norm']].values
    y = np.zeros(len(event_times))
    for i in range(len(event_times)):
        y[i] = np.sum((event_times > event_times[i]) & (event_times <= event_times[i] + 10))
    mlp = MLPRegressor(hidden_layer_sizes=(n_neurons,), max_iter=500, random_state=42, alpha=0.01)
    mlp.fit(marks, y)
    n_params = len(mlp.coefs_[0].flatten()) + len(mlp.coefs_[1].flatten()) + len(mlp.intercepts_[0]) + len(mlp.intercepts_[1])
    return mlp, n_params

def neg_log_likelihood_nn(params, event_times, marks, nn_model):
    mu, alpha = params
    if mu <= 0 or alpha <= 0: return np.inf
    log_likelihood = 0
    T = event_times[-1]
    for i in range(len(event_times)):
        t_i = event_times[i]
        intensity_at_ti = mu
        for j in range(i):
            t_j, mark_j = event_times[j], marks[j]
            kappa = nn_model.predict([mark_j])[0]
            intensity_at_ti += max(0, kappa) * alpha * np.exp(-alpha * (t_i - t_j))
        if intensity_at_ti <= 0: return np.inf
        log_likelihood += np.log(intensity_at_ti)
    integral_term = mu * T
    for i in range(len(event_times)):
        t_i, mark_i = event_times[i], marks[i]
        kappa = nn_model.predict([mark_i])[0]
        integral_term += max(0, kappa) * (1 - np.exp(-alpha * (T - t_i)))
    return -(log_likelihood - integral_term)

# --- Plotting and UI ---
def plot_time_series(df, event_df, threshold_val):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].set_title("Time Series Data & Volatility Events", fontsize=14)
    axs[0].plot(df['time'], df['implied_vol'], label='Implied Volatility', color='cornflowerblue')
    axs[0].axhline(threshold_val, color='red', linestyle='--', label=f'Threshold ({threshold_val:.2f})')
    if not event_df.empty: axs[0].plot(event_df['time'], event_df['implied_vol'], 'ro', label='Events')
    axs[0].set_ylabel('Implied Volatility'); axs[0].legend(); axs[0].grid(True, alpha=0.3)
    axs[1].plot(df['time'], df['sentiment'], label='Sentiment', color='seagreen')
    axs[1].plot(df['time'], df['odds_change'], label='Odds Change', color='darkorange', alpha=0.8)
    axs[1].set_xlabel('Time (Days)'); axs[1].set_ylabel('Feature Value'); axs[1].legend(); axs[1].grid(True, alpha=0.3)
    return fig

# --- Sidebar ---
st.sidebar.header("Master Controls")
if st.sidebar.button("Synthesize New Data"):
    st.cache_data.clear()

estimator_type = st.sidebar.selectbox("Estimator Type", ["Neural Network (Non-Parametric)", "Linear (Parametric)"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Event Detection")
threshold_slider = st.sidebar.slider('Volatility Threshold (Std Devs)', 0.5, 3.0, 1.5, 0.1)

if estimator_type == "Neural Network (Non-Parametric)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Neural Network")
    n_neurons_slider = st.sidebar.slider('Number of Hidden Neurons', 2, 50, 10, 1)

# --- Main Logic ---
df_data = synthesize_data()
event_df, event_times, marks_normalized, threshold_val = get_events_and_marks(df_data, threshold_slider)

st.header("1. Event Data")
if event_df.empty:
    st.warning("No events detected. Please lower the threshold.")
    st.stop()

st.success(f"{len(event_df)} events detected above threshold {threshold_val:.2f}.")
st.dataframe(event_df[['time', 'implied_vol', 'sentiment', 'odds_change']].reset_index(drop=True))
st.pyplot(plot_time_series(df_data, event_df, threshold_val))

# --- Model Fitting and Analysis ---
st.header("2. Model Fitting & Explanation")
logL, n_params = None, None

if estimator_type == "Neural Network (Non-Parametric)":
    st.subheader("Non-Parametric Model (Neural Network)")
    nn_model, n_params_nn = train_nn_model(event_df, n_neurons_slider)
    if nn_model:
        res = minimize(neg_log_likelihood_nn, [0.1, 1.0], args=(event_times, marks_normalized, nn_model), bounds=[(1e-6, None), (1e-6, None)])
        mu_est, alpha_est = res.x
        logL = -res.fun
        n_params = n_params_nn + 2 # Add mu and alpha
        
        st.write(f"Estimated `μ` (baseline): {mu_est:.4f}, `α` (decay): {alpha_est:.4f}")
        
        # SHAP Analysis
        st.markdown("### SHAP Explanation")
        with st.spinner("Calculating SHAP values..."):
            # Use a smaller sample for KernelExplainer for performance
            background_sample = shap.sample(marks_normalized, min(50, len(marks_normalized)))
            explainer = shap.KernelExplainer(nn_model.predict, background_sample)
            shap_values = explainer.shap_values(marks_normalized)
        
        # SHAP Summary Plot
        st.write("**Global Feature Importance:**")
        st.markdown("This plot ranks features by their total impact and shows how high/low values of a feature affect the model's output (predicted influence).")
        fig_summary, ax_summary = plt.subplots()
        shap.summary_plot(shap_values, marks_normalized, feature_names=['Sentiment', 'Odds Change'], show=False)
        st.pyplot(fig_summary)
        
    else:
        st.error("Could not train the NN model (not enough data).")

elif estimator_type == "Linear (Parametric)":
    st.subheader("Parametric Model (Linear)")
    res = minimize(neg_log_likelihood_parametric, [0.1, 0.5, 0.5, 1.0], args=(event_times, marks_normalized), bounds=[(1e-6, None), (None, None), (None, None), (1e-6, None)])
    if res.success:
        mu_est, beta1_est, beta2_est, alpha_est = res.x
        logL = -res.fun
        n_params = 4
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("μ (Baseline)", f"{mu_est:.3f}")
        col2.metric("β₁ (Sentiment)", f"{beta1_est:.3f}")
        col3.metric("β₂ (Odds Change)", f"{beta2_est:.3f}")
        col4.metric("α (Decay)", f"{alpha_est:.3f}")
    else:
        st.error("Linear model optimization failed.")

# --- Comparison ---
st.header("3. Model Comparison")
if logL is not None and n_params is not None:
    n_events = len(event_df)
    aic = 2 * n_params - 2 * logL
    bic = n_params * np.log(n_events) - 2 * logL
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Log-Likelihood", f"{logL:.2f}", help="Higher is better.")
    col2.metric("AIC", f"{aic:.2f}", help="Akaike Info Criterion. Lower is better.")
    col3.metric("BIC", f"{bic:.2f}", help="Bayesian Info Criterion. Lower is better (stronger penalty for complexity).")
    st.info("Use AIC and BIC to compare the two models. A lower score indicates a better model, balancing fit and complexity.")
else:
    st.warning("Model has not been fit yet.")

