# neural_network_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import itertools

@st.cache_resource
def load_model_and_scaler():
    """Loads the pre-trained model and scaler from disk."""
    try:
        model = joblib.load('nn_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        return None, None

def render():
    """Renders the entire UI for the Neural Network tab."""
    st.header("🧠 Neural Network Strategy Predictor")
    st.markdown(
        """
        This tool uses a pre-trained neural network to predict the profitability of different strategy parameters based on the **current market conditions** you provide. 
        It answers the question: *'Given the market today, what settings are likely to perform best for my next trade?'*
        """
    )
    st.info(
        """
        **How to Use:**
        1. Enter the current market state (IV Rank, Trend) in the sidebar.
        2. Define the potential strategy parameters you're considering.
        3. Click 'Predict Optimal Parameters' to see the model's P/L forecast for each combination.
        """,
        icon="💡"
    )
    
    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.error(
            "Model files not found. Please run `train_model.py` from your terminal to generate 'nn_model.joblib' and 'scaler.joblib'.",
            icon="🚨"
        )
        return

    # --- UI for Inputs ---
    st.sidebar.header("NN Predictor Setup")

    # 1. User inputs for CURRENT market state
    st.sidebar.subheader("Current Market State")
    iv_rank = st.sidebar.slider("Implied Volatility (IV) Rank", 0, 100, 30)
    market_trend = st.sidebar.slider("Market Trend (Price vs. 50d MA)", 0.95, 1.05, 1.01, step=0.01)

    # 2. Candidate strategy parameters to test
    st.sidebar.subheader("Candidate Parameters to Predict")
    candidate_dte = st.sidebar.multiselect("Candidate DTEs", [15, 30, 45, 60], default=[30, 45])
    candidate_delta = st.sidebar.multiselect("Candidate Deltas", [5, 10, 15, 20], default=[10, 15])
    
    if st.sidebar.button("🤖 Predict Optimal Parameters", type="primary"):
        # Create all combinations of candidate parameters
        param_combinations = list(itertools.product(candidate_dte, candidate_delta))
        
        if not param_combinations:
            st.warning("Please select at least one DTE and one Delta to predict.")
            return
            
        # Prepare the input data for the model
        prediction_data = []
        for dte, delta in param_combinations:
            prediction_data.append([iv_rank, market_trend, dte, delta])
            
        features_df = pd.DataFrame(prediction_data, columns=['iv_rank', 'market_trend_ma', 'dte', 'delta'])
        
        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(features_df)
        
        # --- Make Predictions ---
        predicted_pnl = model.predict(scaled_features)
        
        # --- Display Results ---
        results_df = features_df.copy()
        results_df['Predicted P/L (%)'] = predicted_pnl
        
        st.subheader("Prediction Results")
        
        sorted_results = results_df.sort_values(by="Predicted P/L (%)", ascending=False).reset_index(drop=True)
        
        st.dataframe(sorted_results.style.format({
            "Predicted P/L (%)": "{:.2f}%"
        }))
        
        best_params = sorted_results.iloc[0]
        st.success(
            f"""
            **Model Recommendation:** Based on the current market state, a strategy with **{best_params['dte']} DTE** and a **{best_params['delta']} Delta** is predicted to have the highest profitability.
            """
        )
