# main.py
#
# To run this backend:
# 1. Save the file as main.py
# 2. Install dependencies: pip install "fastapi[all]" numpy pandas scikit-learn py_vollib
# 3. Run the server: uvicorn main:app --reload

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor # Used to simulate a pre-trained model
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks import analytical
from py_vollib.black_scholes.implied_volatility import implied_volatility
import random
import time
from datetime import datetime

# --- Helper function to fix JSON serialization issues with NumPy types ---
def convert_numpy_types(obj):
    """ Recursively converts numpy types to native Python types for JSON serialization. """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

# --- 1. Neural Network Simulation ---
# In a real-world scenario, these models would be complex and pre-trained.
# Here, we simulate their prediction behavior by calling the analytical solution.

def nn_predict_iv(price, S, K, t, r, flag):
    try:
        # In a real system: return your_iv_model.predict(...)
        return implied_volatility(price, S, K, t, r, flag)
    except Exception:
        return random.uniform(0.2, 0.6)

def nn_predict_greeks(iv, S, K, t, r, flag):
    try:
        # In a real system: return your_greeks_model.predict(...)
        return {
            'delta': analytical.delta(flag, S, K, t, r, iv),
            'gamma': analytical.gamma(flag, S, K,t, r, iv),
            'vega': analytical.vega(flag, S, K, t, r, iv),
            'theta': analytical.theta(flag, S, K, t, r, iv)
        }
    except Exception:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

# --- 2. Synthetic Data Generation Engine ---
def generate_synthetic_data(ticker: str):
    """
    Generates a full, realistic option chain and data for all surface plots.
    """
    # --- Market Conditions ---
    S = round(random.uniform(150, 350), 2)
    r = 0.05
    today = datetime(2025, 7, 12)
    
    # --- Generate data for multiple expiries to create surfaces ---
    expiries_days = np.array([20, 40, 60, 90, 120, 180])
    times_to_expiry = expiries_days / 365.25
    
    # --- Generate strikes and a volatility smile ---
    atm_strike = round(S / 5) * 5
    strikes = np.arange(atm_strike * 0.75, atm_strike * 1.25, 5.0)
    
    # --- Prepare data structures for surfaces ---
    surface_data = {
        'iv': [], 'delta': [], 'gamma': [], 'vega': [], 'theta': []
    }
    
    # --- Loop through each expiry and strike to build surfaces ---
    for t in times_to_expiry:
        moneyness = np.log(strikes / S) / (np.sqrt(t) if t > 0 else 1)
        atm_vol = random.uniform(0.30, 0.55) - t * 0.1 # Simple time decay
        vol_smile = atm_vol + 0.15 * moneyness**2 # Parabolic smile
        
        iv_row, delta_row, gamma_row, vega_row, theta_row = [], [], [], [], []
        
        for K, vol in zip(strikes, vol_smile):
            # Use call options for the surface visualizations
            price = black_scholes('c', S, K, t, r, vol)
            iv = nn_predict_iv(price, S, K, t, r, 'c')
            greeks = nn_predict_greeks(iv, S, K, t, r, 'c')
            
            iv_row.append(iv)
            delta_row.append(greeks['delta'])
            gamma_row.append(greeks['gamma'])
            vega_row.append(greeks['vega'])
            theta_row.append(greeks['theta'])
            
        surface_data['iv'].append(iv_row)
        surface_data['delta'].append(delta_row)
        surface_data['gamma'].append(gamma_row)
        surface_data['vega'].append(vega_row)
        surface_data['theta'].append(theta_row)

    # --- Generate Option Chain for a single expiry (e.g., the 3rd one) ---
    single_expiry_t = times_to_expiry[2]
    expiry_date = today + pd.to_timedelta(expiries_days[2], 'd')
    expiry_name = expiry_date.strftime("%d %b %Y").upper()
    option_chain = []
    
    moneyness = np.log(strikes / S) / (np.sqrt(single_expiry_t) if single_expiry_t > 0 else 1)
    vol_smile = atm_vol + 0.15 * moneyness**2
    
    for K, vol in zip(strikes, vol_smile):
        call_price = black_scholes('c', S, K, single_expiry_t, r, vol)
        put_price = black_scholes('p', S, K, single_expiry_t, r, vol)
        call_iv = nn_predict_iv(call_price, S, K, single_expiry_t, r, 'c')
        put_iv = nn_predict_iv(put_price, S, K, single_expiry_t, r, 'p')
        call_greeks = nn_predict_greeks(call_iv, S, K, single_expiry_t, r, 'c')
        put_greeks = nn_predict_greeks(put_iv, S, K, single_expiry_t, r, 'p')
        
        option_chain.append({
            "strike": round(K, 2),
            "call_iv": call_iv, "call_delta": call_greeks['delta'],
            "call_bid": round(call_price * 0.995, 2), "call_ask": round(call_price * 1.005, 2),
            "call_vol": random.randint(50, 5000), "call_oi": random.randint(1000, 50000),
            "put_iv": put_iv, "put_delta": put_greeks['delta'],
            "put_bid": round(put_price * 0.995, 2), "put_ask": round(put_price * 1.005, 2),
            "put_vol": random.randint(50, 5000), "put_oi": random.randint(1000, 50000),
        })

    # --- Generate data for 2D IV analysis charts ---
    skew_deltas = np.linspace(0.1, 0.9, 15)
    skew_ivs = atm_vol + 0.3 * (skew_deltas - 0.5)**2 + random.uniform(-0.02, 0.02)
    term_labels = [f'{d}D' for d in expiries_days]
    term_ivs = [np.mean(iv_row) for iv_row in surface_data['iv']]

    return {
        "ticker": ticker,
        "underlyingPrice": S,
        "expiry": expiry_name,
        "optionChain": option_chain,
        "surfacePlots": {
            "strikes": strikes,
            "expiries": expiries_days,
            "iv": surface_data['iv'],
            "delta": surface_data['delta'],
            "gamma": surface_data['gamma'],
            "vega": surface_data['vega'],
            "theta": surface_data['theta'],
        },
        "ivSkew": {"deltas": skew_deltas, "ivs": skew_ivs},
        "ivTerm": {"expiries": term_labels, "ivs": term_ivs},
        "metrics": {
            "vix": round(random.uniform(12, 25), 2),
            "spx": round(random.uniform(5400, 5600), 2),
            "skewIdx": round(random.uniform(110, 140), 2),
            "ivr": f"{random.randint(10, 90)}%",
            "ivp": f"{random.randint(10, 90)}%",
        }
    }

# --- 3. FastAPI Application ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/data")
def get_option_data(ticker: str = "SYNTH"):
    """
    The main API endpoint that the frontend will call.
    It generates all data, cleans it for JSON, and returns it.
    """
    start_time = time.time()
    data = generate_synthetic_data(ticker)
    processing_time = (time.time() - start_time) * 1000
    print(f"Generated data for '{ticker}' in {processing_time:.2f} ms")
    
    # Apply the conversion before returning the data to avoid serialization errors
    return convert_numpy_types(data)

