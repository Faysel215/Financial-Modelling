# train_model.py
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import itertools

# --- 1. Synthetic Data Generation ---
def generate_synthetic_trades(n_trades=10000):
    """Generates a large dataset of simulated trades with market features."""
    print("Generating synthetic dataset...")
    np.random.seed(42)
    
    # Market Features
    data = {
        'iv_rank': np.random.randint(0, 101, n_trades),
        'market_trend_ma': np.random.uniform(0.95, 1.05, n_trades), # Price vs 50d MA
    }
    
    # Strategy Parameters (the "knobs" we can turn)
    param_space = {
        'dte': np.random.choice([15, 30, 45, 60], n_trades),
        'delta': np.random.choice([5, 10, 15, 20], n_trades)
    }
    data.update(param_space)
    
    df = pd.DataFrame(data)
    
    # --- 2. Target Variable (P/L Calculation) ---
    # This is a simplified model of reality.
    # High IV and a stable market are good for selling premium.
    # We model P/L based on these principles.
    pnl = (df['iv_rank'] / 50) - (np.abs(1 - df['market_trend_ma']) * 10) \
          - (df['dte'] / 45) + (df['delta'] / 10) \
          + np.random.normal(0, 1.5, n_trades) # Add random noise
          
    # Cap P/L to a realistic range, e.g., -100% to +25%
    df['pnl'] = np.clip(pnl * 10, -100, 25)
    
    print("Dataset generated successfully.")
    return df

# --- 3. Model Training ---
def train_and_save_model():
    df = generate_synthetic_trades()
    
    # Define Features (X) and Target (y)
    features = ['iv_rank', 'market_trend_ma', 'dte', 'delta']
    X = df[features]
    y = df['pnl']

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features - This is crucial for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Neural Network...")
    # Initialize the Multi-Layer Perceptron (MLP) Regressor
    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32), # 2 hidden layers with 64 and 32 neurons
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True, # Stop training if performance doesn't improve
        n_iter_no_change=15
    )

    # Train the model
    nn_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    score = nn_model.score(X_test_scaled, y_test)
    print(f"Model training complete. R-squared score on test data: {score:.4f}")

    # --- 4. Save the Model and Scaler ---
    joblib.dump(nn_model, 'nn_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("\n✅ Model and scaler have been saved to 'nn_model.joblib' and 'scaler.joblib'.")


if __name__ == "__main__":
    train_and_save_model()
