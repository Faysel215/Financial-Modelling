# app.py
import torch
import torch.nn as nn
from torch.optim import Adagrad
import numpy as np
import pandas as pd
from math import sqrt
import streamlit as st
import time

# --- Model Architecture and Helper Functions (Unchanged) ---

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, node_features, adj_matrix):
        support = torch.matmul(node_features, self.weight)
        output = torch.matmul(adj_matrix, support)
        return output

class GCNNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_prob):
        super(GCNNetwork, self).__init__()
        self.gc1 = GCNLayer(in_features, hidden_features)
        self.gc2 = GCNLayer(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
    def forward(self, x, adj):
        x = self.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x

class LSTMGCN(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers,
                 gcn_hidden_features, gcn_out_features, dropout_prob):
        super(LSTMGCN, self).__init__()
        self.lstm_network = LSTMNetwork(lstm_input_size, lstm_hidden_size, lstm_num_layers, dropout_prob)
        self.gcn_network = GCNNetwork(lstm_hidden_size, gcn_hidden_features, gcn_out_features, dropout_prob)
        self.fc = nn.Linear(gcn_out_features, 1)
    def forward(self, x):
        lstm_out = self.lstm_network(x)
        node_features = lstm_out[:, -1, :]
        adj_matrix = self.construct_adjacency_matrix(node_features)
        gcn_out = self.gcn_network(node_features, adj_matrix)
        prediction = self.fc(gcn_out)
        return prediction.squeeze(-1)
    def construct_adjacency_matrix(self, node_features):
        features = node_features.detach().cpu().numpy()
        num_stocks = features.shape[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(features)
        corr_matrix = np.nan_to_num(corr_matrix)
        A = torch.from_numpy(corr_matrix).float().to(node_features.device)
        I = torch.eye(num_stocks, device=node_features.device)
        A_hat = A + I
        D_hat = torch.diag(torch.sum(A_hat, 1))
        D_hat_inv_sqrt = torch.pow(D_hat, -0.5)
        D_hat_inv_sqrt[D_hat_inv_sqrt == float('inf')] = 0
        adj_normalized = torch.matmul(torch.matmul(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
        return adj_normalized

class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
    def forward(self, predicted, target):
        t = torch.abs(predicted - target)
        return torch.mean(torch.where(t < 1, 0.5 * t ** 2, t - 0.5))

@st.cache_data
def load_and_preprocess_data(num_stocks, num_days, num_features, input_len):
    st.write("Simulating data loading and preprocessing...")
    data = np.random.rand(num_days, num_stocks, num_features)
    data[:, :, 3] = np.cumsum(np.random.randn(num_days, num_stocks) * 0.1, axis=0) + 20
    training_sets, testing_sets, scalers = [], [], []
    train_increment, test_period, start_day = 44, 44, 0
    while start_day + input_len + test_period <= num_days:
        train_end_day = start_day + train_increment
        test_end_day = train_end_day + test_period
        if test_end_day > num_days:
            test_end_day = num_days
            train_end_day = test_end_day - test_period
            if train_end_day <= start_day: break
        train_data_raw = data[start_day:train_end_day, :, :]
        test_data_raw = data[train_end_day-input_len:test_end_day, :, :]
        train_data_reshaped = train_data_raw.reshape(-1, num_features)
        max_vals, min_vals = np.max(train_data_reshaped, axis=0) * 1.1, np.min(train_data_reshaped, axis=0) * 0.9
        scaler = max_vals - min_vals
        scaler[scaler == 0] = 1.0
        def normalize(d): return (d - min_vals) / scaler
        def create_sequences(d, seq_len):
            xs, ys, refs = [], [], []
            for i in range(len(d) - seq_len):
                xs.append(d[i:(i + seq_len)])
                ys.append(d[i + seq_len, :, 3])
                refs.append(d[i + seq_len - 1, :, 3])
            return np.array(xs), np.array(ys), np.array(refs)
        train_normalized, test_normalized = normalize(train_data_raw), normalize(test_data_raw)
        train_X, train_y, _ = create_sequences(train_normalized, input_len)
        test_X, test_y, test_ref_prices = create_sequences(test_normalized, input_len)
        training_sets.append((np.transpose(train_X, (0, 2, 1, 3)), train_y))
        testing_sets.append((np.transpose(test_X, (0, 2, 1, 3)), test_y, test_ref_prices))
        scalers.append({'min': min_vals[3], 'range': scaler[3]})
        start_day += train_increment
    st.write(f"Created {len(training_sets)} sliding window(s).")
    return training_sets, testing_sets, scalers

def train_model(model, training_sets, epochs, learning_rate, weight_decay, device):
    criterion, optimizer = SmoothL1Loss(), Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        total_loss, total_samples = 0, 0
        for i, (train_X, train_y) in enumerate(training_sets):
            for j in range(train_X.shape[0]):
                inputs = torch.from_numpy(train_X[j]).float().to(device)
                targets = torch.from_numpy(train_y[j]).float().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_samples += 1
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        status_text.text(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")
        progress_bar.progress((epoch + 1) / epochs)
    
    status_text.text(f"Training complete. Final loss: {avg_loss:.6f}")


def evaluate_performance(y_true, y_pred, y_ref):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    true_move, pred_move = (y_true > y_ref).astype(int), (y_pred > y_ref).astype(int)
    tp, tn = np.sum((pred_move == 1) & (true_move == 1)), np.sum((pred_move == 0) & (true_move == 0))
    fp, fn = np.sum((pred_move == 1) & (true_move == 0)), np.sum((pred_move == 0) & (true_move == 1))
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'mae': mae, 'mse': mse, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def run_backtest(predictions, true_prices, ref_prices, num_stocks):
    initial_capital_per_stock, buy_threshold, profit_sell_threshold, hold_period_sell, annual_risk_free_rate = 300000, 0.05, 0.05, 10, 0.025
    num_days = predictions.shape[0]
    cash, shares, entry_price, days_held = np.full(num_stocks, initial_capital_per_stock), np.zeros(num_stocks), np.zeros(num_stocks), np.zeros(num_stocks)
    portfolio_values, initial_total_capital = [], initial_capital_per_stock * num_stocks
    for t in range(num_days):
        current_price, predicted_price_next = ref_prices[t], predictions[t]
        for i in range(num_stocks):
            if shares[i] > 0:
                profit_margin = (current_price[i] - entry_price[i]) / entry_price[i] if entry_price[i] > 0 else 0
                if profit_margin > profit_sell_threshold or days_held[i] >= hold_period_sell:
                    cash[i] += shares[i] * current_price[i]
                    shares[i] = 0
            if shares[i] == 0:
                predicted_increase = (predicted_price_next[i] - current_price[i]) / current_price[i] if current_price[i] > 0 else 0
                if predicted_increase > buy_threshold:
                    shares_to_buy = cash[i] // current_price[i] if current_price[i] > 0 else 0
                    if shares_to_buy > 0:
                        shares[i], cost, entry_price[i], days_held[i] = shares_to_buy, shares_to_buy * current_price[i], current_price[i], 0
                        cash[i] -= cost
            if shares[i] > 0: days_held[i] += 1
        portfolio_values.append(np.sum(cash + shares * current_price))
    portfolio_values = np.array(portfolio_values)
    final_portfolio_value = portfolio_values[-1] if len(portfolio_values) > 0 else initial_total_capital
    tr_money = final_portfolio_value - initial_total_capital
    daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1] if len(portfolio_values) > 1 else np.array([])
    num_years = num_days / 252.0
    total_return_rate = (final_portfolio_value - initial_total_capital) / initial_total_capital if initial_total_capital > 0 else 0
    aar = ((1 + total_return_rate) ** (1/num_years) - 1) if num_years > 0 and total_return_rate > -1 else 0
    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/252.0) - 1
    excess_returns = daily_returns - daily_risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak if len(peak) > 0 else np.array([])
    mdd = np.min(drawdown) if len(drawdown) > 0 else 0
    return {'tr_money': tr_money, 'aar': aar, 'sharpe_ratio': sharpe_ratio, 'mdd': mdd}

# --- Streamlit UI ---

def display_results(results):
    """Renders the results using Streamlit components."""
    
    st.markdown("""
    <style>
    .metric-card {
        background-color: #27272a;
        border: 1px solid #3f3f46;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-card h3 {
        font-size: 1.1rem;
        color: #a1a1aa;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .positive { color: #4ade80; }
    .negative { color: #f87171; }
    .neutral { color: #60a5fa; }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Prediction Performance Evaluation")
    perf = results['performance']
    p1, p2, p3 = st.columns(3)
    
    with p1:
        st.markdown(f"""<div class="metric-card"><h3>Mean Absolute Error (MAE)</h3><p class="neutral">{perf['mae']:.4f}</p></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class="metric-card"><h3>Mean Square Error (MSE)</h3><p class="neutral">{perf['mse']:.4f}</p></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown(f"""<div class="metric-card"><h3>Accuracy (Acc)</h3><p class="positive">{perf['accuracy']:.2%}</p></div>""", unsafe_allow_html=True)
    
    st.write("") # Spacer
    
    p4, p5, p6 = st.columns(3)
    with p4:
        st.markdown(f"""<div class="metric-card"><h3>Precision (Pre)</h3><p class="positive">{perf['precision']:.4f}</p></div>""", unsafe_allow_html=True)
    with p5:
        st.markdown(f"""<div class="metric-card"><h3>Recall (Rec)</h3><p class="positive">{perf['recall']:.4f}</p></div>""", unsafe_allow_html=True)
    with p6:
        st.markdown(f"""<div class="metric-card"><h3>F-measure (F1)</h3><p class="positive">{perf['f1']:.4f}</p></div>""", unsafe_allow_html=True)

    st.divider()

    st.subheader("Stock Trading Backtest")
    backtest = results['backtest']
    b1, b2 = st.columns(2)

    tr_money_color = "positive" if backtest['tr_money'] > 0 else "negative"
    aar_color = "positive" if backtest['aar'] > 0 else "negative"
    mdd_color = "negative"
    sr_color = "positive" if backtest['sharpe_ratio'] > 1 else "neutral" if backtest['sharpe_ratio'] > 0 else "negative"

    with b1:
        st.markdown(f"""<div class="metric-card"><h3>Total Returns (TRMoney)</h3><p class="{tr_money_color}">{backtest['tr_money']:,.2f} RMB</p></div>""", unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""<div class="metric-card"><h3>Sharpe Ratio (SR)</h3><p class="{sr_color}">{backtest['sharpe_ratio']:.4f}</p></div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""<div class="metric-card"><h3>Average Annual Return (AAR)</h3><p class="{aar_color}">{backtest['aar']:.2%}</p></div>""", unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""<div class="metric-card"><h3>Maximum Drawdown (MDD)</h3><p class="{mdd_color}">{backtest['mdd']:.2%}</p></div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="LSTM-GCN Model", layout="wide")
    st.title("📈 LSTM-GCN Stock Trend Prediction")
    st.markdown("An interactive dashboard to run the LSTM-GCN model simulation and view its performance based on the methodology described.")

    with st.sidebar:
        st.header("Simulation Settings")
        NUM_STOCKS = st.slider("Number of Stocks", 10, 50, 35)
        NUM_DAYS = st.slider("Number of Days for Simulation", 100, 1500, 300)
        EPOCHS = st.slider("Training Epochs", 1, 20, 5)
        INPUT_LEN = st.slider("Input Length (Days)", 5, 30, 12)
        
        st.header("Model Hyperparameters")
        LSTM_HIDDEN_SIZE = st.select_slider("LSTM Hidden Size", [32, 64, 128, 256], 128)
        LEARNING_RATE = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05], 0.01)
        DROPOUT_PROB = st.slider("Dropout Probability", 0.0, 0.5, 0.2)


    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulation... This may take a few minutes."):
            # --- Settings ---
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            NUM_FEATURES = 6
            LSTM_NUM_LAYERS = 1
            GCN_HIDDEN_FEATURES = 64
            GCN_OUT_FEATURES = 32
            WEIGHT_DECAY = 5e-4
            
            # --- Data ---
            training_sets, testing_sets, scalers = load_and_preprocess_data(
                num_stocks=NUM_STOCKS, num_days=NUM_DAYS, num_features=NUM_FEATURES, input_len=INPUT_LEN)

            # --- Model ---
            model = LSTMGCN(
                lstm_input_size=NUM_FEATURES, lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_num_layers=LSTM_NUM_LAYERS,
                gcn_hidden_features=GCN_HIDDEN_FEATURES, gcn_out_features=GCN_OUT_FEATURES, dropout_prob=DROPOUT_PROB)
            
            # --- Training & Evaluation ---
            if training_sets:
                st.subheader("Model Training Progress")
                train_model(model, training_sets, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE)
                
                st.subheader("Final Evaluation")
                with st.spinner("Evaluating on test set..."):
                    model.eval()
                    all_preds, all_trues, all_refs = [], [], []
                    with torch.no_grad():
                        test_X, test_y, test_ref = testing_sets[-1]
                        scaler_info = scalers[-1]
                        for j in range(test_X.shape[0]):
                            inputs = torch.from_numpy(test_X[j]).float().to(DEVICE)
                            outputs = model(inputs).cpu()
                            pred_denorm = outputs.numpy() * scaler_info['range'] + scaler_info['min']
                            true_denorm = test_y[j] * scaler_info['range'] + scaler_info['min']
                            ref_denorm = test_ref[j] * scaler_info['range'] + scaler_info['min']
                            all_preds.append(pred_denorm)
                            all_trues.append(true_denorm)
                            all_refs.append(ref_denorm)
                    
                    all_preds, all_trues, all_refs = np.array(all_preds), np.array(all_trues), np.array(all_refs)
                    
                    performance_results = evaluate_performance(all_trues, all_preds, all_refs)
                    backtest_results = run_backtest(all_preds, all_trues, all_refs, NUM_STOCKS)
                    
                    st.session_state['results'] = {
                        'performance': performance_results,
                        'backtest': backtest_results
                    }
                st.success("Simulation Complete!")

            else:
                st.error("Not enough data to create training/testing windows with the selected parameters.")

    if 'results' in st.session_state:
        st.divider()
        st.header("📊 Simulation Results")
        display_results(st.session_state['results'])

if __name__ == '__main__':
    main()

