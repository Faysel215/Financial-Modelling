import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. Model Architecture: Transformer-Modulated Hawkes Process (TMHP) ---
# This model implants a Transformer into a Hawkes process to capture complex,
# long-range dependencies in event sequences, inspired by the AIPM concept.
# The core idea is based on the Transformer Hawkes Process (THP) literature.

class TransformerHawkesProcess(nn.Module):
    """
    A Transformer-based model for a self-exciting point process.
    The intensity function is modulated by the output of a Transformer encoder
    that processes the history of events.
    """
    def __init__(self, num_event_types, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Component 1: Event Embedding
        self.event_embedding = nn.Embedding(num_event_types, d_model)
        self.time_encoder = nn.Linear(1, d_model)

        # Component 2: Transformer Encoder Core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Component 3: Neural Intensity Function
        self.intensity_layer = nn.Linear(d_model, num_event_types)
        self.softplus = nn.Softplus()

    def forward(self, event_times, event_types):
        """
        Processes a sequence of events to compute hidden states.
        """
        time_intervals = torch.cat([torch.zeros_like(event_times[:, :1]), 
                                    event_times[:, 1:] - event_times[:, :-1]], dim=1)
        
        time_enc = self.time_encoder(time_intervals.unsqueeze(-1))
        type_emb = self.event_embedding(event_types)
        
        x = time_enc + type_emb
        
        seq_len = event_times.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(event_times.device)
        
        hidden_states = self.transformer_encoder(x, mask=causal_mask)
        return hidden_states

    def compute_intensity(self, history_hidden_state, time_since_last_event):
        """
        Computes the conditional intensity λ(t) at a given time.
        """
        time_decay_enc = self.time_encoder(time_since_last_event)
        current_hidden_state = history_hidden_state + time_decay_enc
        intensity = self.softplus(self.intensity_layer(current_hidden_state))
        return intensity

# --- 2. Dummy Data Generation ---
def generate_dummy_data(seq_len=50, num_event_types=2):
    """Generates a single synthetic event sequence."""
    event_times = [np.random.rand() * 0.1]
    event_types = [np.random.randint(0, num_event_types)]
    
    for _ in range(seq_len - 1):
        # Subsequent events occur after some random interval
        # The interval gets smaller as more events occur, simulating self-excitation
        interval = np.random.exponential(scale=1.0) / (1 + len(event_times) * 0.1)
        event_times.append(event_times[-1] + interval)
        event_types.append(np.random.randint(0, num_event_types))
    
    return {
        'times': torch.tensor(event_times, dtype=torch.float32),
        'types': torch.tensor(event_types, dtype=torch.long)
    }

# --- 3. Simulation using Ogata's Thinning Algorithm ---
@st.cache_data
def simulate_tmhp(_model, history_times, history_types, max_time, num_event_types):
    """
    Generates a sequence of events from the trained TMHP model.
    Note: Using st.cache_data to avoid re-running simulation on every UI interaction.
    The underscore in _model tells Streamlit to not hash the model object itself.
    """
    _model.eval()
    with torch.no_grad():
        sim_times = history_times.tolist()
        sim_types = history_types.tolist()
        
        history_times_tensor = torch.tensor(history_times).unsqueeze(0)
        history_types_tensor = torch.tensor(history_types, dtype=torch.long).unsqueeze(0)
        
        hidden_states = _model(history_times_tensor, history_types_tensor)
        last_hidden_state = hidden_states[:, -1, :]
        
        current_time = sim_times[-1]

        while current_time < max_time:
            t_since_last = torch.linspace(0, 1, 100).unsqueeze(1)
            intensities = _model.compute_intensity(last_hidden_state, t_since_last)
            lambda_max = torch.sum(intensities, dim=1).max().item() + 0.1

            time_interval_candidate = np.random.exponential(scale=1.0 / lambda_max)
            current_time += time_interval_candidate
            
            if current_time >= max_time:
                break

            t_since_last_event = torch.tensor([[current_time - sim_times[-1]]], dtype=torch.float32)
            true_intensity_at_candidate = _model.compute_intensity(last_hidden_state, t_since_last_event)
            total_intensity = torch.sum(true_intensity_at_candidate).item()

            if np.random.rand() < total_intensity / lambda_max:
                type_probabilities = true_intensity_at_candidate.squeeze() / total_intensity
                new_event_type = torch.multinomial(type_probabilities, 1).item()
                
                sim_times.append(current_time)
                sim_types.append(new_event_type)
                
                new_times_tensor = torch.tensor(sim_times).unsqueeze(0)
                new_types_tensor = torch.tensor(sim_types, dtype=torch.long).unsqueeze(0)
                hidden_states = _model(new_times_tensor, new_types_tensor)
                last_hidden_state = hidden_states[:, -1, :]

    return sim_times, sim_types

# --- 4. Streamlit User Interface ---
def main():
    st.set_page_config(layout="wide")
    st.title("Transformer-Modulated Hawkes Process (TMHP) Simulator")
    st.markdown("""
    This application simulates a **Transformer-Modulated Hawkes Process (TMHP)**, a sophisticated model for analyzing event sequences like trades or price ticks. 
    It combines the **Hawkes Process**, which models self-exciting events, with a **Transformer network**, inspired by the Artificial Intelligence Pricing Model (AIPM).
    
    Use the sidebar to configure the model and simulation parameters, then click **'Run Simulation'** to see the results.
    """)

    # --- Sidebar for Parameter Configuration ---
    st.sidebar.title("Configuration")
    st.sidebar.header("Model Hyperparameters")
    d_model = st.sidebar.slider("Model Dimension (d_model)", 32, 256, 64, step=32)
    nhead = st.sidebar.slider("Number of Attention Heads (nhead)", 1, 8, 4)
    num_layers = st.sidebar.slider("Number of Transformer Layers", 1, 6, 2)
    num_event_types = st.sidebar.number_input("Number of Event Types", min_value=2, max_value=10, value=2, step=1)

    st.sidebar.header("Simulation Parameters")
    seq_len = st.sidebar.slider("Generated History Length", 20, 200, 50)
    history_len_for_sim = st.sidebar.slider("Events to Use as History", 10, seq_len, 20)
    simulation_duration = st.sidebar.slider("Simulation Duration (Time Units)", 1.0, 20.0, 5.0)

    if st.sidebar.button("Run Simulation"):
        with st.spinner("Generating data, training model, and running simulation..."):
            # --- Data and Model Initialization ---
            dummy_data = generate_dummy_data(seq_len=seq_len, num_event_types=num_event_types)
            model = TransformerHawkesProcess(
                num_event_types=num_event_types, 
                d_model=d_model, 
                nhead=nhead, 
                num_encoder_layers=num_layers
            )
            
            # --- Conceptual Training Loop ---
            # A simplified training process for demonstration purposes.
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss() # Placeholder loss
            model.train()
            for _ in range(5): # Dummy training for 5 epochs
                optimizer.zero_grad()
                hidden_states = model(dummy_data['times'].unsqueeze(0), dummy_data['types'].unsqueeze(0))
                predicted_intensities = model.softplus(model.intensity_layer(hidden_states))
                time_intervals = torch.cat([torch.zeros(1), dummy_data['times'][1:] - dummy_data['times'][:-1]])
                target = 1.0 / (time_intervals + 1e-6)
                loss = loss_fn(torch.sum(predicted_intensities.squeeze(), dim=1), target)
                loss.backward()
                optimizer.step()

            # --- Simulation ---
            history_times = dummy_data['times'][:history_len_for_sim].numpy()
            history_types = dummy_data['types'][:history_len_for_sim].numpy()
            max_simulation_time = history_times[-1] + simulation_duration
            
            sim_times, sim_types = simulate_tmhp(model, history_times, history_types, max_simulation_time, num_event_types)
            
            st.success(f"Simulation finished. Generated {len(sim_times) - len(history_times)} new events.")

            # --- Visualization ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # Plot 1: Event Timestamps
            ax1.plot(history_times, np.zeros_like(history_times), 'bo', label='History Events', markersize=8, alpha=0.6)
            sim_new_times = np.array(sim_times[len(history_times):])
            if len(sim_new_times) > 0:
                ax1.plot(sim_new_times, np.zeros_like(sim_new_times), 'rx', label='Simulated Events', markersize=8, mew=2)
            ax1.set_title('Event Timestamps on Timeline', fontsize=16)
            ax1.set_xlabel('Time')
            ax1.set_yticks([]) # CORRECTED LINE
            ax1.legend()
            ax1.grid(True, axis='x', linestyle='--', alpha=0.6)

            # Plot 2: Intensity Function
            plot_times = np.linspace(0, max_simulation_time, 500)
            intensities_over_time = []
            
            model.eval()
            with torch.no_grad():
                current_idx = 0
                hidden_states_full = model(torch.tensor(sim_times).unsqueeze(0), torch.tensor(sim_types, dtype=torch.long).unsqueeze(0)).squeeze(0)
                for t in plot_times:
                    while current_idx + 1 < len(sim_times) and t > sim_times[current_idx + 1]:
                        current_idx += 1
                    
                    last_hidden_state = hidden_states_full[current_idx, :].unsqueeze(0)
                    time_since_last = torch.tensor([[t - sim_times[current_idx]]], dtype=torch.float32)
                    intensity = model.compute_intensity(last_hidden_state, time_since_last)
                    intensities_over_time.append(intensity.squeeze().numpy())

            intensities_over_time = np.array(intensities_over_time)
            total_intensity = np.sum(intensities_over_time, axis=1)
            
            for i in range(num_event_types):
                ax2.plot(plot_times, intensities_over_time[:, i], label=f'Intensity Type {i}', lw=2)
            ax2.plot(plot_times, total_intensity, 'k--', label='Total Intensity', lw=2, alpha=0.8)
            
            for t in sim_times:
                ax2.axvline(x=t, color='gray', linestyle=':', alpha=0.7)

            ax2.set_title('Simulated Conditional Intensity λ(t)', fontsize=16)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_ylabel('Intensity', fontsize=12)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.set_ylim(bottom=0)
            
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to begin.")

if __name__ == '__main__':
    main()
