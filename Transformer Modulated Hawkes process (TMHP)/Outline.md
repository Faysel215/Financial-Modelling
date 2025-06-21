# Project: Transformer-Modulated Hawkes Process (TMHP) Simulator
**Objective:** To implement and visualize a Transformer-Modulated Hawkes Process, enabling users to interactively simulate event sequences and explore the model's behavior under different parameter configurations.
## Part 1: Foundational Components (Python & PyTorch)
### 1.1. Model Architecture: `TransformerHawkesProcess` Class
- **Objective:** Create a PyTorch `nn.Module` that encapsulates the core logic of the TMHP.
- **Inputs:** `num_event_types`, `d_model`, `nhead`, `num_encoder_layers`.
- **Key Sub-Modules:**
    - `nn.Embedding`: To convert discrete event types (e.g., buy/sell) into dense vector representations.
    - `nn.Linear` (Time Encoder): To encode the continuous time intervals between events into the model's dimension.
    - `nn.TransformerEncoder`: The core of the model. This will process the sequence of event embeddings and time encodings.
    - Crucially, a causal (triangular) attention mask must be applied to ensure the model only uses past information to predict the future.
- **Key Methods:**
    - `forward(event_times, event_types)`: The main method to process a history of events and produce a sequence of hidden states (`h_t`).
    - `compute_intensity(history_hidden_state, time_since_last_event)`: A separate method to calculate the conditional intensity λ(t) given the most recent hidden state and the time elapsed since the last event. This will use a final linear layer followed by a `Softplus` activation to ensure positivity.

## 1.2. Data Generation: `generate_dummy_data` Function
- **Objective:** Create a function to generate synthetic event sequences for training and initializing the simulation.
- **Logic:**
    - Start with a random initial event.
    - Iteratively generate subsequent events. The time interval between events should follow an exponential distribution, with the rate parameter increasing as a function of the number of past events to simulate a simple self-exciting process.
- **Output:** A dictionary containing two PyTorch tensors: `times` (float) and `types` (long).

## 1.3. Simulation Engine: simulate_tmhp Function (Ogata's Thinning Algorithm)
- **Objective:** Implement Ogata's thinning algorithm to generate new events from a "trained" TMHP model.
- Inputs: A model instance, historical event times and types, a maximum simulation time, and the number of event types.
- Algorithm Steps:
    1. Initialize sim_times and sim_types with the provided history.
    2. Feed the initial history through the model to get the final hidden state, h_N.
    3. Start a loop that continues until the current_time exceeds `max_simulation_time`.
    4. Find an upper bound ($\lambda_\max$): Calculate the intensity over a short future interval to find a maximum value. This is the "majorant" process.
    5. Propose a new event: Draw a candidate time interval from an exponential distribution with rate λ_max. Add this to current_time.
    6. Accept or Reject (Thinning): Calculate the true total intensity at the candidate time. Draw a random number $u$ from $U[0, 1]$. If $\mu < \lambda_\text{true} / \lambda_\max$, accept the event.
    7. If Accepted:
- Determine the type of the new event by sampling from the categorical distribution defined by the individual event intensities at that moment.
- Append the new time and type to the simulation lists.
- Crucially, update the hidden state by feeding the entire new sequence back into the model to get the next h_N+1.
8. Repeat from step 4.
- Output: The complete list of historical and newly simulated event times and types.

## Part 2: Interactive User Interface (Streamlit)
### 2.1. Main Application Structure: `main` Function
- **Objective:** Set up the Streamlit application layout and control flow.
- **Layout:**
    - Use `st.set_page_config(layout="wide")` for a better user experience.
    - A main area for titles, explanations, and plots.
    - A sidebar (`st.sidebar`) for all user-configurable parameters.

### 2.2. Parameter Configuration (Sidebar)
- **Objective:** Allow users to test different model and simulation configurations.
- **Model Hyperparameters:** Use `st.sidebar.slider` and `st.sidebar.number_input` for d_model, nhead, num_layers, and num_event_types.
- **Simulation Parameters:** Use st.sidebar.slider for seq_len (history generation), history_len_for_sim (how much history to feed the model), and simulation_duration.
- **Control Flow:** Wrap the main logic in an if st.sidebar.button("Run Simulation"): block to ensure the simulation only runs on user command.

### 2.3. Execution and Visualization (Main Area)
- **Objective:** Run the simulation and display the results graphically when the button is pressed.
- Steps (within the if block):
    1. Display a st.spinner to indicate that computation is in progress.
    2. Initialization: Call generate_dummy_data and initialize the TransformerHawkesProcess model with the parameters from the sidebar.
    3. Conceptual Training: Implement a placeholder training loop. For this simulation, it's not necessary to train to convergence. A few optimization steps on the dummy data are sufficient to ensure the model weights are not random.
    4. Simulation: Call the simulate_tmhp function with the appropriate data slices.
    5. Plotting (Matplotlib):
- Create a figure with two subplots (`subplots(2, 1)`).
- **Plot 1 (Event Timeline):**
    - Plot historical events as blue circles (`bo`).
    - Plot simulated events as red crosses (`rx`).
    - Remove y-axis ticks for clarity: `ax1.set_yticks([])`.
- **Plot 2 (Intensity Function):**
    - Create a fine-grained time axis (`np.linspace`).
    - Loop through this time axis, calculating the total conditional intensity $\ambda(t)$ at each point using the compute_intensity method. This will require carefully tracking which hidden state to use based on the most recent event before time $t$.
    - Plot the intensity for each event type and the total intensity.
    - Draw vertical lines (`axvline`) at each event time to show the correspondence between events and intensity jumps.
    - Display the final plot in Streamlit using `st.pyplot(fig)`.

## Part 3: Future Enhancements & Refinements
Implement Proper Training: Replace the placeholder training loop with a proper negative log-likelihood loss function for point processes.
Real Data Integration: Add functionality to upload real-world financial event data (e.g., from a CSV file) instead of using generated data.
Performance Metrics: Display quantitative metrics about the simulation, such as the total number of generated events, average intensity, etc.
Advanced Visualizations: Add interactive plots (e.g., using Plotly) that allow users to zoom in on the timeline and hover over points to see details.
Model Caching: Use st.cache_data on the simulation function to prevent re-computation when only UI elements are changed, improving responsiveness.
