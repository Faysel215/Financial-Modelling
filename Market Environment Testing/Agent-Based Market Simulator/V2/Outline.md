
This document provides a full, end-to-end implementation plan for the WGAN-powered, agent-based market simulator. It is structured into five distinct phases, moving from foundational data management to the final interactive application and experimentation.

## Phase 1: Foundation - Data Management & Environment Setup

> [!success] Objective
>  To establish a robust and efficient data persistence layer and configure the development environment. This phase is the bedrock upon which all subsequent modeling and simulation work is built.

### 1.1. Environment Configuration
- **Language:** Python 3.8+
- **Virtual Environment:** Set up a dedicated virtual environment (e.g., venv) to manage dependencies.
- **Core Libraries:** Install necessary packages from requirements.txt:
    - `pandas`, `numpy` for data manipulation.
    - `psycopg2-binary`, `SQLAlchemy` for database interaction.
    - torch, `torchvision` for deep learning models.
    - `matplotlib` for plotting.
    - `tkinter` (standard library) for the GUI.
    - `tqdm` for progress bars.

### 1.2. Database Setup
- **Technology:** TimescaleDB (a PostgreSQL extension for time-series data).
- **Deployment:** Use Docker for a consistent, cross-platform setup. 
```
docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb:latest-pg14  
```
- **Database Creation:** Implement a Python function to automatically create the `market_sim_lob` database if it doesn't exist.

### 1.3. Data Ingestion and Migration (`migrate_db.py`)
- **Source:** The local SQLite database file (`lob-btc.db`).
- **Destination:** The TimescaleDB instance.
- **Process:**
	1. Connect to both the source SQLite DB and the destination TimescaleDB.
	2. Define the schema in TimescaleDB: a single `lob_snapshots` table with a TIMESTAMPTZ primary key and JSONB columns for bids and asks.
	3. Convert the `lob_snapshots` table into a TimescaleDB hypertable for performance 
	4. Read data in batches from the SQLite items, bids, and asks tables.
	5. For each snapshot, aggregate bids and asks into JSON arrays.
	6. Efficiently bulk-insert the transformed data into the TimescaleDB hypertable.
    

## Phase 2: Generative Modeling - Market Texture (WGAN)

> [!success] Objective
>  To train a deep generative model that learns the static probability distribution of limit order book states from our historical data.

### 2.1. Script: `wgan_trainer.py`
### 2.2. Data Loading & Preprocessing
1. Connect to the TimescaleDB from Phase 1.
2. Stream the LOB data in manageable chunks to handle large datasets efficiently.
3. Snapshot Transformation: For each LOB snapshot:
	- Calculate the mid-price.
	- Convert the LOB into a fixed-size 2D array (an "image"), where rows represent price levels relative to the mid-price and the single column represents volume.
	- Normalize the volume data in each image (e.g., by dividing by the maximum volume in that snapshot).
### 2.3. Model Architecture (WGAN-GP in PyTorch)
- **Generator:** A deep convolutional neural network using transposed convolution layers (`ConvTranspose2d`) to upscale a random noise vector (`z_dim`) into a synthetic LOB image.
- **Critic (Discriminator):** A deep convolutional neural network (`Conv2d`) that takes a LOB image (real or fake) and outputs a scalar value representing its "realness."
- **WGAN-GP:** Utilize the Wasserstein loss with Gradient Penalty for stable training and to avoid mode collapse.
### 2.4. Training & Validation
1. Implement the WGAN-GP training loop, iterating between training the Critic and the Generator.
2. Leverage a CUDA-enabled GPU for training.
3. During training, periodically save sample generated LOB images to a directory (/training_snapshots) for visual inspection of learning progress.
4. Output: A trained model file, `generator.pth`.

## Phase 3: Dynamic Modeling - Market Flow (NHP)

> [!success] Objective
>  To train a model that learns the temporal dynamics of market events, creating the "pulse" of the market
### 3.1. Script: `nhp_trainer.py`
### 3.2. Synthetic Data Generation for Training
1. Load the trained WGAN Generator (`generator.pth`) from Phase 2.
2. Use the generator to create a long, continuous sequence of synthetic LOB snapshots. This provides a clean, stationary dataset for the NHP to learn from.

### 3.3. Event Stream Inference
- Implement a "diffing" algorithm that processes the sequence of synthetic LOBs.
- By comparing consecutive snapshots (LOB at time t vs. t+1), infer the most likely event that occurred (e.g., limit order placement, cancellation, or market order).
- Create a sequence of event tuples: (`event_type`, `time_delta`).

### 3.4. Model Architecture (Neural Hawkes Process in PyTorch)
- Use a Recurrent Neural Network (RNN), such as a GRU, as the core of the NHP.
- The model takes a history of recent events as input.
- It outputs a prediction for both the time to the next event and the type of the next event.

### 3.5. Training & Output
1. Train the NHP model on the inferred event stream using a combined loss function (e.g., log-likelihood for time + cross-entropy for type).
2. **Output:** A trained model file, `nhp_background_trader.pth`.

## Phase 4: Integration & Simulation - The "Algo-Gym" GUI

> [!success] Objective
> To combine all components into a single, interactive desktop application for running simulations and testing strategies
### 4.1. Script: `market_sim_gui.py`
### 4.2. Core Components (Python/Tkinter)
- **LimitOrderBook Class:** The matching engine. Manages bids and asks using heaps and executes trades when orders cross.
- **Agent Base Class:** An abstract class defining the interface for all agents (id, pnl, inventory, act() method, etc.).
- **BackgroundTrader Class:** A simplified, stochastic implementation of the NHP logic that generates a continuous stream of both passive and aggressive orders to create a dynamic market.
- **StrategicAgent Class:** A placeholder class that can be easily modified or subclassed. This is where user-defined strategies will be implemented.

### 4.3. User Interface (GUI)
- Controls Panel: Input fields to set strategic agent parameters (e.g., order size, moving average windows).
- Live Blotter: A ttk.Treeview widget that streams every order and trade in real-time, with color-coding for buys, sells, and trades.
- Status Display: Live updates of the best bid, best ask, and spread.
- Results Panel: Displays the final P&L and total trades for the strategic agent after the simulation ends.
- Analysis Tab: Integrates a matplotlib canvas to plot the strategic agent's mark-to-market PnL over the course of the simulation.
- Features: Include a "Toggle Dark Mode" button for usability.
#### Screenshot

![GUI Screenshot](/img/gui)
### Phase 5: Experimentation & Analysis

> [!success] Objective
> To use the completed "Algo-Gym" to implement, test, and analyze trading strategies.

### 5.1. Implementing New Strategies
- Create new agent classes that inherit from the Agent base class.
- Example: MomentumAgent
	1. The agent maintains internal lists to track the history of mid-prices.
	2. In its `act()` method, it calculates the short-term (e.g., 5-period) and long-term (e.g., 20-period) moving averages.
	3. It checks for crossover signals (short MA > long MA is a buy signal; short MA < long MA is a sell signal).
	4. It holds a position (long or short) based on the current signal and only trades when a new crossover occurs, reversing its position.
### 5.2. Running Counterfactual Experiments
1. Baseline Run: Run the simulation with a "dummy" strategic agent that does nothing. Save the resulting PnL plot (which will be zero) and market data logs.
2. Experimental Run: Run the simulation with your active strategic agent (e.g., MomentumAgent).
3. Analyze: Compare the PnL plot and other metrics from the experimental run to the baseline. This isolates the strategy's performance and impact.
### 5.3. Parameter Tuning
- Use the GUI controls to easily run multiple simulations with different agent parameters (e.g., test the MomentumAgent with different moving average windows) to find the optimal configuration.