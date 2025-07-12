### **Project: Options Strategy Analysis & Optimization Toolkit**

This document outlines the complete architecture of the Streamlit application, from the database foundation to the advanced neural network predictor.

### **Part 1: Foundational Layer (Infrastructure & Data)**

This layer is responsible for data persistence, configuration, and the environment.

- **1.1. Database Environment**
    
    - **Technology:** Docker Container.
        
    - **Image:** `timescale/timescaledb:latest-pg16` (PostgreSQL with TimescaleDB extension).
        
    - **Purpose:** To provide a consistent, isolated, and persistent database server.
    - **Setup:**
        
        - Instantiated using the `docker run` command.
            
        - Configured with environment variables (`-e`) for `POSTGRES_PASSWORD` and `POSTGRES_DB`.
            
        - Data persistence is ensured by mapping a Docker volume (`-v`) to the container's data directory.
            
        - The container's port is mapped (`-p`) to the host machine's port for external connections.
            
- **1.2. Database Schema & Seeding**
    
    - **File:** `seed_db.py`
        
    - **Purpose:** A one-time setup script to initialize the database schema and populate it with necessary data.
        
    - **Workflow:**
        
        1. Connects to the database using `psycopg2`.
            
        2. Executes `CREATE TABLE IF NOT EXISTS strategies (...)` to define the table structure for storing options strategy definitions.
            
        3. Holds a Python list of dictionaries (`strategies_data`), where each dictionary defines a strategy (name, description, parameters).
            
        4. Iterates through this list and executes `INSERT INTO strategies (...)` for each one, using `ON CONFLICT DO NOTHING` to prevent duplicates on subsequent runs.
            
- **1.3. Secure Configuration**
    
    - **File:** `.streamlit/secrets.toml`
        
    - **Purpose:** To securely store sensitive database credentials, preventing them from being hardcoded in the application.
        
    - **Mechanism:** Streamlit automatically detects and loads this file. The application accesses credentials via the `st.secrets` object.
        

### **Part 2: Core Application (UI & Orchestration)**

This is the central part of the application that the user interacts with directly.

- **File:** `app.py`
    
- **Purpose:** To serve as the main entry point, manage the overall UI layout, and orchestrate the different feature modules.
    
- **Key Components:**
    
    - **Tabbed Interface:** Uses `st.tabs()` to create the main navigation: "Payoff Analyzer", "Strategy Optimizer", and "Neural Network Predictor".
        
    - **Module Imports:** Imports the `render()` functions from `optimization_tab.py` and `neural_network_tab.py` to draw the content for those tabs.
        
    - **Data Loading:** Contains the `get_strategies()` function, which uses **SQLAlchemy** to create a database engine and read the strategy list into a Pandas DataFrame. This function is cached with `@st.cache_data` for performance.
        
    - **Payoff Logic:** Contains the `calculate_payoff()` function, which holds the mathematical logic for calculating the profit/loss of various static strategies.
        

### **Part 3: Feature Modules (The Tabs)**

Each tab is a self-contained feature module with a distinct purpose.

- **3.1. Module 1: Payoff Analyzer**
    
    - **Code Location:** `app.py` (within `with tab1:`)
        
    - **Goal:** To provide a static, visual analysis of a single strategy's P/L profile.
        
    - **Workflow:**
        
        1. User selects a strategy from a dropdown menu (populated by `get_strategies()`).
            
        2. User inputs specific, static parameters (e.g., strike price, premium).
            
        3. The `calculate_payoff()` function is called.
            
        4. A Plotly line chart is generated and displayed alongside key metrics (Max Profit/Loss).
            
- **3.2. Module 2: Strategy Optimizer**
    
    - **File:** `optimization_tab.py`
        
    - **Goal:** To find the best-performing static parameters for a strategy over a historical period.
        
    - **Method:** **Grid Search** (exhaustive brute-force testing).
        
    - **Workflow:**
        
        1. User defines a _range_ of parameters to test (e.g., multiple DTEs).
            
        2. User selects an objective function (e.g., "Maximize Sharpe Ratio").
            
        3. On-the-fly synthetic historical price data is generated.
            
        4. A backtest is run for every possible parameter combination.
            
        5. Results are displayed in a sorted Pandas DataFrame, highlighting the optimal set.
            
- **3.3. Module 3: Neural Network Predictor**
    
    - **Goal:** To provide dynamic, forward-looking parameter recommendations based on current market conditions.
        
    - **Method:** Predictive Modeling.
        
    - **Workflow is split into two phases:**
        
        - **Phase A: Offline Training (File: `train_model.py`)**
            
            1. A large synthetic dataset is generated, mapping market features and strategy parameters to a P/L outcome.
                
            2. Features are scaled using `StandardScaler`.
                
            3. A `scikit-learn` `MLPRegressor` (Neural Network) is trained on this data.
                
            4. The trained model and the scaler are saved to disk as `.joblib` files.
                
        - **Phase B: Online Prediction (File: `neural_network_tab.py`)**
            
            1. The `render()` function loads the pre-trained model and scaler from the `.joblib` files.
                
            2. User inputs _current_ market conditions (e.g., IV Rank).
                
            3. User inputs a set of _candidate_ parameters they are considering.
                
            4. The model receives these inputs, scales them, and _predicts_ the P/L for each candidate.
                
            5. A ranked list of recommendations is displayed to the user.