# Options Strategy Analysis & Optimization Toolkit

This project is a comprehensive Streamlit web application designed for analyzing, optimizing, and predicting the performance of various options trading strategies. It provides a suite of tools ranging from static payoff visualization to dynamic, AI-powered parameter recommendations.

## Features

The toolkit is organized into three main modules, accessible through a tabbed interface:

- **Payoff Analyzer:** A tool for static analysis. Select a classic options strategy, input your specific parameters (strike prices, premium), and instantly visualize the profit/loss profile at expiration.
    
- **Strategy Optimizer:** A backtesting tool that uses a **Grid Search** methodology to find the optimal static parameters for a strategy. It iterates through every possible combination of user-defined ranges (e.g., different DTEs and deltas) to find the best performer based on metrics like Sharpe Ratio or Total P/L.
    
- **Neural Network Predictor:** An advanced, forward-looking tool that uses a pre-trained neural network to recommend the best strategy parameters based on _current_ market conditions. Instead of finding what worked best in the past, it predicts what will work best for your next trade.
    

## Architecture Overview

The application is built on a modern Python stack and is designed to be modular and scalable.

- **Frontend:** Streamlit
    
- **Backend & Modeling:** Python, Pandas, Scikit-learn
    
- **Database:** PostgreSQL with TimescaleDB extension, running in a Docker container for consistency and isolation.
    
- **Database Connector:** SQLAlchemy for robust and efficient communication between the application and the database.
    

## Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/ "null") must be installed and running.
    
- [Python 3.9+](https://www.python.org/downloads/ "null") must be installed.
    

### Step 1: Clone the Repository

```
git clone <your-repository-url>
cd <your-repository-folder>
```

### Step 2: Set Up the Database

1. Start the TimescaleDB Container:
    
    Run the following command in your terminal to start the database in a Docker container.
    
    ```
    docker run --name some-timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -e POSTGRES_DB=opt-strat -v tsdata:/var/lib/postgresql/data -d timescale/timescaledb:latest-pg16
    ```
    
2. Seed the Database:
    
    This one-time script creates the strategies table and populates it.
    
    ```
    python seed_db.py
    ```
    

### Step 3: Configure Secrets

1. Create a folder named `.streamlit` in the root of your project directory.
    
2. Inside that folder, create a file named `secrets.toml`.
    
3. Add the following content to `secrets.toml`:
    
    ```
    [database]
    host = "localhost"
    user = "postgres"
    password = "mysecretpassword"
    port = 5432
    dbname = "opt-strat"
    ```
    

### Step 4: Install Python Dependencies

It is highly recommended to use a virtual environment.

```
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install required libraries
pip install streamlit pandas psycopg2-binary sqlalchemy scikit-learn joblib
```

### Step 5: Train the Neural Network Model

This one-time script trains the predictive model and saves it to a file for the application to use.

```
python train_model.py
```

### Step 6: Run the Streamlit Application

You are now ready to launch the app!

```
streamlit run app.py
```

## File Structure

A brief overview of the key files in this repository:

- `app.py`: The main entry point for the Streamlit application. It builds the UI and orchestrates the different tabs.
    
- `seed_db.py`: A one-time script to set up and populate the database.
    
- `optimization_tab.py`: Contains all the UI and backend logic for the "Strategy Optimizer" tab.
    
- `neural_network_tab.py`: Contains all the UI and backend logic for the "Neural Network Predictor" tab.
    
- `train_model.py`: The offline script used to train the neural network model.
    
- `.streamlit/secrets.toml`: The secure configuration file for database credentials.
    

## Disclaimer

This toolkit is designed for educational and research purposes only. The backtesting and prediction modules use synthetically generated data and simplified pricing models. The results are not indicative of real-world performance and should not be used for making live trading decisions.