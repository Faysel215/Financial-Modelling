# 📈 Dynamic Liquidity Nexus: An S&P 500 Limit Order Book Simulation
Welcome to the Dynamic Liquidity Nexus, an agent-based simulation of a financial market's Limit Order Book (LOB). This interactive tool, built with Python and Streamlit, models the behavior of different trading agents and their impact on market dynamics, providing a visual and intuitive understanding of price discovery, liquidity, and agent profitability.
## ✨ Features
- **Realistic Limit Order Book:** A core implementation of a LOB with standard functionalities like adding, canceling, and processing market orders.
- **Agent-Based Modeling:** The simulation is populated by two distinct types of agents:
  - **Risk-Averse Liquidity Providers (LPs):** These agents aim to profit from the bid-ask spread while managing inventory risk. They continuously quote buy and sell orders around the mid-price.
  - **Risk-Lover Liquidity Takers (LTs):** These agents act on market signals, executing market orders that consume liquidity and impact the market price.
- **Interactive Web Interface:** A user-friendly dashboard built with Streamlit allows you to:
  - Tune simulation parameters in real-time.
  - Visualize market data through interactive Plotly charts.
  - Analyze the outcomes without modifying the source code.
-  Dynamic Market Price: The simulation incorporates a market impact factor, allowing large trades from Liquidity Takers to dynamically influence the reference market price.

## 🚀 How to Run the Simulation
To get the simulation running on your local machine, follow these simple steps.
### 1. Prerequisites
Ensure you have Python 3.7+ installed on your system.
### 2. Clone the Repository
```
git clone <your-repository-url>
cd <repository-directory>
```

### 3. Install Dependencies
Install the necessary Python libraries using pip.
```
pip install streamlit plotly pandas
```

### 4. Run the Streamlit App
Launch the application from your terminal.
```
streamlit run app.py
```

This command will start the Streamlit server and open the interactive simulation in your default web browser.

## ⚙️ Simulation Parameters
The sidebar in the web interface provides several parameters to control the simulation:
- **Number of Liquidity Providers (LPs):** Adjust the number of risk-averse agents providing liquidity to the market.
- **Number of Liquidity Takers (LTs):** Adjust the number of risk-loving agents taking liquidity from the market.
- **Simulation Steps:** Define the number of time steps the simulation will run for.
- **Initial Asset Price:** Set the starting price for the asset (e.g., S&P 500).
- **Market Impact Factor:** A coefficient that determines how much a market order's volume will move the market price.
- **Tick Size:** The minimum price increment for orders.

## 📊 Understanding the Results
The application displays several plots and data tables to help you analyze the simulation run:
- **Price Evolution:** A line chart showing the evolution of the mid-price over time.
- **Agent PnL:** A multi-line chart tracking the Profit and Loss (PnL) for each individual agent.
- **Executed Trades vs. Mid Price:** The price evolution chart overlaid with scatter points representing individual trades. The size of the point indicates the trade quantity, and the color indicates the trade side (green for buy, red for sell).
- **Final Limit Order Book State:** A snapshot of the top bid and ask levels at the end of the simulation.
- **Final Agent Status:** A summary table detailing the final cash, shares, portfolio value, and PnL for each agent.

## 📂 Code Structure
The simulation is contained within a single Python script (`app.py`), organized as follows:
- **Utility Functions & Constants:** Basic helper functions and configuration variables.
- **Order Book Implementation:** The `Order` and `LimitOrderBook` classes that define the market structure.
- **Agent Classes:** The `Agent` base class, along with `RiskAverseLP` and `RiskLoverLT` for specialized agent behaviors.
- **Simulation Orchestration:** The `run_simulation()` function, which initializes the LOB and agents and executes the main simulation loop. This function is cached by Streamlit for performance.
- **Streamlit UI:** The final section of the script, which uses `st.*` commands to build the interactive web dashboard.

## 💡 Future Work & Contributing
This simulation provides a solid foundation for exploring market microstructure. Contributions are welcome! Potential areas for expansion include:
- **More Sophisticated Agent Strategies:** Introduce new agent types with different behaviors (e.g., momentum traders, mean-reversion traders).
- **Exogenous Shocks:** Model the impact of external news events on market sentiment and price.
- **8Advanced Analytics:** Add more detailed metrics like slippage analysis, volatility calculations, and deeper PnL attribution.
- **Historical Data Integration:** Allow the simulation to be initialized with real historical market data.
Feel free to fork the repository, make your changes, and submit a pull request!
