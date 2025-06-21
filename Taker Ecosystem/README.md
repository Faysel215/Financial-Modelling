# Agent-Based Market Evolution Simulator
An interactive agent-based simulation built with Python and Streamlit to explore how different trading behaviors affect financial market stability.

## 🌍 Overview
This project simulates a financial market populated by an ecosystem of autonomous trading agents. It provides an interactive laboratory to test hypotheses about market dynamics. The central question explored is:
Does risk-aversion or risk-loving behavior lead to a more stable market over time?
The simulation features an evolutionary dynamic: underperforming agents are culled, and successful agents are replicated. This allows us to observe which trading strategies are most viable and how the overall "personality" of the market evolves.

## ✨ Features
- **Interactive UI:** A user-friendly control panel built with Streamlit allows for real-time parameter tuning.
- **Multiple Agent Types:**
    - **Prudent Traders:** Risk-averse agents who prioritize stability and low slippage.
    - **Aggressive Traders:** Risk-loving agents who thrive on momentum and high volatility.
    - **Random Traders:** Noise traders who provide a chaotic baseline for the market.
- **Realistic Market Impact:** Features a simplified Limit Order Book (LOB) where large trades directly impact the asset price by consuming liquidity.
- **Evolutionary Dynamics:** A "survival of thefittest" mechanic where successful agent strategies proliferate and unsuccessful ones are removed.
- **Rich Visualization:** Key metrics like population demographics, price history, volatility, and bid-ask spread are plotted in real-time.

## 🛠️ How It Works

The simulation is composed of three core components:
1. **The Agents:** Each agent type has a unique `take_action` method that defines its trading logic based on a view of the market (price, momentum, volatility, etc.).
2. **The Market (`LimitOrderBook`):** At each time step, a new LOB is seeded with liquidity from simulated "market makers." When agents place market orders, they consume this liquidity, and the code calculates the resulting execution price and market impact.
3. **The Simulation Engine:** This orchestrator manages the simulation loop. At each "tick," it prompts agents to act in a random order, executes their trades against the LOB, and logs the results. At the end of each "year" (a set number of ticks), it runs the `evolve()` function to update the agent population based on their performance.

## 🚀 Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites
You need to have Python 3.7+ installed.

#### Installation
1. Clone the repository to your local machine:
```
git clone www.github.com/faysel215/Financial-Models
cd www.github.com/faysel215/Financial-Models
```
2. Install the required Python packages. It's recommended to use a virtual environment.
```
pip install streamlit pandas numpy
```

### Usage
To run the application, navigate to the project directory in your terminal and execute the following command:
```
streamlit run market_simulation_app.py
```

Your web browser should automatically open a new tab with the running application.

## 🔬 Configuration
All simulation parameters can be configured directly from the sidebar in the user interface.
- **Population Setup:** Control the initial number of each agent type.
- **Financial Setup:** Set the starting cash and asset holdings for each agent.
- **Market & Simulation Setup:**
    - `Initial Bid-Ask Spread`: The initial gap between the best buy and sell prices.
    - `Market Maker Depth`: The number of order levels on each side of the LOB.
    - `Number of 'Years'`: The total number of evolutionary cycles to run.
    - `Evolutionary Pressure`: The percentage of the population to cull and reproduce at the end of each year.
Adjust these parameters and click "Run Simulation" to start a new experiment.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License
This project is licensed under the MIT License - see the LICENSE.md file for details.
