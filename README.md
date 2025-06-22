# Dynamic Liquidity Nexus (DLN)
A framework for high-fidelity financial market simulation, focusing on the complex dynamics of liquidity. This repository contains the official implementation of the Transformer-Modulated Hawkes Process (TMHP) and the Dynamic Liquidity Nexus (DLN) model, all within a comprehensive Agent-Based Market Evolution Simulator.
## Abstract
Modern financial markets are complex adaptive systems where the actions of heterogeneous agents give rise to emergent properties, with liquidity being one of the most crucial and least understood. This project introduces a novel framework for modeling and simulating market microstructure dynamics with a primary focus on liquidity. We leverage an Agent-Based Model (ABM) to simulate the interplay of diverse trading strategies. The core of our event-stream modeling is a Transformer-Modulated Hawkes Process (TMHP), which captures the long-range, self-exciting nature of market events with unprecedented sophistication. To manage the computational complexity of calibrating our ABM, we employ Gaussian Process Surrogates (GPS), enabling efficient exploration of the parameter space. The entire system is designed to study the Dynamic Liquidity Nexus (DLN), our term for the interconnected, time-varying nature of market liquidity.
### Core Components
This project is built upon several key pillars:
#### 1. Agent-Based Market Evolution Simulator
The foundation of our research is a powerful agent-based model. It simulates a realistic electronic market environment (e.g., a limit order book) populated by heterogeneous agents, each with distinct strategies, risk profiles, and information sets. This allows us to study market evolution from the "bottom-up."
#### 2. Transformer-Modulated Hawkes Process (TMHP)
To model the stream of market events (trades, quotes, cancellations), we move beyond traditional stochastic models. The TMHP is a significant innovation over the standard Marked Hawkes Process (MHP).
**Marked Hawkes Process (MHP):** Captures the self-exciting nature of events (i.e., trades beget more trades) and associates a "mark" (e.g., trade volume, price impact) with each event.
Transformer Modulation: We introduce a Transformer architecture to modulate the intensity function of the Hawkes process. The Transformer's self-attention mechanism allows the model to learn complex, long-range dependencies and conditionalities from the sequence of past events. This means the probability and nature of the next market event are conditioned on a rich, learned representation of market history, not just a simple exponential decay of influence.
#### 3. Gaussian Process Surrogates (GPS)
Calibrating the agent-based simulator to replicate real-world market statistics is a computationally intensive, high-dimensional problem. We use Gaussian Process Surrogates to create a fast, accurate approximation of the simulator's input-output map (i.e., agent parameters -> market statistics). This allows for efficient Bayesian optimization and sensitivity analysis, drastically reducing the time required for model calibration.
#### 4. Dynamic Liquidity Nexus (DLN)
The DLN is not a single model but the central concept our simulator is built to explore. It represents our view of liquidity as a multi-faceted, interconnected system encompassing:
- **Time:** Liquidity availability over different time horizons.
- **Price:** The cost of consuming liquidity across different depths of the order book.
- **Resilience:** The speed at which liquidity is replenished after being consumed.

## System Architecture
The components work in a synergistic loop, enabling a cycle of simulation, analysis, and calibration.
```
+------------------+      +-------------------------+      +-------------------+
|    Real Market   |----->|   TMHP & GPS Model      |----->|   Calibrated      |
|       Data       |      |       Calibration       |      | Agent Parameters  |
+------------------+      +-------------------------+      +--------+----------+
                                                                     |
                                                                     |
+------------------+      +-------------------------+      +--------v----------+
|  DLN Analysis &  |<-----|     Simulated Market    |<-----|  Agent-Based      |
|   Visualization  |      |   Data (Order Books)    |      |   Simulator       |
+------------------+      +-------------------------+      +-------------------+
```

## Repository Structure
```
├── Dynamic Liquidity Nexus (DLQ)/                
│   ├── Agentic LOB simulation/
│   │   ├── Outline.md
│   │   └── README.md
|   |   └── agentic_lob_sim.py
│   └── LOB simulation/
│   │   ├── Overview.md
│   │   └── README.md
|   |   └── lob_sim.py
├── Gaussian Process Surrogates (GPS)/
│   ├── README.md             
│   ├── gps_greeks_estim.ipynb      
├── Marked Hawkes process (MHP)/              
│   └── images/
│   │   ├── mhp-screenshot.png
│   │   └── thinning-algo_optim.png
│   ├── mhp_trading_sim.py 
│   ├── README.md             
│   ├── agentic_mhp_trading_sim.py
│   ├── mhp_thinning-algo_optim.ipynb   
├── Taker Ecosystem/                   
│   ├── README.md            
│   ├── LOB_sim_logic.md       
│   ├── img/            
│   │   ├── tmhp.py
│   │   └── gps.py
│   ├── lob_takers.py                        
├── Tranformer Modulated HAwkes process (MHP)
│   ├── img/            
│   │   ├── tmhp-dashboard.png
│   ├── README.md            
│   ├── Outline.md
│   ├── tmhp-app.py
```
