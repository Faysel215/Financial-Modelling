# 🤖 Dynamic Liquidity Nexus: An Agentic AI LOB Simulation
Welcome to the agentic version of the Dynamic Liquidity Nexus, a simulation of a financial market's Limit Order Book (LOB). This project moves beyond hard-coded logic and demonstrates how modern AI agents, driven by natural language instructions, can simulate complex market behaviors.
This implementation is conceptually powered by Google's Agent Development Kit (ADK) and a Gemini-family model. Instead of defining agent behavior in rigid if/else statements, we provide them with instructions in plain English, and they decide which actions to take by calling a set of available "tools."

## ✨ Key Features
Instruction-Driven Agents: Agent behavior is not coded, but instructed. Modifying an agent's strategy is as simple as changing its prompt.
Agentic AI Framework: Conceptually built on Google's ADK, showcasing a modern, flexible approach to building autonomous systems.
Tools on Top of a Simulation: The core market mechanics (placing limit or market orders) are exposed as tools that the AI agents can choose to use.
Interactive Dashboard: A user-friendly interface built with Streamlit allows you to configure and launch the simulation and visualize the results.

## ⚠️ A Note on the "Conceptual" Implementation
This simulation runs in an environment where external libraries like google-adk cannot be installed and run. Therefore, the agent's decision-making process is simulated.
What this means:
The code correctly formulates the instructions and the state-aware prompts that would be sent to the Gemini model at each step.
Instead of making a real API call, the simulation prints the intended prompt to your console.
The agents do not actually execute trades on the order book. The price chart evolves based on a simple random walk for demonstration purposes.
This approach allows you to inspect the core logic of an agentic system (prompt engineering, state management, tool definition) even without a live LLM connection.

## 🚀 How to Run the Simulation
To get the simulation running on your local machine, follow these steps.
1. Prerequisites
Ensure you have Python 3.7+ installed on your system.
2. Clone the Repository
```
git clone www.github.com/faysel215/Financial-Models
cd Financial-Models
```

3. Install Dependencies
Install the necessary Python libraries. You do not need to install google-adk to run this conceptual demo.
```
pip install streamlit plotly pandas
```

4. Run the Streamlit App
Launch the application from your terminal.
```
streamlit run app.py
```

This will open the interactive simulation in your web browser. Open the terminal where you launched the command to see the simulated agent prompts being printed in real-time.

## ⚙️ How It Works
Agent Definition: Two types of agents (LiquidityProvider and LiquidityTaker) are created. Their entire strategy is defined in a multi-line string called an instruction.
Tool Definition: Functions like place_limit_order and place_market_order are defined and marked with a @tool decorator. This makes them available actions for the agents.
Simulation Loop: At each time step:
a. The simulation gathers the current state of the market (prices, LOB depth) and the agent's internal state (cash, shares).
b. It formats this information into a prompt.
c. It simulates calling the agent with this prompt. In a real implementation, the ADK would send this prompt to the Gemini model.
d. The model would then reason based on its instructions and choose to call one of the available tools (e.g., place_limit_order).
e. The simulation would execute the tool's return action. (Note: This final step is not implemented in the conceptual demo).

## 📂 Code Structure
The simulation is contained within a single Python script (app.py), organized as follows:
ADK Placeholders: Mock classes for ADKAgent and tool to allow the code to be syntactically correct without the library.
Core Market Mechanics: The Order and LimitOrderBook classes.
Agent Tools: The @tool-decorated functions that wrap the LimitOrderBook methods.
Agent Instructions: The multi-line string constants (LP_INSTRUCTION, LT_INSTRUCTION) that define agent behavior.
Simulation Orchestration: The run_simulation() function containing the main loop.
Streamlit UI: The final section that builds the interactive web dashboard.

## 💡 Future Work & Full Implementation
To turn this conceptual demo into a fully functional agentic simulation, the following steps would be required:
Full ADK Integration: Run the code in an environment where google-adk can be installed and an API key for the Gemini API is available.
Asynchronous Execution: The agent.run(prompt) call should be made asynchronously to avoid blocking the simulation while waiting for the LLM.
State Updates from Tool Results: The agent's portfolio (cash, shares) and the LOB must be updated based on the actual results returned by the tool calls.
Richer Tools and Instructions: More complex tools (e.g., get_news_sentiment, analyze_chart_patterns) and more nuanced instructions could be developed to create incredibly sophisticated agent behaviors.
