### **Project Outline: Agentic AI LOB Simulation**

1.  **Setup and Configuration**
    * **Imports:** Import standard libraries (`streamlit`, `pandas`, `plotly`, `random`, etc.).
    * **ADK Placeholders:** A `try...except` block to handle the `google-adk` import, with placeholder classes (`ADKAgent`, `tool`) if the library isn't installed. This ensures the script is runnable for demonstration.
    * **Global Constants:** Define default simulation parameters like `INITIAL_PRICE`, `TICK_SIZE`, `GEMINI_MODEL`, etc.

2.  **Core Market Mechanics**
    * **`Order` Class:** A simple data class to represent a single order (ID, agent, side, price, quantity).
    * **`LimitOrderBook` (LOB) Class:**
        * Manages the `bids` and `asks` using `OrderedDict`.
        * Contains methods to `add_order`, `cancel_order`, and `process_market_order`.
        * This class represents the central state of the market.

3.  **Agentic AI Framework Integration (ADK Concept)**
    * **Tool Definition:**
        * The LOB's functionalities are exposed to AI agents as "tools."
        * `@tool` decorator is used on functions like `place_limit_order` and `place_market_order`.
        * These tools directly manipulate a single, global `LOB_INSTANCE`.
    * **Agent Definition via Instructions:**
        * Agent logic is abstracted away from Python code into natural language prompts.
        * `LP_INSTRUCTION`: A detailed string describing the strategy for a Risk-Averse Liquidity Provider.
        * `LT_INSTRUCTION`: A string describing the strategy for a Risk-Loving Liquidity Taker.

4.  **Simulation Orchestration (`run_simulation` function)**
    * **Initialization:**
        * Clears and re-initializes the `LOB_INSTANCE` for a fresh run.
        * Instantiates `ADKAgent` objects, passing the corresponding instructions (`LP_INSTRUCTION` or `LT_INSTRUCTION`).
        * Creates simple `AgentPortfolio` objects to hold the state (cash, shares) for each agent externally.
    * **Main Loop (per time step):**
        * Iterates through the simulation steps.
        * For each agent, it dynamically constructs a detailed **prompt** containing:
            * The agent's current portfolio state (cash, shares).
            * The current market state (mid-price, best bid/ask).
        * **Conceptual ADK Call:** It simulates calling `agent.run(prompt)`, which in a real implementation would have the Gemini model interpret the prompt and choose which tool to execute.
    * **State Updates:** In the conceptual model, this part is simplified to a random price walk, as actual tool execution is not possible.

5.  **User Interface (Streamlit)**
    * **Page Configuration:** Sets the title and layout.
    * **Informational Text:** Includes Markdown text explaining the agentic AI concept and the simulation's limitations.
    * **Sidebar Controls:** Uses `st.slider` and other widgets to allow users to configure `num_lps`, `num_lts`, etc.
    * **Main Panel:**
        * A "Run Simulation" button triggers the `run_simulation` function.
        * Uses `st.plotly_chart` to display the results (e.g., price evolution).
        * Displays the final state of the LOB and a summary of agent status.