import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import deque

# --- Core Simulation Classes ---

# 1. The Market: Limit Order Book (LOB)
class LimitOrderBook:
    """
    A simplified Limit Order Book to simulate market dynamics.
    - Bids and Asks are lists of (price, size) tuples.
    - Market makers provide background liquidity.
    """
    def __init__(self, initial_price, spread, liquidity_depth):
        self.initial_price = initial_price
        self.spread = spread
        self.liquidity_depth = liquidity_depth
        self.bids = []  # List of (price, size)
        self.asks = []  # List of (price, size)
        self._seed_book()

    def _seed_book(self):
        """Initializes the LOB with liquidity from background market makers."""
        # Clear existing orders
        self.bids.clear()
        self.asks.clear()

        # Get best bid and ask
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        # Create a ladder of orders around the current price
        for i in range(self.liquidity_depth):
            # Bids descending from the best bid
            price_bid = best_bid - i
            size_bid = random.randint(5, 20)
            self.bids.append((price_bid, size_bid))

            # Asks ascending from the best ask
            price_ask = best_ask + i
            size_ask = random.randint(5, 20)
            self.asks.append((price_ask, size_ask))
            
        self.bids.sort(key=lambda x: x[0], reverse=True)
        self.asks.sort(key=lambda x: x[0])

    def get_best_bid(self):
        """Returns the highest price a buyer is willing to pay."""
        return self.bids[0][0] if self.bids else self.initial_price - self.spread / 2

    def get_best_ask(self):
        """Returns the lowest price a seller is willing to accept."""
        return self.asks[0][0] if self.asks else self.initial_price + self.spread / 2

    def get_market_depth(self):
        """Calculates the total size of all bids and asks."""
        total_bid_size = sum(size for _, size in self.bids)
        total_ask_size = sum(size for _, size in self.asks)
        return total_bid_size, total_ask_size

    def execute_market_order(self, size, side):
        """
        Executes a market order, consuming liquidity from the book.
        This is where market impact happens.
        Returns the average execution price and total cost/revenue.
        """
        if side == 'buy':
            book = self.asks
            sign = 1
        else: # sell
            book = self.bids
            sign = -1

        if not book:
            return self.get_best_ask() if side == 'buy' else self.get_best_bid(), 0

        filled_size = 0
        total_cost = 0
        
        orders_to_remove = []

        for i, (price, order_size) in enumerate(book):
            if filled_size >= size:
                break
            
            fillable_size = min(size - filled_size, order_size)
            filled_size += fillable_size
            total_cost += fillable_size * price
            
            # Update order size
            book[i] = (price, order_size - fillable_size)
        
        # Remove orders that have been fully filled
        book[:] = [order for order in book if order[1] > 0]
        
        if filled_size == 0:
            avg_price = book[0][0] if book else self.initial_price
            return avg_price, 0

        avg_price = total_cost / filled_size
        return avg_price, total_cost


# 2. The Players: Agent Classes
class Agent:
    """Base class for all trading agents."""
    def __init__(self, agent_id, cash, assets):
        self.agent_id = agent_id
        self.cash = cash
        self.assets = assets
        self.initial_wealth = cash + assets * 100 # Approx initial wealth

    def get_wealth(self, current_price):
        """Calculates the current total wealth of the agent."""
        return self.cash + self.assets * current_price

    def take_action(self, market_view):
        """Abstract method for agent's decision-making logic."""
        raise NotImplementedError

class RandomTrader(Agent):
    """Makes random buy or sell decisions."""
    def take_action(self, market_view):
        decision = random.choice(['buy', 'sell', 'hold'])
        if decision == 'hold':
            return None
        size = random.randint(1, 5)
        return {'side': decision, 'size': size}

class PrudentTrader(Agent):
    """A risk-averse agent. Prefers stability and low slippage."""
    def take_action(self, market_view):
        # This agent is sensitive to volatility and book depth.
        volatility = market_view['volatility']
        bid_depth, ask_depth = market_view['depth']

        # If market is too volatile or the book is too thin, it holds.
        if volatility > 2.0 or (bid_depth < 20 and ask_depth < 20):
            return None

        # Prefers small, low-impact trades
        size = random.randint(1, 3)
        
        # Simple trend-following, but cautiously
        if market_view['momentum'] > 0.5:
             return {'side': 'buy', 'size': size}
        elif market_view['momentum'] < -0.5:
            return {'side': 'sell', 'size': size}
        return None

class AggressiveTrader(Agent):
    """A risk-loving agent. Tries to capitalize on momentum."""
    def take_action(self, market_view):
        # This agent loves momentum and is less scared of volatility.
        momentum = market_view['momentum']

        # If it sees strong momentum, it makes a large trade to accelerate it.
        if momentum > 1.0:
            size = random.randint(10, 25)
            return {'side': 'buy', 'size': size}
        elif momentum < -1.0:
            size = random.randint(10, 25)
            return {'side': 'sell', 'size': size}
        
        return None


# 3. The Orchestrator: Simulation Class
class Simulation:
    """Manages the entire simulation process, state, and data logging."""
    def __init__(self, params):
        self.params = params
        self.market = LimitOrderBook(params['initial_price'], params['spread'], params['liquidity_depth'])
        self.agents = self._create_population()
        self.history = []
        self.price_history = deque(maxlen=50) # For calculating rolling metrics

    def _create_population(self):
        """Initializes the agent population based on parameters."""
        agents = []
        p = self.params
        for i in range(p['prudent_traders']):
            agents.append(PrudentTrader(f'prudent_{i}', p['initial_capital'], p['initial_assets']))
        for i in range(p['aggressive_traders']):
            agents.append(AggressiveTrader(f'aggressive_{i}', p['initial_capital'], p['initial_assets']))
        for i in range(p['random_traders']):
            agents.append(RandomTrader(f'random_{i}', p['initial_capital'], p['initial_assets']))
        return agents

    def run_tick(self, tick):
        """Runs a single time-step (tick) of the simulation."""
        # 1. Market makers provide liquidity
        self.market._seed_book()

        # 2. Agents act in random order to avoid bias
        random.shuffle(self.agents)

        current_price = (self.market.get_best_bid() + self.market.get_best_ask()) / 2
        self.price_history.append(current_price)
        
        # 3. Create a market view for agents
        price_series = pd.Series(self.price_history)
        market_view = {
            'price': current_price,
            'volatility': price_series.rolling(window=20).std().iloc[-1] if len(price_series) > 20 else 0,
            'momentum': price_series.diff().rolling(window=10).mean().iloc[-1] if len(price_series) > 10 else 0,
            'depth': self.market.get_market_depth()
        }

        # 4. Agents make decisions
        for agent in self.agents:
            action = agent.take_action(market_view)
            if action:
                # 5. Execute trades and update agent state
                price, cost = self.market.execute_market_order(action['size'], action['side'])
                if action['side'] == 'buy':
                    agent.cash -= cost
                    agent.assets += action['size']
                else: # sell
                    agent.cash += cost
                    agent.assets -= action['size']

        # 6. Log data for this tick
        bid_depth, ask_depth = self.market.get_market_depth()
        self.history.append({
            'tick': tick,
            'price': current_price,
            'bid_ask_spread': self.market.get_best_ask() - self.market.get_best_bid(),
            'volatility': market_view['volatility'],
            'market_depth': bid_depth + ask_depth,
            'prudent_count': sum(1 for a in self.agents if isinstance(a, PrudentTrader)),
            'aggressive_count': sum(1 for a in self.agents if isinstance(a, AggressiveTrader)),
            'random_count': sum(1 for a in self.agents if isinstance(a, RandomTrader)),
        })


    def evolve(self):
        """Evolves the population based on performance (wealth)."""
        current_price = (self.market.get_best_bid() + self.market.get_best_ask()) / 2
        
        # Rank agents by final wealth
        self.agents.sort(key=lambda a: a.get_wealth(current_price), reverse=True)
        
        num_to_evolve = int(len(self.agents) * self.params['evolution_pressure'])
        
        # Remove the worst performers
        survivors = self.agents[:-num_to_evolve]
        
        # Reproduce the best performers
        top_performers = self.agents[:num_to_evolve]
        new_agents = []
        for agent_to_clone in top_performers:
            new_agent = type(agent_to_clone)(
                f"clone_{agent_to_clone.agent_id}", 
                self.params['initial_capital'], 
                self.params['initial_assets']
            )
            new_agents.append(new_agent)

        self.agents = survivors + new_agents


    def run(self):
        """Main simulation runner."""
        num_years = self.params['num_years']
        ticks_per_year = self.params['ticks_per_year']

        progress_bar = st.progress(0)
        status_text = st.empty()

        for year in range(num_years):
            for tick in range(ticks_per_year):
                total_ticks = year * ticks_per_year + tick
                self.run_tick(total_ticks)
                
                # Update progress bar
                progress_val = total_ticks / (num_years * ticks_per_year)
                progress_bar.progress(progress_val)
                
            status_text.text(f"Year {year + 1}/{num_years} complete. Evolving population...")
            self.evolve()
        
        status_text.text("Simulation complete!")
        progress_bar.empty()
        return pd.DataFrame(self.history)


# --- Streamlit User Interface ---

st.set_page_config(layout="wide", page_title="Market Evolution Simulation")

st.title("🌍 Ecosystem of Takers: Market Stability Simulation")
st.markdown("""
This application simulates a financial market populated by different types of trading agents. 
Use the controls in the sidebar to configure the experiment. Press **Run Simulation** to see how the market and its population evolve over time.
The core question: **Does risk-aversion or risk-loving behavior lead to more stable markets?**
""")

# --- Sidebar Controls ---
st.sidebar.title("🔬 Experiment Controls")

st.sidebar.header("Population Setup")
params = {}
params['prudent_traders'] = st.sidebar.slider("Number of Prudent Traders", 1, 50, 10)
params['aggressive_traders'] = st.sidebar.slider("Number of Aggressive Traders", 1, 50, 10)
params['random_traders'] = st.sidebar.slider("Number of Random Traders (Noise)", 1, 200, 80)

st.sidebar.header("Financial Setup")
params['initial_capital'] = st.sidebar.number_input("Initial Agent Capital ($)", 1000, 100000, 10000)
params['initial_assets'] = st.sidebar.number_input("Initial Agent Assets (units)", 10, 1000, 100)
params['initial_price'] = 100.0 # Not user configurable for simplicity

st.sidebar.header("Market & Simulation Setup")
params['spread'] = st.sidebar.slider("Initial Bid-Ask Spread ($)", 0.1, 5.0, 1.0, 0.1)
params['liquidity_depth'] = st.sidebar.slider("Market Maker Depth (orders)", 5, 50, 10)
params['num_years'] = st.sidebar.slider("Number of 'Years' to Simulate", 1, 50, 10)
params['ticks_per_year'] = 252 # Standard trading days
params['evolution_pressure'] = st.sidebar.slider("Evolutionary Pressure (%)", 1, 50, 20) / 100.0

# --- Main Dashboard Logic ---
if st.sidebar.button("🚀 Run Simulation"):
    with st.spinner("The market is evolving... Please wait."):
        sim = Simulation(params)
        results_df = sim.run()

    st.success("Simulation Complete!")

    st.subheader("📈 Population Demographics Over Time")
    st.markdown("This chart shows which agent strategies are succeeding and dominating the population.")
    
    pop_df = results_df[['tick', 'prudent_count', 'aggressive_count', 'random_count']].set_index('tick')
    st.area_chart(pop_df)
    
    st.subheader("⚖️ Market Stability Metrics")
    col1, col2, col3 = st.columns(3)
    
    # Final values for metrics
    final_spread = results_df['bid_ask_spread'].iloc[-1]
    final_volatility = results_df['volatility'].iloc[-1]
    final_depth = results_df['market_depth'].iloc[-1]

    col1.metric("Final Bid-Ask Spread", f"${final_spread:.2f}")
    col2.metric("Final Volatility (Std. Dev)", f"{final_volatility:.2f}")
    col3.metric("Final Market Depth", f"{int(final_depth)} units")

    # Time series charts
    st.line_chart(results_df.set_index('tick')[['price']], use_container_width=True)
    st.line_chart(results_df.set_index('tick')[['volatility', 'bid_ask_spread']], use_container_width=True)

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to begin.")

