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
        self.bids.clear()
        self.asks.clear()

        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        # Create a ladder of orders around the current price
        for i in range(self.liquidity_depth):
            price_bid = best_bid - i
            # REFINEMENT: Increased liquidity to reduce market impact of trades.
            size_bid = random.randint(25, 75)
            self.bids.append((price_bid, size_bid))

            price_ask = best_ask + i
            # REFINEMENT: Increased liquidity to reduce market impact of trades.
            size_ask = random.randint(25, 75)
            self.asks.append((price_ask, size_ask))
            
        self.bids.sort(key=lambda x: x[0], reverse=True)
        self.asks.sort(key=lambda x: x[0])

    def get_best_bid(self):
        return self.bids[0][0] if self.bids else self.initial_price - self.spread / 2

    def get_best_ask(self):
        return self.asks[0][0] if self.asks else self.initial_price + self.spread / 2

    def get_market_depth(self):
        total_bid_size = sum(size for _, size in self.bids)
        total_ask_size = sum(size for _, size in self.asks)
        return total_bid_size, total_ask_size

    def execute_market_order(self, size, side):
        """Executes a market order, consuming liquidity from the book."""
        if side == 'buy':
            book = self.asks
        else: # sell
            book = self.bids

        if not book:
            return self.get_best_ask() if side == 'buy' else self.get_best_bid(), 0

        filled_size = 0
        total_cost = 0

        for i, (price, order_size) in enumerate(book):
            if filled_size >= size:
                break
            
            fillable_size = min(size - filled_size, order_size)
            filled_size += fillable_size
            total_cost += fillable_size * price
            book[i] = (price, order_size - fillable_size)
        
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
        self.initial_wealth = cash + assets * 100 

    def get_wealth(self, current_price):
        return self.cash + self.assets * current_price

    def take_action(self, market_view):
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
        # REFINEMENT: Relaxed conditions to increase trading frequency.
        volatility = market_view['volatility']
        bid_depth, ask_depth = market_view['depth']

        if volatility > 3.5 or (bid_depth < 50 and ask_depth < 50):
            return None # Hold if market is too unstable or illiquid

        # REFINEMENT: Lowered momentum threshold to trade more often on perceived trends.
        if market_view['momentum'] > 0.3:
             return {'side': 'buy', 'size': random.randint(1, 4)}
        elif market_view['momentum'] < -0.3:
            return {'side': 'sell', 'size': random.randint(1, 4)}
        
        # REFINEMENT: Added a small chance to make a random trade to ensure activity.
        if random.random() < 0.1: # 10% chance
            return {'side': random.choice(['buy', 'sell']), 'size': 1}

        return None

class AggressiveTrader(Agent):
    """A risk-loving agent. Tries to capitalize on momentum."""
    def take_action(self, market_view):
        momentum = market_view['momentum']

        # REFINEMENT: Lowered momentum threshold to increase trading frequency.
        if momentum > 0.6:
            # REFINEMENT: Adjusted trade size to be impactful but not market-breaking.
            size = random.randint(8, 20)
            return {'side': 'buy', 'size': size}
        elif momentum < -0.6:
            size = random.randint(8, 20)
            return {'side': 'sell', 'size': size}
        
        # REFINEMENT: Added a small chance to trade to increase activity.
        if random.random() < 0.15: # 15% chance
            return {'side': random.choice(['buy', 'sell']), 'size': random.randint(2, 5)}
        
        return None


# 3. The Orchestrator: Simulation Class
class Simulation:
    """Manages the entire simulation process, state, and data logging."""
    def __init__(self, params):
        self.params = params
        self.market = LimitOrderBook(params['initial_price'], params['spread'], params['liquidity_depth'])
        self.agents = self._create_population()
        self.history = []
        self.price_history = deque(maxlen=50) 

    def _create_population(self):
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
        self.market._seed_book()
        random.shuffle(self.agents)
        current_price = (self.market.get_best_bid() + self.market.get_best_ask()) / 2
        self.price_history.append(current_price)
        
        price_series = pd.Series(self.price_history)
        market_view = {
            'price': current_price,
            'volatility': price_series.rolling(window=20).std().iloc[-1] if len(price_series) > 20 else 0,
            'momentum': price_series.diff().rolling(window=10).mean().iloc[-1] if len(price_series) > 10 else 0,
            'depth': self.market.get_market_depth()
        }

        for agent in self.agents:
            action = agent.take_action(market_view)
            if action:
                price, cost = self.market.execute_market_order(action['size'], action['side'])
                if action['side'] == 'buy':
                    agent.cash -= cost
                    agent.assets += action['size']
                else: 
                    agent.cash += cost
                    agent.assets -= action['size']

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
        self.agents.sort(key=lambda a: a.get_wealth(current_price), reverse=True)
        num_to_evolve = int(len(self.agents) * self.params['evolution_pressure'])
        survivors = self.agents[:-num_to_evolve]
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
                progress_val = (total_ticks + 1) / (num_years * ticks_per_year)
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

st.sidebar.title("🔬 Experiment Controls")

st.sidebar.header("Population Setup")
params = {}
params['prudent_traders'] = st.sidebar.slider("Number of Prudent Traders", 1, 50, 10)
params['aggressive_traders'] = st.sidebar.slider("Number of Aggressive Traders", 1, 50, 10)
params['random_traders'] = st.sidebar.slider("Number of Random Traders (Noise)", 1, 200, 80)

st.sidebar.header("Financial Setup")
params['initial_capital'] = st.sidebar.number_input("Initial Agent Capital ($)", 1000, 100000, 10000)
params['initial_assets'] = st.sidebar.number_input("Initial Agent Assets (units)", 10, 1000, 100)
params['initial_price'] = 100.0

st.sidebar.header("Market & Simulation Setup")
params['spread'] = st.sidebar.slider("Initial Bid-Ask Spread ($)", 0.1, 5.0, 0.5, 0.1)
params['liquidity_depth'] = st.sidebar.slider("Market Maker Depth (orders)", 5, 50, 15)
params['num_years'] = st.sidebar.slider("Number of 'Years' to Simulate", 1, 50, 10)
params['ticks_per_year'] = 252
params['evolution_pressure'] = st.sidebar.slider("Evolutionary Pressure (%)", 1, 50, 20) / 100.0

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
    final_spread = results_df['bid_ask_spread'].iloc[-1]
    final_volatility = results_df['volatility'].iloc[-1]
    final_depth = results_df['market_depth'].iloc[-1]
    col1.metric("Final Bid-Ask Spread", f"${final_spread:.2f}")
    col2.metric("Final Volatility (Std. Dev)", f"{final_volatility:.2f}")
    col3.metric("Final Market Depth", f"{int(final_depth)} units")
    st.line_chart(results_df.set_index('tick')[['price']], use_container_width=True)
    st.line_chart(results_df.set_index('tick')[['volatility', 'bid_ask_spread']], use_container_width=True)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to begin.")

