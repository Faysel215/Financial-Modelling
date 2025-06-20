import collections
import random
import time
import math
import streamlit as st
import plotly.express as px
import pandas as pd

# --- Constants and Configuration (can be overridden by Streamlit) ---
# Global variables for simplicity with Streamlit, but generally prefer passing as parameters
DEFAULT_TICK_SIZE = 0.01
DEFAULT_INITIAL_PRICE = 4500.00
INITIAL_CASH = 1000000
MAX_ORDER_SIZE = 100
MIN_ORDER_SIZE = 10
DEFAULT_SIMULATION_STEPS = 500 # Reduced for quicker UI interaction
DEFAULT_MARKET_IMPACT_FACTOR = 0.0001

# --- Utility Functions ---
def get_mid_price(order_book, default_price):
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    if best_bid is None and best_ask is None:
        return default_price # Fallback to a default if LOB is empty
    elif best_bid is None:
        return best_ask.price
    elif best_ask is None:
        return best_bid.price
    else:
        return (best_bid.price + best_ask.price) / 2

# --- Order Book Implementation (Same as before) ---
class Order:
    def __init__(self, order_id, agent_id, side, price, quantity, timestamp):
        self.order_id = order_id
        self.agent_id = agent_id
        self.side = side  # 'buy' or 'sell'
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp

    def __repr__(self):
        return f"Order(ID:{self.order_id}, Agent:{self.agent_id}, {self.side.upper()} {self.quantity}@{self.price})"

class LimitOrderBook:
    def __init__(self):
        self.bids = collections.OrderedDict()
        self.asks = collections.OrderedDict()
        self.next_order_id = 0
        self.trades = [] # To record executed trades

    def add_order(self, order):
        self.next_order_id += 1
        order.order_id = self.next_order_id
        
        if order.side == 'buy':
            if order.price not in self.bids:
                self.bids[order.price] = []
            self.bids[order.price].append(order)
            self.bids = collections.OrderedDict(sorted(self.bids.items(), reverse=True))
        else: # sell
            if order.price not in self.asks:
                self.asks[order.price] = []
            self.asks[order.price].append(order)
            self.asks = collections.OrderedDict(sorted(self.asks.items()))
        return order.order_id

    def cancel_order(self, order_id):
        for price, orders in list(self.bids.items()):
            for order in orders:
                if order.order_id == order_id:
                    orders.remove(order)
                    if not orders:
                        del self.bids[price]
                    return True
        for price, orders in list(self.asks.items()):
            for order in orders:
                if order.order_id == order_id:
                    orders.remove(order)
                    if not orders:
                        del self.asks[price]
                    return True
        return False

    def process_market_order(self, agent_id, side, quantity, current_time, market_impact_factor, current_market_price_ref):
        executed_quantity = 0
        executed_value = 0
        
        if side == 'buy':
            book = self.asks
        else: # sell
            book = self.bids

        # Capture the price before execution for slippage calculation
        original_best_price = self.get_best_ask().price if side == 'buy' and self.get_best_ask() else \
                              (self.get_best_bid().price if side == 'sell' and self.get_best_bid() else current_market_price_ref)
        
        prices_to_remove = []
        for price, orders_at_price in list(book.items()):
            if executed_quantity >= quantity:
                break
            
            for order_in_book in list(orders_at_price):
                if executed_quantity >= quantity:
                    break
                
                trade_qty = min(quantity - executed_quantity, order_in_book.quantity)
                
                self.trades.append({
                    'buyer_id': agent_id if side == 'buy' else order_in_book.agent_id,
                    'seller_id': order_in_book.agent_id if side == 'buy' else agent_id,
                    'price': order_in_book.price,
                    'quantity': trade_qty,
                    'time': current_time,
                    'type': 'buy' if side == 'buy' else 'sell'
                })

                executed_quantity += trade_qty
                executed_value += trade_qty * order_in_book.price
                order_in_book.quantity -= trade_qty

                if order_in_book.quantity == 0:
                    orders_at_price.remove(order_in_book)
            
            if not orders_at_price:
                prices_to_remove.append(price)
        
        for price in prices_to_remove:
            del book[price]

        price_impact = 0
        if executed_quantity > 0:
            avg_executed_price = executed_value / executed_quantity
            # Calculate price impact on the market
            price_impact = (market_impact_factor * executed_quantity) * (1 if side == 'buy' else -1)
            
        return executed_quantity, executed_value, price_impact

    def get_best_bid(self):
        if not self.bids:
            return None
        return self.bids[max(self.bids.keys())][0]

    def get_best_ask(self):
        if not self.asks:
            return None
        return self.asks[min(self.asks.keys())][0]

    def display_simple(self):
        display_str = "--- Limit Order Book ---\n"
        display_str += "Asks:\n"
        ask_prices = sorted(self.asks.keys(), reverse=True)
        for price in ask_prices[:5]: # Show top 5 levels
            total_qty = sum(order.quantity for order in self.asks[price])
            display_str += f"  {total_qty} @ {price:.2f}\n"
        
        display_str += "------------------------\n"
        display_str += "Bids:\n"
        bid_prices = sorted(self.bids.keys(), reverse=True)
        for price in bid_prices[:5]: # Show top 5 levels
            total_qty = sum(order.quantity for order in self.bids[price])
            display_str += f"  {total_qty} @ {price:.2f}\n"
        display_str += "------------------------"
        return display_str
        
# --- Agent Classes (Slight modifications for passing parameters) ---
class Agent:
    def __init__(self, agent_id, cash, initial_shares, initial_price):
        self.agent_id = agent_id
        self.cash = cash
        self.shares = initial_shares
        self.open_orders = {}
        self.pnl_history = []
        self.portfolio_value_history = []
        self.initial_cash_start = cash # Store initial cash for PnL base

    def update_portfolio_value(self, current_price):
        self.portfolio_value = self.cash + (self.shares * current_price)
        self.portfolio_value_history.append(self.portfolio_value)

    def calculate_pnl(self, current_price):
        current_pnl = self.cash + (self.shares * current_price) - self.initial_cash_start
        self.pnl_history.append(current_pnl)
        return current_pnl

class RiskAverseLP(Agent):
    def __init__(self, agent_id, cash, initial_shares, initial_price, risk_aversion_factor, tick_size):
        super().__init__(agent_id, cash, initial_shares, initial_price)
        self.risk_aversion_factor = risk_aversion_factor
        self.tick_size = tick_size
        self.inventory_target = 0
        self.last_traded_price = initial_price

    def utility(self, wealth):
        if wealth <= 0: return -float('inf')
        return (wealth**(1 - self.risk_aversion_factor)) / (1 - self.risk_aversion_factor)

    def decide(self, lob, current_time, current_market_price_ref):
        mid_price = get_mid_price(lob, current_market_price_ref)
        
        orders_to_cancel = []
        for order_id, order in list(self.open_orders.items()):
            if abs(order.price - mid_price) > (2 * self.tick_size):
                orders_to_cancel.append(order_id)
            elif self.shares > self.inventory_target + MAX_ORDER_SIZE and order.side == 'buy':
                orders_to_cancel.append(order_id)
            elif self.shares < self.inventory_target - MAX_ORDER_SIZE and order.side == 'sell':
                orders_to_cancel.append(order_id)
                
        for order_id in orders_to_cancel:
            lob.cancel_order(order_id)
            if order_id in self.open_orders:
                del self.open_orders[order_id]

        spread = self.tick_size * (2 + (random.random() * self.risk_aversion_factor))
        
        bid_price = round(mid_price - spread / 2, 2)
        ask_price = round(mid_price + spread / 2, 2)

        bid_price = round(bid_price / self.tick_size) * self.tick_size
        ask_price = round(ask_price / self.tick_size) * self.tick_size

        qty = random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE)
        
        if self.shares < self.inventory_target + MAX_ORDER_SIZE and self.cash >= bid_price * qty:
            order = Order(None, self.agent_id, 'buy', bid_price, qty, current_time)
            order_id = lob.add_order(order)
            self.open_orders[order_id] = order

        if self.shares > self.inventory_target - MAX_ORDER_SIZE and self.shares >= qty:
            order = Order(None, self.agent_id, 'sell', ask_price, qty, current_time)
            order_id = lob.add_order(order)
            self.open_orders[order_id] = order

    def process_trade(self, side, price, quantity):
        if side == 'buy':
            self.cash -= price * quantity
            self.shares += quantity
        else: # sell
            self.cash += price * quantity
            self.shares -= quantity
        self.last_traded_price = price
        
class RiskLoverLT(Agent):
    def __init__(self, agent_id, cash, initial_shares, initial_price, risk_love_factor, tick_size):
        super().__init__(agent_id, cash, initial_shares, initial_price)
        self.risk_love_factor = risk_love_factor
        self.tick_size = tick_size
        self.position = 0
        self.target_position = 0
        self.last_trade_price = initial_price

    def utility(self, wealth):
        if wealth <= 0: return -float('inf')
        return wealth**(1 + self.risk_love_factor)

    def decide(self, lob, current_time, market_impact_factor, current_market_price_ref):
        mid_price = get_mid_price(lob, current_market_price_ref)
        
        if abs(mid_price - self.last_trade_price) < self.tick_size * 2:
            return

        qty = random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE)
        if random.random() < 0.2:
            qty = random.randint(MAX_ORDER_SIZE, MAX_ORDER_SIZE * 2)

        if mid_price > self.last_trade_price:
            if self.position < MAX_ORDER_SIZE * 2 and self.cash >= mid_price * qty:
                executed_qty, executed_value, price_impact = lob.process_market_order(self.agent_id, 'buy', qty, current_time, market_impact_factor, current_market_price_ref)
                if executed_qty > 0:
                    self.process_trade('buy', executed_value / executed_qty, executed_qty)
                    # Return price_impact to the simulation loop to update the global market price
                    return price_impact

        elif mid_price < self.last_trade_price:
            if self.position > -MAX_ORDER_SIZE * 2 and self.shares >= qty:
                executed_qty, executed_value, price_impact = lob.process_market_order(self.agent_id, 'sell', qty, current_time, market_impact_factor, current_market_price_ref)
                if executed_qty > 0:
                    self.process_trade('sell', executed_value / executed_qty, executed_qty)
                    # Return price_impact to the simulation loop
                    return price_impact

        if self.position != self.target_position and random.random() < 0.1:
            rebalance_qty = abs(self.position - self.target_position)
            if self.position > self.target_position:
                if self.shares > 0:
                    sell_qty = min(rebalance_qty, self.shares, random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE))
                    executed_qty, executed_value, price_impact = lob.process_market_order(self.agent_id, 'sell', sell_qty, current_time, market_impact_factor, current_market_price_ref)
                    if executed_qty > 0:
                        self.process_trade('sell', executed_value / executed_qty, executed_qty)
                        return price_impact
            else:
                if self.cash > 0:
                    buy_qty = min(rebalance_qty, int(self.cash / mid_price), random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE))
                    if buy_qty > 0:
                        executed_qty, executed_value, price_impact = lob.process_market_order(self.agent_id, 'buy', buy_qty, current_time, market_impact_factor, current_market_price_ref)
                        if executed_qty > 0:
                            self.process_trade('buy', executed_value / executed_qty, executed_qty)
                            return price_impact
        return 0 # No price impact if no market order executed

    def process_trade(self, side, price, quantity):
        if side == 'buy':
            self.cash -= price * quantity
            self.shares += quantity
            self.position += quantity
        else: # sell
            self.cash += price * quantity
            self.shares -= quantity
            self.position -= quantity
        self.last_trade_price = price

# --- Simulation Orchestration (Wrapped for Streamlit) ---
@st.cache_data(show_spinner=True) # Cache results to avoid re-running if params unchanged
def run_simulation(num_lps, num_lts, num_steps, initial_price_param, market_impact_factor_param, tick_size_param):
    lob = LimitOrderBook()
    agents = []
    
    current_market_price = initial_price_param # This will be the dynamic reference price

    # Create agents
    for i in range(num_lps):
        agents.append(RiskAverseLP(f"LP-{i+1}", INITIAL_CASH, 0, initial_price_param, 
                                   risk_aversion_factor=random.uniform(1.5, 3.0), tick_size=tick_size_param))
    for i in range(num_lts):
        agents.append(RiskLoverLT(f"LT-{i+1}", INITIAL_CASH, 0, initial_price_param, 
                                   risk_love_factor=random.uniform(0.3, 0.8), tick_size=tick_size_param))

    # Initial seeding of LOB
    for agent in [a for a in agents if isinstance(a, RiskAverseLP)]:
        for _ in range(5):
            side = random.choice(['buy', 'sell'])
            price_offset = random.uniform(tick_size_param, tick_size_param * 5)
            price = initial_price_param - price_offset if side == 'buy' else initial_price_param + price_offset
            price = round(price / tick_size_param) * tick_size_param
            qty = random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE)
            order = Order(None, agent.agent_id, side, price, qty, 0)
            order_id = lob.add_order(order)
            agent.open_orders[order_id] = order

    price_history = []
    
    # Store agent data for plotting PnL, etc.
    agent_pnl_data = {agent.agent_id: [] for agent in agents}
    agent_pv_data = {agent.agent_id: [] for agent in agents}

    # Streamlit progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(num_steps):
        current_time = step

        price_impact_this_step = 0

        # Agents decide their actions
        random.shuffle(agents)
        for agent in agents:
            if isinstance(agent, RiskAverseLP):
                agent.decide(lob, current_time, current_market_price)
            elif isinstance(agent, RiskLoverLT):
                impact = agent.decide(lob, current_time, market_impact_factor_param, current_market_price)
                if impact is not None:
                    price_impact_this_step += impact

        # Apply total market impact from all LTs in this step
        current_market_price += price_impact_this_step

        mid_price = get_mid_price(lob, current_market_price)
        price_history.append(mid_price)
        
        # Update agent PnL and PV history for plotting
        for agent in agents:
            agent.update_portfolio_value(mid_price)
            agent.calculate_pnl(mid_price)
            agent_pnl_data[agent.agent_id].append(agent.pnl_history[-1])
            agent_pv_data[agent.agent_id].append(agent.portfolio_value_history[-1])

        # Update progress bar
        progress_bar.progress((step + 1) / num_steps)
        if step % 50 == 0: # Update status text less frequently
            status_text.text(f"Simulation running... Step {step+1}/{num_steps}, Current Mid Price: {mid_price:.2f}")

    status_text.text("Simulation complete!")

    # Prepare data for plotting
    price_df = pd.DataFrame({'Step': range(num_steps), 'Price': price_history})
    
    pnl_dfs = []
    for agent_id, pnl_list in agent_pnl_data.items():
        pnl_dfs.append(pd.DataFrame({'Step': range(len(pnl_list)), 'Agent': agent_id, 'PnL': pnl_list}))
    all_pnl_df = pd.concat(pnl_dfs) if pnl_dfs else pd.DataFrame(columns=['Step', 'Agent', 'PnL'])

    trade_df = pd.DataFrame(lob.trades)

    return price_df, all_pnl_df, trade_df, lob, agents

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="LOB Simulation")

st.title("📈 Dynamic Liquidity Nexus: S&P 500 LOB Simulation")

st.markdown("""
Welcome to the Limit Order Book Simulation! Here, you can experiment with how different types of agents
(Risk-Averse Liquidity Providers and Risk-Lover Liquidity Takers) interact and shape the market price.
Adjust the parameters below and hit 'Run Simulation' to see the market come alive!
""")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")

num_lps = st.sidebar.slider("Number of Liquidity Providers (LPs)", 1, 10, 3)
num_lts = st.sidebar.slider("Number of Liquidity Takers (LTs)", 1, 10, 2)
sim_steps = st.sidebar.slider("Simulation Steps", 100, 2000, DEFAULT_SIMULATION_STEPS)
initial_price_ui = st.sidebar.number_input("Initial Asset Price", value=DEFAULT_INITIAL_PRICE, step=100.0, format="%.2f")
market_impact_factor_ui = st.sidebar.slider("Market Impact Factor", 0.00001, 0.001, DEFAULT_MARKET_IMPACT_FACTOR, format="%.5f")
tick_size_ui = st.sidebar.number_input("Tick Size", value=DEFAULT_TICK_SIZE, min_value=0.01, max_value=1.0, format="%.2f")


# Main area for plots and results
st.header("Simulation Results")

if st.sidebar.button("Run Simulation"):
    st.subheader("Running Simulation...")
    price_history_df, pnl_history_df, trades_df, final_lob, final_agents = run_simulation(
        num_lps, num_lts, sim_steps, initial_price_ui, market_impact_factor_ui, tick_size_ui
    )

    st.subheader("Price Evolution")
    fig_price = px.line(price_history_df, x='Step', y='Price', title='Simulated Mid Price Over Time')
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Agent PnL")
    if not pnl_history_df.empty:
        fig_pnl = px.line(pnl_history_df, x='Step', y='PnL', color='Agent', title='Agent PnL Over Time')
        st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.write("No PnL data available (perhaps no agents or no trades).")

    st.subheader("Executed Trades vs. Mid Price")
    if not trades_df.empty:
        # Overlay trades on price history
        fig_trades_price = px.line(price_history_df, x='Step', y='Price', title='Executed Trades (Size indicates Quantity)')
        
        # Add trades as scatter points
        # Map trade type to color, use quantity for size
        fig_trades_price.add_scatter(x=trades_df['time'], y=trades_df['price'], 
                                   mode='markers', 
                                   marker=dict(size=trades_df['quantity']/5, 
                                               opacity=0.6, 
                                               color=trades_df['type'].map({'buy': 'green', 'sell': 'red'})),
                                   name='Trades',
                                   hovertext=trades_df.apply(lambda row: f"Agent: {row['buyer_id'] if row['type']=='buy' else row['seller_id']}<br>Price: {row['price']:.2f}<br>Qty: {row['quantity']}", axis=1)
                                  )
        st.plotly_chart(fig_trades_price, use_container_width=True)
    else:
        st.write("No trades executed in this simulation run.")

    st.subheader("Final Limit Order Book State")
    st.text(final_lob.display_simple())

    st.subheader("Final Agent Status")
    agent_status_data = []
    for agent in final_agents:
        agent_status_data.append({
            "Agent ID": agent.agent_id,
            "Type": "LP" if isinstance(agent, RiskAverseLP) else "LT",
            "Cash": f"{agent.cash:,.2f}",
            "Shares": agent.shares,
            "Final PV": f"{agent.portfolio_value:,.2f}",
            "Final PnL": f"{agent.pnl_history[-1]:,.2f}" if agent.pnl_history else "N/A"
        })
    st.dataframe(pd.DataFrame(agent_status_data))

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start!")