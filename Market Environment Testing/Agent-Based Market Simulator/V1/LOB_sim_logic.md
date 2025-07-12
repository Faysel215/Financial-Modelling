LOB is handled by the `LimitOrderBook` class with the breakdown:

1.  **Initialization (`__init__`)**:
    * When a simulation starts, a `LimitOrderBook` object is created.
    * It's given an initial price, a starting spread, and a "liquidity depth" (how many price levels of orders to create).

2.  **Liquidity Seeding (`_seed_book`)**: This is the core of the LOB simulation. This method is called at the beginning of *every single tick* of the simulation to represent a fresh state of the market provided by background "market makers."
    * It first clears any old orders from the previous tick.
    * It then creates a "ladder" of buy orders (bids) and sell orders (asks) around the current market price.
    * **Bids:** It creates a series of buy orders at prices stepping *down* from the best bid.
    * **Asks:** It creates a series of sell orders at prices stepping *up* from the best ask.
    * The number of these price levels is determined by the `liquidity_depth` parameter you set in the UI, and the size of each order is a random number between 5 and 20.

3.  **Order Execution (`execute_market_order`)**: This is how the agents interact with the LOB.
    * When an agent decides to 'buy', their order is matched against the `asks` side of the book, starting with the cheapest one.
    * When an agent decides to 'sell', their order is matched against the `bids` side, starting with the most expensive one.
    * The code "walks the book," consuming liquidity from each price level until the agent's order is filled. If a large order consumes all the shares at one price level, it moves to the next, which causes **price slippage** and **market impact**—a key part of the simulation!