import psycopg2
import json

# --- Database Connection Details (replace with your own) ---
DB_NAME = "opt-strat"
DB_USER = "postgres"
DB_PASS = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = "5432"

# --- Comprehensive List of Options Strategies ---
strategies_data = [
    # Bullish
    {
        "name": "Long Call", "outlook": "Bullish", "volatility_view": "Long", "description": "A simple bet on price increase. Max loss is the premium paid.",
        "parameters": ["Strike Price", "Premium Paid"]
    },
    {
        "name": "Bull Call Spread", "outlook": "Moderately Bullish", "volatility_view": "N/A", "description": "Buy a call and sell a higher-strike call to reduce cost. Caps profit and loss.",
        "parameters": ["Long Call Strike", "Short Call Strike", "Net Debit"]
    },
    {
        "name": "Bull Put Spread", "outlook": "Moderately Bullish", "volatility_view": "Short", "description": "Sell a put and buy a lower-strike put for protection. Collect a credit, hoping the price stays above the short strike.",
        "parameters": ["Short Put Strike", "Long Put Strike", "Net Credit"]
    },
    {
        "name": "Covered Call", "outlook": "Neutral to Bullish", "volatility_view": "Short", "description": "Own the underlying stock and sell a call to generate income. Caps upside profit.",
        "parameters": ["Stock Purchase Price", "Short Call Strike", "Premium Received"]
    },
    # Bearish
    {
        "name": "Long Put", "outlook": "Bearish", "volatility_view": "Long", "description": "A simple bet on price decrease. Max loss is the premium paid.",
        "parameters": ["Strike Price", "Premium Paid"]
    },
    {
        "name": "Bear Put Spread", "outlook": "Moderately Bearish", "volatility_view": "N/A", "description": "Buy a put and sell a lower-strike put to reduce cost. Caps profit and loss.",
        "parameters": ["Long Put Strike", "Short Put Strike", "Net Debit"]
    },
    {
        "name": "Bear Call Spread", "outlook": "Moderately Bearish", "volatility_view": "Short", "description": "Sell a call and buy a higher-strike call for protection. Collect a credit, hoping the price stays below the short strike.",
        "parameters": ["Short Call Strike", "Long Call Strike", "Net Credit"]
    },
    # Neutral - Long Volatility
    {
        "name": "Long Straddle", "outlook": "Neutral", "volatility_view": "Long", "description": "Buy a call and a put at the same strike. Profits from a large price move in either direction.",
        "parameters": ["Strike Price", "Total Premium Paid"]
    },
    {
        "name": "Long Strangle", "outlook": "Neutral", "volatility_view": "Long", "description": "Buy an out-of-the-money call and put. Cheaper than a straddle, but requires a larger move to profit.",
        "parameters": ["Call Strike", "Put Strike", "Total Premium Paid"]
    },
    # Neutral - Short Volatility
    {
        "name": "Short Strangle", "outlook": "Neutral", "volatility_view": "Short", "description": "Sell an out-of-the-money call and put. Profits if the stock stays between the strikes. Unlimited risk.",
        "parameters": ["Call Strike", "Put Strike", "Total Premium Received"]
    },
    {
        "name": "Iron Condor", "outlook": "Neutral", "volatility_view": "Short", "description": "A combination of a bull put spread and a bear call spread. A high-probability strategy with defined risk.",
        "parameters": ["Long Put Strike", "Short Put Strike", "Short Call Strike", "Long Call Strike", "Net Credit"]
    },
    {
        "name": "Iron Butterfly", "outlook": "Neutral", "volatility_view": "Short", "description": "Sell an at-the-money call and put, and buy protective options. Profits from a very narrow price range.",
        "parameters": ["Protective Put Strike", "Short Strike", "Protective Call Strike", "Net Credit"]
    },
     # Hedging
    {
        "name": "Collar", "outlook": "Hedging", "volatility_view": "N/A", "description": "Protect a long stock position by buying a put and selling a call to finance it.",
        "parameters": ["Stock Purchase Price", "Long Put Strike", "Short Call Strike", "Net Cost/Credit"]
    }
]

def setup_and_seed_database():
    """Connects to the DB, creates the strategies table, and populates it."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        print("✅ Database connection successful!")
        cur = conn.cursor()

        # Create table for strategy definitions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                outlook VARCHAR(50),
                volatility_view VARCHAR(50),
                description TEXT,
                parameters JSONB
            );
        """)
        print("✅ 'strategies' table exists or was created.")

        # Insert data
        for strategy in strategies_data:
            # Use ON CONFLICT to avoid errors if run multiple times
            cur.execute(
                """
                INSERT INTO strategies (name, outlook, volatility_view, description, parameters)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING;
                """,
                (
                    strategy["name"],
                    strategy["outlook"],
                    strategy["volatility_view"],
                    strategy["description"],
                    json.dumps(strategy["parameters"]),
                ),
            )
        
        conn.commit()
        print(f"✅ Database seeded with {len(strategies_data)} strategies.")
        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    setup_and_seed_database()
