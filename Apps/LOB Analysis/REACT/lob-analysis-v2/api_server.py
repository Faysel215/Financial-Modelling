import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from flask import Flask, jsonify
from flask_cors import CORS

# --- Configuration ---
# IMPORTANT: Update these with your PostgreSQL credentials.
# The user needs CREATEDB permissions to run this script for the first time.
DB_USER = "postgres"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "quant_data" # The database we will create and use
TABLE_NAME = "lob_metrics"
# This file is created by process_data.py and tells us which CSV to load.
POINTER_FILENAME = 'last_processed_file.txt'

# --- Database Setup Logic ---

def setup_database():
    """
    Connects to the default postgres database to check for and create
    our target database if it doesn't exist.
    """
    try:
        conn = psycopg2.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, dbname='postgres')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if the target database exists
        cursor.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [DB_NAME])
        if not cursor.fetchone():
            print(f"Database '{DB_NAME}' does not exist. Creating it...")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error during database setup: {e}")
        print("Please ensure your user has CREATEDB permissions and the PostgreSQL server is running.")
        exit(1)

def initialize_table():
    """
    Connects to our target database and creates the table and hypertable.
    """
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        print(f"Setting up table '{TABLE_NAME}' in database '{DB_NAME}'...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # This schema matches the output of the process_data.py script
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                time          TIMESTAMPTZ       NOT NULL,
                type          SMALLINT,
                order_id      BIGINT,
                size          BIGINT,
                price         DOUBLE PRECISION,
                direction     SMALLINT,
                mid_price     DOUBLE PRECISION,
                spread        DOUBLE PRECISION,
                market_depth  BIGINT,
                obi           DOUBLE PRECISION
            );
        """)

        # Check if the table is already a hypertable before trying to create one
        cursor.execute(f"""
            SELECT 1 FROM timescaledb_information.hypertables 
            WHERE hypertable_name = '{TABLE_NAME}';
        """)
        if not cursor.fetchone():
            print(f"Converting '{TABLE_NAME}' to a hypertable...")
            cursor.execute(f"SELECT create_hypertable('{TABLE_NAME}', 'time');")
        else:
            print(f"'{TABLE_NAME}' is already a hypertable.")
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error during table initialization: {e}")
        exit(1)

def ingest_data_if_needed():
    """Checks the pointer file and ingests data from the specified CSV if the table is empty."""
    try:
        with open(POINTER_FILENAME, 'r') as f:
            csv_to_ingest = f.read().strip()
    except FileNotFoundError:
        print(f"Info: Pointer file '{POINTER_FILENAME}' not found. Run process_data.py first. Skipping ingestion.")
        return

    if not os.path.exists(csv_to_ingest):
        print(f"Info: Processed CSV '{csv_to_ingest}' not found. Skipping ingestion.")
        return

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        if cursor.fetchone()[0] > 0:
            print(f"Table '{TABLE_NAME}' already contains data. Skipping ingestion.")
            return

        print(f"Table is empty. Ingesting data from '{csv_to_ingest}'...")
        with open(csv_to_ingest, 'r') as f:
            cursor.copy_expert(
                sql=f"COPY {TABLE_NAME} FROM STDIN WITH CSV HEADER",
                file=f
            )
        conn.commit()
        print("Data ingestion successful.")
    except Exception as e:
        print(f"Error during data ingestion: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

# --- Flask Application ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for the React UI

def get_db_connection():
    """Establishes a connection for API endpoints."""
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

@app.route('/api/metrics/1s', methods=['GET'])
def get_aggregated_metrics():
    """Endpoint to fetch key metrics, aggregated by second for performance."""
    query = f"""
    SELECT
        time_bucket('1 second', time) AS "second",
        AVG(mid_price) AS "avg_mid_price",
        AVG(spread) AS "avg_spread",
        AVG(market_depth) AS "avg_market_depth",
        AVG(obi) AS "avg_obi"
    FROM
        {TABLE_NAME}
    GROUP BY "second" ORDER BY "second";
    """
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['second'] = df['second'].astype(str)
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/volume/executed', methods=['GET'])
def get_volume_data():
    """Endpoint to fetch the total executed volume by trade direction."""
    query = f"""
    SELECT
        CASE
            WHEN direction = 1 THEN 'Buy Limit (Seller Initiated)'
            WHEN direction = -1 THEN 'Sell Limit (Buyer Initiated)'
        END AS trade_type,
        SUM(size) AS total_executed_volume
    FROM {TABLE_NAME}
    WHERE type = 4
    GROUP BY trade_type;
    """
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return jsonify(df.to_dict(orient='records'))

# --- Main Execution Block ---
if __name__ == '__main__':
    # This block runs when you execute 'python api_server.py'
    print("--- Initializing Backend Server & Database ---")
    setup_database()
    initialize_table()
    ingest_data_if_needed()
    print("--- Initialization Complete. Starting Flask API... ---")
    
    # Start the web server
    app.run(debug=True, port=5001)

