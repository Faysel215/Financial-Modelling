import pandas as pd
import numpy as np
import os
import re

# --- Configuration ---
# These substrings are used to find your raw data files.
MESSAGE_FILE_SUBSTRING = '_message_'
ORDERBOOK_FILE_SUBSTRING = '_orderbook_'

# Processes data in chunks to avoid running out of memory on large files.
CHUNK_SIZE = 10000

# Set to a value less than 100 (e.g., 10) to process only the first 10% of the data for quick testing.
# Set to 100 to process the entire file.
DATA_PERCENTAGE_TO_PROCESS = 15 

# This file will store the name of the output CSV for the API server to find.
POINTER_FILENAME = 'last_processed_file.txt' 

def parse_date_from_filename(filename):
    """Robustly parses the filename to extract the date (e.g., '2012-06-21')."""
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if not date_match:
        print(f"Warning: Could not automatically determine date from filename '{filename}'.")
        return None
    return date_match.group(1)

def process_chunk(msg_chunk, ob_chunk, date_str, levels):
    """Processes a single chunk of data, calculating all necessary metrics."""
    ob_cols = []
    for i in range(1, levels + 1):
        ob_cols.extend([f'ask_price_{i}', f'ask_size_{i}', f'bid_price_{i}', f'bid_size_{i}'])
    
    ob_chunk.columns = ob_cols
    df = pd.concat([msg_chunk, ob_chunk], axis=1)

    # --- Metric Calculations ---
    ask_p1 = df['ask_price_1'] / 10000.0
    bid_p1 = df['bid_price_1'] / 10000.0
    df['mid_price'] = (ask_p1 + bid_p1) / 2
    df['spread'] = ask_p1 - bid_p1

    # Calculate L5 Market Depth and Order Book Imbalance
    ask_vols = [f'ask_size_{i}' for i in range(1, min(6, levels + 1))]
    bid_vols = [f'bid_size_{i}' for i in range(1, min(6, levels + 1))]
    df['total_ask_volume'] = df[ask_vols].sum(axis=1)
    df['total_bid_volume'] = df[bid_vols].sum(axis=1)
    df['market_depth'] = df['total_ask_volume'] + df['total_bid_volume']
    df['obi'] = (df['total_bid_volume'] - df['total_ask_volume']) / (df['total_bid_volume'] + df['total_ask_volume'])
    
    # Clean and Transform original data columns
    df['time'] = pd.to_datetime(df['time'], unit='s', origin=date_str)
    df['price'] = np.where(df['type'] == 7, np.nan, df['price'] / 10000.0)
    
    # Select only the columns needed for the final output
    final_cols = ['time', 'type', 'order_id', 'size', 'price', 'direction', 
                  'mid_price', 'spread', 'market_depth', 'obi']
    
    return df[final_cols]

def process_lobster_data(message_file, orderbook_file):
    """Main function to orchestrate the processing of raw LOBSTER data."""
    print("--- Starting Stage 1: Data Processing with Metric Calculation ---")

    date_str = parse_date_from_filename(message_file)
    if not date_str:
        print("Halting processing due to filename format issue.")
        return

    try:
        header_df = pd.read_csv(orderbook_file, header=None, nrows=1)
        levels = int(header_df.shape[1] / 4)
    except Exception as e:
        print(f"Error reading orderbook file to determine levels: {e}")
        return

    rows_to_process = None
    if DATA_PERCENTAGE_TO_PROCESS < 100:
        print(f"Sampling enabled: Processing the first {DATA_PERCENTAGE_TO_PROCESS}% of the data.")
        with open(message_file) as f:
            total_lines = sum(1 for line in f)
        rows_to_process = int(total_lines * (DATA_PERCENTAGE_TO_PROCESS / 100.0))
        print(f"Total rows in source file: {total_lines}. Will process approximately {rows_to_process} rows.")

    msg_cols = ['time', 'type', 'order_id', 'size', 'price', 'direction']
    
    msg_iterator = pd.read_csv(message_file, header=None, names=msg_cols, chunksize=CHUNK_SIZE)
    ob_iterator = pd.read_csv(orderbook_file, header=None, chunksize=CHUNK_SIZE)
    
    base_name = message_file.split(MESSAGE_FILE_SUBSTRING)[0]
    output_filename = f"{base_name}_processed_metrics.csv"

    is_first_chunk = True
    total_rows_processed = 0
    
    # Overwrite pointer file at the start of processing
    with open(POINTER_FILENAME, 'w') as f:
        f.write(output_filename)
    print(f"Pointer file '{POINTER_FILENAME}' created, pointing to '{output_filename}'")
    
    print(f"Processing data with {levels} levels in chunks of {CHUNK_SIZE} rows...")
    for i, (msg_chunk, ob_chunk) in enumerate(zip(msg_iterator, ob_iterator)):
        print(f"  Processing chunk {i+1}...")
        processed_chunk = process_chunk(msg_chunk, ob_chunk, date_str, levels)
        
        if is_first_chunk:
            processed_chunk.to_csv(output_filename, index=False, mode='w', header=True)
            is_first_chunk = False
        else:
            processed_chunk.to_csv(output_filename, index=False, mode='a', header=False)
        
        total_rows_processed += len(msg_chunk)

        if rows_to_process and total_rows_processed >= rows_to_process:
            print(f"\nReached target of {rows_to_process} rows. Stopping processing.")
            break

    print(f"\nSuccessfully processed all chunks. Total rows processed: {total_rows_processed}")
    print(f"--- Stage 1 Complete: Clean data saved to '{output_filename}' ---")

if __name__ == "__main__":
    try:
        msg_file = next(f for f in os.listdir('.') if MESSAGE_FILE_SUBSTRING in f)
        ob_file = next(f for f in os.listdir('.') if ORDERBOOK_FILE_SUBSTRING in f)
        
        print(f"Found message file: {msg_file}")
        print(f"Found orderbook file: {ob_file}")
        
        process_lobster_data(msg_file, ob_file)
        
    except StopIteration:
        print("\n--- ERROR ---")
        print("Could not find the LOBSTER message and orderbook CSV files in this directory.")
        print("-------------")

