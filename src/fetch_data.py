"""
Stock Market Data Fetcher
Downloads historical stock price data for NASDAQ-listed companies using yfinance.
"""

import os
import argparse
import sys
from datetime import datetime
import time
from tqdm import tqdm

# Try to import required libraries with fallback options
try:
    import pandas as pd
except ImportError:
    print("Error: pandas library is missing. Please install it using 'pip install pandas'")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance library is missing. Please install it using 'pip install yfinance'")
    sys.exit(1)

try:
    import requests
    from io import StringIO
except ImportError:
    print("Error: requests library is missing. Please install it using 'pip install requests'")
    sys.exit(1)


def get_nasdaq_tickers(limit=None):
    """
    Retrieve a list of NASDAQ-listed tickers from the NASDAQ website.
    
    Args:
        limit (int, optional): Limit the number of tickers to fetch (for testing).
        
    Returns:
        list: List of ticker symbols.
    """
    url = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the text file
        content = StringIO(response.text)
        df = pd.read_csv(content, sep="|")
        
        # Extract the ticker symbols from the 'Symbol' column
        tickers = df["Symbol"].tolist()
        
        # Remove the file header and footer entries
        tickers = [ticker for ticker in tickers if not ticker.startswith('Symbol')]
        
        if limit:
            return tickers[:limit]
        return tickers
    except Exception as e:
        print(f"Error fetching NASDAQ tickers: {e}")
        # Provide a fallback list of popular tickers if we can't fetch the full list
        fallback_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "CSCO"]
        print(f"Using fallback list of {len(fallback_tickers)} popular tickers instead.")
        
        if limit:
            return fallback_tickers[:limit]
        return fallback_tickers


def fetch_and_save(ticker_list, start_date, end_date, dest_folder):
    """
    Download historical price data for a list of tickers and save to CSV files.
    
    Args:
        ticker_list (list): List of ticker symbols to download.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        dest_folder (str): Directory to save the CSV files.
    """
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for symbol in tqdm(ticker_list, desc="Downloading stock data"):
        try:
            # Check if file already exists to avoid re-downloading
            file_path = os.path.join(dest_folder, f"{symbol}.csv")
            if os.path.exists(file_path):
                print(f"File for {symbol} already exists, skipping download.")
                successful += 1
                continue
                
            # Download data
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Skip if no data was retrieved
            if df.empty:
                print(f"No data available for {symbol}, skipping...")
                failed += 1
                continue
                
            # Save to CSV
            df.to_csv(file_path)
            successful += 1
            
            # Brief pause to avoid hitting rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            failed += 1
    
    print(f"Download complete. Successfully downloaded {successful} stocks. Failed: {failed}.")


def main():
    """Main function to parse arguments and execute the data fetch."""
    parser = argparse.ArgumentParser(description="Download historical stock data for NASDAQ tickers")
    parser.add_argument("--symbol", type=str, help="Specific ticker symbol to download (e.g., AAPL)")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default="../data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of tickers to download (for testing)")
    
    args = parser.parse_args()
    
    if args.symbol:
        # Download a specific symbol
        print(f"Downloading data for {args.symbol} from {args.start} to {args.end}...")
        fetch_and_save([args.symbol], args.start, args.end, args.out)
    else:
        # Download multiple symbols from NASDAQ list
        print(f"Fetching NASDAQ ticker list...")
        tickers = get_nasdaq_tickers(limit=args.limit)
        print(f"Found {len(tickers)} tickers.")
        
        if not tickers:
            print("No tickers found. Exiting.")
            return
        
        print(f"Downloading data from {args.start} to {args.end}...")
        fetch_and_save(tickers, args.start, args.end, args.out)


if __name__ == "__main__":
    main()