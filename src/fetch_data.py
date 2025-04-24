import yfinance as yf
import os

def download_stock_data(ticker, start_date, end_date, output_dir):
    """
    Downloads historical stock data for a given ticker symbol and saves it as a CSV file.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.
        output_dir (str): Directory to save the CSV file.
    """
    # Fetch data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save data to CSV
    output_path = os.path.join(output_dir, f"{ticker}.csv")
    data.to_csv(output_path)
    print(f"Data for {ticker} saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "GOOGL", "MSFT"]  # Add more tickers as needed
    start_date = "2010-01-01"
    end_date = "2023-12-31"
    output_dir = "data"

    for ticker in tickers:
        download_stock_data(ticker, start_date, end_date, output_dir)