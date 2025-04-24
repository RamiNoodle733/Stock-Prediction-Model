import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

def fetch_stock_data(ticker, start_date, end_date=None, interval='1d'):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame with stock data
    """
    # If no end date is provided, use today
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download data
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    # Check if data was successfully downloaded
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} in the specified date range")
    
    # Reset index to make Date a column
    data.reset_index(inplace=True)
    
    return data

def plot_stock_data(data, ticker, save_path=None):
    """
    Plot stock price and volume
    
    Args:
        data: DataFrame with stock data
        ticker: Stock ticker symbol
        save_path: Path to save the figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot closing price
    ax1.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    ax1.set_title(f'{ticker} Stock Price')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot trading volume
    ax2.bar(data['Date'], data['Volume'], color='gray', alpha=0.7)
    ax2.set_title(f'{ticker} Trading Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    
    # Adjust layout and date formatting
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for stock data
    
    Args:
        data: DataFrame with stock data
    
    Returns:
        DataFrame with added technical indicators
    """
    df = data.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over 14 periods
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def prepare_for_ml(data, feature_columns=None, target_column='Close'):
    """
    Prepare data for machine learning
    
    Args:
        data: DataFrame with stock data
        feature_columns: List of feature columns to use (None uses all available)
        target_column: Target column to predict
        
    Returns:
        X, y, feature_names
    """
    if feature_columns is None:
        # Use all columns except Date and target column
        feature_columns = [col for col in data.columns if col not in ['Date', target_column]]
    
    # Select features and target
    X = data[feature_columns].values
    y = data[target_column].values
    
    return X, y, feature_columns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and process stock data')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1d, 1wk, 1mo)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--plot', action='store_true', help='Plot the stock data')
    
    args = parser.parse_args()
    
    # Fetch data
    data = fetch_stock_data(args.ticker, args.start_date, args.end_date, args.interval)
    
    # Add technical indicators
    data_with_indicators = calculate_technical_indicators(data)
    
    # Save to CSV if output path is provided
    if args.output:
        data_with_indicators.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")
    
    # Plot if requested
    if args.plot:
        plot_stock_data(data, args.ticker)
    
    # Display summary
    print(f"\nData Summary for {args.ticker}:")
    print(f"Date Range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Total Records: {len(data)}")
    print("\nSample Data:")
    print(data.head())