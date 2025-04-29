"""
Stock Market Prediction - Data Loader Module
This module handles loading and preprocessing stock market data.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def download_stock_data(ticker, start_date=None, end_date=None, period="5y"):
    """
    Downloads historical stock data using the yfinance library.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        period (str, optional): Time period to download if start_date and end_date are not specified
        
    Returns:
        pd.DataFrame: Historical stock data
    """
    print(f"Downloading {ticker} stock data...")
    
    if start_date and end_date:
        data = yf.download(ticker, start=start_date, end=end_date)
    else:
        data = yf.download(ticker, period=period)
    
    print(f"Downloaded {len(data)} rows of data for {ticker}")
    return data

def save_stock_data(data, ticker, output_dir="data"):
    """
    Saves stock data to a CSV file.
    
    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        output_dir (str): Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ticker}_data.csv")
    data.to_csv(file_path)
    print(f"Data saved to {file_path}")

def load_stock_data(ticker, data_dir="data"):
    """
    Loads stock data from a CSV file or downloads it if not available.
    
    Args:
        ticker (str): Stock ticker symbol
        data_dir (str): Directory containing the data files
        
    Returns:
        pd.DataFrame: Stock data with standard format
    """
    file_path = os.path.join(data_dir, f"{ticker}_data.csv")
    
    if os.path.exists(file_path):
        print(f"Loading {ticker} data from {file_path}")
        
        try:
            # Check the structure of the CSV by reading the first few lines
            with open(file_path, 'r') as f:
                first_lines = [next(f).strip() for _ in range(5)]
            
            # Check if this is the custom format by looking for "Ticker" in second row
            if any('Ticker' in line for line in first_lines[:2]):
                print("Detected non-standard CSV format. Transforming data...")
                # Skip the first three rows and read as transposed data
                df = pd.read_csv(file_path, skiprows=3, header=None)
                
                # First column contains dates
                dates = df.iloc[:, 0].values
                
                # Second column contains Close prices
                close_prices = df.iloc[:, 1].values
                
                # Create a proper DataFrame with Date and Close columns
                transformed_data = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Close': close_prices,
                })
                
                # Set Date as index
                transformed_data.set_index('Date', inplace=True)
                return transformed_data
            else:
                # Standard format
                data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                return data
                
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            print("Attempting to download fresh data...")
            data = download_stock_data(ticker)
            save_stock_data(data, ticker, data_dir)
            return data
    else:
        print(f"No local data found for {ticker}, downloading...")
        data = download_stock_data(ticker)
        save_stock_data(data, ticker, data_dir)
        return data

def create_sequences(data, sequence_length=60):
    """
    Creates input sequences for time series prediction.
    
    Args:
        data (np.ndarray): The preprocessed stock price data
        sequence_length (int): Number of time steps to use for input sequence
        
    Returns:
        tuple: (X, y) where X is input sequences and y is target values
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)

def prepare_data(data, target_col='Close', sequence_length=60, test_size=0.2, predict_returns=False):
    """
    Prepares stock data for modeling by:
    1. Selecting the target column
    2. Optionally converting to returns (percent change)
    3. Scaling the data
    4. Creating sequences
    5. Splitting into train and test sets
    
    Args:
        data (pd.DataFrame): Stock data
        target_col (str): Target column to predict
        sequence_length (int): Number of time steps to use for input sequence
        test_size (float): Proportion of data to use for testing
        predict_returns (bool): If True, predict returns instead of price levels
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # If predicting returns, calculate percent changes first
    if predict_returns:
        print("Preparing data to predict returns instead of price levels")
        # Calculate returns - pct_change returns NaN for first row
        returns = data[target_col].pct_change().fillna(0)
        # Use returns as our target data
        target_data = returns.values.reshape(-1, 1)
    else:
        # Select target column and convert to numpy array for price level prediction
        target_data = data[target_col].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target_data)
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Reshape for modeling (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def visualize_stock_data(data, ticker, output_dir="data"):
    """
    Creates visualization of stock data.
    
    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save visualization
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ticker}_stock_price.png")
        
        # Use the most basic plotting approach possible
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'].values)
        ax.set_title(f"{ticker} Stock Price")
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Close Price (USD)')
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Stock price visualization saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not visualize stock data: {str(e)}")
        print("Continuing with model training without visualization...")

def load_multiple_stocks(tickers, start_date=None, end_date=None, period="5y", data_dir="data"):
    """
    Loads data for multiple stock tickers.
    
    Args:
        tickers (list): List of stock ticker symbols
        start_date (str, optional): Start date
        end_date (str, optional): End date
        period (str, optional): Time period
        data_dir (str): Data directory
        
    Returns:
        dict: Dictionary mapping ticker symbols to DataFrames
    """
    stocks_data = {}
    
    for ticker in tickers:
        try:
            # Try to load from file first
            file_path = os.path.join(data_dir, f"{ticker}_data.csv")
            if os.path.exists(file_path):
                # Use same improved date handling as in load_stock_data
                data = pd.read_csv(file_path)
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.set_index('Date', inplace=True)
                stocks_data[ticker] = data
            else:
                # Download if not available
                data = download_stock_data(ticker, start_date, end_date, period)
                save_stock_data(data, ticker, data_dir)
                stocks_data[ticker] = data
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
    
    return stocks_data

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    data = load_stock_data(ticker)
    print(data.head())
    visualize_stock_data(data, ticker)