import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def download_stock_data(ticker='AAPL', start_date=None, end_date=None, period='5y'):
    """
    Download historical stock data using yfinance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (default: 'AAPL' for Apple)
    start_date : str
        Start date in 'YYYY-MM-DD' format (default: None)
    end_date : str
        End date in 'YYYY-MM-DD' format (default: None)
    period : str
        Period to download if start_date and end_date are not specified
        (default: '5y' for 5 years)
    
    Returns:
    --------
    DataFrame: Historical stock data
    """
    print(f"Downloading {ticker} stock data...")
    
    # If dates are not specified, use period
    if start_date is None or end_date is None:
        data = yf.download(ticker, period=period)
    else:
        data = yf.download(ticker, start=start_date, end=end_date)
    
    print(f"Downloaded {len(data)} rows of data for {ticker}")
    
    # Save to CSV file
    os.makedirs('../data', exist_ok=True)
    csv_file = f"../data/{ticker}_stock_data.csv"
    data.to_csv(csv_file)
    print(f"Data saved to {csv_file}")
    
    return data

def load_stock_data(ticker='AAPL', force_download=False):
    """
    Load stock data from CSV file or download if not available
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (default: 'AAPL' for Apple)
    force_download : bool
        If True, force download even if file exists
        
    Returns:
    --------
    DataFrame: Historical stock data
    """
    csv_file = f"../data/{ticker}_stock_data.csv"
    
    if os.path.exists(csv_file) and not force_download:
        print(f"Loading {ticker} data from {csv_file}")
        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        return data
    else:
        return download_stock_data(ticker=ticker)

def prepare_data_for_training(data, target_column='Close', sequence_length=60, train_split=0.8):
    """
    Prepare data for training a machine learning model
    
    Parameters:
    -----------
    data : DataFrame
        Historical stock data
    target_column : str
        Column to predict (default: 'Close' price)
    sequence_length : int
        Number of previous days to use for prediction (default: 60 days)
    train_split : float
        Percentage of data to use for training (default: 80%)
    
    Returns:
    --------
    tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # Select the target column and convert to numpy array
    dataset = data[target_column].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create sequences
    X = []
    y = []
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler

def plot_stock_data(data, title="Stock Price History"):
    """
    Plot the stock price history
    
    Parameters:
    -----------
    data : DataFrame
        Historical stock data
    title : str
        Title for the plot
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../data/stock_price_history.png')
    plt.show()

if __name__ == "__main__":
    # Example usage
    data = load_stock_data(ticker='AAPL', force_download=True)
    plot_stock_data(data, title="Apple Stock Price History")