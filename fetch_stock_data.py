import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

def fetch_apple_stock_data(start_date='2018-01-01', end_date='2023-01-01'):
    """
    Fetch Apple stock data from Yahoo Finance
    
    Args:
        start_date: Start date for data retrieval (string in 'YYYY-MM-DD' format)
        end_date: End date for data retrieval (string in 'YYYY-MM-DD' format)
        
    Returns:
        DataFrame with stock data
    """
    ticker = 'AAPL'
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    print(f"Downloaded {len(df)} rows of data")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values. Filling with forward fill method.")
        df.fillna(method='ffill', inplace=True)
        
    return df

def plot_apple_stock_data(data, save_path=None):
    """
    Plot stock data
    
    Args:
        data: DataFrame with stock data
        save_path: Path to save the plot (if None, plot is displayed)
    """
    ticker = 'AAPL'
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot closing price
    ax1.plot(data.index, data['Close'], label='Close Price')
    ax1.set_title(f'{ticker} Stock Price')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot volume
    ax2.bar(data.index, data['Volume'] / 1000000)
    ax2.set_title(f'{ticker} Trading Volume')
    ax2.set_ylabel('Volume (millions)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def calculate_returns(data):
    """
    Calculate daily and cumulative returns
    
    Args:
        data: DataFrame with stock data
        
    Returns:
        DataFrame with original data and return columns added
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # Calculate cumulative returns
    df['Cum_Return'] = (1 + df['Daily_Return']/100).cumprod() - 1
    df['Cum_Return'] = df['Cum_Return'] * 100  # Convert to percentage
    
    return df

def analyze_apple_stock_statistics(data, save_dir=None):
    """
    Analyze stock statistics and create plots
    
    Args:
        data: DataFrame with stock data
        save_dir: Directory to save plots (if None, plots are displayed)
        
    Returns:
        DataFrame with statistical analysis
    """
    ticker = 'AAPL'
    # Calculate returns
    df = calculate_returns(data)
    
    # Calculate statistics
    stats = {
        'Ticker': ticker,
        'Start_Date': df.index[0].strftime('%Y-%m-%d'),
        'End_Date': df.index[-1].strftime('%Y-%m-%d'),
        'Trading_Days': len(df),
        'Start_Price': df['Close'].iloc[0],
        'End_Price': df['Close'].iloc[-1],
        'Min_Price': df['Close'].min(),
        'Max_Price': df['Close'].max(),
        'Avg_Price': df['Close'].mean(),
        'Total_Return_Pct': df['Cum_Return'].iloc[-1],
        'Avg_Daily_Return_Pct': df['Daily_Return'].mean(),
        'StdDev_Daily_Return_Pct': df['Daily_Return'].std(),
        'Positive_Days_Pct': (df['Daily_Return'] > 0).mean() * 100
    }
    
    stats_df = pd.DataFrame([stats])
    
    # Create plots if save_dir is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot 1: Closing Price and Volume
        plot_apple_stock_data(df, save_path=f"{save_dir}/{ticker}_price_volume.png")
        
        # Plot 2: Daily Returns
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Daily_Return'])
        plt.title(f'{ticker} Daily Returns')
        plt.ylabel('Daily Return (%)')
        plt.grid(True)
        plt.savefig(f"{save_dir}/{ticker}_daily_returns.png")
        plt.close()
        
        # Plot 3: Cumulative Returns
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Cum_Return'])
        plt.title(f'{ticker} Cumulative Returns')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True)
        plt.savefig(f"{save_dir}/{ticker}_cumulative_returns.png")
        plt.close()
        
        # Plot 4: Return Distribution
        plt.figure(figsize=(14, 7))
        plt.hist(df['Daily_Return'].dropna(), bins=50, alpha=0.75)
        plt.title(f'{ticker} Daily Returns Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"{save_dir}/{ticker}_return_distribution.png")
        plt.close()
        
        # Save statistics to CSV
        stats_df.to_csv(f"{save_dir}/{ticker}_statistics.csv", index=False)
        print(f"Analysis saved to {save_dir} directory")
    
    return stats_df

if __name__ == "__main__":
    # Create a directory for raw data analysis
    analysis_dir = "raw_data_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Fetch Apple stock data for the past 5 years
    apple_data = fetch_apple_stock_data('2018-01-01', '2023-01-01')
    
    # Perform statistical analysis and create plots
    stats = analyze_apple_stock_statistics(apple_data, save_dir=analysis_dir)
    
    # Print summary statistics
    print("\nApple Stock Summary Statistics:")
    print(stats)