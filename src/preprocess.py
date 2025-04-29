"""
Data Preprocessing Module
Clean raw stock data and engineer features for ML models.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def preprocess_stock_data(df, symbol):
    """
    Clean and engineer features from raw stock price data.
    
    Args:
        df (pd.DataFrame): Raw stock price data.
        symbol (str): Stock ticker symbol.
        
    Returns:
        pd.DataFrame: Processed DataFrame with engineered features.
    """
    # Make a copy to avoid modifying the original data
    df = df.copy()
    
    # Reset index to make Date a column if it's the index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    # Ensure Date column exists and is properly formatted
    if 'Date' not in df.columns and 0 in df.columns:
        df = df.rename(columns={0: 'Date'})
    
    # Convert numeric columns to float
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for null values and handle them
    if df.isnull().sum().sum() > 0:
        print(f"Filling null values for {symbol}")
        df = df.fillna(method='ffill')  # Forward fill
        df = df.fillna(method='bfill')  # Backward fill any remaining NAs
    
    # Check if we have at least 252 trading days (1 year)
    if len(df) < 252:
        print(f"Warning: {symbol} has less than a year of data ({len(df)} days)")
    
    # Calculate daily returns
    df['Return'] = df['Close'].pct_change()
    
    # Calculate rolling statistics
    for window in [5, 10, 20, 50]:
        # Rolling mean of close price
        df[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Rolling standard deviation of close price
        df[f'Close_STD_{window}'] = df['Close'].rolling(window=window).std()
        
        # Rolling mean of volume
        df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Price momentum (percentage change over window)
        df[f'Momentum_{window}'] = df['Close'].pct_change(periods=window)
    
    # Calculate price difference between high and low (volatility indicator)
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    
    # Calculate price difference between open and close
    df['OC_PCT'] = (df['Open'] - df['Close']) / df['Close'] * 100.0
    
    # Add lag features (previous N days' close prices)
    for i in range(1, 6):  # Previous 5 days
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    # Calculate RSI (Relative Strength Index) - 14 days
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop rows with NaN values (due to rolling calculations)
    df = df.dropna()
    
    # Keep Date column for reference
    return df


def create_supervised_data(df, window_size=20, forecast_horizon=1):
    """
    Create a supervised learning dataset with historical window and target.
    
    Args:
        df (pd.DataFrame): Processed stock data.
        window_size (int): Number of days in the lookback window.
        forecast_horizon (int): Number of days ahead to predict.
        
    Returns:
        tuple: (features array, target array, dates array)
    """
    features = []
    targets = []
    dates = []
    
    # Select columns to use as features
    feature_cols = [col for col in df.columns if col not in ['Date']]
    
    for i in range(len(df) - window_size - forecast_horizon + 1):
        # Extract the window
        window = df.iloc[i:i+window_size][feature_cols].values
        
        # Extract the target (future price)
        target = df.iloc[i + window_size + forecast_horizon - 1]['Close']
        
        # Extract the date for reference
        date = df.iloc[i + window_size + forecast_horizon - 1]['Date']
        
        features.append(window)
        targets.append(target)
        dates.append(date)
    
    return np.array(features), np.array(targets), np.array(dates)


def process_and_save(input_folder, output_folder, window_size=20, min_data_completeness=0.9):
    """
    Process all stock data files and save the processed datasets.
    
    Args:
        input_folder (str): Directory containing raw CSV files.
        output_folder (str): Directory to save processed files.
        window_size (int): Lookback window size for feature generation.
        min_data_completeness (float): Minimum fraction of days with data required.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    # Create a summary DataFrame to track processing statistics
    summary_data = []
    
    for csv_file in tqdm(csv_files, desc="Processing stock files"):
        try:
            # Extract symbol name from filename
            symbol = os.path.splitext(csv_file)[0]
            
            # Load raw data with special handling for unusual CSV structure
            file_path = os.path.join(input_folder, csv_file)
            # Skip the first 3 rows (unusual header structure) and use the 4th row onwards as data
            df = pd.read_csv(file_path, skiprows=3, header=None, 
                            names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
            
            # Check if data is sufficient
            expected_days = 252 * 10  # ~10 years of trading
            actual_days = len(df)
            completeness = actual_days / expected_days
            
            if completeness < min_data_completeness:
                print(f"Skipping {symbol}: Insufficient data ({completeness:.2%} complete)")
                continue
            
            # Preprocess data
            processed_df = preprocess_stock_data(df, symbol)
            
            # Save processed DataFrame
            processed_path = os.path.join(output_folder, f"{symbol}_processed.csv")
            processed_df.to_csv(processed_path, index=False)
            
            # Create supervised learning data
            X, y, dates = create_supervised_data(processed_df, window_size=window_size)
            
            # Save as a structured dataset
            structured_path = os.path.join(output_folder, f"{symbol}_ml_ready.npz")
            np.savez(structured_path, X=X, y=y, dates=dates)
            
            # Add to summary
            summary_data.append({
                'Symbol': symbol,
                'Days': len(processed_df),
                'Start_Date': processed_df['Date'].min(),
                'End_Date': processed_df['Date'].max(),
                'Features': X.shape,
                'Samples': len(y)
            })
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Save the summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_folder, "processing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Processing complete. Summary saved to {summary_path}")


def main():
    """Main function to parse arguments and execute the preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess stock market data")
    parser.add_argument("--in_folder", type=str, default="data/raw", help="Input directory with raw CSV files")
    parser.add_argument("--out_folder", type=str, default="data/processed", help="Output directory for processed files")
    parser.add_argument("--window_size", type=int, default=20, help="Lookback window size (days)")
    parser.add_argument("--min_completeness", type=float, default=0.8, help="Minimum data completeness (0.0-1.0)")
    
    args = parser.parse_args()
    
    print(f"Processing stock data from {args.in_folder}")
    process_and_save(args.in_folder, args.out_folder, args.window_size, args.min_completeness)


if __name__ == "__main__":
    main()