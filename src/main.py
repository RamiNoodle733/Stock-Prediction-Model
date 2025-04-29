"""
Stock Market Prediction - Main Script
This script runs the complete workflow for stock price prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Import our modules
from data_loader import load_stock_data, prepare_data, visualize_stock_data
from models import (
    NaiveBaseline,
    LinearRegressionModel, 
    evaluate_trading_metrics,
    plot_predictions
)

# Set plot style
plt.style.use('ggplot')
sns.set_palette('deep')

def run_stock_prediction(ticker='AAPL', sequence_length=60, test_size=0.2, output_dir='results'):
    """
    Run the complete stock prediction workflow with returns-based analysis.
    
    Args:
        ticker: Stock ticker symbol
        sequence_length: Number of days to use for prediction
        test_size: Proportion of data to use for testing
        output_dir: Directory to save results
    """
    print("\n" + "="*80)
    print(f"STOCK PRICE PREDICTION: {ticker}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    print("\nStep 1: Loading stock data...")
    data = load_stock_data(ticker)
    
    # Step 2: Visualize data
    print("\nStep 2: Visualizing stock data...")
    visualize_stock_data(data, ticker, output_dir)
    
    # Create a copy of the data to work with for returns analysis
    print("\nStep 3: Preparing data for returns-based prediction...")
    df_returns = data.copy()
    
    # Calculate returns and all required features
    df_returns['Returns'] = df_returns['Close'].pct_change()
    df_returns['Prev_Returns'] = df_returns['Returns'].shift(1)
    
    # Create additional lagged returns as features
    for i in range(2, 6):
        df_returns[f'Prev_Returns_{i}'] = df_returns['Returns'].shift(i)
    
    # Add other potential features
    df_returns['Prev_Volume'] = df_returns['Volume'].shift(1) if 'Volume' in df_returns.columns else 0
    df_returns['Volume_Change'] = df_returns['Volume'].pct_change() if 'Volume' in df_returns.columns else 0
    
    # Remove rows with NaN values
    df_returns = df_returns.dropna()
    
    # Function to create features and target for returns-based model
    def create_features_target(data, target_col='Returns', feature_cols=None):
        if feature_cols is None:
            feature_cols = ['Prev_Returns', 'Prev_Returns_2', 'Prev_Returns_3', 
                           'Prev_Returns_4', 'Prev_Returns_5', 'Prev_Volume', 'Volume_Change']
        
        # Ensure all required columns exist
        available_cols = [col for col in feature_cols if col in data.columns]
        X = data[available_cols]
        y = data[target_col]
        
        return X, y

    # Create features and target
    X, y = create_features_target(df_returns)

    # Split data into train and test sets (using a simple time-based split for illustration)
    train_size = int(len(X) * 0.8)
    X_returns_train, X_returns_test = X[:train_size], X[train_size:]
    y_returns_train, y_returns_test = y[:train_size], y[train_size:]

    # Step 4: Train models
    print("\nStep 4: Training returns-based Linear Regression model...")
    from sklearn.linear_model import LinearRegression
    model_returns = LinearRegression()
    model_returns.fit(X_returns_train, y_returns_train)

    # Make predictions
    y_returns_train_pred = model_returns.predict(X_returns_train)
    y_returns_test_pred = model_returns.predict(X_returns_test)

    # Calculate metrics for returns prediction
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    train_rmse = np.sqrt(mean_squared_error(y_returns_train, y_returns_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_returns_test, y_returns_test_pred))
    train_r2 = r2_score(y_returns_train, y_returns_train_pred)
    test_r2 = r2_score(y_returns_test, y_returns_test_pred)

    # Print results
    print("Returns-based Linear Regression Model Performance:")
    print(f"Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
    print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    # Convert returns predictions back to price predictions for visualization
    test_dates = df_returns.index[train_size:]
    actual_prices = df_returns['Close'][train_size:].values
    predicted_prices = np.zeros(len(test_dates))

    # First predicted price is the last known price times (1 + predicted return)
    predicted_prices[0] = df_returns['Close'].iloc[train_size-1] * (1 + y_returns_test_pred[0])

    # Calculate the rest of the predicted prices
    for i in range(1, len(predicted_prices)):
        predicted_prices[i] = predicted_prices[i-1] * (1 + y_returns_test_pred[i])

    # Calculate RMSE and R² on the reconstructed prices
    price_rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    price_r2 = r2_score(actual_prices, predicted_prices)

    print("\nReconstructed Price Prediction Performance:")
    print(f"RMSE: {price_rmse:.4f}")
    print(f"R²: {price_r2:.4f}")

    # Visualize actual vs predicted prices from returns-based model
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, actual_prices, label='Actual Price')
    plt.plot(test_dates, predicted_prices, label='Predicted Price', alpha=0.7)
    plt.title('Actual vs Predicted Prices (Returns-based Model)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{ticker}_returns_reconstructed_prices.png"))
    plt.close()

    # Calculate and visualize directional accuracy
    correct_direction = np.sign(y_returns_test) == np.sign(y_returns_test_pred)
    direction_accuracy = np.mean(correct_direction) * 100
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Correct', 'Incorrect'], 
            [direction_accuracy, 100-direction_accuracy],
            color=['green', 'red'])
    plt.title(f"{ticker} Return Direction Prediction Accuracy")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, f"{ticker}_returns_direction_accuracy.png"))
    plt.close()
    
    print(f"\nDirectional Accuracy: {direction_accuracy:.2f}%")
    
    # Step 5: Add a naive baseline model (persistence)
    print("\nStep 5: Training naive baseline model...")
    
    # For comparison, prepare sequence-based data too (this is needed for the NaiveBaseline model)
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        data, 
        target_col='Close',
        sequence_length=sequence_length, 
        test_size=test_size
    )
    
    naive_model = NaiveBaseline()
    naive_model.fit(X_train, y_train)  # Doesn't actually train, just sets up the model
    
    # Evaluate the naive model
    naive_metrics, naive_pred = naive_model.evaluate(X_test, y_test, scaler)
    plot_predictions(
        y_test, naive_pred, 
        title=f"{ticker} Stock Price: Naive Baseline Predictions",
        save_path=os.path.join(output_dir, f"{ticker}_naive_baseline_predictions.png")
    )
    
    # Step 6: Analyze trading performance and economic metrics
    print("\nStep 6: Analyzing trading performance...")
    
    # Calculate trading-specific metrics
    
    # Prepare data for returns-based model with our common framework
    X_train_ret, X_test_ret, y_train_ret, y_test_ret, scaler_ret = prepare_data(
        data, 
        target_col='Close',
        sequence_length=sequence_length, 
        test_size=test_size,
        predict_returns=True
    )
    
    # Create a returns-based wrapper model to use our framework
    lr_returns_model = LinearRegressionModel()
    lr_returns_model.name = "Linear Regression (Returns)"
    lr_returns_model.fit(X_train_ret, y_train_ret)
    lr_returns_metrics, lr_returns_pred = lr_returns_model.evaluate(X_test_ret, y_test_ret, scaler_ret)
    
    # Plot returns predictions
    plot_predictions(
        y_test_ret, lr_returns_pred, 
        title=f"{ticker} Stock Returns: Linear Regression Predictions",
        save_path=os.path.join(output_dir, f"{ticker}_linear_regression_returns_predictions.png")
    )
    
    # Calculate and print trading-specific metrics
    trading_metrics = evaluate_trading_metrics(y_test_ret, lr_returns_pred, scaler_ret)
    print(f"\nTrading-specific metrics for Linear Regression (Returns):")
    for metric, value in trading_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Create a simple trading strategy based on returns predictions
    if len(y_returns_test) > 0 and len(y_returns_test_pred) > 0:
        # 1 = long, -1 = short, 0 = no position
        strategy_positions = np.where(y_returns_test_pred > 0, 1, -1)
        
        # Calculate strategy returns - multiply positions by actual returns
        strategy_returns = strategy_positions * y_returns_test
        market_returns = y_returns_test  # Buy and hold returns
        
        # Calculate cumulative returns
        cumulative_strategy = (1 + strategy_returns).cumprod() - 1
        cumulative_market = (1 + market_returns).cumprod() - 1
        
        # Plot strategy performance
        plt.figure(figsize=(14, 7))
        plt.plot(test_dates, cumulative_strategy * 100, label='Trading Strategy')
        plt.plot(test_dates, cumulative_market * 100, label='Buy and Hold')
        plt.title('Cumulative Returns: Strategy vs Buy-and-Hold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ticker}_trading_strategy.png"))
        plt.close()
        
        # Calculate additional trading metrics
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) * 100
        print(f"\nTrading Strategy Performance:")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Final Strategy Return: {cumulative_strategy[-1]*100:.2f}%")
        print(f"  Final Buy-and-Hold Return: {cumulative_market[-1]*100:.2f}%")
    
    print("\nStock prediction analysis complete!")

if __name__ == "__main__":
    # Choose which analysis to run
    run_single = True
    
    # Run prediction on a single stock (Apple)
    if run_single:
        run_stock_prediction(ticker='AAPL', output_dir='results')