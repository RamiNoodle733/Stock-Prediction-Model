import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
from datetime import datetime
from models_sklearn import (
    preprocess_stock_data, 
    train_linear_regression,
    train_ridge_regression,
    train_random_forest,
    train_gradient_boosting,
    evaluate_model,
    plot_predictions,
    plot_residuals,
    plot_training_loss
)

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with stock data
    """
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Data shape: {data.shape}")
    return data

def analyze_stock(ticker, start_date, end_date):
    """
    Analyze a stock using multiple models
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dictionary with model results
    """
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, y_scaler = preprocess_stock_data(data)
    
    # Dictionary to store results
    results = {}
    metrics = []
    
    # Train and evaluate Linear Regression
    print("Training Linear Regression model...")
    lr_model, lr_preds, lr_time = train_linear_regression(X_train, y_train, X_test, y_test)
    lr_metrics = evaluate_model(y_test, lr_preds)
    
    # Convert predictions back to original scale
    lr_preds_orig = y_scaler.inverse_transform(lr_preds.reshape(-1, 1)).flatten()
    y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Save plots
    print("Saving Linear Regression plots...")
    plot_predictions(y_test_orig, lr_preds_orig, title=f"{ticker} Linear Regression Predictions",
                   save_path=f"results/{ticker}_linear_regression_predictions.png")
    plot_residuals(y_test_orig, lr_preds_orig, title=f"{ticker} Linear Regression Residuals",
                 save_path=f"results/{ticker}_linear_regression_residuals.png")
    
    # Add results
    results["Linear Regression"] = {
        "model": lr_model,
        "predictions": lr_preds_orig,
        "metrics": lr_metrics,
        "training_time": lr_time
    }
    
    metrics.append({
        "Model": "Linear Regression",
        "MSE": lr_metrics["MSE"],
        "RMSE": lr_metrics["RMSE"],
        "MAE": lr_metrics["MAE"],
        "R2": lr_metrics["R2"],
        "Training Time (s)": lr_time
    })
    
    # Train and evaluate Random Forest
    print("Training Random Forest model...")
    rf_model, rf_preds, rf_time = train_random_forest(X_train, y_train, X_test, y_test)
    rf_metrics = evaluate_model(y_test, rf_preds)
    
    # Convert predictions back to original scale
    rf_preds_orig = y_scaler.inverse_transform(rf_preds.reshape(-1, 1)).flatten()
    
    # Save plots
    print("Saving Random Forest plots...")
    plot_predictions(y_test_orig, rf_preds_orig, title=f"{ticker} Random Forest Predictions",
                   save_path=f"results/{ticker}_random_forest_predictions.png")
    plot_residuals(y_test_orig, rf_preds_orig, title=f"{ticker} Random Forest Residuals",
                 save_path=f"results/{ticker}_random_forest_residuals.png")
    
    # Add results
    results["Random Forest"] = {
        "model": rf_model,
        "predictions": rf_preds_orig,
        "metrics": rf_metrics,
        "training_time": rf_time
    }
    
    metrics.append({
        "Model": "Random Forest",
        "MSE": rf_metrics["MSE"],
        "RMSE": rf_metrics["RMSE"],
        "MAE": rf_metrics["MAE"],
        "R2": rf_metrics["R2"],
        "Training Time (s)": rf_time
    })
    
    # Train and evaluate Gradient Boosting
    print("Training Gradient Boosting model...")
    gb_model, gb_history, gb_preds, gb_time = train_gradient_boosting(X_train, y_train, X_test, y_test)
    gb_metrics = evaluate_model(y_test, gb_preds)
    
    # Convert predictions back to original scale
    gb_preds_orig = y_scaler.inverse_transform(gb_preds.reshape(-1, 1)).flatten()
    
    # Save plots
    print("Saving Gradient Boosting plots...")
    plot_predictions(y_test_orig, gb_preds_orig, title=f"{ticker} Gradient Boosting Predictions",
                   save_path=f"results/{ticker}_gradient_boosting_predictions.png")
    plot_residuals(y_test_orig, gb_preds_orig, title=f"{ticker} Gradient Boosting Residuals",
                 save_path=f"results/{ticker}_gradient_boosting_residuals.png")
    plot_training_loss(gb_history, title=f"{ticker} Gradient Boosting Training Loss",
                     save_path=f"results/{ticker}_gradient_boosting_loss.png")
    
    # Add results
    results["Gradient Boosting"] = {
        "model": gb_model,
        "predictions": gb_preds_orig,
        "metrics": gb_metrics,
        "training_time": gb_time
    }
    
    metrics.append({
        "Model": "Gradient Boosting",
        "MSE": gb_metrics["MSE"],
        "RMSE": gb_metrics["RMSE"],
        "MAE": gb_metrics["MAE"],
        "R2": gb_metrics["R2"],
        "Training Time (s)": gb_time
    })
    
    # Create a DataFrame with metrics
    metrics_df = pd.DataFrame(metrics)
    
    # Save metrics to CSV
    metrics_df.to_csv(f"results/{ticker}_model_metrics.csv", index=False)
    
    print(f"{ticker} model metrics:")
    print(metrics_df)
    
    return results, metrics_df

def main():
    # Define parameters
    ticker = "AAPL"  # Apple stock
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    # Analyze the stock
    print(f"Analyzing {ticker} stock...")
    results, metrics_df = analyze_stock(ticker, start_date, end_date)
    
    print(f"Analysis for {ticker} complete. Results saved to the 'results' directory.")

if __name__ == "__main__":
    main()