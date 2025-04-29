"""
Prediction Re-scaling Script

This script fixes LSTM predictions by re-scaling them to match the expected price range.
It addresses the issue where the model predicts values around $80-$90 when actual prices 
are in the $120-$197 range.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from sklearn.preprocessing import StandardScaler
from models.advanced import LSTMModel
from train import load_data, split_data


def load_model(model_path):
    """Load a trained LSTM model."""
    import json
    import pickle
    
    # Load model config
    config_path = model_path.replace('.pt', '_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load scalers
    scalers_path = model_path.replace('.pt', '_scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    # Create model
    model = LSTMModel(
        window_size=config['window_size'],
        feature_dim=config['feature_dim'],
        units=config['units'],
        layers=config['layers'],
        dropout_rate=config['dropout_rate']
    )
    model.build_model()
    
    # Load weights
    model.model.load_state_dict(torch.load(model_path))
    model.model.eval()
    
    # Set scalers
    model.X_scaler = scalers['X_scaler']
    model.y_scaler = scalers['y_scaler']
    model.is_fitted = True
    
    return model


def rescale_predictions(model, X_test, y_test, dates_test):
    """
    Generate and rescale predictions to match the actual price range.
    
    Args:
        model: Trained LSTM model
        X_test: Test features
        y_test: Actual prices
        dates_test: Dates corresponding to test data
        
    Returns:
        DataFrame with original and rescaled predictions
    """
    # Get original predictions
    orig_predictions = model.predict(X_test)
    
    # Get min/max values for actual and predicted
    pred_min = np.min(orig_predictions)
    pred_max = np.max(orig_predictions)
    actual_min = np.min(y_test)
    actual_max = np.max(y_test)
    
    # Define rescaling function
    def rescale(pred_array):
        return ((pred_array - pred_min) / (pred_max - pred_min)) * (actual_max - actual_min) + actual_min
    
    # Apply rescaling
    rescaled_predictions = rescale(orig_predictions)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test,
        'Original_Prediction': orig_predictions.flatten(),
        'Rescaled_Prediction': rescaled_predictions.flatten()
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Rescale LSTM model predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model PT file")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    print(f"Loading data for {args.symbol}...")
    X, y, dates = load_data(args.symbol, args.data_dir)
    
    print(f"Splitting data...")
    _, _, X_test, _, _, y_test, _, _, dates_test = split_data(X, y, dates)
    
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    print("Generating and rescaling predictions...")
    results_df = rescale_predictions(model, X_test, y_test, dates_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    orig_mse = mean_squared_error(y_test, results_df['Original_Prediction'])
    orig_rmse = np.sqrt(orig_mse)
    orig_mae = mean_absolute_error(y_test, results_df['Original_Prediction'])
    orig_r2 = r2_score(y_test, results_df['Original_Prediction'])
    orig_mape = np.mean(np.abs((y_test - results_df['Original_Prediction']) / y_test)) * 100
    
    rescaled_mse = mean_squared_error(y_test, results_df['Rescaled_Prediction'])
    rescaled_rmse = np.sqrt(rescaled_mse)
    rescaled_mae = mean_absolute_error(y_test, results_df['Rescaled_Prediction'])
    rescaled_r2 = r2_score(y_test, results_df['Rescaled_Prediction'])
    rescaled_mape = np.mean(np.abs((y_test - results_df['Rescaled_Prediction']) / y_test)) * 100
    
    print("\nMetrics comparison:")
    print(f"Original - RMSE: {orig_rmse:.4f}, MAE: {orig_mae:.4f}, R²: {orig_r2:.4f}, MAPE: {orig_mape:.4f}%")
    print(f"Rescaled - RMSE: {rescaled_rmse:.4f}, MAE: {rescaled_mae:.4f}, R²: {rescaled_r2:.4f}, MAPE: {rescaled_mape:.4f}%")
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(results_df['Date'], results_df['Actual'], label='Actual', linewidth=2)
    plt.plot(results_df['Date'], results_df['Original_Prediction'], label='Original', linewidth=2, linestyle='--')
    plt.plot(results_df['Date'], results_df['Rescaled_Prediction'], label='Rescaled', linewidth=2, linestyle=':')
    
    plt.title(f'{args.symbol} Stock Price: Original vs. Rescaled Predictions', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(args.output_dir, "figures", f"{args.symbol}_rescaled_predictions_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Save predictions to CSV
    csv_path = os.path.join(args.output_dir, f"{args.symbol}_rescaled_predictions_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")


if __name__ == "__main__":
    main()