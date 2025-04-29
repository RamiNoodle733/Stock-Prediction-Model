"""
Diagnostic script to troubleshoot prediction scaling issues
"""

import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.advanced import LSTMModel
from train import load_data, split_data

def main():
    parser = argparse.ArgumentParser(description="Diagnose prediction issues in stock price models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Loading {args.symbol} data...")
    X, y, dates = load_data(args.symbol, args.data_dir)

    # Split data to get test set (same as in training)
    print('Splitting data...')
    _, _, X_test, _, _, y_test, _, _, dates_test = split_data(X, y, dates)

    # Load model config and scalers
    print('Loading model config and scalers...')
    model_path = args.model_path
    config_path = model_path.replace('.pt', '_config.json')
    scalers_path = model_path.replace('.pt', '_scalers.pkl')

    # Load config
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load scalers
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    # Create model
    print('Creating model...')
    model = LSTMModel(
        window_size=config['window_size'],
        feature_dim=config['feature_dim'],
        units=config['units'],
        layers=config['layers'],
        dropout_rate=config['dropout_rate']
    )
    model.build_model()

    # Load weights
    print('Loading model weights...')
    model.model.load_state_dict(torch.load(model_path))
    model.model.eval()

    # Set scalers
    model.X_scaler = scalers['X_scaler']
    model.y_scaler = scalers['y_scaler']
    model.is_fitted = True

    # Manually get the raw scaled predictions
    print('Getting raw scaled predictions...')
    # Scale features
    if len(X_test.shape) == 2:
        # Handle 2D data (samples, features)
        print("Input data is 2D, reshaping to 3D for LSTM...")
        n_samples, n_features = X_test.shape
        n_timesteps = 1  # Just one timestep when data is 2D
        X_scaled = model.X_scaler.transform(X_test)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    else:
        # Handle 3D data (samples, timesteps, features)
        n_samples, n_timesteps, n_features = X_test.shape
        X_reshaped = X_test.reshape(-1, n_features)
        X_scaled = model.X_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

    # Convert to PyTorch tensor and predict
    X_tensor = torch.FloatTensor(X_scaled).to(model.device)
    with torch.no_grad():
        y_pred_scaled = model.model(X_tensor).cpu().numpy()

    # Check shapes and values at each step
    print(f'X_test shape: {X_test.shape}')
    print(f'X_scaled shape: {X_scaled.shape}')
    print(f'Raw scaled predictions shape: {y_pred_scaled.shape}')
    print(f'Raw scaled predictions range: [{np.min(y_pred_scaled):.4f}, {np.max(y_pred_scaled):.4f}]')

    # Inverse transform the scaled predictions - method 1 (as in the model's predict method)
    print('Method 1: Using model predict method:')
    y_pred_method1 = model.predict(X_test)
    print(f'Method 1 shape: {y_pred_method1.shape}')
    print(f'Method 1 range: [{np.min(y_pred_method1):.4f}, {np.max(y_pred_method1):.4f}]')

    # Inverse transform the scaled predictions - method 2 (manual approach)
    print('Method 2: Manual inverse transform:')
    y_pred_method2 = model.y_scaler.inverse_transform(y_pred_scaled)
    print(f'Method 2 shape: {y_pred_method2.shape}')
    print(f'Method 2 range: [{np.min(y_pred_method2):.4f}, {np.max(y_pred_method2):.4f}]')

    # Directly inverse transform the raw scaled values
    print('Method 3: Reshape and inverse_transform:')
    reshaped_pred = y_pred_scaled.reshape(-1, 1)
    y_pred_method3 = model.y_scaler.inverse_transform(reshaped_pred)
    print(f'Method 3 shape: {y_pred_method3.shape}')
    print(f'Method 3 range: [{np.min(y_pred_method3):.4f}, {np.max(y_pred_method3):.4f}]')

    # Print actual target values
    print(f'Actual y_test shape: {y_test.shape}')
    print(f'Actual y_test range: [{np.min(y_test):.4f}, {np.max(y_test):.4f}]')

    # Print some sample predictions vs actual
    print('\nSample predictions:')
    for i in range(min(5, len(y_test))):
        print(f'Sample {i}:')
        print(f'  Actual: {y_test[i][0]:.4f}')
        print(f'  Predicted Method 1: {y_pred_method1[i]:.4f}')
        print(f'  Predicted Method 2: {y_pred_method2[i][0]:.4f}')
        print(f'  Predicted Method 3: {y_pred_method3[i][0]:.4f}')

    # Create a plot comparing the predictions from each method against actual values
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, y_test, label='Actual', linewidth=2)
    plt.plot(dates_test, y_pred_method1, label='Method 1 (model.predict)', linewidth=2, linestyle='--')
    plt.plot(dates_test, y_pred_method2, label='Method 2 (direct inverse_transform)', linewidth=2, linestyle=':')
    plt.plot(dates_test, y_pred_method3, label='Method 3 (reshape + inverse_transform)', linewidth=2, linestyle='-.')

    plt.title('AAPL Stock Price: Actual vs Predicted with Different Methods', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'diagnosis_plot.png'))
    print('\nDiagnostic plot saved to diagnosis_plot.png')

    # Calculate metrics for each prediction method
    mse1 = mean_squared_error(y_test, y_pred_method1)
    mae1 = mean_absolute_error(y_test, y_pred_method1)
    mse2 = mean_squared_error(y_test, y_pred_method2)
    mae2 = mean_absolute_error(y_test, y_pred_method2)
    mse3 = mean_squared_error(y_test, y_pred_method3)
    mae3 = mean_absolute_error(y_test, y_pred_method3)

    print('\nMetrics:')
    print(f'Method 1 - MSE: {mse1:.4f}, MAE: {mae1:.4f}')
    print(f'Method 2 - MSE: {mse2:.4f}, MAE: {mae2:.4f}')
    print(f'Method 3 - MSE: {mse3:.4f}, MAE: {mae3:.4f}')

    # Check if we actually have a differencing issue (predictions are changes instead of absolute prices)
    print('\nTesting if this might be a differencing issue:')
    # Convert predictions to price levels using cumulative sum
    first_price = y_test[0]
    y_pred_returns = y_pred_scaled.flatten()
    y_pred_cumsum = first_price * (1 + y_pred_returns.cumsum())
    print(f'Cumsum from first price shape: {y_pred_cumsum.shape}')
    print(f'Cumsum from first price range: [{np.min(y_pred_cumsum):.4f}, {np.max(y_pred_cumsum):.4f}]')

    # Try another approach - reconstructing from returns
    # Plot the returns-reconstructed predictions
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, y_test, label='Actual', linewidth=2)
    plt.plot(dates_test, y_pred_cumsum, label='Reconstructed from returns', linewidth=2, linestyle='--')
    plt.title('AAPL Stock Price: Testing Returns-based Reconstruction', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'returns_reconstruction_plot.png'))
    print('\nReturns reconstruction plot saved to returns_reconstruction_plot.png')

    # Print any other useful diagnostic information
    print('\nAdditional info:')
    print(f'X_scaler mean: {model.X_scaler.mean_}')
    print(f'X_scaler var: {model.X_scaler.var_}')
    print(f'y_scaler mean: {model.y_scaler.mean_}')
    print(f'y_scaler var: {model.y_scaler.var_}')
    
    # Check the preprocess.py file to see if we are using returns instead of prices
    print('\nNow check the preprocess.py file to see how the data was prepared')

if __name__ == "__main__":
    main()