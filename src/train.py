"""
Model Training Module

This module handles the training of various stock prediction models,
including configuration, training, and saving of model artifacts.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
import time
import json
from tqdm import tqdm

from models.baseline import LinearRegressionModel, ARIMAModel
from models.advanced import LSTMModel, TransformerModel


def load_data(symbol, data_dir):
    """
    Load preprocessed data for a specific stock symbol.
    
    Args:
        symbol (str): Stock symbol
        data_dir (str): Directory containing the processed data
        
    Returns:
        tuple: X features, y targets, dates array
    """
    train_file = os.path.join(data_dir, f"{symbol}_train_scaled.npz")
    val_file = os.path.join(data_dir, f"{symbol}_val_scaled.npz")
    test_file = os.path.join(data_dir, f"{symbol}_test_scaled.npz")
    
    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        raise FileNotFoundError(f"Processed data files not found for {symbol} in {data_dir}")
    
    train_data = np.load(train_file)
    val_data = np.load(val_file)
    test_data = np.load(test_file)
    
    X_train = train_data['features']
    y_train = train_data['targets']
    dates_train = train_data['dates']
    
    X_val = val_data['features']
    y_val = val_data['targets']
    dates_val = val_data['dates']
    
    X_test = test_data['features']
    y_test = test_data['targets']
    dates_test = test_data['dates']
    
    # Combine all data for compatibility with existing split logic
    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)
    dates = np.concatenate([dates_train, dates_val, dates_test], axis=0)
    
    return X, y, dates


def split_data(X, y, dates, test_size=0.2, validation_size=0.1, random_state=42, time_based=True):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X (numpy.ndarray): Feature array
        y (numpy.ndarray): Target array
        dates (numpy.ndarray): Array of dates
        test_size (float): Proportion of data for testing
        validation_size (float): Proportion of training data for validation
        random_state (int): Random seed for reproducibility
        time_based (bool): If True, use time-based split; otherwise, use random split
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test
    """
    if time_based:
        # Time-based split (no shuffling)
        n_samples = len(X)
        test_idx = int(n_samples * (1 - test_size))
        
        X_temp, X_test = X[:test_idx], X[test_idx:]
        y_temp, y_test = y[:test_idx], y[test_idx:]
        dates_temp, dates_test = dates[:test_idx], dates[test_idx:]
        
        n_temp = len(X_temp)
        val_idx = int(n_temp * (1 - validation_size))
        
        X_train, X_val = X_temp[:val_idx], X_temp[val_idx:]
        y_train, y_val = y_temp[:val_idx], y_temp[val_idx:]
        dates_train, dates_val = dates_temp[:val_idx], dates_temp[val_idx:]
    else:
        # Random split (shuffled)
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=test_size, random_state=random_state
        )
        
        X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(
            X_train, y_train, dates_train, test_size=validation_size/(1-test_size),
            random_state=random_state
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test


def walk_forward_cv(X, y, n_splits=5):
    """
    Perform walk-forward cross-validation for time-series data.

    Args:
        X (numpy.ndarray): Feature array
        y (numpy.ndarray): Target array
        n_splits (int): Number of splits for cross-validation

    Yields:
        tuple: (X_train, X_val, y_train, y_val)
    """
    n_samples = len(X)
    split_size = n_samples // (n_splits + 1)

    for i in range(n_splits):
        train_end = split_size * (i + 1)
        val_end = train_end + split_size

        X_train, X_val = X[:train_end], X[train_end:val_end]
        y_train, y_val = y[:train_end], y[train_end:val_end]

        yield X_train, X_val, y_train, y_val


def train_model(model_type, X_train, y_train, X_val=None, y_val=None, **model_params):
    """
    Train a model of the specified type.
    
    Args:
        model_type (str): Type of model ('linear', 'arima', 'lstm', 'transformer')
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_val (numpy.ndarray, optional): Validation features
        y_val (numpy.ndarray, optional): Validation targets
        **model_params: Additional parameters for the specific model
        
    Returns:
        tuple: Trained model, training history, training time
    """
    start_time = time.time()
    history = None
    
    if model_type == 'linear':
        model = LinearRegressionModel()
        model.fit(X_train, y_train)
        
    elif model_type == 'arima':
        order = model_params.get('order', (5, 1, 0))
        model = ARIMAModel(order=order)
        model.fit(y_train, X_train)
        
    elif model_type == 'lstm':
        # Check dimensions and reshape if needed
        if len(X_train.shape) == 2:
            # Reshape 2D data to 3D: (samples, 1, features)
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        n_samples, n_timesteps, n_features = X_train.shape
        
        units = model_params.get('units', 64)
        layers = model_params.get('layers', 2)
        dropout_rate = model_params.get('dropout_rate', 0.2)
        epochs = model_params.get('epochs', 50)
        batch_size = model_params.get('batch_size', 32)
        patience = model_params.get('patience', 10)
        
        model = LSTMModel(
            window_size=n_timesteps,
            feature_dim=n_features,
            units=units,
            layers=layers,
            dropout_rate=dropout_rate
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            validation_split=0.1 if X_val is None else 0.0
        )
        
    elif model_type == 'transformer':
        # Check dimensions and reshape if needed
        if len(X_train.shape) == 2:
            # Reshape 2D data to 3D: (samples, 1, features)
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        n_samples, n_timesteps, n_features = X_train.shape
        
        head_size = model_params.get('head_size', 256)
        num_heads = model_params.get('num_heads', 4)
        ff_dim = model_params.get('ff_dim', 512)
        num_transformer_blocks = model_params.get('num_transformer_blocks', 2)
        mlp_units = model_params.get('mlp_units', [128, 64])
        dropout_rate = model_params.get('dropout_rate', 0.2)
        epochs = model_params.get('epochs', 50)
        batch_size = model_params.get('batch_size', 32)
        patience = model_params.get('patience', 10)
        
        model = TransformerModel(
            window_size=n_timesteps,
            feature_dim=n_features,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            dropout_rate=dropout_rate
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            validation_split=0.1 if X_val is None else 0.0
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    training_time = time.time() - start_time
    
    return model, history, training_time


def save_model(model, model_type, model_dir, symbol):
    """
    Save the trained model and its metadata.
    
    Args:
        model: The trained model
        model_type (str): Type of model ('linear', 'arima', 'lstm', 'transformer')
        model_dir (str): Directory to save the model
        symbol (str): Stock symbol
        
    Returns:
        str: Path to the saved model
    """
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{symbol}_{model_type}_{timestamp}"
    
    model_path = os.path.join(model_dir, model_filename)
    
    if model_type in ['linear', 'arima']:
        # Save scikit-learn or statsmodels model with pickle
        with open(f"{model_path}.pkl", 'wb') as f:
            pickle.dump(model, f)
        return f"{model_path}.pkl"
    elif model_type in ['lstm', 'transformer']:
        # For PyTorch models, save the state_dict
        import torch
        torch.save(model.model.state_dict(), f"{model_path}.pt")
        
        # Save model hyperparameters separately for reconstruction
        model_params = model.__dict__.copy()
        model_params.pop('model', None)
        model_params.pop('history', None)
        model_params.pop('X_scaler', None)
        model_params.pop('y_scaler', None)
        
        # Remove device object which is not JSON serializable
        if 'device' in model_params:
            model_params['device'] = str(model_params['device'])
        
        # Save scalers separately
        with open(f"{model_path}_scalers.pkl", 'wb') as f:
            pickle.dump({'X_scaler': model.X_scaler, 'y_scaler': model.y_scaler}, f)
            
        # Save model config, with special handling for non-serializable types
        with open(f"{model_path}_config.json", 'w') as f:
            json_params = {}
            for k, v in model_params.items():
                if isinstance(v, np.ndarray):
                    json_params[k] = v.tolist()
                elif isinstance(v, (str, int, float, bool, type(None))):
                    json_params[k] = v
                else:
                    # Convert other non-serializable types to string representation
                    json_params[k] = str(v)
            json.dump(json_params, f)
        
        return f"{model_path}.pt"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_training_history(history, model_type, output_dir, symbol):
    """
    Save training history and generate loss curve plots.
    
    Args:
        history: Training history object (dictionary)
        model_type (str): Type of model
        output_dir (str): Directory to save the history
        symbol (str): Stock symbol
        
    Returns:
        str: Path to the saved history plot
    """
    if history is None or model_type in ['linear', 'arima']:
        return None
    
    # Create directory for figures if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Save history to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Fix for different length arrays: ensure all arrays have the same length
    max_length = max(len(v) for v in history.values())
    history_fixed = {}
    for key, values in history.items():
        if len(values) < max_length:
            # Pad shorter arrays with NaN values
            history_fixed[key] = values + [float('nan')] * (max_length - len(values))
        else:
            history_fixed[key] = values
    
    history_df = pd.DataFrame(history_fixed)
    history_csv_path = os.path.join(output_dir, f"{symbol}_{model_type}_history_{timestamp}.csv")
    history_df.to_csv(history_csv_path, index=False)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{symbol} - {model_type.upper()} Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE if available
    if 'mae' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            plt.plot(history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title(f'{symbol} - {model_type.upper()} Training MAE')
        plt.legend()
        plt.grid(True)
    
    # Save the figure
    plot_path = os.path.join(output_dir, 'figures', f"{symbol}_{model_type}_loss_curve_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def main():
    """Main function to parse arguments and run model training."""
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument("--model", type=str, default="lstm", 
                        choices=["linear", "arima", "lstm", "transformer"],
                        help="Model type to train")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to train on")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs for neural models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for neural models")
    parser.add_argument("--window_size", type=int, default=20, help="Window size for time series data")
    
    # Advanced model parameters
    parser.add_argument("--units", type=int, default=64, help="Number of units in LSTM layers")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers in neural networks")
    parser.add_argument("--head_size", type=int, default=256, help="Head size for transformer models")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for testing")
    parser.add_argument("--time_based", type=bool, default=True, help="Use time-based train/test split")
    
    args = parser.parse_args()
    
    # Create output directory
    model_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading data for {args.symbol}...")
    X, y, dates = load_data(args.symbol, args.data_dir)
    
    print(f"Splitting data (time-based: {args.time_based})...")
    X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = split_data(
        X, y, dates, 
        test_size=args.test_size, 
        time_based=args.time_based
    )
    
    # Extract model-specific parameters
    model_params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'units': args.units,
        'layers': args.layers,
        'head_size': args.head_size,
        'num_heads': args.num_heads,
    }
    
    if args.time_based:
        print("Performing walk-forward cross-validation...")
        for fold, (X_train, X_val, y_train, y_val) in enumerate(walk_forward_cv(X, y)):
            print(f"Fold {fold + 1}: Train set: {X_train.shape}, Validation set: {X_val.shape}")

            model, history, training_time = train_model(
                args.model, 
                X_train, y_train, 
                X_val, y_val, 
                **model_params
            )

            print(f"Fold {fold + 1} training completed in {training_time:.2f} seconds")

            # Save model and history for each fold
            model_path = save_model(model, args.model, model_dir, f"{args.symbol}_fold{fold + 1}")
            print(f"Model for fold {fold + 1} saved to {model_path}")

            if history is not None:
                history_path = save_training_history(history, args.model, args.output_dir, f"{args.symbol}_fold{fold + 1}")
                print(f"Training history for fold {fold + 1} saved to {history_path}")
    else:
        print(f"Training {args.model} model for {args.symbol}...")
        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        model, history, training_time = train_model(
            args.model, 
            X_train, y_train, 
            X_val, y_val, 
            **model_params
        )
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model and training history
        model_path = save_model(model, args.model, model_dir, args.symbol)
        print(f"Model saved to {model_path}")
        
        if history is not None:
            history_path = save_training_history(history, args.model, args.output_dir, args.symbol)
            print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()