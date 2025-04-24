import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import time
import yfinance as yf
import os

# Import our modules
from models import evaluate_model

def preprocess_data(data, lookback=60, test_size=0.2):
    """
    Preprocess stock data for LSTM model
    
    Args:
        data: DataFrame with stock prices
        lookback: Number of previous days to use for prediction
        test_size: Proportion of data to use for testing
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Use only the 'Close' price for prediction
    data = data[['Close']]
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create features and labels
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # Split into training and testing sets
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def build_model(input_shape, lstm_units=50, lstm_layers=2, dropout_rate=0.2):
    """
    Build an LSTM model with configurable hyperparameters
    
    Args:
        input_shape: Shape of input data (lookback, features)
        lstm_units: Number of LSTM units per layer
        lstm_layers: Number of LSTM layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer
    if lstm_layers > 1:
        model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    else:
        model.add(LSTM(units=lstm_units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers
    for i in range(1, lstm_layers):
        if i < lstm_layers - 1:
            model.add(LSTM(units=lstm_units, return_sequences=True))
        else:
            model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def hyperparameter_tuning(ticker='AAPL', start_date='2018-01-01', end_date='2023-01-01'):
    """
    Perform hyperparameter tuning for the LSTM model
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for stock data
        end_date: End date for stock data
        
    Returns:
        Results dataframe with performance metrics
    """
    # Fetch stock data
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Reshape input to be 3D [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Define hyperparameter grid
    param_grid = {
        'lstm_units': [50, 100],
        'lstm_layers': [1, 2],
        'dropout_rate': [0.2, 0.3],
        'batch_size': [16, 32],
        'epochs': [50, 100]
    }
    
    # Generate all combinations of hyperparameters
    grid = list(ParameterGrid(param_grid))
    print(f"Testing {len(grid)} hyperparameter combinations...")
    
    # Create a directory for results if it doesn't exist
    results_dir = 'tuning_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Store results
    results = []
    
    for i, params in enumerate(grid):
        print(f"\nTraining model {i+1}/{len(grid)} with parameters: {params}")
        
        # Build model with current hyperparameters
        model = build_model(
            input_shape=(X_train.shape[1], 1),
            lstm_units=params['lstm_units'],
            lstm_layers=params['lstm_layers'],
            dropout_rate=params['dropout_rate']
        )
        
        # Train model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_test, y_test),
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Evaluate model
        metrics = evaluate_model(y_test, predictions)
        
        # Record results
        result = {
            'model_id': i+1,
            **params,
            'mse': metrics['MSE'],
            'rmse': metrics['RMSE'],
            'r2': metrics['R2'],
            'training_time': training_time
        }
        results.append(result)
        
        # Create loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model {i+1} Loss - LSTM Units: {params["lstm_units"]}, Layers: {params["lstm_layers"]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{results_dir}/model_{i+1}_loss.png')
        plt.close()
        
        # Create predictions plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.title(f'Model {i+1} Predictions - LSTM Units: {params["lstm_units"]}, Layers: {params["lstm_layers"]}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price (Normalized)')
        plt.legend()
        plt.savefig(f'{results_dir}/model_{i+1}_predictions.png')
        plt.close()
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{results_dir}/hyperparameter_tuning_results.csv', index=False)
    
    # Print best model by MSE
    best_model = results_df.loc[results_df['mse'].idxmin()]
    print("\nBest model by MSE:")
    print(best_model)
    
    return results_df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run hyperparameter tuning
    results = hyperparameter_tuning('AAPL', '2018-01-01', '2023-01-01')
    
    # Plot training time vs MSE
    plt.figure(figsize=(10, 6))
    plt.scatter(results['training_time'], results['mse'], alpha=0.7)
    plt.title('Training Time vs MSE')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('tuning_results/training_time_vs_mse.png')
    plt.close()
    
    # Plot number of layers vs MSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='lstm_layers', y='mse', data=results)
    plt.title('Number of LSTM Layers vs MSE')
    plt.xlabel('Number of LSTM Layers')
    plt.ylabel('Mean Squared Error')
    plt.savefig('tuning_results/layers_vs_mse.png')
    plt.close()
    
    print("Hyperparameter tuning completed. Results saved to 'tuning_results' directory.")