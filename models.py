import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """
    Build an LSTM model for time series forecasting
    
    Args:
        input_shape: Shape of input data (time steps, features)
        units: Number of LSTM units
        dropout_rate: Dropout rate to prevent overfitting
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer with dropout
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer with dropout
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def build_deep_lstm_model(input_shape, units=50, dropout_rate=0.2, layers=3):
    """
    Build a deep LSTM model with multiple stacked LSTM layers
    
    Args:
        input_shape: Shape of input data (time steps, features)
        units: Number of LSTM units
        dropout_rate: Dropout rate to prevent overfitting
        layers: Number of LSTM layers
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=units, return_sequences=True if layers > 1 else False, 
                  input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Middle LSTM layers
    for i in range(1, layers-1):
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    # Last LSTM layer (if more than one layer)
    if layers > 1:
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32, verbose=1):
    """
    Train an LSTM model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity level
        
    Returns:
        Trained model, training history, predictions, training time
    """
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        restore_best_weights=True
    )
    
    # Train model and measure time
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=verbose
    )
    
    training_time = time.time() - start_time
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return model, history, predictions, training_time

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_training_loss(history, title='Model Loss During Training', save_path=None):
    """
    Plot training and validation loss
    
    Args:
        history: Training history from model.fit()
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_predictions(y_true, y_pred, title='Actual vs Predicted', save_path=None):
    """
    Plot actual vs predicted values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_residuals(y_true, y_pred, title='Prediction Residuals', save_path=None):
    """
    Plot prediction residuals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure
    """
    residuals = y_true - y_pred.reshape(-1)
    
    plt.figure(figsize=(12, 10))
    
    # Residual plot
    plt.subplot(2, 1, 1)
    plt.plot(residuals, marker='o', linestyle='None', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'{title} Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.grid(True)
    
    # Histogram
    plt.subplot(2, 1, 2)
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual Value')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def preprocess_stock_data(data, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                        target_column='Close', sequence_length=60, train_split=0.8):
    """
    Preprocess stock data for LSTM model
    
    Args:
        data: DataFrame with stock data
        feature_columns: List of feature columns to use
        target_column: Target column to predict
        sequence_length: Number of time steps to use for prediction
        train_split: Fraction of data to use for training
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Select features
    features = data[feature_columns].values
    
    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X = []
    y = []
    
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i+sequence_length])
        # We're predicting the closing price
        target_idx = feature_columns.index(target_column)
        y.append(features_scaled[i+sequence_length, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into training and testing sets
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler