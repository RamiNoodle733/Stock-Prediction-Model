import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time

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

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Train a Linear Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        
    Returns:
        Trained model, predictions, training time
    """
    # Create model
    model = LinearRegression()
    
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return model, predictions, training_time

def train_ridge_regression(X_train, y_train, X_test, y_test, alpha=1.0):
    """
    Train a Ridge Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        alpha: Regularization strength
        
    Returns:
        Trained model, predictions, training time
    """
    # Create model
    model = Ridge(alpha=alpha)
    
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return model, predictions, training_time

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None):
    """
    Train a Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        
    Returns:
        Trained model, predictions, training time
    """
    # Create model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return model, predictions, training_time

def train_gradient_boosting(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Train a Gradient Boosting model (advanced model replacement for LSTM)
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        n_estimators: Number of boosting stages
        learning_rate: Learning rate
        max_depth: Maximum depth of trees
        
    Returns:
        Trained model, predictions, training history, training time
    """
    # Create model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create a mock history object similar to Keras for plotting
    class MockHistory:
        def __init__(self, model):
            self.model = model
            self.history = {}
            
            # Get training and validation losses (approximation)
            train_losses = []
            val_losses = []
            
            # Using staged_predict to get losses at each stage
            for i, train_pred_stage in enumerate(model.staged_predict(X_train)):
                train_mse = mean_squared_error(y_train, train_pred_stage)
                train_losses.append(train_mse)
                
                test_pred_stage = model.predict(X_test)
                val_mse = mean_squared_error(y_test, test_pred_stage)
                val_losses.append(val_mse)
                
                if i >= 10:  # Just use first 10 stages for performance
                    break
            
            self.history['loss'] = train_losses
            self.history['val_loss'] = val_losses
    
    history = MockHistory(model)
    
    return model, history, predictions, training_time

def plot_training_loss(history, title='Model Loss During Training', save_path=None):
    """
    Plot training and validation loss
    
    Args:
        history: Training history object
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Iterations/Trees')
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
    residuals = y_true - y_pred
    
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
                        target_column='Close', sequence_length=10, train_split=0.8):
    """
    Preprocess stock data for ML models
    
    Args:
        data: DataFrame with stock data
        feature_columns: List of feature columns to use
        target_column: Target column to predict
        sequence_length: Number of time steps to use for prediction
        train_split: Fraction of data to use for training
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Reset index to handle any potential MultiIndex issues
    df = data.copy()
    
    # If we have a MultiIndex in columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Create lag features directly in the dataframe
    for col in feature_columns:
        if col in df.columns:  # Only create features for columns that exist
            for lag in range(1, sequence_length + 1):
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Add some technical indicators
    # Moving averages
    if 'Close' in df.columns:  # Make sure the Close column exists
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Drop rows with NaN (due to lag creation)
    df = df.dropna()
    
    # Prepare features and target
    feature_cols = []
    
    # Add original feature columns (except target)
    for col in feature_columns:
        if col in df.columns and col != target_column:
            feature_cols.append(col)
    
    # Add lag columns and MA columns by checking column names
    for col in df.columns:
        col_name = str(col)  # Convert to string to be safe
        if '_lag_' in col_name:
            feature_cols.append(col)
        elif col_name in ['MA5', 'MA20']:
            feature_cols.append(col)
    
    # Make sure all feature columns exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].values
    y = df[target_column].values
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a separate scaler for the target
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split into training and testing sets
    split_idx = int(len(X_scaled) * train_split)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    return X_train, X_test, y_train, y_test, y_scaler