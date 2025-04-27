"""
Stock Market Prediction - Models Module
This module implements various machine learning models for stock price prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Try importing TensorFlow/Keras if available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. LSTM model will not be available.")
    TENSORFLOW_AVAILABLE = False

class BaseModel:
    """Base class for all stock prediction models"""
    
    def __init__(self, name="BaseModel"):
        self.name = name
        self.model = None
        self.training_time = None
        self.history = None
    
    def fit(self, X_train, y_train):
        """Train the model"""
        raise NotImplementedError("Subclass must implement abstract method")
    
    def predict(self, X_test):
        """Make predictions"""
        raise NotImplementedError("Subclass must implement abstract method")
    
    def evaluate(self, X_test, y_test, scaler=None):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True target values
            scaler: Scaler used for inverse transformation
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        start_time = time.time()
        y_pred = self.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Training Time': self.training_time,
            'Inference Time': inference_time
        }
        
        # Print evaluation results
        print(f"\n{self.name} Model Evaluation:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Training Time: {self.training_time:.2f} seconds")
        print(f"Inference Time: {inference_time:.6f} seconds")
        
        return metrics, y_pred
    
    def save(self, filepath):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")


class LinearRegressionModel(BaseModel):
    """Linear Regression model for stock price prediction"""
    
    def __init__(self):
        super().__init__(name="Linear Regression")
        self.model = LinearRegression()
        
    def fit(self, X_train, y_train):
        """Train the model"""
        # Record start time
        start_time = time.time()
        
        # Reshape input if needed
        if len(X_train.shape) == 3:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_reshaped = X_train
        
        # Train the model
        self.model.fit(X_train_reshaped, y_train)
        
        # Calculate training time
        self.training_time = time.time() - start_time
        print(f"{self.name} model trained in {self.training_time:.2f} seconds")
        
    def predict(self, X_test):
        """Make predictions"""
        # Reshape input if needed
        if len(X_test.shape) == 3:
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_reshaped = X_test
        
        return self.model.predict(X_test_reshaped)


class RandomForestModel(BaseModel):
    """Random Forest model for stock price prediction"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__(name="Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
    def fit(self, X_train, y_train):
        """Train the model"""
        # Record start time
        start_time = time.time()
        
        # Reshape input if needed
        if len(X_train.shape) == 3:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_reshaped = X_train
        
        # Train the model
        self.model.fit(X_train_reshaped, y_train)
        
        # Calculate training time
        self.training_time = time.time() - start_time
        print(f"{self.name} model trained in {self.training_time:.2f} seconds")
        
    def predict(self, X_test):
        """Make predictions"""
        # Reshape input if needed
        if len(X_test.shape) == 3:
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_reshaped = X_test
        
        return self.model.predict(X_test_reshaped)
    
    def feature_importance(self, sequence_length):
        """
        Get feature importance from the Random Forest model
        
        Args:
            sequence_length: Length of input sequences
            
        Returns:
            DataFrame with feature importances
        """
        if hasattr(self.model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'Feature': [f'Day-{i+1}' for i in range(sequence_length)],
                'Importance': self.model.feature_importances_[:sequence_length]
            })
            return feature_imp.sort_values(by='Importance', ascending=False)
        else:
            return None


class LSTMModel(BaseModel):
    """LSTM model for stock price prediction"""
    
    def __init__(self, input_shape, units=50, dropout=0.2, learning_rate=0.001):
        super().__init__(name="LSTM")
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow is not available. Using a simple model instead.")
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(units, units//2),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=learning_rate,
                max_iter=200
            )
            return
            
        self.model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units=units),
            Dropout(dropout),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Train the model"""
        # Record start time
        start_time = time.time()
        
        # Train the model
        if TENSORFLOW_AVAILABLE and not hasattr(self.model, 'fit_transform'):
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            self.history = history.history
        else:
            # For scikit-learn model
            # Reshape input if needed
            if len(X_train.shape) == 3:
                X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            else:
                X_train_reshaped = X_train
            
            self.model.fit(X_train_reshaped, y_train)
            if hasattr(self.model, 'loss_curve_'):
                self.history = {'loss': self.model.loss_curve_}
        
        # Calculate training time
        self.training_time = time.time() - start_time
        print(f"{self.name} model trained in {self.training_time:.2f} seconds")
        
    def predict(self, X_test):
        """Make predictions"""
        if TENSORFLOW_AVAILABLE and not hasattr(self.model, 'fit_transform'):
            return self.model.predict(X_test, verbose=0).flatten()
        else:
            # For scikit-learn model
            # Reshape input if needed
            if len(X_test.shape) == 3:
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            else:
                X_test_reshaped = X_test
            
            return self.model.predict(X_test_reshaped)
    
    def save(self, filepath):
        """Save the model to disk"""
        if TENSORFLOW_AVAILABLE and not hasattr(self.model, 'fit_transform'):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"LSTM model saved to {filepath}")
        else:
            super().save(filepath)
    
    def load(self, filepath):
        """Load the model from disk"""
        if TENSORFLOW_AVAILABLE and os.path.isdir(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"LSTM model loaded from {filepath}")
        else:
            super().load(filepath)


def plot_predictions(y_true, y_pred, title="Stock Price Prediction", save_path=None):
    """
    Plot actual vs predicted values
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue', linewidth=2)
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def plot_training_loss(history, title="Training Loss", save_path=None):
    """
    Plot training loss
    
    Args:
        history: Training history dictionary with 'loss' key
        title: Plot title
        save_path: Path to save the plot
    """
    if history is None or not isinstance(history, dict) or 'loss' not in history:
        print("Valid training history not available")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
    
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def compare_models(models_list, X_test, y_test, scaler=None, save_path=None):
    """
    Compare multiple models' performance
    
    Args:
        models_list: List of trained model objects
        X_test: Test features
        y_test: Test targets
        scaler: Scaler for inverse transformation
        save_path: Path to save comparison plot
        
    Returns:
        DataFrame with model comparison metrics
    """
    results = {}
    predictions = {}
    
    for model in models_list:
        metrics, y_pred = model.evaluate(X_test, y_test, scaler)
        results[model.name] = metrics
        predictions[model.name] = y_pred
    
    # Plot comparison of predictions
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual', color='black', linewidth=2)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, y_pred) in enumerate(predictions.items()):
        plt.plot(y_pred, label=f'{name}', color=colors[i % len(colors)], linestyle='--')
    
    plt.title('Model Comparison: Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()
    
    # Return comparison DataFrame
    return pd.DataFrame(results).T