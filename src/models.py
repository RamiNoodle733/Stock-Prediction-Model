import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

class BaseStockModel:
    """Base class for stock prediction models"""
    
    def __init__(self, name="BaseModel"):
        self.name = name
        self.model = None
        self.training_time = None
        self.history = None
    
    def train(self, X_train, y_train):
        """Train the model"""
        start_time = time.time()
        self.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        print(f"{self.name} trained in {self.training_time:.2f} seconds")
    
    def fit(self, X_train, y_train):
        """Implement in subclass"""
        raise NotImplementedError("Subclass must implement abstract method")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Training Time': self.training_time
        }
        
        print(f"\n{self.name} Model Evaluation:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Training Time: {self.training_time:.2f} seconds")
        
        return results, y_pred
    
    def save_model(self, filepath):
        """Save model to disk"""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")


class LinearRegressionModel(BaseStockModel):
    """Linear Regression model for stock prediction"""
    
    def __init__(self):
        super().__init__(name="Linear Regression")
        self.model = LinearRegression()
    
    def fit(self, X_train, y_train):
        # Linear regression expects 2D input, so reshape if necessary
        if len(X_train.shape) == 3:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_reshaped = X_train
        
        self.model.fit(X_train_reshaped, y_train)
    
    def predict(self, X):
        # Reshape input if necessary
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        return self.model.predict(X_reshaped)


class RandomForestModel(BaseStockModel):
    """Random Forest model for stock prediction"""
    
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__(name="Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            random_state=42
        )
    
    def fit(self, X_train, y_train):
        # Random forest expects 2D input
        if len(X_train.shape) == 3:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_reshaped = X_train
        
        self.model.fit(X_train_reshaped, y_train)
    
    def predict(self, X):
        # Reshape input if necessary
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        return self.model.predict(X_reshaped)


class LSTMModel(BaseStockModel):
    """LSTM model for stock prediction using scikit-learn API"""
    
    def __init__(self, input_shape, units=50, dropout=0.2):
        super().__init__(name="LSTM")
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.history = None
        
        try:
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(units, units//2),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=200,
                shuffle=True,
                random_state=42,
                early_stopping=True
            )
        except ImportError:
            print("Note: Using MLPRegressor as a substitute for LSTM since TensorFlow is not available")
            self.model = None
    
    def fit(self, X_train, y_train):
        # MLPRegressor expects 2D input
        if len(X_train.shape) == 3:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_reshaped = X_train
        
        if self.model is not None:
            self.model.fit(X_train_reshaped, y_train)
            # Store loss curve as history
            self.history = {'loss': self.model.loss_curve_}
            
    def predict(self, X):
        # Reshape input if necessary
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        return self.model.predict(X_reshaped)


def plot_predictions(actual, predictions, title="Stock Price Prediction", xlabel="Time", ylabel="Price"):
    """
    Plot actual vs predicted values
    
    Parameters:
    -----------
    actual : array
        Actual values
    predictions : array
        Predicted values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', linewidth=2)
    plt.plot(predictions, label='Predicted', linewidth=2, linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../data/{title.replace(" ", "_").lower()}.png')
    plt.show()


def plot_training_history(history, title="Training Loss Over Time"):
    """
    Plot training history (loss curve)
    
    Parameters:
    -----------
    history : dict
        Training history dictionary with 'loss' key
    title : str
        Plot title
    """
    if history is None or 'loss' not in history:
        print("No training history available")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Loss')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../data/training_loss_curve.png')
    plt.show()


def compare_models(models, X_test, y_test, scaler=None, original_data=None):
    """
    Compare multiple models' performance
    
    Parameters:
    -----------
    models : list
        List of trained model objects
    X_test : array
        Test features
    y_test : array
        Test targets
    scaler : MinMaxScaler
        Scaler used to normalize data
    original_data : DataFrame
        Original stock data
    
    Returns:
    --------
    DataFrame: Comparison of model metrics
    """
    results = {}
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual', linewidth=2)
    
    for model in models:
        metrics, y_pred = model.evaluate(X_test, y_test)
        plt.plot(y_pred, label=f'{model.name} Prediction', linestyle='--')
        results[model.name] = metrics
    
    plt.title('Model Comparison: Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../data/model_comparison.png')
    plt.show()
    
    # Convert results to DataFrame for easier comparison
    return pd.DataFrame(results).T