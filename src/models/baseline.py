"""
Baseline Models for Stock Price Prediction

This module implements simple baseline models for stock price prediction,
including Linear Regression and ARIMA models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class LinearRegressionModel:
    """Linear Regression model implementation."""

    def __init__(self):
        """Initialize the Linear Regression model."""
        self.model = LinearRegression()
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Train the Linear Regression model on the given data.
        
        Args:
            X_train (numpy.ndarray): Training features of shape (n_samples, n_timesteps, n_features)
            y_train (numpy.ndarray): Training targets of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        # Reshape 3D data to 2D for sklearn
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(n_samples, n_timesteps * n_features)
        
        # Scale the data
        X_train_scaled = self.X_scaler.fit_transform(X_train_2d)
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Fit the linear model
        self.model.fit(X_train_scaled, y_train_scaled)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (numpy.ndarray): Features to predict on, shape (n_samples, n_timesteps, n_features)
                               
        Returns:
            numpy.ndarray: Predicted stock prices
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Reshape 3D data to 2D for sklearn
        n_samples, n_timesteps, n_features = X.shape
        X_2d = X.reshape(n_samples, n_timesteps * n_features)
        
        # Scale the data
        X_scaled = self.X_scaler.transform(X_2d)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform to original scale
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred


class ARIMAModel:
    """ARIMA (AutoRegressive Integrated Moving Average) model implementation."""

    def __init__(self, order=(5, 1, 0)):
        """
        Initialize the ARIMA model.
        
        Args:
            order (tuple): ARIMA order parameters: (p, d, q)
        """
        self.order = order
        self.model = None
        self.is_fitted = False
        self.last_values = None
    
    def fit(self, y_train, X_train=None):
        """
        Train the ARIMA model on the given data.
        
        Args:
            y_train (numpy.ndarray): Training targets of shape (n_samples,)
            X_train (numpy.ndarray, optional): Exogenous variables (not used in basic ARIMA)
            
        Returns:
            self: The fitted model
        """
        # Convert to pandas Series for statsmodels
        train_series = pd.Series(y_train)
        
        # Fit ARIMA model
        self.model = sm.tsa.ARIMA(train_series, order=self.order)
        self.fit_result = self.model.fit()
        
        # Store the last values for forecasting
        self.last_values = y_train
        self.is_fitted = True
        
        return self
    
    def predict(self, steps=None, X=None):
        """
        Make predictions using the trained model.
        
        Args:
            steps (int): Number of steps to forecast
            X (numpy.ndarray, optional): Exogenous variables (not used in basic ARIMA)
                               
        Returns:
            numpy.ndarray: Predicted stock prices
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if steps is None and X is not None:
            steps = len(X)
        elif steps is None:
            steps = 1
        
        # Make predictions
        y_pred = self.fit_result.forecast(steps=steps).values
        
        return y_pred