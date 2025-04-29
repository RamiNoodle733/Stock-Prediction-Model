"""
Advanced Models for Stock Price Prediction

This module implements advanced deep learning models for stock price prediction,
including LSTM and Transformer models, using PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import time


class LSTMModel:
    """LSTM model implementation using PyTorch."""

    def __init__(self, window_size, feature_dim, units=64, layers=2, dropout_rate=0.2):
        """
        Initialize the LSTM model.
        
        Args:
            window_size (int): Number of time steps in each input sequence
            feature_dim (int): Number of features at each time step
            units (int): Number of LSTM units in each layer
            layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
        """
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.units = units
        self.layers = layers
        self.dropout_rate = dropout_rate
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.history = None
        self.is_fitted = False
        
        self.build_model()
    
    def build_model(self):
        """Build the LSTM neural network architecture."""
        self.model = LSTMNetwork(
            input_dim=self.feature_dim,
            hidden_dim=self.units,
            num_layers=self.layers,
            dropout=self.dropout_rate
        ).to(self.device)
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, patience=10):
        """
        Train the LSTM model on the given data.
        
        Args:
            X_train (numpy.ndarray): Training features of shape (n_samples, n_timesteps, n_features)
            y_train (numpy.ndarray): Training targets of shape (n_samples,)
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion of training data to use for validation
            patience (int): Number of epochs to wait for improvement before early stopping
            
        Returns:
            dict: Training history
        """
        # Scale features (reshape to 2D for scaling, then back to 3D)
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = self.X_scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Scale targets
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Split into train and validation sets
        if validation_split > 0:
            val_size = int(n_samples * validation_split)
            train_size = n_samples - val_size
            
            X_val = X_train_scaled[-val_size:]
            y_val = y_train_scaled[-val_size:]
            X_train_scaled = X_train_scaled[:train_size]
            y_train_scaled = y_train_scaled[:train_size]
        else:
            X_val = None
            y_val = None
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        
        # Create DataLoader for batching
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor.unsqueeze(1))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        # Initialize tracking variables for early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_mae = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - y_batch)).item()
            
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0
                val_mae = 0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                        val_mae += torch.mean(torch.abs(outputs - y_batch)).item()
                
                val_loss /= len(val_loader)
                val_mae /= len(val_loader)
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model weights
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        # Restore best weights
                        self.model.load_state_dict(best_model_state)
                        break
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
            
            history['loss'].append(train_loss)
            history['mae'].append(train_mae)
        
        self.history = history
        self.is_fitted = True
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (numpy.ndarray): Features to predict on, shape (n_samples, n_features) or (n_samples, n_timesteps, n_features)
                               
        Returns:
            numpy.ndarray: Predicted stock prices
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Check if X is 2D or 3D, and reshape if needed
        if len(X.shape) == 2:
            # Reshape 2D data to 3D: (samples, 1, features)
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Scale features
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.X_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
            
        # Print raw scaled output range for debugging
        print(f"Raw scaled output range: [{np.min(y_pred_scaled):.4f}, {np.max(y_pred_scaled):.4f}]")
        
        # Reshape to ensure it's a 2D array with shape (n_samples, 1)
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        # Inverse transform to original scale
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        
        # Flatten to 1D array
        y_pred = y_pred.flatten()
        
        # Print inverse-scaled output range for debugging
        print(f"After inverse scaling: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
        
        # Check for prediction collapse
        if y_pred.max() - y_pred.min() < 10:
            print("WARNING: Predictions collapsed to a narrow band; check your scaler or model capacity.")
        
        return y_pred


class TransformerModel:
    """Transformer model implementation using PyTorch."""

    def __init__(self, window_size, feature_dim, head_size=256, num_heads=4, 
                 ff_dim=512, num_transformer_blocks=2, mlp_units=[128, 64], dropout_rate=0.2):
        """
        Initialize the Transformer model.
        
        Args:
            window_size (int): Number of time steps in each input sequence
            feature_dim (int): Number of features at each time step
            head_size (int): Size of attention heads
            num_heads (int): Number of attention heads
            ff_dim (int): Hidden layer size in feed forward network inside transformer
            num_transformer_blocks (int): Number of transformer blocks
            mlp_units (list): List of units in MLP layers
            dropout_rate (float): Dropout rate for regularization
        """
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.history = None
        self.is_fitted = False
        
        self.build_model()
    
    def build_model(self):
        """Build the Transformer neural network architecture."""
        self.model = TransformerNetwork(
            input_dim=self.feature_dim,
            d_model=self.head_size,
            nhead=self.num_heads,
            num_layers=self.num_transformer_blocks,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout_rate,
            mlp_units=self.mlp_units
        ).to(self.device)
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, patience=10):
        """
        Train the Transformer model on the given data.
        
        Args:
            X_train (numpy.ndarray): Training features of shape (n_samples, n_timesteps, n_features)
            y_train (numpy.ndarray): Training targets of shape (n_samples,)
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion of training data to use for validation
            patience (int): Number of epochs to wait for improvement before early stopping
            
        Returns:
            dict: Training history
        """
        # Scale features (reshape to 2D for scaling, then back to 3D)
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = self.X_scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Scale targets
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Split into train and validation sets
        if validation_split > 0:
            val_size = int(n_samples * validation_split)
            train_size = n_samples - val_size
            
            X_val = X_train_scaled[-val_size:]
            y_val = y_train_scaled[-val_size:]
            X_train_scaled = X_train_scaled[:train_size]
            y_train_scaled = y_train_scaled[:train_size]
        else:
            X_val = None
            y_val = None
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        
        # Create DataLoader for batching
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor.unsqueeze(1))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        # Initialize tracking variables for early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_mae = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - y_batch)).item()
            
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0
                val_mae = 0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                        val_mae += torch.mean(torch.abs(outputs - y_batch)).item()
                
                val_loss /= len(val_loader)
                val_mae /= len(val_loader)
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model weights
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        # Restore best weights
                        self.model.load_state_dict(best_model_state)
                        break
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
            
            history['loss'].append(train_loss)
            history['mae'].append(train_mae)
        
        self.history = history
        self.is_fitted = True
        
        return history
    
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
        
        # Scale features
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.X_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
            
        # Print raw scaled output range for debugging
        print(f"Raw scaled output range: [{np.min(y_pred_scaled):.4f}, {np.max(y_pred_scaled):.4f}]")
        
        # Reshape to ensure it's a 2D array with shape (n_samples, 1)
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        # Inverse transform to original scale
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        
        # Flatten to 1D array
        y_pred = y_pred.flatten()
        
        # Print inverse-scaled output range for debugging
        print(f"After inverse scaling: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
        
        # Check for prediction collapse
        if y_pred.max() - y_pred.min() < 10:
            print("WARNING: Predictions collapsed to a narrow band; check your scaler or model capacity.")
        
        return y_pred


# PyTorch model implementations
class LSTMNetwork(nn.Module):
    """PyTorch LSTM network implementation."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(LSTMNetwork, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Get the last time step's output
        last_time_step = lstm_out[:, -1, :]
        x = self.dropout(last_time_step)
        x = self.fc(x)
        return x


class TransformerNetwork(nn.Module):
    """PyTorch Transformer network implementation."""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, mlp_units):
        super(TransformerNetwork, self).__init__()
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Global averaging
        self.global_avg_pool = nn.AvgPool1d(kernel_size=1000, stride=1)  # Will be adapted in forward pass
        
        # MLP layers
        layers = []
        input_size = d_model
        for units in mlp_units:
            layers.append(nn.Linear(input_size, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = units
        
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(mlp_units[-1] if mlp_units else d_model, 1)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # MLP processing
        if hasattr(self, 'mlp') and len(list(self.mlp.children())) > 0:
            x = self.mlp(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)