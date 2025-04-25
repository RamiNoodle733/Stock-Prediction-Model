import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import time
import yfinance as yf
import os

# Import our modules
from models_sklearn import evaluate_model

def preprocess_data(data, lookback=60, test_size=0.2):
    """
    Preprocess stock data for machine learning models
    
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

def hyperparameter_tuning(start_date='2018-01-01', end_date='2023-01-01'):
    """
    Perform hyperparameter tuning for Gradient Boosting on Apple stock
    
    Args:
        start_date: Start date for stock data
        end_date: End date for stock data
        
    Returns:
        Results dataframe with performance metrics
    """
    # Define ticker - only Apple stock
    ticker = 'AAPL'
    
    # Fetch stock data
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    # Generate all combinations of hyperparameters
    grid = list(ParameterGrid(param_grid))
    print(f"Testing {len(grid)} hyperparameter combinations...")
    
    # Create a directory for results if it doesn't exist
    results_dir = 'tuning_results_sklearn'
    os.makedirs(results_dir, exist_ok=True)
    
    # Store results
    results = []
    
    for i, params in enumerate(grid):
        print(f"\nTraining model {i+1}/{len(grid)} with parameters: {params}")
        
        # Build model with current hyperparameters
        model = GradientBoostingRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            random_state=42
        )
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
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
        
        # Create predictions plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.title(f'Model {i+1} - n_est: {params["n_estimators"]}, max_depth: {params["max_depth"]}, lr: {params["learning_rate"]}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price (Normalized)')
        plt.legend()
        plt.savefig(f'{results_dir}/model_{i+1}_predictions.png')
        plt.close()
        
        # Create feature importance plot
        if i < 5:  # Only plot for the first few models to save time
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Feature Importances for Model {i+1}')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.xlabel('Relative Importance')
            plt.ylabel('Feature Index (Time Lag)')
            plt.savefig(f'{results_dir}/model_{i+1}_importances.png')
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
    
    # Run hyperparameter tuning for Apple
    results = hyperparameter_tuning('2018-01-01', '2023-01-01')
    
    # Plot training time vs MSE
    plt.figure(figsize=(10, 6))
    plt.scatter(results['training_time'], results['mse'], alpha=0.7)
    plt.title('Training Time vs MSE')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('tuning_results_sklearn/training_time_vs_mse.png')
    plt.close()
    
    # Plot number of estimators vs MSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='n_estimators', y='mse', data=results)
    plt.title('Number of Estimators vs MSE')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Squared Error')
    plt.savefig('tuning_results_sklearn/estimators_vs_mse.png')
    plt.close()
    
    # Plot max depth vs MSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='max_depth', y='mse', data=results)
    plt.title('Max Depth vs MSE')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Squared Error')
    plt.savefig('tuning_results_sklearn/depth_vs_mse.png')
    plt.close()
    
    print("Hyperparameter tuning completed. Results saved to 'tuning_results_sklearn' directory.")