import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import time
import os

# Import our custom modules
from models import build_lstm_model, train_lstm_model, evaluate_model, plot_training_loss, plot_predictions
from hyperparameter_tuning import preprocess_data

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data using yfinance
    """
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Data shape: {data.shape}")
    return data

def analyze_apple_stock(start_date, end_date):
    """
    Analyze Apple stock with Linear Regression and LSTM models
    """
    ticker = 'AAPL'
    print(f"\n{'='*50}")
    print(f"Analyzing {ticker}")
    print(f"{'='*50}")
    
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # ---- LINEAR REGRESSION MODEL ----
    print("\nTraining Linear Regression model...")
    lr_start_time = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_training_time = time.time() - lr_start_time
    
    # Make predictions
    lr_predictions = lr_model.predict(X_test)
    
    # Evaluate model
    lr_metrics = evaluate_model(y_test, lr_predictions)
    print(f"Linear Regression - MSE: {lr_metrics['MSE']:.6f}, RMSE: {lr_metrics['RMSE']:.6f}, R²: {lr_metrics['R2']:.6f}")
    print(f"Training time: {lr_training_time:.2f} seconds")
    
    # ---- LSTM MODEL ----
    print("\nTraining LSTM model...")
    # Reshape input for LSTM [samples, time steps, features]
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Train LSTM model
    lstm_model, lstm_history, lstm_predictions, lstm_training_time = train_lstm_model(
        X_train_lstm, y_train, X_test_lstm, y_test, epochs=50, batch_size=32
    )
    
    # Evaluate model
    lstm_metrics = evaluate_model(y_test, lstm_predictions)
    print(f"LSTM - MSE: {lstm_metrics['MSE']:.6f}, RMSE: {lstm_metrics['RMSE']:.6f}, R²: {lstm_metrics['R2']:.6f}")
    print(f"Training time: {lstm_training_time:.2f} seconds")
    
    # Create results directory for this ticker
    results_dir = f'results/{ticker}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(lstm_history.history['loss'], label='Training Loss')
    plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
    plt.title(f'{ticker} - LSTM Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/lstm_loss.png')
    plt.close()
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(lr_predictions, label='Linear Regression')
    plt.plot(lstm_predictions, label='LSTM')
    plt.title(f'{ticker} - Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Stock Price (Normalized)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/predictions_comparison.png')
    plt.close()
    
    # Store results in dictionary
    results = {
        'ticker': ticker,
        'lr_mse': lr_metrics['MSE'],
        'lr_rmse': lr_metrics['RMSE'],
        'lr_r2': lr_metrics['R2'],
        'lr_training_time': lr_training_time,
        'lstm_mse': lstm_metrics['MSE'],
        'lstm_rmse': lstm_metrics['RMSE'],
        'lstm_r2': lstm_metrics['R2'],
        'lstm_training_time': lstm_training_time
    }
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame([results])
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/apple_model_comparison.csv', index=False)
    
    # Create model comparison plots
    
    # RMSE comparison
    plt.figure(figsize=(8, 6))
    models = ['Linear Regression', 'LSTM']
    rmse_values = [lr_metrics['RMSE'], lstm_metrics['RMSE']]
    
    plt.bar(models, rmse_values)
    plt.title('Model Performance Comparison (RMSE)')
    plt.ylabel('RMSE (lower is better)')
    plt.grid(True, axis='y')
    plt.savefig('results/apple_rmse_comparison.png')
    plt.close()
    
    # Training time comparison
    plt.figure(figsize=(8, 6))
    time_values = [lr_training_time, lstm_training_time]
    
    plt.bar(models, time_values)
    plt.title('Model Training Time Comparison')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, axis='y')
    plt.savefig('results/apple_training_time_comparison.png')
    plt.close()
    
    return results_df

def run_overfitting_analysis(ticker='AAPL', start_date='2018-01-01', end_date='2023-01-01'):
    """
    Run an analysis of overfitting with different dropout rates
    """
    print(f"\n{'='*50}")
    print("Running overfitting analysis with different dropout rates")
    print(f"{'='*50}")
    
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Reshape input for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Different dropout rates to test
    dropout_rates = [0.0, 0.2, 0.5]
    results = []
    
    for dropout_rate in dropout_rates:
        print(f"\nTraining model with dropout rate: {dropout_rate}")
        
        # Build model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        
        # Store results
        results.append({
            'dropout_rate': dropout_rate,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'gap': train_mse - test_mse,
            'history': history
        })
        
        print(f"Training MSE: {train_mse:.6f}")
        print(f"Testing MSE: {test_mse:.6f}")
    
    # Create results directory
    os.makedirs('results/overfitting', exist_ok=True)
    
    # Plot training and validation loss for each dropout rate
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(len(dropout_rates), 1, i+1)
        plt.plot(result['history'].history['loss'], label='Training Loss')
        plt.plot(result['history'].history['val_loss'], label='Validation Loss')
        plt.title(f'Dropout Rate: {result["dropout_rate"]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/overfitting/loss_comparison.png')
    plt.close()
    
    # Plot MSE comparison
    plt.figure(figsize=(10, 6))
    
    dropout_labels = [str(rate) for rate in dropout_rates]
    x = np.arange(len(dropout_rates))
    width = 0.35
    
    plt.bar(x - width/2, [r['train_mse'] for r in results], width, label='Training MSE')
    plt.bar(x + width/2, [r['test_mse'] for r in results], width, label='Testing MSE')
    
    plt.xlabel('Dropout Rate')
    plt.ylabel('Mean Squared Error')
    plt.title('Effect of Dropout Rate on Overfitting')
    plt.xticks(x, dropout_labels)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('results/overfitting/mse_comparison.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create main results directory
    os.makedirs('results', exist_ok=True)
    
    # Define date range
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    
    # Run Apple stock analysis
    print("\nRunning Apple stock analysis...")
    apple_results = analyze_apple_stock(start_date, end_date)
    
    # Run overfitting analysis
    print("\nRunning overfitting analysis...")
    overfitting_analysis = run_overfitting_analysis('AAPL', start_date, end_date)
    
    print("\nAll analyses complete. Results saved to 'results' directory.")