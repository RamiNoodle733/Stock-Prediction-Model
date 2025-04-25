import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

# Import our custom modules
from models_sklearn import evaluate_model
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
    Analyze Apple stock with Linear Regression model
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
    
    # Create results directory for this ticker
    results_dir = f'results/{ticker}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(lr_predictions, label='Linear Regression')
    plt.title(f'{ticker} - Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Stock Price (Normalized)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/lr_predictions.png')
    plt.close()
    
    # Store results
    results = {
        'ticker': ticker,
        'lr_mse': lr_metrics['MSE'],
        'lr_rmse': lr_metrics['RMSE'],
        'lr_r2': lr_metrics['R2'],
        'lr_training_time': lr_training_time
    }
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame([results])
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/apple_linear_model.csv', index=False)
    
    return results_df

def run_overfitting_analysis(ticker='AAPL', start_date='2018-01-01', end_date='2023-01-01'):
    """
    Run an analysis of overfitting with different hyperparameters using Random Forest
    """
    from sklearn.ensemble import RandomForestRegressor
    
    print(f"\n{'='*50}")
    print("Running overfitting analysis with different hyperparameters")
    print(f"{'='*50}")
    
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Different max_depth values to test
    max_depths = [1, 3, 10]
    results = []
    
    for max_depth in max_depths:
        depth_name = "Simple" if max_depth == 1 else "Medium" if max_depth == 3 else "Complex"
        print(f"\nTraining model with {depth_name} Model (max_depth={max_depth})")
        
        # Build model
        model = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        
        # Store results
        results.append({
            'max_depth': max_depth,
            'model_complexity': depth_name,
            'train_mse': train_mse,
            'test_mse': test_mse
        })
        
        print(f"Training MSE: {train_mse:.6f}")
        print(f"Testing MSE: {test_mse:.6f}")
    
    # Create results directory
    os.makedirs('results/overfitting', exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/overfitting/rf_metrics.csv', index=False)
    
    # Plot MSE comparison
    plt.figure(figsize=(10, 6))
    
    model_labels = [r['model_complexity'] for r in results]
    x = np.arange(len(model_labels))
    width = 0.35
    
    plt.bar(x - width/2, [r['train_mse'] for r in results], width, label='Training MSE')
    plt.bar(x + width/2, [r['test_mse'] for r in results], width, label='Testing MSE')
    
    plt.xlabel('Model Complexity')
    plt.ylabel('Mean Squared Error')
    plt.title('Effect of Model Complexity on Overfitting')
    plt.xticks(x, model_labels)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('results/overfitting/rf_mse_comparison.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create main results directory
    os.makedirs('results', exist_ok=True)
    
    # Define ticker and date range
    ticker = 'AAPL'  # Only analyze Apple stock
    
    # Define date range
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    
    # Run Apple stock analysis
    print("\nRunning Apple stock analysis...")
    apple_results = analyze_apple_stock(start_date, end_date)
    
    # Run overfitting analysis
    print("\nRunning overfitting analysis...")
    overfitting_analysis = run_overfitting_analysis(ticker, start_date, end_date)
    
    print("\nAll analyses complete. Results saved to 'results' directory.")