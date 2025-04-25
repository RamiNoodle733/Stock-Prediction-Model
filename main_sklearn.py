import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import time
import os

# Import our custom modules
from models_sklearn import (
    evaluate_model,
    train_linear_regression,
    train_ridge_regression, 
    train_random_forest,
    train_gradient_boosting,
    plot_predictions,
    plot_residuals,
    plot_training_loss,
    preprocess_stock_data
)

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data using yfinance
    """
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Data shape: {data.shape}")
    return data

def analyze_stock(ticker, start_date, end_date):
    """
    Analyze a single stock with multiple models
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {ticker}")
    print(f"{'='*50}")
    
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess data with lag features
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, y_scaler = preprocess_stock_data(data)
    
    results = {}
    
    # ---- LINEAR REGRESSION MODEL ----
    print("\nTraining Linear Regression model...")
    lr_model, lr_predictions, lr_training_time = train_linear_regression(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    lr_metrics = evaluate_model(y_test, lr_predictions)
    print(f"Linear Regression - MSE: {lr_metrics['MSE']:.6f}, RMSE: {lr_metrics['RMSE']:.6f}, R²: {lr_metrics['R2']:.6f}")
    print(f"Training time: {lr_training_time:.2f} seconds")
    
    results['Linear Regression'] = {
        'model': lr_model,
        'predictions': lr_predictions,
        'metrics': lr_metrics,
        'training_time': lr_training_time
    }
    
    # ---- RIDGE REGRESSION MODEL ----
    print("\nTraining Ridge Regression model...")
    ridge_model, ridge_predictions, ridge_training_time = train_ridge_regression(X_train, y_train, X_test, y_test, alpha=0.1)
    
    # Evaluate model
    ridge_metrics = evaluate_model(y_test, ridge_predictions)
    print(f"Ridge Regression - MSE: {ridge_metrics['MSE']:.6f}, RMSE: {ridge_metrics['RMSE']:.6f}, R²: {ridge_metrics['R2']:.6f}")
    print(f"Training time: {ridge_training_time:.2f} seconds")
    
    results['Ridge Regression'] = {
        'model': ridge_model,
        'predictions': ridge_predictions,
        'metrics': ridge_metrics,
        'training_time': ridge_training_time
    }
    
    # ---- RANDOM FOREST MODEL ----
    print("\nTraining Random Forest model...")
    rf_model, rf_predictions, rf_training_time = train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10)
    
    # Evaluate model
    rf_metrics = evaluate_model(y_test, rf_predictions)
    print(f"Random Forest - MSE: {rf_metrics['MSE']:.6f}, RMSE: {rf_metrics['RMSE']:.6f}, R²: {rf_metrics['R2']:.6f}")
    print(f"Training time: {rf_training_time:.2f} seconds")
    
    results['Random Forest'] = {
        'model': rf_model,
        'predictions': rf_predictions,
        'metrics': rf_metrics,
        'training_time': rf_training_time
    }
    
    # ---- GRADIENT BOOSTING MODEL (Advanced model replacement for LSTM) ----
    print("\nTraining Gradient Boosting model...")
    gb_model, gb_history, gb_predictions, gb_training_time = train_gradient_boosting(
        X_train, y_train, X_test, y_test, 
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3
    )
    
    # Evaluate model
    gb_metrics = evaluate_model(y_test, gb_predictions)
    print(f"Gradient Boosting - MSE: {gb_metrics['MSE']:.6f}, RMSE: {gb_metrics['RMSE']:.6f}, R²: {gb_metrics['R2']:.6f}")
    print(f"Training time: {gb_training_time:.2f} seconds")
    
    results['Gradient Boosting'] = {
        'model': gb_model,
        'predictions': gb_predictions,
        'metrics': gb_metrics,
        'training_time': gb_training_time,
        'history': gb_history
    }
    
    # Create results directory
    results_dir = f'results/{ticker}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot training loss for Gradient Boosting (replacement for LSTM loss)
    plot_training_loss(
        gb_history, 
        title=f'{ticker} - Gradient Boosting Training Loss', 
        save_path=f'{results_dir}/gb_loss.png'
    )
    
    # Plot predictions for all models
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual')
    
    colors = ['blue', 'green', 'orange', 'red']
    for i, (model_name, model_results) in enumerate(results.items()):
        plt.plot(model_results['predictions'], label=model_name, color=colors[i], alpha=0.7)
    
    plt.title(f'{ticker} - Stock Price Predictions Comparison')
    plt.xlabel('Time')
    plt.ylabel('Stock Price (Normalized)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{results_dir}/predictions_comparison.png')
    plt.close()
    
    # Plot residuals for the best model (Gradient Boosting)
    plot_residuals(
        y_test, 
        gb_predictions, 
        title=f'{ticker} - Gradient Boosting Residuals',
        save_path=f'{results_dir}/gb_residuals.png'
    )
    
    # Plot metrics comparison
    metrics_df = pd.DataFrame({
        model_name: {
            'MSE': results[model_name]['metrics']['MSE'],
            'RMSE': results[model_name]['metrics']['RMSE'],
            'R²': results[model_name]['metrics']['R2'],
            'Training Time (s)': results[model_name]['training_time']
        } for model_name in results.keys()
    }).T
    
    # Save metrics to CSV
    metrics_df.to_csv(f'{results_dir}/model_metrics.csv')
    
    # Plot RMSE comparison
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df.index, metrics_df['RMSE'])
    plt.title(f'{ticker} - RMSE Comparison')
    plt.ylabel('RMSE (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/rmse_comparison.png')
    plt.close()
    
    # Plot R² comparison
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df.index, metrics_df['R²'])
    plt.title(f'{ticker} - R² Comparison')
    plt.ylabel('R² (higher is better)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/r2_comparison.png')
    plt.close()
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df.index, metrics_df['Training Time (s)'])
    plt.title(f'{ticker} - Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/training_time_comparison.png')
    plt.close()
    
    return results, metrics_df

def run_overfitting_analysis(ticker='AAPL', start_date='2018-01-01', end_date='2023-01-01'):
    """
    Run an analysis of overfitting with different hyperparameters for Gradient Boosting
    """
    print(f"\n{'='*50}")
    print("Running overfitting analysis with different hyperparameters")
    print(f"{'='*50}")
    
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, y_scaler = preprocess_stock_data(data)
    
    # Different hyperparameter settings
    settings = [
        {'max_depth': 1, 'label': 'Simple Model (max_depth=1)'},
        {'max_depth': 3, 'label': 'Medium Model (max_depth=3)'},
        {'max_depth': 10, 'label': 'Complex Model (max_depth=10)'},
    ]
    
    results = []
    
    for setting in settings:
        print(f"\nTraining model with {setting['label']}")
        
        # Train model
        model, history, predictions, training_time = train_gradient_boosting(
            X_train, y_train, X_test, y_test,
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=setting['max_depth']
        )
        
        # Make training predictions
        train_predictions = model.predict(X_train)
        
        # Calculate metrics
        train_metrics = evaluate_model(y_train, train_predictions)
        test_metrics = evaluate_model(y_test, predictions)
        
        # Store results
        results.append({
            **setting,
            'train_mse': train_metrics['MSE'],
            'test_mse': test_metrics['MSE'],
            'gap': train_metrics['MSE'] - test_metrics['MSE'],
            'history': history,
            'model': model
        })
        
        print(f"Training MSE: {train_metrics['MSE']:.6f}")
        print(f"Testing MSE: {test_metrics['MSE']:.6f}")
    
    # Create results directory
    os.makedirs('results/overfitting', exist_ok=True)
    
    # Plot training and validation loss for each model complexity
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(len(settings), 1, i+1)
        plt.plot(result['history'].history['loss'], label='Training Loss')
        plt.plot(result['history'].history['val_loss'], label='Validation Loss')
        plt.title(f"{result['label']}")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/overfitting/loss_comparison.png')
    plt.close()
    
    # Plot MSE comparison
    plt.figure(figsize=(10, 6))
    
    model_labels = [r['label'] for r in results]
    x = np.arange(len(model_labels))
    width = 0.35
    
    plt.bar(x - width/2, [r['train_mse'] for r in results], width, label='Training MSE')
    plt.bar(x + width/2, [r['test_mse'] for r in results], width, label='Testing MSE')
    
    plt.xlabel('Model Complexity')
    plt.ylabel('Mean Squared Error')
    plt.title('Effect of Model Complexity on Overfitting')
    plt.xticks(x, model_labels, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('results/overfitting/mse_comparison.png')
    plt.close()
    
    # Create a metrics dataframe
    metrics_df = pd.DataFrame({
        'Model': [r['label'] for r in results],
        'Training MSE': [r['train_mse'] for r in results],
        'Testing MSE': [r['test_mse'] for r in results],
        'Gap': [r['gap'] for r in results],
    })
    
    # Save to CSV
    metrics_df.to_csv('results/overfitting/metrics.csv', index=False)
    
    return results, metrics_df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create main results directory
    os.makedirs('results', exist_ok=True)
    
    # Define ticker and date range
    ticker = 'AAPL'  # Only analyze Apple stock
    
    # Define date range (3 years)
    end_date = '2023-01-01'
    start_date = '2020-01-01'
    
    # Run Apple stock analysis
    print("\nRunning Apple stock analysis...")
    apple_results, apple_metrics = analyze_stock(ticker, start_date, end_date)
    
    # Run overfitting analysis on Apple stock
    print("\nRunning overfitting analysis...")
    overfitting_results, overfitting_metrics = run_overfitting_analysis(ticker, start_date, end_date)
    
    print("\nAll analyses complete. Results saved to 'results' directory.")
    print("\nSummary of models for Apple stock:")
    print(apple_metrics)