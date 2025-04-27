"""
Stock Market Prediction - Main Script
This script runs the complete workflow for stock price prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Import our modules
from data_loader import load_stock_data, prepare_data, visualize_stock_data
from models import (
    LinearRegressionModel, 
    RandomForestModel, 
    LSTMModel, 
    plot_predictions, 
    plot_training_loss, 
    compare_models
)

# Set plot style
plt.style.use('ggplot')
sns.set_palette('deep')

def run_stock_prediction(ticker='AAPL', sequence_length=60, test_size=0.2, output_dir='results'):
    """
    Run the complete stock prediction workflow.
    
    Args:
        ticker: Stock ticker symbol
        sequence_length: Number of days to use for prediction
        test_size: Proportion of data to use for testing
        output_dir: Directory to save results
    """
    print("\n" + "="*80)
    print(f"STOCK PRICE PREDICTION: {ticker}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    print("\nStep 1: Loading stock data...")
    data = load_stock_data(ticker)
    
    # Step 2: Visualize data
    print("\nStep 2: Visualizing stock data...")
    visualize_stock_data(data, ticker, output_dir)
    
    # Step 3: Prepare data for training
    print("\nStep 3: Preparing data for training...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        data, 
        target_col='Close',
        sequence_length=sequence_length, 
        test_size=test_size
    )
    
    # Step 4: Train models
    print("\nStep 4: Training models...")
    
    # Linear Regression
    print("\nTraining Linear Regression model...")
    lr_model = LinearRegressionModel()
    lr_model.fit(X_train, y_train)
    
    # Random Forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestModel(n_estimators=100, max_depth=20)
    rf_model.fit(X_train, y_train)
    
    # LSTM Neural Network
    print("\nTraining LSTM Neural Network model...")
    input_shape = (X_train.shape[1], 1)
    lstm_model = LSTMModel(input_shape)
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # Step 5: Evaluate individual models
    print("\nStep 5: Evaluating individual models...")
    
    # Evaluate Linear Regression
    lr_metrics, lr_pred = lr_model.evaluate(X_test, y_test)
    plot_predictions(
        y_test, lr_pred, 
        title=f"{ticker} Stock Price: Linear Regression Predictions",
        save_path=os.path.join(output_dir, f"{ticker}_linear_regression_predictions.png")
    )
    
    # Evaluate Random Forest
    rf_metrics, rf_pred = rf_model.evaluate(X_test, y_test)
    plot_predictions(
        y_test, rf_pred, 
        title=f"{ticker} Stock Price: Random Forest Predictions",
        save_path=os.path.join(output_dir, f"{ticker}_random_forest_predictions.png")
    )
    
    # Evaluate LSTM
    lstm_metrics, lstm_pred = lstm_model.evaluate(X_test, y_test)
    plot_predictions(
        y_test, lstm_pred, 
        title=f"{ticker} Stock Price: LSTM Predictions",
        save_path=os.path.join(output_dir, f"{ticker}_lstm_predictions.png")
    )
    
    # Plot training loss for LSTM
    if lstm_model.history:
        plot_training_loss(
            lstm_model.history, 
            title="LSTM Training Loss",
            save_path=os.path.join(output_dir, "lstm_training_loss.png")
        )
    
    # Step 6: Compare models
    print("\nStep 6: Comparing model performance...")
    models = [lr_model, rf_model, lstm_model]
    comparison_df = compare_models(
        models, X_test, y_test, 
        save_path=os.path.join(output_dir, f"{ticker}_model_comparison.png")
    )
    
    # Save comparison results to CSV
    comparison_df.to_csv(os.path.join(output_dir, f"{ticker}_model_comparison.csv"))
    print(f"\nModel comparison saved to {os.path.join(output_dir, f'{ticker}_model_comparison.csv')}")
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df[['MSE', 'RMSE', 'R²', 'Training Time']])
    
    # Step 7: Feature importance analysis (for Random Forest)
    print("\nStep 7: Analyzing feature importance...")
    feature_importance = rf_model.feature_importance(sequence_length)
    if feature_importance is not None:
        # Plot top 10 most important features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title(f"Top 10 Most Important Features for {ticker} Stock Prediction")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ticker}_feature_importance.png"))
        plt.close()
        
        # Save feature importance to CSV
        feature_importance.to_csv(os.path.join(output_dir, f"{ticker}_feature_importance.csv"))
    
    # Step 8: Ablation study on sequence length
    print("\nStep 8: Ablation study on sequence length...")
    run_ablation_study(data, ticker, output_dir)
    
    # Step 9: Make future predictions
    print("\nStep 9: Making future price predictions...")
    make_future_predictions(data, models, scaler, sequence_length, ticker)
    
    print("\nStock prediction analysis complete!")

def run_ablation_study(data, ticker, output_dir):
    """
    Run an ablation study on the effect of sequence length.
    
    Args:
        data: Stock data
        ticker: Stock ticker symbol
        output_dir: Directory to save results
    """
    sequence_lengths = [10, 30, 60, 90, 120]
    results = []
    
    for seq_len in sequence_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Prepare data with this sequence length
        X_train, X_test, y_train, y_test, _ = prepare_data(
            data, 
            target_col='Close',
            sequence_length=seq_len, 
            test_size=0.2
        )
        
        # Train and evaluate Linear Regression model (fastest)
        model = LinearRegressionModel()
        model.fit(X_train, y_train)
        metrics, _ = model.evaluate(X_test, y_test)
        
        # Store results
        results.append({
            'Sequence Length': seq_len,
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'R²': metrics['R²'],
            'Training Time': metrics['Training Time']
        })
    
    # Create DataFrame and save results
    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv(os.path.join(output_dir, f"{ticker}_ablation_study.csv"))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(ablation_df['Sequence Length'], ablation_df['RMSE'], marker='o')
    plt.title('Effect of Sequence Length on RMSE')
    plt.xlabel('Sequence Length (days)')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(ablation_df['Sequence Length'], ablation_df['R²'], marker='o')
    plt.title('Effect of Sequence Length on R²')
    plt.xlabel('Sequence Length (days)')
    plt.ylabel('R² Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{ticker}_ablation_study.png"))
    plt.close()

def make_future_predictions(data, models, scaler, sequence_length, ticker):
    """
    Make future price predictions using trained models.
    
    Args:
        data: Stock data
        models: List of trained models
        scaler: Fitted scaler for data transformation
        sequence_length: Sequence length used for prediction
        ticker: Stock ticker symbol
    """
    # Get the last sequence from the data
    last_sequence = data['Close'].values[-sequence_length:].reshape(1, -1, 1)
    
    # Predict using each model
    print("\nPredictions for next trading day:")
    for model in models:
        # Make prediction
        prediction = model.predict(last_sequence)
        
        # Inverse transform to get actual price
        prediction_rescaled = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        
        print(f"  {model.name}: ${prediction_rescaled:.2f}")

def run_multi_stock_analysis():
    """
    Run stock prediction on multiple stocks for comparison.
    """
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    for ticker in tickers:
        output_dir = os.path.join('results', ticker)
        run_stock_prediction(ticker=ticker, output_dir=output_dir)

if __name__ == "__main__":
    # Run prediction on a single stock (Apple)
    run_stock_prediction(ticker='AAPL', output_dir='results')