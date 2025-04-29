"""
Model Evaluation Module

This module evaluates trained stock prediction models using various metrics
and generates visualizations to compare model performance.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.baseline import LinearRegressionModel, ARIMAModel
from models.advanced import LSTMModel, TransformerModel
from train import load_data, split_data


def load_model(model_path, model_type):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        model_type (str): Type of the model ('linear', 'arima', 'lstm', 'transformer')
        
    Returns:
        object: Loaded model
    """
    if model_type in ['linear', 'arima']:
        # Load scikit-learn or statsmodels model
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_type in ['lstm', 'transformer']:
        # For deep learning models, need to reconstruct the model and load weights
        
        # Load model config
        config_path = model_path.replace('.pt', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load scalers
        scalers_path = model_path.replace('.pt', '_scalers.pkl')
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        # Create the appropriate model
        if model_type == 'lstm':
            model = LSTMModel(
                window_size=config['window_size'],
                feature_dim=config['feature_dim'],
                units=config['units'],
                layers=config['layers'],
                dropout_rate=config['dropout_rate']
            )
        else:  # transformer
            model = TransformerModel(
                window_size=config['window_size'],
                feature_dim=config['feature_dim'],
                head_size=config['head_size'],
                num_heads=config['num_heads'],
                ff_dim=config['ff_dim'],
                num_transformer_blocks=config['num_transformer_blocks'],
                mlp_units=config['mlp_units'],
                dropout_rate=config['dropout_rate']
            )
        
        # Build the model architecture
        model.build_model()
        
        # Load weights (using PyTorch's load method instead of TensorFlow's load_weights)
        import torch
        model.model.load_state_dict(torch.load(model_path))
        model.model.eval()  # Set the model to evaluation mode
        
        # Set scalers
        model.X_scaler = scalers['X_scaler']
        model.y_scaler = scalers['y_scaler']
        
        # Set is_fitted flag to True so predict() can work
        model.is_fitted = True
        
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, X_test, y_test, model_type):
    """
    Evaluate a model on test data.
    
    Args:
        model: The trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
        model_type (str): Type of model
        
    Returns:
        tuple: Predictions, inference time, and evaluation metrics
    """
    # Measure inference time
    start_time = time.time()
    
    # Make predictions
    if model_type == 'arima':
        y_pred = model.predict(steps=len(y_test))
    else:
        y_pred = model.predict(X_test)
    
    inference_time = time.time() - start_time
    
    # Ensure predictions and actual values are 1D arrays
    if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    if hasattr(y_test, 'shape') and len(y_test.shape) > 1:
        y_test = y_test.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Return predictions and metrics
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'inference_time': inference_time
    }
    
    return y_pred, inference_time, metrics


def plot_predictions(y_true, y_pred, dates, symbol, model_type, output_dir):
    """
    Create visualization of model predictions vs actual values.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        dates (numpy.ndarray): Dates corresponding to the values
        symbol (str): Stock symbol
        model_type (str): Type of model
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved plot
    """
    # Create directory for figures if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    
    # Convert date strings to datetime if they're strings
    if isinstance(dates[0], str):
        dates = [pd.to_datetime(d) for d in dates]
    
    # Create DataFrame for easier plotting with dates
    df = pd.DataFrame({
        'Date': dates,
        'Actual': y_true,
        'Predicted': y_pred
    })
    df = df.set_index('Date')
    
    # Plot the predictions
    plt.plot(df.index, df['Actual'], label='Actual', linewidth=2)
    plt.plot(df.index, df['Predicted'], label='Predicted', linewidth=2, linestyle='--')
    
    # Add shaded region for prediction error
    plt.fill_between(df.index, df['Actual'], df['Predicted'], 
                     color='lightgrey', alpha=0.3, label='Error')
    
    # Formatting
    plt.title(f'{symbol} Stock Price Prediction using {model_type.upper()}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(figures_dir, f"{symbol}_{model_type}_predictions_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def plot_metrics_comparison(metrics_dict, output_dir):
    """
    Create bar plots comparing model performance across different metrics.
    
    Args:
        metrics_dict (dict): Dictionary of model metrics keyed by model name
        output_dir (str): Directory to save the plots
        
    Returns:
        list: Paths to the saved plots
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get all metrics and models
    all_models = list(metrics_dict.keys())
    metric_names = list(set().union(*[set(m.keys()) for m in metrics_dict.values()]))
    metric_names = [m for m in metric_names if m != 'inference_time']  # Handle inference time separately
    
    # Prepare data for plotting
    metric_values = {metric: [metrics_dict[model].get(metric, np.nan) for model in all_models] 
                     for metric in metric_names}
    
    # Create comparison bar plots for each metric
    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        sns.barplot(x=all_models, y=metric_values[metric])
        
        # Add value labels on top of each bar
        for i, v in enumerate(metric_values[metric]):
            plt.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
        
        plt.title(f'Comparison of {metric} across Models', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plot_path = os.path.join(figures_dir, f"comparison_{metric}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
    
    # Create inference time comparison
    if 'inference_time' in metrics_dict[all_models[0]]:
        plt.figure(figsize=(10, 6))
        
        # Get inference times for all models
        inference_times = [metrics_dict[model]['inference_time'] for model in all_models]
        
        # Create bar plot
        sns.barplot(x=all_models, y=inference_times)
        
        # Add value labels
        for i, v in enumerate(inference_times):
            plt.text(i, v, f"{v:.4f}s", ha='center', va='bottom', fontsize=10)
        
        plt.title('Model Inference Time Comparison', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Inference Time (seconds)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plot_path = os.path.join(figures_dir, f"comparison_inference_time_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
    
    return plot_paths


def run_ablation_study(model_type, X_train, y_train, X_test, y_test, param_grid, output_dir, symbol):
    """
    Run ablation studies to analyze the effect of different hyperparameters.
    
    Args:
        model_type (str): Type of model to study
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
        param_grid (dict): Dictionary of parameter values to test
        output_dir (str): Directory to save results
        symbol (str): Stock symbol
        
    Returns:
        tuple: Results DataFrame and path to the saved plot
    """
    from train import train_model
    
    # Prepare DataFrame to store results
    results = []
    
    # Create a progress bar for ablation runs
    total_runs = np.prod([len(values) for values in param_grid.values()])
    print(f"Running ablation study with {total_runs} configurations...")
    
    # Generate all parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))
        
        # Print progress
        print(f"Run {i+1}/{total_runs}: Testing {params}")
        
        try:
            # Train the model with these parameters
            model, history, training_time = train_model(
                model_type, X_train, y_train, **params
            )
            
            # Evaluate on test data
            _, _, metrics = evaluate_model(model, X_test, y_test, model_type)
            
            # Get final training loss and validation loss if available
            if history and hasattr(history, 'history'):
                final_train_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            else:
                final_train_loss = None
                final_val_loss = None
            
            # Get parameter count for model complexity
            if hasattr(model, 'model') and hasattr(model.model, 'count_params'):
                param_count = model.model.count_params()
            else:
                param_count = None
            
            # Store the results
            result = {
                'model_type': model_type,
                'train_time': training_time,
                'param_count': param_count,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                **metrics,
                **params  # include the tested parameters
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error in ablation run with params {params}: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results as CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{symbol}_{model_type}_ablation_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Create visualization of ablation results
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # For each parameter, create a plot showing its effect on metrics
    plot_paths = []
    
    for param in param_names:
        if param in results_df.columns and len(results_df[param].unique()) > 1:
            plt.figure(figsize=(12, 8))
            
            # Create subplots for different metrics
            metrics_to_plot = ['RMSE', 'MAE', 'R²']
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in results_df.columns:
                    plt.subplot(len(metrics_to_plot), 1, i+1)
                    
                    # Group by the parameter and calculate mean and std of the metric
                    grouped = results_df.groupby(param)[metric].agg(['mean', 'std']).reset_index()
                    
                    # Plot with error bars
                    plt.errorbar(grouped[param], grouped['mean'], yerr=grouped['std'], 
                                marker='o', linestyle='-', capsize=5)
                    
                    plt.title(f'Effect of {param} on {metric}')
                    plt.xlabel(param)
                    plt.ylabel(metric)
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure
            plot_path = os.path.join(figures_dir, f"{symbol}_{model_type}_ablation_{param}_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(plot_path)
    
    return results_df, plot_paths


def save_results_table(metrics_dict, output_dir, symbol):
    """
    Save a comparative table of model metrics.
    
    Args:
        metrics_dict (dict): Dictionary of model metrics keyed by model name
        output_dir (str): Directory to save the table
        symbol (str): Stock symbol
        
    Returns:
        str: Path to the saved CSV file
    """
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Reset index and rename it
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{symbol}_model_comparison_{timestamp}.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    return csv_path


def main():
    """Main function to parse arguments and run model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate stock prediction models")
    parser.add_argument("--models", type=str, nargs='+', required=True, 
                       help="List of model types to evaluate (linear, arima, lstm, transformer)")
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                       help="Paths to the trained model files (one for each model type)")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to evaluate on")
    parser.add_argument("--data_dir", type=str, default="../data/processed", help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="../results", help="Output directory for results")
    parser.add_argument("--ablation", action="store_true", help="Run ablation studies")
    
    args = parser.parse_args()
    
    # Ensure we have the same number of model types and paths
    if len(args.models) != len(args.model_paths):
        raise ValueError("Number of model types must match number of model paths")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading data for {args.symbol}...")
    X, y, dates = load_data(args.symbol, args.data_dir)
    
    # Split data to get test set (same split as in training)
    _, _, X_test, _, _, y_test, _, _, dates_test = split_data(X, y, dates)
    
    # Dictionary to store metrics for all models
    all_metrics = {}
    
    # Evaluate each model
    for i, (model_type, model_path) in enumerate(zip(args.models, args.model_paths)):
        print(f"Evaluating {model_type} model ({i+1}/{len(args.models)})...")
        
        try:
            # Load the model
            model = load_model(model_path, model_type)
            
            # Evaluate the model
            y_pred, inference_time, metrics = evaluate_model(model, X_test, y_test, model_type)
            
            print(f"  Metrics for {model_type}:")
            for metric_name, metric_value in metrics.items():
                print(f"    {metric_name}: {metric_value:.4f}")
            
            # Store metrics for comparison
            all_metrics[model_type] = metrics
            
            # Plot predictions vs actual
            plot_path = plot_predictions(
                y_test, y_pred, dates_test, 
                args.symbol, model_type, args.output_dir
            )
            print(f"  Prediction plot saved to: {plot_path}")
            
            # Run ablation study if requested
            if args.ablation and model_type in ['lstm', 'transformer']:
                print(f"Running ablation study for {model_type}...")
                
                # Define parameter grid for ablation study
                if model_type == 'lstm':
                    param_grid = {
                        'units': [32, 64, 128],
                        'layers': [1, 2, 3],
                        'dropout_rate': [0.1, 0.2, 0.3]
                    }
                else:  # transformer
                    param_grid = {
                        'num_heads': [1, 2, 4],
                        'head_size': [128, 256],
                        'num_transformer_blocks': [1, 2, 3]
                    }
                
                # Get training data for ablation
                X_train, _, _, y_train, _, _, _, _, _ = split_data(X, y, dates)
                
                # Run the ablation study
                results_df, ablation_plots = run_ablation_study(
                    model_type, X_train, y_train, X_test, y_test, 
                    param_grid, args.output_dir, args.symbol
                )
                
                print(f"  Ablation study results saved to CSV and plots generated:")
                for plot in ablation_plots:
                    print(f"    {plot}")
                
        except Exception as e:
            print(f"Error evaluating {model_type} model: {e}")
    
    # Compare models if we have multiple
    if len(all_metrics) > 1:
        print("\nGenerating model comparison plots...")
        
        comparison_plots = plot_metrics_comparison(all_metrics, args.output_dir)
        for plot in comparison_plots:
            print(f"  Comparison plot saved to: {plot}")
        
        # Save metrics table
        metrics_table = save_results_table(all_metrics, args.output_dir, args.symbol)
        print(f"Metrics comparison table saved to: {metrics_table}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()