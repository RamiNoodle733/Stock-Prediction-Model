import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Import our modules
from data_loader import load_stock_data, prepare_data_for_training, plot_stock_data
from models import (
    LinearRegressionModel,
    RandomForestModel,
    LSTMModel,
    plot_predictions,
    plot_training_history,
    compare_models
)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")

def run_stock_prediction():
    """Run the complete stock prediction workflow"""
    
    print("\n" + "="*80)
    print("STOCK PRICE PREDICTION PROJECT")
    print("="*80 + "\n")
    
    # Step 1: Load the data
    print("\nStep 1: Loading stock data...")
    ticker = 'AAPL'  # Apple stock
    data = load_stock_data(ticker=ticker, force_download=True)
    
    # Step 2: Visualize the data
    print("\nStep 2: Visualizing stock data...")
    plot_stock_data(data, title=f"{ticker} Stock Price History")
    
    # Step 3: Prepare data for training
    print("\nStep 3: Preparing data for training...")
    sequence_length = 60  # Use 60 days of historical data to predict the next day
    X_train, y_train, X_test, y_test, scaler = prepare_data_for_training(
        data, target_column='Close', sequence_length=sequence_length
    )
    
    # Step 4: Train models
    print("\nStep 4: Training models...")
    
    # Linear Regression
    lr_model = LinearRegressionModel()
    lr_model.train(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestModel(n_estimators=100, max_depth=20)
    rf_model.train(X_train, y_train)
    
    # LSTM (using MLPRegressor as a substitute since TensorFlow isn't available)
    lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
    lstm_model.train(X_train, y_train)
    
    # Step 5: Evaluate and compare models
    print("\nStep 5: Evaluating and comparing models...")
    models = [lr_model, rf_model, lstm_model]
    comparison_df = compare_models(models, X_test, y_test)
    
    # Save comparison results
    os.makedirs('../models', exist_ok=True)
    comparison_df.to_csv('../models/model_comparison_results.csv')
    print(f"Model comparison results saved to 'models/model_comparison_results.csv'")
    
    # Plot training history for LSTM (neural network) model
    if lstm_model.history is not None:
        plot_training_history(lstm_model.history, title="Neural Network Training Loss")
    
    # Step 6: Save the best model
    print("\nStep 6: Saving models...")
    best_model_name = comparison_df['R²'].idxmax()
    print(f"Best model: {best_model_name} (highest R² score)")
    
    # Find best model and save it
    for model in models:
        if model.name == best_model_name:
            model.save_model(f'../models/{model.name.lower().replace(" ", "_")}_model.pkl')
            break
    
    # Step 7: Feature importance analysis for Random Forest model
    print("\nStep 7: Analyzing feature importance...")
    if hasattr(rf_model.model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Day': [f'Day-{i+1}' for i in range(sequence_length)],
            'Importance': rf_model.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Day', data=feature_importance.head(10))
        plt.title('Top 10 Most Important Days for Prediction')
        plt.tight_layout()
        plt.savefig('../data/feature_importance.png')
        plt.show()
    
    # Step 8: Future price prediction
    print("\nStep 8: Making future price predictions...")
    # Use the last sequence_length days for prediction
    last_sequence = data['Close'].values[-sequence_length:].reshape(1, -1, 1)
    
    # Get predictions from each model
    for model in models:
        prediction = model.predict(last_sequence)
        # Rescale the prediction
        prediction_rescaled = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        print(f"{model.name} predicts next day's price: ${prediction_rescaled:.2f}")
    
    print("\nStock prediction completed successfully!")
    
    return comparison_df

if __name__ == "__main__":
    run_stock_prediction()