import matplotlib.pyplot as plt
import pandas as pd

def plot_aapl_lstm_loss():
    # Load the training history for AAPL LSTM
    history_file = 'models/AAPL_lstm_history_20250429_124312.csv'
    history = pd.read_csv(history_file)

    # Extract training and validation loss
    epochs = range(1, len(history['loss']) + 1)  # Generate epoch numbers
    train_loss = history['loss']
    val_loss = history['val_loss']

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('AAPL LSTM Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = 'reports/figures/AAPL_lstm_loss_curve_20250429_124312.png'
    plt.savefig(output_path)
    plt.close()

def plot_msft_lstm_loss():
    # Load the training history for MSFT LSTM
    history_file = 'models/MSFT_fold1_lstm_history_20250429_170349.csv'
    history = pd.read_csv(history_file)

    # Extract training and validation loss
    epochs = range(1, len(history['loss']) + 1)  # Generate epoch numbers
    train_loss = history['loss']
    val_loss = history['val_loss']

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('MSFT LSTM Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = 'reports/figures/MSFT_fold1_lstm_loss_curve_20250429_170349.png'
    plt.savefig(output_path)
    plt.close()

def plot_rmse_r2_comparison():
    # Load the RMSE and R² comparison data
    rmse_data = {
        'AAPL': {'LSTM': 12.22, 'Linear': 11.26},
        'MSFT': {'LSTM': 16.84, 'Linear': 9.75},
        'AMD': {'LSTM': 14.37, 'Linear': 10.51}
    }

    r2_data = {
        'AAPL': {'LSTM': -0.29, 'Linear': -0.10},
        'MSFT': {'LSTM': -0.24, 'Linear': 0.58},
        'AMD': {'LSTM': -0.32, 'Linear': 0.12}
    }

    # Plot RMSE comparison
    plt.figure(figsize=(10, 6))
    stocks = list(rmse_data.keys())
    lstm_rmse = [rmse_data[stock]['LSTM'] for stock in stocks]
    linear_rmse = [rmse_data[stock]['Linear'] for stock in stocks]

    x = range(len(stocks))
    plt.bar(x, lstm_rmse, width=0.4, label='LSTM', color='blue', align='center')
    plt.bar([p + 0.4 for p in x], linear_rmse, width=0.4, label='Linear', color='orange', align='center')
    plt.xticks([p + 0.2 for p in x], stocks)
    plt.xlabel('Stocks')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison Across Models and Stocks')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/comparison_RMSE_20250429_182659.png')
    plt.close()

    # Plot R² comparison
    plt.figure(figsize=(10, 6))
    lstm_r2 = [r2_data[stock]['LSTM'] for stock in stocks]
    linear_r2 = [r2_data[stock]['Linear'] for stock in stocks]

    plt.bar(x, lstm_r2, width=0.4, label='LSTM', color='blue', align='center')
    plt.bar([p + 0.4 for p in x], linear_r2, width=0.4, label='Linear', color='orange', align='center')
    plt.xticks([p + 0.2 for p in x], stocks)
    plt.xlabel('Stocks')
    plt.ylabel('R²')
    plt.title('R² Comparison Across Models and Stocks')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/comparison_R²_20250429_182659.png')
    plt.close()

def plot_prediction_vs_actual():
    # Load prediction and actual data for MSFT
    msft_data = pd.read_csv('results/MSFT_linear_fixed_metrics_20250430_074006.csv')
    msft_actual = msft_data['actual']
    msft_predicted = msft_data['predicted']

    # Plot MSFT prediction vs. actual
    plt.figure(figsize=(10, 6))
    plt.plot(msft_actual, label='Actual', color='blue')
    plt.plot(msft_predicted, label='Predicted', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title('MSFT Actual vs. Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/MSFT_lstm_predictions_20250429_170820.png')
    plt.close()

    # Load prediction and actual data for AMD
    amd_data = pd.read_csv('results/AMD_linear_fixed_metrics_20250430_074007.csv')
    amd_actual = amd_data['actual']
    amd_predicted = amd_data['predicted']

    # Plot AMD prediction vs. actual
    plt.figure(figsize=(10, 6))
    plt.plot(amd_actual, label='Actual', color='blue')
    plt.plot(amd_predicted, label='Predicted', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title('AMD Actual vs. Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/AMD_lstm_predictions_20250429_170821.png')
    plt.close()

if __name__ == "__main__":
    plot_aapl_lstm_loss()
    plot_msft_lstm_loss()
    plot_rmse_r2_comparison()
    plot_prediction_vs_actual()