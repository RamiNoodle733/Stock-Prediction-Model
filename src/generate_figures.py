import matplotlib.pyplot as plt
import pandas as pd
import glob, os

def plot_aapl_lstm_loss():
    history_file = 'models/AAPL_lstm_history_20250429_124312.csv'
    history = pd.read_csv(history_file)
    epochs = range(1, len(history['mae']) + 1)
    train_mae = history['mae']
    val_mae = history['val_mae']
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mae, label='Training MAE', color='blue')
    plt.plot(epochs, val_mae, label='Validation MAE', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('AAPL LSTM Training and Validation MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/AAPL_lstm_mae_curve_20250429_124312.png')
    plt.close()

def plot_msft_lstm_loss():
    history_file = 'models/MSFT_fold1_lstm_history_20250429_170349.csv'
    history = pd.read_csv(history_file)
    epochs = range(1, len(history['mae']) + 1)
    train_mae = history['mae']
    val_mae = history['val_mae']
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mae, label='Training MAE', color='blue')
    plt.plot(epochs, val_mae, label='Validation MAE', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('MSFT LSTM Training and Validation MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/MSFT_lstm_mae_curve_20250429_170349.png')
    plt.close()

def plot_rmse_r2_comparison():
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
    plt.savefig('reports/figures/comparison_R2_20250429_182659.png')
    plt.close()

def plot_prediction_vs_actual():
    files = glob.glob('results/*rescaled_predictions*.csv')
    for f in files:
        df = pd.read_csv(f, parse_dates=['Date'])
        stock = os.path.basename(f).split('_')[0]
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['Actual'], label='Actual', color='blue')
        plt.plot(df['Date'], df['Rescaled_Prediction'], label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title(f'{stock} Actual vs. Predicted Prices')
        plt.legend()
        plt.grid(True)
        out_path = f'reports/figures/{stock}_predicted_vs_actual.png'
        plt.savefig(out_path)
        plt.close()

if __name__ == "__main__":
    plot_aapl_lstm_loss()
    plot_msft_lstm_loss()
    plot_rmse_r2_comparison()
    plot_prediction_vs_actual()