# Apple Stock Price Prediction Model

A machine learning project focused on predicting Apple (AAPL) stock prices using various algorithms including Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, and LSTM neural networks.

## Features

- Apple stock data fetching using yfinance API
- Data preprocessing and feature engineering
- Multiple prediction models:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - LSTM (requires TensorFlow)
- Performance metrics calculation (MSE, RMSE, MAE, R²)
- Visualization of predictions and model performance
- Hyperparameter tuning and overfitting analysis

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   If you want to use LSTM models, uncomment the TensorFlow dependency in requirements.txt.

## Usage

### Using scikit-learn models (no TensorFlow required):
```
python main_sklearn.py
```

### Using all models including LSTM (requires TensorFlow):
```
python main.py
```

### Using a simplified version with basic models:
```
python main_simple.py
```

## Results

Results are saved in the `results` directory, including:
- Performance metrics in CSV format
- Prediction plots comparing actual vs. predicted values
- Residual plots for error analysis
- Training loss plots
- Model comparison visualizations

## Project Structure

- `main.py`: Main script with LSTM models (requires TensorFlow)
- `main_sklearn.py`: Main script with scikit-learn models only
- `main_simple.py`: Simplified version of the script
- `main_no_tf.py`: Basic implementation without TensorFlow
- `models.py`: LSTM model implementation
- `models_sklearn.py`: scikit-learn model implementation
- `fetch_stock_data.py`: Functions for retrieving stock data
- `hyperparameter_tuning.py`: Hyperparameter optimization code
- `results/`: Directory containing all output files and visualizations
- `requirements.txt`: Required packages for the project

## License

MIT License