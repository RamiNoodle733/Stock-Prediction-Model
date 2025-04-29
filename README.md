# Stock Market Prediction Model

This project implements a comprehensive stock market prediction system using various machine learning and deep learning techniques. It provides tools for data collection, preprocessing, model training, and evaluation, allowing you to predict stock prices and compare different prediction methodologies.

## Project Structure

```
stock-prediction-model/
├── data/
│   ├── raw/            # Raw stock price data from Yahoo Finance
│   └── processed/      # Processed data with engineered features
├── notebooks/
│   ├── 1_data_exploration.ipynb    # Data analysis and visualization
│   └── 2_model_prototyping.ipynb   # Model development and comparison
├── models/
│   ├── linear/         # Linear regression models (.pkl)
│   └── lstm/           # LSTM model files (.pt, _config.json, _scalers.pkl)
├── results/            # Evaluation results and predictions
│   └── figures/        # Generated prediction plots
├── src/
│   ├── fetch_data.py             # Script to download stock data
│   ├── preprocess.py             # Data cleaning and feature engineering
│   ├── train.py                  # Model training functionality
│   ├── evaluate.py               # Model evaluation and comparison
│   ├── rescale_predictions.py    # Fix and rescale predictions
│   ├── diagnose_predictions.py   # Tools to diagnose prediction issues
│   └── models/
│       ├── baseline.py           # Linear Regression models
│       └── advanced.py           # LSTM and Transformer models
└── requirements.txt              # Project dependencies
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stock-prediction-model.git
cd stock-prediction-model
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Complete Guide to Running and Testing the Project

This section provides a step-by-step guide to run the entire pipeline from data collection to model evaluation.

### Step 1: Data Collection

Download historical stock data from Yahoo Finance:

```bash
# On Windows
python src/fetch_data.py --symbols AAPL MSFT GOOGL AMZN NVDA INTC META CSCO TSLA --start 2010-01-01 --end 2023-01-01 --out data/raw
```

Options:
- `--symbol` or `--symbols`: One or more stock symbols (e.g., AAPL, MSFT, GOOG)
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format
- `--out`: Output directory for the downloaded data

### Step 2: Data Preprocessing

Process the raw data and generate features:

```bash
python src/preprocess.py --in_folder data/raw --out_folder data/processed
```

Options:
- `--in_folder`: Directory containing raw CSV files
- `--out_folder`: Output directory for processed files
- `--window_size`: Lookback window size in days (default: 20)
- `--min_completeness`: Minimum data completeness requirement (default: 0.8)

This step creates:
- Processed and scaled data in NPZ files for each stock
- A processing_summary.csv with data statistics

### Step 3: Model Training

Train a prediction model using one of the available algorithms:

```bash
# Train an LSTM model for AAPL stock
python src/train.py --model lstm --symbol AAPL --data_dir data/processed --output_dir models

# Train a linear model for AAPL stock 
python src/train.py --model linear --symbol AAPL --data_dir data/processed --output_dir models
```

Models available:
- `linear`: Linear Regression model
- `lstm`: Long Short-Term Memory neural network

Important options:
- `--model`: Type of model to train
- `--symbol`: Stock symbol to train on
- `--data_dir`: Directory with processed data (containing .npz files)
- `--output_dir`: Output directory for trained models
- `--epochs`: Number of training epochs for neural models (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--window_size`: Window size for time series data (default: 20)
- `--time_based`: Whether to use time-based train/test split (default: True)

When training completes, the model files will be saved in the output directory with the following naming pattern:
- LSTM models (.pt): `{symbol}_fold{n}_lstm_{timestamp}.pt`
- Linear models (.pkl): `{symbol}_fold{n}_linear_{timestamp}.pkl`
- Model configuration: `{symbol}_fold{n}_lstm_{timestamp}_config.json`
- Data scalers: `{symbol}_fold{n}_lstm_{timestamp}_scalers.pkl`
- Training history: `{symbol}_fold{n}_lstm_history_{timestamp}.csv`

### Step 4: Model Evaluation

Evaluate trained models and generate performance metrics and visualizations:

```bash
# Evaluate a single LSTM model
python src/evaluate.py --models lstm --model_paths models/lstm/AAPL_fold5_lstm_20250429_161416.pt --symbol AAPL --data_dir data/processed --output_dir results

# Compare LSTM and linear models
python src/evaluate.py --models lstm linear --model_paths models/lstm/AAPL_fold5_lstm_20250429_161416.pt models/linear/AAPL_fold5_linear_20250429_161448.pkl --symbol AAPL --data_dir data/processed --output_dir results
```

Important options:
- `--models`: List of model types being evaluated (lstm, linear)
- `--model_paths`: Paths to the trained model files
- `--symbol`: Stock symbol to evaluate on
- `--data_dir`: Directory with processed data
- `--output_dir`: Output directory for results and figures

The evaluation will output:
- Performance metrics (MSE, RMSE, MAE, R², MAPE)
- Inference time measurement
- Prediction plots saved to the output directory

### Step 5: Generate Rescaled Predictions

Generate and evaluate price predictions in their original scale:

```bash
python src/rescale_predictions.py --model_path models/lstm/AAPL_fold5_lstm_20250429_161416.pt --symbol AAPL --data_dir data/processed --output_dir results
```

This will produce:
- CSV file with actual and predicted prices
- Plot comparing actual vs. predicted prices
- Metrics for both original and rescaled predictions

### Step 6: Diagnose Prediction Issues (Optional)

If you need to diagnose problems with predictions:

```bash
python src/diagnose_predictions.py --model_path models/lstm/AAPL_fold5_lstm_20250429_161416.pt --symbol AAPL --data_dir data/processed --output_dir results
```

This creates detailed diagnostic plots to help identify issues with the predictions.

## Known Issues and Limitations

1. **AMD Data Processing**: The AMD.csv file contains values that are too large to process correctly. This issue can be resolved by manually cleaning the data.

2. **Linear Model Performance**: Linear regression models sometimes show suspiciously perfect performance (R²=1.0, MSE=0.0), which may indicate potential data leakage issues in the training/evaluation pipeline.

3. **LSTM Prediction Range**: The LSTM model predictions sometimes collapse to a narrow range of values. This is indicated by the warning message about predictions collapsing to a narrow band.

## Interpreting Results

The evaluation produces several metrics:
- **MSE/RMSE**: Lower values indicate better fit
- **MAE**: Average absolute error in price predictions
- **R²**: Value close to 1 indicates good fit; negative values indicate poor performance
- **MAPE**: Error as percentage of actual values; lower is better

Prediction plots show:
- Actual prices (solid line)
- Predicted prices (dashed line)
- Training loss curves are saved during the LSTM training process

## Meeting the Assignment Requirements

This project fulfills the requirements outlined in the assignmentRubric.txt by:
- Implementing multiple ML models for stock price prediction
- Providing a complete pipeline from data collection to evaluation
- Including advanced models (LSTM)
- Generating training loss curves and model comparisons
- Analyzing model performance and computational requirements
- Supporting analysis of overfitting/underfitting issues

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.