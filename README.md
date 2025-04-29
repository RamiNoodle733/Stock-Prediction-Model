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
├── models/             # Saved model files
│   └── lstm/           # LSTM model files (.pt, _config.json, _scalers.pkl)
├── results/            # Evaluation results
│   └── figures/        # Generated prediction plots
├── reports/
│   └── figures/        # Generated plots and visualizations
├── src/
│   ├── fetch_data.py   # Script to download stock data
│   ├── preprocess.py   # Data cleaning and feature engineering
│   ├── train.py        # Model training functionality
│   ├── evaluate.py     # Model evaluation and comparison
│   └── models/
│       ├── baseline.py # Linear Regression and ARIMA models
│       └── advanced.py # LSTM and Transformer models
└── requirements.txt    # Project dependencies
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
python src/fetch_data.py --symbol AAPL --start 2010-01-01 --end 2023-01-01 --out data/raw

# Download multiple stocks at once
python src/fetch_data.py --symbols AAPL MSFT GOOGL AMZN --start 2010-01-01 --end 2023-01-01 --out data/raw
```

Options:
- `--symbol` or `--symbols`: One or more stock symbols (e.g., AAPL, MSFT, GOOG)
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format
- `--out`: Output directory for the downloaded data

### Step 2: Data Preprocessing

Process the raw data and generate features:

```bash
# On Windows
python src\preprocess.py --in_folder data\raw --out_folder data\processed
```

Options:
- `--in_folder`: Directory containing raw CSV files
- `--out_folder`: Output directory for processed files
- `--window_size`: Lookback window size in days (default: 20)
- `--min_completeness`: Minimum data completeness requirement (default: 0.8)

This step creates:
- Processed CSV files with technical indicators
- NPZ files with machine learning-ready data
- A processing_summary.csv with data statistics

### Step 3: Model Training

Train a prediction model using one of the available algorithms:

```bash
# For Windows PowerShell
python src\train.py --model lstm --symbol AAPL --data_dir data\processed --output_dir models

# For Windows Command Prompt
python src\train.py --model lstm --symbol AAPL --data_dir data\processed --output_dir models
```

Models available:
- `linear`: Linear Regression (baseline)
- `arima`: ARIMA time series model
- `lstm`: Long Short-Term Memory neural network
- `transformer`: Transformer-based neural network

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
- PyTorch models (.pt): `{symbol}_{model_type}_{timestamp}.pt`
- Model configuration: `{symbol}_{model_type}_{timestamp}_config.json`
- Data scalers: `{symbol}_{model_type}_{timestamp}_scalers.pkl`

### Step 4: Model Evaluation

Evaluate trained models and generate performance metrics and visualizations:

```bash
# For Windows PowerShell
python src\evaluate.py --models lstm --model_paths models\lstm\AAPL_lstm_TIMESTAMP.pt --symbol AAPL --data_dir data\processed --output_dir results

# For Windows Command Prompt
python src\evaluate.py --models lstm --model_paths models\lstm\AAPL_lstm_TIMESTAMP.pt --symbol AAPL --data_dir data\processed --output_dir results
```

Replace `TIMESTAMP` with the actual timestamp in your model filename (e.g., `AAPL_lstm_20250429_032712.pt`).

Important options:
- `--models`: List of model types being evaluated
- `--model_paths`: Paths to the trained model files
- `--symbol`: Stock symbol to evaluate on
- `--data_dir`: Directory with processed data
- `--output_dir`: Output directory for results and figures

The evaluation will output:
- Performance metrics (MSE, RMSE, MAE, R², MAPE)
- Inference time measurement
- Prediction plots saved to the output directory

Example output:
```
Evaluating LSTM model for AAPL...
MSE: 5880.8608
RMSE: 76.6868
MAE: 74.3769
R²: -15.3825
MAPE: 46.7773%
Inference time: 0.1006 seconds
Saved prediction plot to results\figures\AAPL_lstm_predictions_20250429_035606.png
```

### Step 5: Comparing Multiple Models

To compare different models side by side:

```bash
# For Windows Command Prompt
python src\evaluate.py --models linear lstm transformer --model_paths models\linear\AAPL_linear.pkl models\lstm\AAPL_lstm_TIMESTAMP.pt models\transformer\AAPL_transformer_TIMESTAMP.pt --symbol AAPL --data_dir data\processed --output_dir results
```

This will generate:
- Individual metrics for each model
- Comparison bar charts for key metrics
- Side-by-side prediction plots

### Working with Jupyter Notebooks

For interactive exploration and analysis:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `notebooks` folder and open:
   - `1_data_exploration.ipynb` for data analysis
   - `2_model_prototyping.ipynb` for interactive model development

## Troubleshooting

### Common Issues and Solutions

1. **Path Issues on Windows**:
   - Use backslashes (`\`) for paths in Windows Command Prompt
   - In PowerShell, avoid using `&&` for command chaining; use separate commands

2. **Model Loading Errors**:
   - Ensure you're using the correct model path with timestamp
   - Verify that all model files exist (_config.json and _scalers.pkl)

3. **Missing Data Files**:
   - Run the preprocessing step if .npz files are missing
   - Check processing_summary.csv to ensure data quality

4. **CUDA/GPU Issues**:
   - Add `--cpu` flag to train or evaluate commands to force CPU usage

## Interpreting Results

The evaluation produces several metrics:
- **MSE/RMSE**: Lower values indicate better fit
- **MAE**: Average absolute error in price predictions
- **R²**: Value close to 1 indicates good fit; negative values indicate poor performance
- **MAPE**: Error as percentage of actual values; lower is better

Prediction plots show:
- Actual prices (solid line)
- Predicted prices (dashed line)
- Error margin (shaded area)

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.