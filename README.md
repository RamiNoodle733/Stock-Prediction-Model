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

## Usage

### Data Collection

Download historical stock data from Yahoo Finance:

```bash
python src/fetch_data.py --symbol AAPL --start 2010-01-01 --end 2023-01-01 --out data/raw
```

Options:
- `--symbol`: Stock symbol to download (e.g., AAPL, MSFT, GOOG)
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format
- `--out`: Output directory for the downloaded data
- `--limit`: Limit the number of stocks to download (useful for testing)

To download data for multiple NASDAQ stocks:
```bash
python src/fetch_data.py --limit 10  # Download data for 10 NASDAQ stocks
```

### Data Preprocessing

Process the raw data and generate features:

```bash
python src/preprocess.py --in_folder data/raw --out_folder data/processed
```

Options:
- `--in_folder`: Directory containing raw CSV files
- `--out_folder`: Output directory for processed files
- `--window_size`: Lookback window size in days (default: 20)
- `--min_completeness`: Minimum data completeness requirement (default: 0.8)

### Model Training

Train a prediction model:

```bash
python src/train.py --model lstm --symbol AAPL --data_dir data/processed --output_dir models
```

Options:
- `--model`: Model type to train (linear, arima, lstm, transformer)
- `--symbol`: Stock symbol to train on
- `--data_dir`: Directory with processed data
- `--output_dir`: Output directory for trained models
- `--epochs`: Number of training epochs for neural models
- `--batch_size`: Batch size for training
- `--window_size`: Window size for time series data
- `--time_based`: Whether to use time-based train/test split

### Model Evaluation

Evaluate and compare models:

```bash
python src/evaluate.py --models lstm transformer --model_paths models/lstm/AAPL_lstm_timestamp.h5 models/transformer/AAPL_transformer_timestamp.h5 --symbol AAPL
```

Options:
- `--models`: List of model types to evaluate
- `--model_paths`: Paths to the trained model files
- `--symbol`: Stock symbol to evaluate on
- `--data_dir`: Directory with processed data
- `--output_dir`: Output directory for results
- `--ablation`: Run ablation studies

### Jupyter Notebooks

The project includes two Jupyter notebooks:

1. **Data Exploration (1_data_exploration.ipynb)**:
   - Visualize stock price data
   - Analyze statistical properties
   - Examine correlation and time series characteristics
   - Explore feature engineering possibilities

2. **Model Prototyping (2_model_prototyping.ipynb)**:
   - Implement and compare different prediction models
   - Evaluate performance using various metrics
   - Visualize predictions and errors
   - Conduct ablation studies to understand model behavior

To run the notebooks:
```bash
jupyter notebook notebooks/1_data_exploration.ipynb
```

## Models

This project includes four types of stock prediction models:

1. **Linear Regression**: A simple baseline that predicts prices based on linear relationships.

2. **ARIMA**: AutoRegressive Integrated Moving Average - a traditional time series forecasting model.

3. **LSTM**: Long Short-Term Memory networks - specialized RNNs capable of learning long-term dependencies.

4. **Transformer**: Attention-based architecture that can capture complex patterns in sequential data.

## Results

The models are evaluated using multiple metrics:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

Performance comparisons and visualizations are saved to the `reports/figures` directory.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.