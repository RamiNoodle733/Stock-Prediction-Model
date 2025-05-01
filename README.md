# Stock Market Prediction Model

This project implements a comprehensive stock market prediction system using various machine learning and deep learning techniques. It provides tools for data collection, preprocessing, model training, and evaluation, allowing you to predict stock prices and compare different prediction methodologies.

## Project Status

**✅ COMPLETED: April 30, 2025**

This project has been completed and the final report has been prepared using the ACM SIG conference LaTeX template. The report includes:
- Analysis of Linear Regression and LSTM models for stock price prediction
- Performance comparison across multiple stocks (AAPL, MSFT, AMD)
- Literature review of 5 key papers in financial forecasting
- Ablation studies on model architecture and learning rates
- Computational complexity analysis
- Visualizations of prediction performance

## Project Structure

```
stock-prediction-model/
├── acm_template/            # LaTeX templates for the final report
│   └── acmart-primary/      # ACM SIG conference template
├── data/
│   ├── raw/                 # Raw stock price data from Yahoo Finance
│   └── processed/           # Processed data with engineered features
├── models/                  # Trained model files and training histories
│   ├── figures/
│   ├── linear/              # Linear regression models (.pkl)
│   └── lstm/                # LSTM model files (.pt, _config.json, _scalers.pkl)
├── notebooks/               # Jupyter notebooks for analysis
├── reports/                 # Final report documents
│   └── figures/             # Report visualizations
├── results/                 # Model evaluation results and predictions
│   ├── figures/             # Generated prediction plots
│   └── metrics/             # Performance metrics CSVs
├── src/                     # Python source code
│   ├── fetch_data.py             # Script to download stock data
│   ├── preprocess.py             # Data cleaning and feature engineering
│   ├── train.py                  # Model training functionality
│   ├── evaluate.py               # Model evaluation and comparison
│   ├── generate_figures.py       # Create figures for the report
│   ├── rescale_predictions.py    # Fix and rescale predictions
│   ├── diagnose_predictions.py   # Tools to diagnose prediction issues
│   └── models/
│       ├── baseline.py           # Linear Regression models
│       └── advanced.py           # LSTM models
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

### Step 7: Generate Figures for Final Report

Generate all necessary figures for the final report:

```bash
python src/generate_figures.py
```

This script creates:
- Training loss curves for AAPL and MSFT
- RMSE and R² comparison charts across models and stocks
- Actual vs. predicted price plots for multiple stocks
- All figures are saved to the `reports/figures/` directory

### Step 8: Compile the Final Report

Compile the LaTeX report for submission:

```bash
cd acm_template/acmart-primary
pdflatex stock_prediction_report.tex
bibtex stock_prediction_report
pdflatex stock_prediction_report.tex
pdflatex stock_prediction_report.tex
```

This will generate the final PDF report `stock_prediction_report.pdf` in the ACM format.

## Key Findings

Our research revealed several important insights:
1. **Linear vs. LSTM Performance**: Surprisingly, Linear Regression models often outperformed more complex LSTM models for this task
2. **Best Performance**: Best RMSE of 9.75 USD achieved by Linear Regression on MSFT stock data
3. **Computational Efficiency**: Linear models showed 50-60x faster training and inference times compared to LSTMs
4. **Feature Importance**: Recent price history (1-day lag) and short-term moving averages were the most predictive features
5. **Underfitting**: Both model types showed signs of underfitting rather than overfitting, suggesting more feature engineering might be needed

## Known Issues and Limitations

1. **AMD Data Processing**: The AMD dataset required extra cleaning due to extreme outliers
2. **LSTM Prediction Range Collapse**: LSTM predictions tend to fall within a narrower range than actual prices
3. **Data Leakage Issue (Fixed)**: Earlier versions had data leakage in the training/evaluation pipeline, fixed in the final implementation

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

This project fulfills all requirements outlined in the assignmentRubric.txt:
- **Integrity**: All required sections are included in the final report with proper formatting
- **Clarity**: Clear problem description and model definitions
- **Literature Review**: Summary of 5 relevant papers in financial forecasting
- **Results**: Comprehensive evaluation including:
  - Advanced ML model implementation (LSTM)
  - Analysis of model performance metrics
  - Ablation studies on model architecture and hyperparameters
  - Computational complexity analysis
  - Comparison between linear and deep learning approaches

## Contributors

- Rami Razaq (ramiabdelrazzaq@gmail.com)
- Taha Amir (tahashah61@gmail.com)
- Akshnoor Singh (akshnoorsingh987@gmail.com)

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.