# Stock Market Prediction Project

This repository implements a full end-to-end workflow for predicting future stock prices using historical NASDAQ data and machine learning models (Linear Regression, Random Forest, LSTM). The pipeline covers data download, preprocessing, model training, evaluation, comparison, ablation study, feature importance, and future forecasting.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Results Interpretation](#results-interpretation)
- [Customization](#customization)
- [Report](#report)
- [Authors](#authors)
- [License](#license)

## Features
- Download and cache NASDAQ stock data via `yfinance`
- Data preparation: scaling and sequence generation
- Three model implementations: Linear Regression, Random Forest, LSTM (with Keras)
- Training-loss plot for LSTM
- Model comparison table and plot (MSE, RMSE, R², training/inference time)
- Random Forest feature importance analysis
- Ablation study on input sequence length
- Batch prediction across multiple tickers
- One-day-ahead future price forecasts

## Prerequisites
- Python 3.8+ (tested on 3.10/3.12)
- Windows / Linux / macOS

## Installation

```bash
# create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# run full analysis on a single stock (AAPL) and then multiple tickers
python -u src/main.py
```

By default, the script saves all figures and CSVs under `results/`. To change tickers or parameters, edit the bottom of `src/main.py` or wrap it with your own CLI.

## Project Structure
```
LICENSE.txt
README.md
requirements.txt
stock_prediction_analysis.ipynb   # exploratory notebook
stock_prediction_project_rubric.txt

data/                              # cached raw CSVs
results/                           # generated plots and tables
src/                               # source code
  ├── data_loader.py               # download, load, preprocess stock data
  ├── data_processor.py            # (empty/deprecated)
  ├── main.py                      # orchestrates workflow
  ├── models.py                    # model implementations and plotting
  └── utilities.py                 # (empty/deprecated)
```

## Outputs
All output files are saved in `results/`:
- `*_stock_price.png`: raw close-price time series
- `*_predictions.png`: actual vs predicted on test set
- `lstm_training_loss.png`: training and validation loss curves
- `*_model_comparison.png` & `.csv`: comparison metrics
- `*_feature_importance.png` & `.csv`: top RF lag features
- `*_ablation_study.png` & `.csv`: performance vs sequence length
- `multiple/`: batch run across tickers

## Results Interpretation
- **Normalized vs Real-scale metrics**: both reported (MSE/RMSE) for model evaluation
- **Overfitting/Underfitting**: inspect LSTM loss curves for divergence
- **Ablation Study**: understand how lookback window affects accuracy

## Customization
- Change `ticker`, `sequence_length`, `test_size` in `src/main.py`
- Add new tickers or modify models by editing `src/models.py`

## Report
Write your 3+ page final report (Word or LaTeX) including:
- Group members & contributions
- Introduction & problem description
- Literature review (3+ related papers)
- ML models & methods
- Experiment results (tables/figures)
- Conclusion

Export your notebook or figures into the report and submit PDF before the deadline.

## Authors
- Group member A: major implementation & report writing
- Group member B: model evaluation & analysis
- Group member C: data processing & visualization

## License
This project is MIT licensed. See [LICENSE.txt](LICENSE.txt).