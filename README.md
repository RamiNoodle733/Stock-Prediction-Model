# Stock Market Prediction

This project implements machine learning models to predict stock prices based on historical data, specifically focusing on Apple (AAPL) stock.

## Project Overview

This project uses historical stock price data to train and evaluate several machine learning models for predicting future stock prices. The goal is to compare different approaches and understand their strengths and limitations in the context of financial time series forecasting.

## Features

- Data fetching using `yfinance` library
- Data preprocessing and sequence creation for time series prediction
- Implementation of multiple prediction models:
  - Linear Regression
  - Random Forest
  - Neural Network (MLPRegressor as LSTM substitute)
- Model evaluation and comparison
- Feature importance analysis
- Ablation studies on sequence length effects
- Future price prediction

## Project Structure

```
stock-prediction-model/
├── data/                  # Stock data and generated plots
├── models/                # Saved trained models
├── src/                   # Source code
│   ├── data_loader.py     # Functions for loading and processing stock data
│   ├── models.py          # Model implementations and evaluation functions
│   └── main.py            # Main script to run the complete workflow
├── requirements.txt       # Python dependencies
├── stock_prediction_analysis.ipynb  # Jupyter notebook for analysis
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-prediction-model.git
cd stock-prediction-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script

To run the complete stock prediction workflow:

```bash
python src/main.py
```

This will:
- Download or load Apple stock data
- Visualize the data
- Train and evaluate multiple prediction models
- Compare model performance
- Make predictions for future stock prices

### Exploring with Jupyter Notebook

To run the Jupyter notebook for interactive analysis:

```bash
jupyter notebook stock_prediction_analysis.ipynb
```

## Results

The models are evaluated using several metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Our experiments found that:
- Linear Regression performs surprisingly well for short-term stock price prediction
- Different sequence lengths affect prediction accuracy
- Recent days have more predictive power than older data points

## Contributors

Taha, Akshnoor, Rami

## License

This project is licensed under the MIT License - see the LICENSE file for details.