# VolForecast: Equity Portfolio Risk Forecasting

This project analyzes volatility forecasting for the "Magnificent 7" technology stocks (AAPL, MSFT, GOOG, AMZN, TSLA, META, NVDA) using both traditional time series methods and specialized financial models. The analysis demonstrates why domain-specific approaches outperform generic forecasting for financial volatility.

**Course**: EE 344 | **Term**: Fall 2025 | **Author**: Oorjit Chowdhary

## Project Overview

Financial volatility exhibits unique properties—clustering, persistence, and conditional heteroskedasticity—that generic time series models fail to capture effectively. This project systematically compares traditional econometric approaches (ARIMA, Exponential Smoothing, Prophet) with specialized volatility models (EWMA, GARCH) to demonstrate the importance of domain knowledge in financial modeling.

The analysis covers data from January 2015 to December 2024, implementing rigorous cross-validation to ensure robust findings across different market regimes.

## Key Concepts Explored

### Volatility Characteristics
- **Volatility Clustering**: Periods of high volatility tend to be followed by high volatility periods
- **Mean Reversion**: Volatility tends to revert to long-term averages over time  
- **Conditional Heteroskedasticity**: Variance changes predictably based on past information
- **Leverage Effects**: Negative returns often increase future volatility more than positive returns

### Model Categories
- **Generic Time Series Models**: Treat volatility as any other time series without financial context
- **Econometric Volatility Models**: Explicitly model the conditional variance structure of financial returns

## Notebooks Structure

### 1. Data Collection & Exploration (`01_data_collection_exploration.ipynb`)
- Downloads daily OHLCV data for MAG7 stocks using yfinance (2015-2024)
- Performs comprehensive exploratory data analysis
- Examines price relationships, trading patterns, and correlation structure
- Establishes data quality and alignment across all securities

### 2. Data Preprocessing (`02_data_preprocessing.ipynb`)
- Converts prices to log returns for better statistical properties
- Computes realized volatility using 21-day and 63-day rolling windows (annualized)
- Creates equal-weight portfolio for diversification analysis
- Generates multi-horizon volatility features for modeling
- Performs statistical analysis of return distributions

### 3. Generic Time Series Modeling (`03_generic_timeseries_modeling.ipynb`)
- Applies traditional forecasting methods: **ARIMA**, **Exponential Smoothing**, and **Prophet**
- Conducts hyperparameter grid search and diagnostic analysis
- Treats realized volatility as a generic univariate time series
- Evaluates performance using standard metrics (MAE, RMSE, MAPE, R²)
- Demonstrates limitations when volatility-specific properties are ignored

### 4. Generic Modeling Cross-Validation (`04_generic_modeling_cross_validation.ipynb`)
- Implements rolling window cross-validation with three temporal splits
- Validates findings across different market regimes (70% train, 30% test per fold)
- Computes confidence intervals for performance metrics
- Confirms that poor performance of generic models is consistent, not an artifact
- Applies rigorous model evaluation methodology to ensure generalization

### 5. Quantitative Finance Modeling (`05_quant_finance_modeling.ipynb`)
- Implements specialized volatility models: **EWMA** and **GARCH(1,1)**
- Models conditional variance explicitly through recursive variance equations
- Captures volatility clustering and persistence effects (λ=0.94 for EWMA)
- Compares performance against generic approaches
- Achieves R² scores of 0.79-0.86 vs. near-zero for classical methods

## Technical Requirements

```bash
# Key dependencies
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
statsmodels>=0.13.0
arch>=5.3.0        # GARCH modeling
prophet>=1.1.0     # Facebook Prophet
yfinance>=0.1.87   # Yahoo Finance data
```

## Repository Structure

```
project/
├── 01_data_collection_exploration.ipynb         # Data acquisition and EDA
├── 02_data_preprocessing.ipynb                  # Feature engineering
├── 03_generic_timeseries_modeling.ipynb         # Traditional forecasting methods
├── 04_generic_modeling_cross_validation.ipynb   # Rolling window cross-validation
├── 05_quant_finance_modeling.ipynb              # Volatility-specific models (EWMA, GARCH)
├── requirements.txt                             # Python dependencies
├── README.md                                    # This file
├── assets/
│   ├── proposal.md                              # Initial project proposal
│   └── report.md                                # Comprehensive final report
└── data/
    ├── raw/
    │   └── mag7_prices.csv                      # Original OHLCV data (2015-2024)
    └── processed/                               # Processed datasets for modeling
```

## AI Usage
I used GitHub Copilot and other similar AI coding assistants for mainly implementing the quantitative finance models alongside some boilerpplate code for data manipulation and visualization. All code was reviewed by me. These tools also made it easier to come up with a document structure like this one for the README.