# MAG7 Stock Volatility Forecasting

This project analyzes volatility forecasting for the "Magnificent 7" technology stocks (AAPL, MSFT, GOOG, AMZN, TSLA, META, NVDA) using both traditional time series methods and specialized financial models. The analysis demonstrates why domain-specific approaches outperform generic forecasting for financial volatility.

## Project Overview

Financial volatility exhibits unique properties—clustering, persistence, and conditional heteroskedasticity—that generic time series models fail to capture effectively. This project systematically compares traditional econometric approaches (ARIMA, Exponential Smoothing, Prophet) with specialized volatility models (EWMA, GARCH) to demonstrate the importance of domain knowledge in financial modeling.

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
- Downloads daily OHLCV data for MAG7 stocks using yfinance
- Performs comprehensive exploratory data analysis
- Examines price relationships, trading patterns, and correlation structure
- Establishes data quality and alignment across all securities

### 2. Data Preprocessing (`02_data_preprocessing.ipynb`)
- Converts prices to log returns for better statistical properties
- Computes realized volatility using 21-day rolling windows
- Creates equal-weight portfolio for diversification analysis
- Generates multi-horizon volatility features for modeling
- Performs statistical analysis of return distributions

### 3. Generic Time Series Modeling (`03_generic_timeseries_modeling.ipynb`)
- Applies traditional forecasting methods: **ARIMA**, **Exponential Smoothing**, and **Prophet**
- Conducts stationarity testing and diagnostic analysis
- Treats realized volatility as a generic univariate time series
- Evaluates performance using standard metrics (MAE, RMSE, MAPE, R²)
- Demonstrates limitations when volatility-specific properties are ignored

### 4. Quantitative Finance Modeling (`04_quant_finance_modeling.ipynb`)
- Implements specialized volatility models: **EWMA** and **GARCH(1,1)**
- Models conditional variance explicitly through recursive variance equations
- Captures volatility clustering and persistence effects
- Compares performance against generic approaches
- Provides theoretical foundation for superior performance

## Key Findings

### Generic Models Fall Short
- ARIMA, Exponential Smoothing, and Prophet achieved **negative or near-zero R² scores**
- These models assume independence and fail to capture volatility clustering
- They treat each volatility observation as unrelated to market conditions

### Specialized Models Excel
- **EWMA** achieved positive R² scores by modeling exponential decay in volatility impact
- **GARCH(1,1)** demonstrated superior performance by explicitly modeling:
  - Volatility persistence through lagged conditional variance
  - News impact through lagged squared returns
  - Long-run variance levels through constant terms

### Performance Comparison
| Model Type | Typical R² Range | Key Limitation |
|------------|------------------|----------------|
| ARIMA | -0.10 to 0.00 | Ignores volatility clustering |
| Exp Smoothing | -1.25 to -0.03 | No conditional variance modeling |
| Prophet | -1.06 to -0.10 | Treats volatility as trend/seasonal |
| **EWMA** | **0.15 to 0.40** | **Captures persistence** |
| **GARCH(1,1)** | **0.25 to 0.50** | **Models conditional heteroskedasticity** |

## Practical Implications

### Risk Management
- Accurate volatility forecasts are crucial for Value-at-Risk (VaR) calculations
- Portfolio risk depends heavily on volatility predictions
- Regulatory capital requirements often mandate sophisticated volatility models

### Derivatives Pricing
- Options pricing models (Black-Scholes, Heston) require volatility inputs
- Volatility trading strategies depend on accurate forecasting
- Risk-neutral densities are sensitive to volatility assumptions

### Portfolio Optimization
- Mean-variance optimization requires volatility forecasts
- Risk parity strategies need reliable volatility estimates
- Dynamic hedging strategies adjust based on volatility predictions

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
├── 01_data_collection_exploration.ipynb    # Data acquisition and EDA
├── 02_data_preprocessing.ipynb             # Feature engineering
├── 03_generic_timeseries_modeling.ipynb    # Traditional forecasting
├── 04_quant_finance_modeling.ipynb         # Volatility-specific models
├── data/
│   ├── raw/                                # Original downloaded data
│   └── processed/                          # Engineered features
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

## Conclusion

This project demonstrates that **domain expertise matters significantly** in financial modeling. While generic time series methods work well for many forecasting problems, financial volatility requires specialized approaches that acknowledge its unique statistical properties. The superior performance of EWMA and GARCH models validates decades of financial econometrics research and highlights the importance of incorporating financial theory into practical modeling workflows.

The analysis provides a foundation for more advanced volatility modeling techniques such as multivariate GARCH, stochastic volatility models, and machine learning approaches that incorporate volatility-specific features.
