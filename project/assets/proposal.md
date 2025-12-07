# VolForecast: Equity Portfolio Risk Forecasting
### EE 344 Final Project Proposal
Oorjit Chowdhary (oorjitc@uw.edu)

# Resources (for me)
https://otexts.com/fpp3/
https://facebook.github.io/prophet/

## Project Overview
My goal with this project is to build a data-driven system that forecasts the short term volatility and downside risk of an equity portfolio, using historical price and volume data and applying both classical and modern time series modeling techniques. The system will not predict any future stock prices, but rather forecast the realized volatility (magnitude of price fluctuations) and related risk metrics like Value at Risk (VaR) and Conditional VaR for a portfolio of stocks over the next week. The modeling approach will begin with classical econometric models like ARIMA and exponential smoothing and then extend to Facebook Prophet to compare their performances.

The data will be sourced using Yahoo Finance (either API or the yfinance Python package) to obtain daily open, high, low, close, and volume data for a selected set of stocks over the past 5-10 years, which will be used to compute rolling realized volatility and risk metrics for each stock and the overall portfolio. This volatility time series will become the primary target variable for forecasting.

This model is interesting to me because I'm an ECE and economics double major and I'm very interested in financial markets and personal investing. Also, financial volatility will exhibit memory and clustering effects that might challenge traditional time series models. From the class, this project will apply concepts including curve fitting, cross validation, and model selection principles, extending the time series forecasting techniques we have learned.

Some insights I expect are how well different models can capture volatility dynamics vs Prophet's decomposed approach, whether volatility series exhibit any seasonal patterns or trends that Prophet can learn automatically, and how volatility forecasts propagate to portfolio risk metrics like VaR.

Some challenges I anticipate are non-stationarity, as market structures change over time, which may break model assumptions, and weak signal to noise ratio, as volatility can be quite noisy and hard to predict accurately.

## Timeline
- Nov 10: Dataset & repo ready
- Nov 17: ARIMA / exponential smoothing complete
- Nov 24: Prophet results + comparison
- Dec 1: Portfolio-level volatility & risk forecast integrated
- Dec 5: Final report + presentation submission

## Weekly Availability
I can generally make it to the Wednesday 4:30 pm office hours, but otherwise Monday and Friday after class also work well for me.