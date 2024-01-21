import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin

yfin.pdr_override()


def get_data(stocks, start, end):
    # Get price data for all provided stocks from X days ago to now
    stock_data = pdr.get_data_yahoo(stocks, start, end)
    # Get only the closing price
    stock_data = stock_data['Close']

    # Get the percentage change of the stock price per day
    returns = stock_data.pct_change()

    # Get the mean return over the time period fo every stock
    mean_returns = returns.mean()

    # Get the covariance of the stocks
    # For every stock, get its covariance with every other stock so for X number of stocks
    # this forms an X by X covariance matrix
    # Covariance is essentially how much it moves in the same direction (similar to correlation)
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix


stock_list = ['CBA', 'BHP', ]  # ,'TLS', 'NAB', 'WBC', 'STO'
stocks = [stock + '.AX' for stock in stock_list]
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)

# Get the mean returns and the covariance matrix
mean_returns, cov_matrix = get_data(stocks, start_date, end_date)

# len(meanReturns) is just the number of stocks
weights = np.random.random(len(mean_returns))  # get a random number for each stock, so X random numbers

weights /= np.sum(weights)  # normalise the random numbers so they total to 1 which represents how much of each stock we have invested in

# Monte Carlo Method
mc_sims = 100  # number of simulations to run
T = 100  # timeframe over which we will simulate each run, in days

# make an array of X columns (num of stocks) and T rows (number of days the sim will be run for
# X columns by T rows 2D array
meanM = np.full(shape=(T, len(weights)), fill_value=mean_returns)

# transpose the array i.e. num of rows -> num of columns and vice versa so now
# instead of X columns and T rows, now T columns and X rows
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initial_portfolio = 10000

# Cholesky decomposition allows you to take a matrix and decompose it into
# a lower triangular matrix L
# and its conjugate transpose


for m in range(0, mc_sims):
    # Random variables array of X columns and T rows from normal distribution
    Z = np.random.normal(size=(T, len(weights)))  # Uncorrelated random variables
    L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition to Lower Triangular Matrix
    corr_vars = np.inner(L, Z)  # dot product: uncorrelated and cholesky = correlated
    daily_returns = meanM + corr_vars  # Correlated daily returns for individual stocks
    daily_returns = daily_returns.T  # transpose the array so we can use it
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, daily_returns) + 1) * initial_portfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value (£)')
plt.xlabel('Days')
plt.title('MC simulation of stock portfolio')
plt.show()
