#import stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# define some functions
def calc_portfolio(mu, expected_return, cov_matrix):
    """
    Uses Markowitz portfolio optimization to calculate asset 
    allocation for lowest risk (AKA minimum variance) portfolio 
    given the following variables:
    - a target level of return (mu),
    - a vector of expected returns for available assets (expected_return)
    - a covariance matrix of returns for available assets (cov_matrix)

    Also calculates the total variance and standard deviation for each portfolio.
    """

    # calculate asset allocation
    ones = np.matrix([[1] for item in r])
    x1 = calc_x1(expected_return, cov_matrix, ones)
    x2 = calc_x2(expected_return, cov_matrix)
    x3 = calc_x3(expected_return, cov_matrix, ones)
    lambda1 = calc_lambda1(x1, x2, x3, mu)
    lambda2 = calc_lambda2(x1, x2, x3, mu)
    weights = calc_weights(cov_matrix, lambda1, lambda2, expected_return, ones)

    # add results to dictionary object 
    portfolio_details = {}
    portfolio_details['mu'] = mu
    
    # capture all weights, no matter how many assets
    for i, val in enumerate(weights):
        portfolio_details["w%s" % i] = val[0,0]

    # calculate portfolio variance and standard deviation
    portfolio_details['tot_var'] = (weights.T * C * weights)[0,0]
    portfolio_details['std_dev'] = np.sqrt(portfolio_details['tot_var'])

    return portfolio_details

def calc_x1(expected_return, cov_matrix, ones):
    return expected_return.T * cov_matrix.I * ones

def calc_x2(expected_return, cov_matrix):
    return expected_return.T * cov_matrix.I * expected_return

def calc_x3(expected_return, cov_matrix, ones):
    return ones.T * cov_matrix.I * ones

def calc_lambda1(x1, x2, x3, mu):
    return (x3 * mu - x1) / (x2 * x3 - np.square(x1))

def calc_lambda2(x1, x2, x3, mu):
    return (x2 - x1 * mu) / (x2 * x3 - np.square(x1))

def calc_weights(cov_matrix, lambda1, lambda2, expected_return, ones):
    return cov_matrix.I * (lambda1[0,0] * expected_return + lambda2[0,0] * ones)

def plot_asset_allocation(results, save_dir):
    """
    Plots the asset allocation for all portfolios generated
    along with a dotted line to indicate minimum variance
    portfolio. Plot is saved in relative directory provided as 
    'save_dir'.
    """

    # Find portfolio with lowest risk, and generate x and y values to plot the red line
    lowest_risk_portfolio = results.iloc[np.argmin(results['std_dev'])]
    y = [lowest_risk_portfolio['mu'] for _ in range(len(returns_vector))]
    x = np.arange(-0.2, 1.4, 0.2)

    fig, ax = plt.subplots(figsize=(15,9))
    ax.set_title('Lowest risk asset allocation for a range of expected return values', size=20)
    ax.set_ylabel('percent allocation', fontsize=16)
    ax.set_xlabel('expected return', fontsize=16)
    # plotting the portfolio weights
    weight_col = [col for col in results if col.startswith('w')]
    ax.plot(results['mu'], results[weight_col])
    # plotting the line to show min var portfolio
    ax.plot(y,x,'--')
    weight_col.append('min var portfolio')
    ax.legend(weight_col)

    #save plot to file
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fig.savefig(save_dir + 'PortfolioAllocation.png')

    return None

def plot_risk(results, save_dir):
    """
    Plots the standard deviation of each portfolio
    to show which one has the lowest risk. Plot is saved 
    in relative directory provided as 'save_dir'.
    """

    fig, ax = plt.subplots(figsize=(15,9))
    ax.set_title('Standard deviation of portfolio returns', size=20)
    ax.set_ylabel('standard deviation', fontsize=16)
    ax.set_xlabel('expected return', fontsize=16)
    ax.plot(results['mu'], results['std_dev'])

    #save plot to file
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fig.savefig(save_dir + 'PortfolioStandardDeviation.png')

    return None

#do the thing
if __name__ == '__main__':

    r = np.matrix([[0.0761], [0.0499], [0.0949]])

    C = np.matrix([[0.007870, 0.003085, 0.000989],
                   [0.003085, 0.003696, 0.000129],
                   [0.000989, 0.000129, 0.009587]])

    returns_vector = np.arange(0.05, 0.09, 0.005)

    results = pd.DataFrame()

    for target_return in returns_vector:
        min_var_portfolio = calc_portfolio(target_return, r, C)
        results = results.append(min_var_portfolio, ignore_index=True)
        
    print(results)

    save_dir = './visuals/'

    plot_asset_allocation(results, save_dir)
    plot_risk(results, save_dir)



