import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

###############

def stockplots():
    # Define the set of stocks
    top_100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK-A', 'BABA', 'V', 'JPM', 
        'JNJ', 'WMT', 'MA', 'PG', 'UNH', 'DIS', 'NVDA', 'HD', 'PYPL', 'BAC', 'VZ', 
        'ADBE', 'CMCSA', 'KO', 'NKE', 'MRK', 'PEP', 'PFE', 'T', 'INTC', 'CRM', 'ABT', 
        'ORCL', 'ABBV', 'CSCO', 'TMO', 'AVGO', 'XOM', 'ACN', 'QCOM', 'TXN', 'MCD', 
        'BHP.AX', 'RIO.L', 'TM', 'BA', 'LMT', 'HON', 'UNP', 'SBUX', 'IBM', 'MMM', 
        'GE', 'AMD', 'BLK', 'CAT', 'CVX', 'RTX', 'GS', 'MS', 'C', 'UPS', 'ISRG', 
        'INTU', 'VRTX', 'ITW', 'BDX', 'TGT', 'ANTM', 'TJX', 'SYK', 'NEE', 'AMT', 
        'ADP', 'ILMN', 'CME', 'KMB', 'SPGI', 'USB', 'MDT', 'PNC', 'DUK', 'MU', 
        'MO', 'CSX', 'BKNG', 'ZTS', 'CL', 'PLD', 'GILD', 'CB', 'AXP', 'CCI', 'LIN', 
        'COST', 'SO', 'LOW', 'TFC', 'DHR', 'BK', 'DD', 'BIIB', 'CI', 'BSX', 'TRV', 
        'BAX', 'EOG', 'ATVI', 'GD', 'COF'
    ]

    # Select start and end date
    start_date = st.date_input("Start date", pd.to_datetime('2022-01-01'))
    end_date = st.date_input("End date", pd.to_datetime('2023-01-01'))

    if start_date > end_date:
        st.error('Error: End date must fall after start date.')
        
    # Use the multiselect widget to select stocks
    selected_stocks = st.multiselect("Select stocks to plot:", top_100_tickers, default=top_100_tickers[:4])
    
    df2 = yf.download(selected_stocks, start=start_date, end=end_date)
    df2_close = df2['Adj Close']

    fig, axs = plt.subplots(5, 4)
    fig.set_size_inches(16,22)
    old_df = pd.DataFrame()
    x = 0
    
    for col in df2_close.columns:
    
        selected_series = df2_close[col].values.tolist()
        analysis = pd.DataFrame(selected_series.copy())
        analysis.columns = ["Price"]
        analysis['Returns'] = np.log(analysis["Price"] / analysis["Price"].shift(1))
        analysis = analysis.fillna(0)
        analysis["Price"].plot(title="{} Price Summary".format(col), ax=axs[0,x])
        analysis['Returns'].plot(title="{} Returns".format(col), ax=axs[1,x])
        analysis['Returns'].plot(kind='hist', bins=100, orientation='vertical', title="{} Histogram".format(col), ax=axs[2,x])
        
        if x == 0:
            analysis['Prev Returns'] = analysis['Returns']
        else: analysis['Prev Returns'] = old_df
    
        regression = np.polyfit(analysis['Prev Returns'], analysis['Returns'], deg=1)
        analysis[['Prev Returns', 'Returns']].plot(kind='scatter', title="{} Scatter vs Prev".format(col), x="Prev Returns", y="Returns", ax=axs[3,x])
        axs[3,x].plot(analysis['Prev Returns'], np.polyval(regression, analysis['Prev Returns']), 'r', lw=2)
    
        analysis['Prev Returns'].rolling(window=20).corr(analysis['Returns']).plot(title="{} Correlation to Prev".format(col), ax=axs[4,x])
        # axs[4,x].axhline(analysis['Returns'].corr().iloc[0,1], c='r')
    
        x += 1
        old_df = analysis['Returns']
    
    # fig.suptitle("Portfolio Summary")
    st.pyplot(fig)

    returns = (df2['Adj Close']/ df2['Adj Close'].shift(1))
    returns = returns.fillna(1)
    fig2, axs2 = plt.subplots(1, 5, sharey=True, gridspec_kw={'wspace': 0})
    fig2.set_facecolor("white")
    fig2.set_size_inches(24, 8)
    idx = 0
    start = 0
    
    for idx in range(5):
    
        analysis_cut = returns.iloc[start:]
        analysis_cut = analysis_cut.reset_index(drop=True)
        analysis_cut = analysis_cut.cumprod()
        analysis_cut.plot(ylim=(0.5, 2.5), ax=axs2[idx])
        idx += 1
        start += 50
    
    axs2[0].set_ylabel("Return Rebased to 1 at Beginning of Period")
    fig2.suptitle("Relative % Returns for each Investment over Time Period", y=1)
    st.pyplot(fig2)

###############

def portfolio_simulator():

    # Select up to 10 stocks
    selected_stocks = st.multiselect("Type or select up to 10 stocks for your portfolio:", options=[], default=['AAPL', 'MSFT'])

    for stock in selected_stocks:
        ticker_info = yf.Ticker(stock).info
        if not ticker_info or 'symbol' not in ticker_info:
            st.error(f"Error: {stock} is not a valid stock symbol.")
            return  # Stop execution
    
    # Get the percentage allocation for each selected stock
    allocations = {}
    for stock in selected_stocks:
        allocations[stock] = st.number_input(f"Allocation for {stock} (in %):", min_value=1, max_value=100, value=10)

    # Check if the sum of the allocations is equal to 100
    if sum(allocations.values()) != 100:
        st.error('Error: The sum of the allocations must be equal to 100%.')

    # Select start and end date
    start_date = st.date_input("Purchase date", pd.to_datetime('2022-01-01'))
    end_date = st.date_input("End date", pd.to_datetime('2023-01-01'))

    if start_date > end_date:
        st.error('Error: End date must fall after purchase date.')

    # Download the data for selected stocks over the chosen period
    data = yf.download(selected_stocks, start=start_date, end=end_date)['Adj Close']

    # Calculate the portfolio value over time
    portfolio_value = pd.Series(index=data.index, data=np.zeros(len(data.index)))
    for stock in selected_stocks:
        # Calculate the quantity for each stock
        quantities = (allocations[stock] / 100) * 100 / data[stock].iloc[0]  # Here 100 is the total portfolio size
        # Calculate the portfolio value
        portfolio_value += quantities * data[stock]

    # Plot the portfolio value
    st.subheader('Portfolio Value Over Time')
    portfolio_value.plot(figsize=(12,8))
    st.pyplot(plt.gcf())

    # Calculate and display portfolio return
    portfolio_cost = portfolio_value.iloc[0]
    portfolio_end_value = portfolio_value.iloc[-1]
    portfolio_return = ((portfolio_end_value - portfolio_cost) / portfolio_cost) * 100  # In percentage
    st.write(f"Portfolio return from {start_date} to {end_date} is {portfolio_return:.2f}%.")

    returns = np.log(data['Adj Close']/ data['Adj Close'].shift(1))
    returns.dropna(inplace=True)
    p_ret = []
    p_vol = []
    p_weights = []
    num_assets = len(returns.columns)
    num_iterations = 10000
    
    for portfolio in range(num_iterations):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        ret = np.dot(weights, returns_overall)
    
        p_ret.append(ret)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        stan_dev = np.sqrt(var)
        volatility = stan_dev*np.sqrt(250)
        p_vol.append(volatility)
    
    data2 = {'Returns':p_ret, 'Volatility':p_vol}
    for counter, symbol in enumerate(data.columns.tolist()):
        data2[symbol+' weight'] = [w[counter] for w in p_weights]

    portfolios = pd.DataFrame(data2)
    risk_free_rate = 0.1
    optimal_risky_port = portfolios.iloc[((portfolios['Returns']-risk_free_rate)/portfolios['Volatility']).idxmax()]
    plt.subplots(figsize=(10, 10))
    plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
    st.pyplot(plt)
    st.write(f"Optimised weighting: {optimal_risky_port}")

#########

# Function to generate sample data
def generate_data(samples):
    np.random.seed(0)
    return np.random.normal(10, 2, samples)

# Likelihood and transition model
def likelihood(param, data):
    mu, sigma = param
    if sigma < 0:
        return 0
    else:
        return np.prod(stats.norm(mu, sigma).pdf(data))

def transition_model(param):
    return [np.random.normal(param[0], 0.5), abs(np.random.normal(param[1], 0.5))]

# Metropolis-Hastings algorithm
def metropolis_hastings(likelihood_func, transition_model, param_init, iterations, data):
    param_current = param_init
    param_posterior = []
    for i in range(iterations):
        param_new = transition_model(param_current)
        ratio = likelihood_func(param_new, data) / likelihood_func(param_current, data)
        acceptance = min(1, ratio)
        if np.random.uniform(0,1) < acceptance:
            param_current = param_new
        param_posterior.append(param_current)
    return param_posterior

# Main Metropolis-Hastings demonstration function
def run_metropolis_hastings_demo(samples, iterations, burn_in):
    data = generate_data(samples)
    output = metropolis_hastings(likelihood, transition_model, [0,1], iterations, data)
    estimated_mean = np.mean([param[0] for param in output[burn_in:]])
    estimated_std_dev = np.mean([param[1] for param in output[burn_in:]])

    # Output to Streamlit
    st.title('Metropolis-Hastings Algorithm')
    st.write(f"Estimated Mean: {estimated_mean}")
    st.write(f"Estimated Standard Deviation: {estimated_std_dev}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot([param[0] for param in output], label='Mean')
    plt.plot([param[1] for param in output], label='Standard Deviation')
    plt.axvline(x=burn_in, linestyle='--', color='red', label='Burn-in period')
    plt.legend()
    st.pyplot(plt)

######################### Navigation
st.sidebar.title('NATLANTICS')
page = st.sidebar.radio("Go to", ['Stock Price Plot', 'Portfolio Simulator', 'Metropolis-Hastings Demo'])

if page == 'Stock Price Plot':
    stockplots()
elif page == 'Portfolio Simulator':
    portfolio_simulator()
elif page == 'Metropolis-Hastings Demo':
    st.sidebar.title('Metropolis-Hastings Demo')
    samples = st.sidebar.slider('Number of data samples', 100, 1000, 1000)
    iterations = st.sidebar.slider('Number of iterations', 1000, 10000, 5000)
    burn_in = st.sidebar.slider('Burn-in period', 0, iterations//2, 1000)

    # Call the main function
    run_metropolis_hastings_demo(samples, iterations, burn_in)
