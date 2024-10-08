import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import requests
from bs4 import BeautifulSoup
import re
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
import string

##############

def stockplots():
    # Define the set of stocks
    top_100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-A', 'BABA', 'V', 'JPM', 
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

    # Allow users to manually input stock symbols as a comma-separated string
    stock_input = st.text_input("Type up to 10 stock symbols for your portfolio (separated by commas):", value='AAPL, BP, WMT, BA, PFE')
    selected_stocks = [stock.strip() for stock in stock_input.split(',')]

    # Check if there are more than 10 stocks
    if len(selected_stocks) > 10:
        st.error('Error: You can select up to 10 stocks only.')
        return  # Stop execution

    # Check if the stock symbols are valid by trying to download the data
    for stock in selected_stocks:
        if yf.Ticker(stock).info.get('symbol') != stock:
            st.error(f"Error: {stock} is not a valid stock symbol.")
            return  # Stop execution
    
    # Get the percentage allocation for each selected stock
    allocations = {}
    for stock in selected_stocks:
        allocations[stock] = st.number_input(f"Allocation for {stock} (in %):", min_value=1, max_value=100, value=20)

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

    returns = np.log(data/ data.shift(1))
    returns.dropna(inplace=True)
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()
    returns_overall = returns.mean()
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
    risk_free_rate = 0.08
    optimal_risky_port = portfolios.iloc[((portfolios['Returns']-risk_free_rate)/portfolios['Volatility']).idxmax()]
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    plt.subplots(figsize=(10, 10))
    plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
    st.pyplot(plt)
    st.write("Optimal Risky Portfolio Weighting")
    st.table(optimal_risky_port)
    st.write("Minimum Volatility Portfolio Weighting")
    st.table(min_vol_port)

#########

# Metropolis Hastings - refer to Gist

#########
# Black Scholes for European
def black_scholes(option_type, S, K, r, T, sigma):
    # Black-Scholes formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    elif option_type == 'put':
        option_price = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

def calculate_volatility(stock_symbol, start_date, end_date):
    # Get stock historical data
    stock_data = yf.Ticker(stock_symbol).history(period='1d', start=start_date, end=end_date)['Close']
    # Calculate daily returns
    daily_returns = stock_data.pct_change().dropna()
    # Calculate volatility as the standard deviation of daily returns
    volatility = daily_returns.std()
    return volatility

def european_option_demo(stock_symbol, start_date, end_date, option_type, strike_price, risk_free_rate, expiry_days):
    S0 = yf.Ticker(stock_symbol).history(period='1d', start=start_date, end=end_date)['Close'][-1]
    T = expiry_days / 252  # Convert to years assuming 252 trading days per year
    sigma = calculate_volatility(stock_symbol, start_date, end_date)

    european_option_price_result = black_scholes(option_type, S0, strike_price, risk_free_rate, T, sigma)
    
    st.write(f"The estimated price for the European {option_type} option is: ${european_option_price_result:.2f}.")

#########
def american_option_LSM(stock_symbol, start_date, end_date, K, r, M, N, option_type='call'):
    S0 = yf.Ticker(stock_symbol).history(period='1d', start=start_date, end=end_date)['Close'][-1]
    dt = M / 252  # Convert to years assuming 252 trading days per year
    sigma = calculate_volatility(stock_symbol, start_date, end_date)

    discount_factor = np.exp(-r * dt)
    
    # Simulate asset price paths using Geometric Brownian Motion
    paths = np.zeros((M+1, N))
    paths[0] = S0
    for t in range(1, M+1):
        z = np.random.standard_normal(N)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Initialize payoffs at final time step
    if option_type == 'call':
        payoffs = np.maximum(paths[-1] - K, 0)
    else:
        payoffs = np.maximum(K - paths[-1], 0)

    # Backward induction
    for t in range(M-1, 0, -1):
        in_the_money = (paths[t] > K) if option_type == 'call' else (paths[t] < K)
        if np.any(in_the_money):
            X = paths[t][in_the_money].reshape(-1, 1)
            y = payoffs[in_the_money] * discount_factor
            model = LinearRegression().fit(X, y) # Fit continuation values
            continuation_values = model.predict(X)
            immediate_payoffs = (paths[t] - K) if option_type == 'call' else (K - paths[t])
            exercise_now = immediate_payoffs[in_the_money] > continuation_values
            payoffs[in_the_money] = np.where(exercise_now, immediate_payoffs[in_the_money], payoffs[in_the_money] * discount_factor)

    # Discounting to the present value
    option_price = np.exp(-r * dt) * np.mean(payoffs)

    return option_price

#########
# Simulations for American
def simulate_brownian_motion(stock_symbol, start_date, end_date, forecast_days):
    # Get stock historical data
    stock_data = yf.Ticker(stock_symbol).history(period='1d', start=start_date, end=end_date)['Close']

    # Calculate daily returns
    daily_returns = stock_data.pct_change().dropna()

    # Calculate drift and volatility
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    # Initialize stock price
    S0 = stock_data[-1]
    stock_prices = [S0]

    # Simulate stock price using geometric Brownian motion
    for t in range(forecast_days):
        dW = np.random.normal()
        dS = mu * S0 * 1 + sigma * S0 * dW
        S0 += dS
        stock_prices.append(S0)

    return stock_prices

def simulate_multiple_paths(stock_symbol, start_date, end_date, forecast_days, n_simulations):
    paths = [simulate_brownian_motion(stock_symbol, start_date, end_date, forecast_days) for _ in range(n_simulations)]
    return pd.DataFrame(paths).transpose()

def american_option_price(option_type, simulated_paths, K, r, dt):
    # Determine the optimal early exercise points for each path
    if option_type == 'call':
        exercise_values = (simulated_paths - K).apply(lambda path: path[path > 0])
    elif option_type == 'put':
        exercise_values = (K - simulated_paths).apply(lambda path: path[path > 0])
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Determine the payoffs and discount them
    discounted_payoffs = exercise_values.apply(
    lambda path: np.exp(-r * path.first_valid_index() * dt) * path.dropna().iloc[0] if path.first_valid_index() is not None and not path.dropna().empty else 0, 
    axis=0)
    
    # Average the discounted payoffs
    option_price = discounted_payoffs.mean()

    return option_price

def plot_simulated_paths(stock_symbol, start_date, end_date, simulated_paths, strike_price):
    # Get stock historical data
    historical_data = yf.Ticker(stock_symbol).history(period='1d', start=start_date, end=end_date)['Close']

    # Plot historical price
    plt.figure(figsize=(10, 6))
    plt.plot(historical_data.index, historical_data.values, label='Historical Price')

    # Calculate quantiles for simulated paths
    median_path = np.percentile(simulated_paths, 50, axis=1)
    lower_bound = np.percentile(simulated_paths, 25, axis=1)
    upper_bound = np.percentile(simulated_paths, 75, axis=1)

    # Define x values for simulated paths
    forecasted_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), periods=len(median_path), freq='B')
    
    # Plot the median and interquartile range of simulated paths
    plt.plot(forecasted_dates, median_path, label='Median Simulated Price')
    plt.fill_between(forecasted_dates, lower_bound, upper_bound, color='skyblue', alpha=0.4, label='Interquartile Range (25th-75th percentile)')

    # Plot the strike price
    plt.axhline(y=strike_price, color='r', linestyle='--', label='Strike Price')

    # Labeling
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Historical and Simulated Stock Price with Strike Price')
    plt.show()

def brownian_motion_demo():
    st.title('Brownian Motion & American Option Pricing')

    stock_symbol = st.text_input("Stock Symbol:")
    start_date = st.date_input("Start date for historical data:")
    end_date = st.date_input("End date for historical data:")
    forecast_days = st.number_input("Number of forecast days:", min_value=1, max_value=365, value=30)
    strike_price = st.number_input("Strike price for the option:", value=100.0)
    risk_free_rate = st.number_input("Risk-free interest rate (annual, in %):", value=0.01) / 100
    option_type = st.selectbox("Option Type:", options=['call', 'put'])

    # Simulate Multiple Stock Price Paths
    n_simulations = st.slider("Number of simulations for American option:", min_value=100, max_value=1000, value=100)
    simulated_paths = simulate_multiple_paths(stock_symbol, start_date, end_date, forecast_days, n_simulations)

    # Plotting
    st.subheader('Simulated Stock Price Paths')
    plot_simulated_paths(stock_symbol, start_date, end_date, simulated_paths, strike_price)
    st.pyplot(plt)

    # Calculate American option price
    american_option_price_result = american_option_price(option_type, simulated_paths, strike_price, risk_free_rate, dt=1/252)

    # Display the American option price
    st.write(f"The estimated price for the American {option_type} option is: ${american_option_price_result:.2f}.")

    # Additional part for European Option
    st.write('American Option Pricing using Least Square Monte Carlo')
    # european_option_demo(stock_symbol, start_date, end_date, option_type, strike_price, risk_free_rate, forecast_days)
    result2 = american_option_LSM(stock_symbol, start_date, end_date, strike_price, risk_free_rate, forecast_days, n_simulations, option_type=option_type)
    st.write(f"The estimated price for the American {option_type} option using Least Square Monte Carlo method is: ${result2:.2f}.")

#########

def fetch_headlines(keywords):
    url = f"https://www.google.com/search?q=site:news.google.com+{'+'.join(keywords)}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [re.sub('<.*?>', '', str(h3)) for h3 in soup.find_all('h3')]

    return headlines

def display_word_cloud(headlines):
    text = ' '.join(headlines)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())

def create_connection_graph(keywords, headlines):
    # Preprocess keywords
    keywords = [keyword.lower() for keyword in keywords]
    G = nx.Graph()

    for keyword in keywords:
        connections = []
        for headline in headlines:
            # Lowercase and remove punctuation
            headline = headline.lower().translate(str.maketrans('', '', string.punctuation))
            if keyword in headline:
                connections.extend([word for word in headline.split() if word != keyword])
        
        # Consider all connections, not just the top 5
        common_words = [word for word, count in Counter(connections).items()]
        G.add_node(keyword, color='blue')
        for word in common_words:
            G.add_edge(keyword, word, color='grey')

    pos = nx.spring_layout(G)
    colors = [G.nodes[node].get('color', 'red') for node in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold', node_size=700)
    plt.show()

def financialheadlines():
    st.title('Google News Headline Search')
    keywords_input = st.text_input('Enter keywords to search for (e.g., company name, topics), separated by commas:')

    if st.button('Search'):
        keywords = [keyword.strip() for keyword in keywords_input.split(',')]
        headlines = fetch_headlines(keywords)
        st.subheader('Search Results')

       # Display the list of headlines as a table
        st.write("Headlines:")
        st.table(pd.DataFrame(headlines, columns=["Headlines"]))  # Using pandas DataFrame to create a table
        
        st.write("Word Cloud:")
        display_word_cloud(headlines)

        st.write("Word Connections:")
        create_connection_graph(keywords, headlines)

######################### Navigation
st.sidebar.title('NATLANTICS')
page = st.sidebar.radio("Go to", ['Stock Price Plot', 'Portfolio Simulator', 'Brownian Motion', 'Financial News Summary'])

if page == 'Stock Price Plot':
    stockplots()
elif page == 'Portfolio Simulator':
    portfolio_simulator()
elif page == 'Brownian Motion':
    brownian_motion_demo()
elif page == 'Financial News Summary':
    financialheadlines()
