import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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
    
    # Use the multiselect widget to select stocks
    selected_stocks = st.multiselect("Select stocks to plot:", top_100_tickers, default=stocks[:4])
    
    df2 = yf.download(selected_stocks, start="2022-03-31", end="2023-03-31")
    df2_close = df2['Adj Close']
    
    fig, axs = plt.subplots(5, 4)
    fig.set_size_inches(16,24)
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
    
    fig.suptitle("Portfolio Summary")
    st.pyplot(fig)

#########

# Function to generate sample data
def generate_data(samples):
    np.random.seed(0)
    return np.random.normal(10, 2, samples)

# Likelihood and transition model
def likelihood(param):
    mu, sigma = param
    if sigma < 0:
        return 0
    else:
        return np.prod(stats.norm(mu, sigma).pdf(data))

def transition_model(param):
    return [np.random.normal(param[0], 0.5), abs(np.random.normal(param[1], 0.5))]

# Metropolis-Hastings algorithm
def metropolis_hastings(likelihood_func, transition_model, param_init, iterations):
    param_current = param_init
    param_posterior = []
    for i in range(iterations):
        param_new = transition_model(param_current)
        ratio = likelihood_func(param_new) / likelihood_func(param_current)
        acceptance = min(1, ratio)
        if np.random.uniform(0,1) < acceptance:
            param_current = param_new
        param_posterior.append(param_current)
    return param_posterior

# Streamlit sidebar
def mcmcdemo():
    samples = st.sidebar.slider('Number of data samples', 100, 1000, 1000)
    iterations = st.sidebar.slider('Number of iterations', 1000, 10000, 5000)
    burn_in = st.sidebar.slider('Burn-in period', 0, iterations//2, 1000)
    
    # Main Streamlit code
    data = generate_data(samples)
    output = metropolis_hastings(likelihood, transition_model, [0,1], iterations)
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

# Navigation
st.sidebar.title('NATLANTICS')
page = st.sidebar.radio("Go to", ['Stock Price Plot', 'Metropolis-Hastings Demo'])

if page == 'Stock Price Plot':
    stockplots()
elif page == 'Metropolis-Hastings Demo':
    mcmcdemo()
