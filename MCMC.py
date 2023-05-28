import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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
st.sidebar.title('Metropolis-Hastings Demo')
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
