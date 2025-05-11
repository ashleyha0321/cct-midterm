import numpy as np
import pandas as pd
import os
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az


# code was written with the assistance of chatGPT

# loading plant_knowledge.csv
def load_plant_knowledge():
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    filepath = os.path.join(base_dir, '../data/plant_knowledge.csv')
    df = pd.read_csv(filepath)
    data = df.drop(columns=['Informant']).values
    return data

X = load_plant_knowledge()
N, M = X.shape
#print(X)

# Implementing model with PyMC
with pm.Model() as model:
    # Defining priors for informants' competence (D) & consensus answers (Z)
    D = pm.Uniform('D', lower=0.5, upper=1, shape=N)  # Competence of each informant (between 0.5 and 1)
    Z = pm.Bernoulli('Z', p=0.5, shape=M)  # Consensus for each question (Bernoulli with p=0.5)
    
    # Reshaping D for broadcasting (N informants & M questions)
    D_reshaped = D[:, None]  # Shape N x 1, for broadcasting
    
    # Defining probability for each informant's response to each question
    p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)  # p[i,j] = P(i answers j correctly)

    # Likelihood (observed data X)
    X_obs = pm.Bernoulli('X_obs', p=p, observed=X)  # Likelihood of the observed data
    
    # Sampling from posterior
    trace = pm.sample(2000, return_inferencedata=False)  # MCMC sampling (2000 draws)

#  Trace plots to assess convergence
pm.plot_trace(trace)
plt.show()

# Posterior summary for D & Z
posterior_summary = pm.summary(trace, var_names=["D", "Z"], hdi_prob=0.95)  # 95% HDI
print(posterior_summary)

# Check R-hat (convergence) & effective sample size (ESS)
rhat = trace.get_rhat()  # R-hat values to check convergence
ess = trace.get_ess()  # Effective sample size for each variable

print("R-hat values:", rhat)
print("Effective sample sizes:", ess)

# Posterior distribution plot for D & Z
pm.plot_posterior(trace, var_names=["D", "Z"])
plt.show()

# Additional visualization for pair plots (relationships between D & Z)
pm.plot_pair(trace, var_names=["D", "Z"])
plt.show()