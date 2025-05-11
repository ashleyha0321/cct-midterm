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
    # Defining priors
    D = pm.Uniform("D", 0, 1, shape=N)
    Z = pm.Bernoulli("Z", p=0.5, shape=M)

    # Computing probability matrix p_ij
    D_reshaped = D[:, None]  # shape (N, 1)
    p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)  # shape (N, M)

    # Likelihood
    Y = pm.Bernoulli("Y", p=p, observed=X)

    # Sampling
    trace = pm.sample(2000, tune=2000, chains=4, target_accept=0.95, return_inferencedata=True)

# Analyzing convergence
summary = az.summary(trace, var_names=["D", "Z"])
print("\nSummary:\n", summary)

# Estimating informant competence
competence_means = summary.loc[summary.index.str.startswith("D"), "mean"]
most_competent = competence_means.idxmax()
least_competent = competence_means.idxmin()

print("\nInformant competence (posterior means):\n", competence_means)
print(f"Most competent: {most_competent}")
print(f"Least competent: {least_competent}")

# Plot competence
az.plot_posterior(trace, var_names=["D"], hdi_prob=0.94)
plt.suptitle("Posterior Distributions of Informant Competence", y=1.02)
plt.tight_layout()
plt.show()

# --- Consensus answers ---
z_means = summary.loc[summary.index.str.startswith("Z"), "mean"]
z_mode = np.round(z_means).astype(int)

print("\nConsensus answer key (rounded posterior mean):\n", z_mode.values)

# Plot consensus answers
az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.94)
plt.suptitle("Posterior Distributions of Consensus Answers", y=1.02)
plt.tight_layout()
plt.show()

# --- Naive aggregation ---
naive_key = (X.mean(axis=0) > 0.5).astype(int)
print("\nNaive majority vote answer key:\n", naive_key)

# Compare naive vs. model consensus
differences = naive_key != z_mode.values
diff_indices = np.where(differences)[0]
print(f"\nQuestions with different answers between naive and model: {diff_indices}")