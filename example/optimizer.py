import sys
import numpy as np
import pandas as pd
from scipy import optimize
from kinetic_models import simple_kinetic_model
import time

sys.path.append('../')
from pfr_class import Reactor_model

# ------------------------------------------------------

# You need to install python 3.12 (easiest is through anaconda)
# and numpy, pandas, and scipy libraries

# ------------------------------------------------------

# Measure time
start_t = time.time()

# Read conditions table
df = pd.read_csv('data/simple_example.csv')
# You need to adapt the columns per exercise!
# w_cat is the catalyst loading in g
# T is the temperaure in K
# p_tot is the pressure in bar
# p_A is the partial pressure of A in bar
# F_in_A is the inlet molar flow rate of A in mol/h
# F_out_A is the outlet molar flow rate of A in mol/h
# A_coverage is the fraction of active sites adsorbed with A: [A*]/[*]_0
columns = ['w_cat', 'T', 'p_tot', 'F_in_Inert', 'F_in_A', 'F_in_B', 'F_in_C', 'F_out_Inert', 'F_out_A', 'F_out_B', 'F_out_C']
print(df)

# Reaction system
# 1: A -> B
# 2: A -> C
# Kinetics in kinetic_models.py

# Reaction conditions from dataframe
# n_species: inert gas, A, B, C
w = df.loc[:, 'w_cat'].to_numpy()  # in g
temp = df.loc[:, 'T'].to_numpy()
ptot = df.loc[:, 'p_tot'].to_numpy()
F_0 = df.loc[:, ['F_in_Inert', 'F_in_A', 'F_in_B', 'F_in_C']].to_numpy()  # (n_exp, n_species)
F_out = df.loc[:, ['F_out_Inert', 'F_out_A', 'F_out_B', 'F_out_C']].to_numpy()  # (n_exp, n_species)

# Stoichiometric constants (n_species, n_reac)
v = np.array([[0, 0],   # Inert gas is not consumed
              [-1, -1],  # A is consumed in both reactions
              [1, 0],   # B is produced in reaction 1
              [0, 1]])  # C is produced in reaction 2 
# Kinetic constants

# Real values used to generate the results
k0 = np.array([1.5e6, 2.0e8])  # (n_k,)
ea = np.array([50000, 75000])  # (n_k,) in J/mol
T_ref = 423  # to estimated k_ref instead of k_0

# Initialize class
# Create a reactor object using kinetic model and reaciton conditions
# Analyze the file pfr_class.py to understand how it works
my_model = Reactor_model(v, simple_kinetic_model, w, temp, ptot, F_0, F_out, T_ref=T_ref)
# Once initialized, it predicts outlet concentrations using k0 and Ea values
# print(my_model.predict_F_out(k0, ea))

# ------------------------------------------------------

# Solve optimization problem

# Initial guess and bounds are very important for the solver to converge rapidly
# k_ref in log space: log(k_ref) is in [-20, 20]
# ea or delta_H in 1/1e4 space: ea/1e4 is in [1, 20]
# Reactions have positive Ea in the range 0-200 kJ/mol
# You need to adapt the bounds depending on the problem!
# e.g. for equilibrium constants where delta_H is in the range -200-200 kJ/mol (exothermic vs endothermic)
guess = np.array([2, 2, 5, 5])
bounds = ([-20, -20, 0, 0], [20, 20, 20, 20])

# Solve optimization using least_squares
# Use the minimal tolerance 1e-15
results = optimize.least_squares(my_model.residuals, guess, bounds=bounds, xtol=1e-15, ftol=1e-15, gtol=1e-15, x_scale='jac')

# Std error of estimated parameters
# cov_matrix_x = sigma**2 = sum(residuals**2) / (n - p) with n data points and p parameters
# cov_matrix_y = sigma**2 * (J.T @ J)**-1 with J the Jacobian matrix
cov = np.linalg.pinv(results.jac.T @ results.jac) * np.sum(results.fun**2) / (results.fun.size - results.x.size)
stderr = np.sqrt(np.diag(cov))

# Print results
n_param = int(len(results.x)/2)
print(f'Solver success after {results.nfev} iterations: ', results.success)
print('Log k_ref: ', results.x[:n_param], '+-', stderr[:n_param])
print('Experimental vs fitted k0 ', k0, np.exp(results.x[:n_param]+ results.x[n_param:]*10000/8.314/my_model.T_ref))
print('Experimental vs fitted Ea ', ea, results.x[n_param:]*10000, '+-', stderr[n_param:]*10000)
# cost and errors on estimated parameters are important insights to check for convergence
# good practice: varying a bit the bounds and initial guess to check how the cost and errors evolve
print('Residual cost', results.cost)  
print('Termination', results.status)

# Other possibility: optimize.curve_fit
# popt, pcov, info_dict, _, _ = optimize.curve_fit(function, xdata, ydata, p0=initial_guess, bounds=bounds, full_output=True)
# perr = np.sqrt(np.diag(pcov))  # calculate error on estimated parameters
# print('Residual cost', info_dict.get('fvec').sum())  # cost

# Print time
print('Solver time: ', f'{time.time() - start_t} seconds')

