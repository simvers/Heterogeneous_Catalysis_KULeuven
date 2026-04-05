# Heterogeneous Catalysis KU Leuven
GitHub repository containing exercise data and scripts examples related to the course Heterogeneous Catalysis.

## Structure
"""
├── data/ # Data used for the exercises 
├── example/ # Kinetic parameters estimation example
└── pfr_class.py # PFR class used to simulate the reactor, containing the odeint and ODE functions.
"""

## Example
"""
example/
├── data/ # Data used for the example
├── figures/ # Figures
├── pfr_check.py # Comparison between PFR analytical and odeint solutions.
├── kinetic_models.py # Contains the (simplified) kinetic model which computes reaction rates from kinetic parameters and partial pressures.
└── optimizer.py # Main script, contains the solver to estimate the kinetic parameters.
"""

## Your turn
Read and understand how the PFR class, kinetic models, and optimizer work.
Adapt the optimizer and the kinetic models to solve your exercise.
You should not need to modify the PFR class.

