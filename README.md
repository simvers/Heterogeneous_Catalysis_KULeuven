# Heterogeneous Catalysis KU Leuven
GitHub repository containing exercise data and scripts examples related to the course Heterogeneous Catalysis.

## Structure
├── data/ contains data used for the exercises.  
├── example/ contains the kinetic parameters estimation example.  
└── pfr_class.py contains the PFR class used to simulate the reactor, with the odeint and ODE functions.  

## Example
example/  
├── data/ contains data used for the example.  
├── figures/ contains figures.  
├── pfr_check.py compares PFR analytical and odeint solutions.  
├── kinetic_models.py contains the (simplified) kinetic model which computes reaction rates from kinetic parameters and partial pressures.  
└── optimizer.py is the optimization script, containing the solver to estimate the kinetic parameters.  

## Your turn
Read and understand how the PFR class, kinetic models, and optimizer work.  
Adapt the optimizer and the kinetic models to solve your exercise.  
You should not need to modify the PFR class.

