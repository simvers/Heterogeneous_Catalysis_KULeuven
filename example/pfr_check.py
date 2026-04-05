import sys
import numpy as np
import matplotlib.pyplot as plt
from kinetic_models import simple_kinetic_model

sys.path.append('../')
from pfr_class import Reactor_model


# Sanity check: ODE solution vs analytical solution for simple reaction

if __name__ == "__main__":
        
    # Reactions
    # 1: A -> B
    # 2: A -> C

    # Reaction conditions
    # Inert, A, B, C
    w = 0.5  # in g
    Ftot_0 = np.array([1])  # (n_f,)
    Ptot = 1
    X_0 = np.array([
        [0.4, 0.6, 0.0, 0.0], 
        [0.5, 0.5, 0.0, 0.0], 
    ])  # (n_x, n_species)
    temp = np.array([423, 523])  # (n_temp,)
    n_exp = len(Ftot_0) * len(X_0) * len(temp)  # n_exp = n_f * n_x * n_temp

    # Concatenate input
    w_concat = np.repeat([w], repeats=n_exp)  # (n_exp,)
    ptot_concat = np.repeat([Ptot], repeats=n_exp)  # (n_exp,)
    x_concat = np.tile(np.tile(X_0, (len(temp), 1)), (len(Ftot_0), 1))  # (n_exp, n_species)
    t_concat = np.tile(np.repeat(temp, repeats=len(X_0)), (len(Ftot_0)))  # (n_exp,)
    ftot_concat = np.repeat(Ftot_0, repeats=len(X_0)*len(temp))  # (n_exp,)
    f0_concat = x_concat * ftot_concat[:, None]  # (n_exp, n_species)

    # Kinetic and stoichiometric constants
    v = np.array([[0, 0], [-1, -1], [1, 0], [0, 1]])  # (n_species, n_reac)
    k0 = np.array([1.5e6, 2.0e8])  # (n_reac,)
    ea = np.array([50000, 75000])  # (n_reac,) in J/mol

    # Kinetic constants for every reaction as function of reaction temperature
    exp_term = -np.outer(1/t_concat, ea) / 8.314  # (n_exp, n_reac) = (n_exp,) x (n_reac,)
    k_t = k0[None, :] * np.exp(exp_term)  # (n_exp, n_reac) = (1, n_reac) * (n_exp, n_reac)
    
    # Integration space
    n_eval = 1001
    Z = np.linspace(np.zeros(n_exp), w_concat, n_eval).T  # (n_exp, n_eval)

    # ------------------------------------------------------

    # Analytical solution of PFR
    
    # Determine concentrations at t
    # F of shape (n_exp, reac_coo) = (n_exp, 1) * (n_exp, reac_coo)
    # k[i] of shape (n_exp,)
    F_an = np.empty((n_exp, len(v), n_eval))  # (n_exp, n_species, reac_coo)
    exp_term = (k_t[:, 0] + k_t[:, 1])[:, None] * Z  # (n_exp, reac_coo) = np.outer((n_exp,), (n, exp, reac_coo,))
    F_an[:, 0, :] = f0_concat[:, [0]] * np.ones((n_exp, n_eval))
    F_an[:, 1, :] = f0_concat[:, [1]] * np.exp(- exp_term)
    F_an[:, 2, :] = f0_concat[:, [1]] * (k_t[:, 0] / (k_t[:, 0] + k_t[:, 1]))[:, None] * (1 - np.exp(- exp_term))
    F_an[:, 3, :] = f0_concat[:, [1]] * (k_t[:, 1] / (k_t[:, 0] + k_t[:, 1]))[:, None] * (1 - np.exp(- exp_term))

    # ------------------------------------------------------

    # ODE solution of PFR

    # ODE resolution
    # Not using PFR model to have the full reactor profile, not only outlet
    # Not possible to parallelize odeint with multiple initial conditions
    # F_ode = np.empty((n_exp, len(v), n_eval))  # (n_exp, n_species, reac_coo)
    # for i in range(n_exp):
    #     sol = integrate.odeint(pfr_ode, f0_concat[i, :], Z[i, :], args=(1, k_t[i, :], v, simple_kinetic_model))  # (reac_coo, n_species)
    #     F_ode[i, :, :] = sol.T  # (n_exp, n_species, reac_coo)
    my_pfr = Reactor_model(v, simple_kinetic_model, w_concat, t_concat, ptot_concat, f0_concat)
    F_ode = my_pfr.predict_F_profile(k0, ea)
    
    # ------------------------------------------------------

    # Visualization

    # Plot k values
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.plot(t_concat, k_t[:, 0], '-', color='DarkBlue', label='k1')
    # ax.plot(t_concat, k_t[:, 1], '-', color='DarkRed', label='k2')
    # ax.set(xlim=(temp[0], temp[-1]), ylim=(0, 5), title='Reaction rate vs T', xlabel='Reaction temperature / K', ylabel='k / ')
    # fig.savefig('figures/kinetic_model_kT.png')

    # Plot analytics
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    colormaps = ['Greys', 'Blues', 'Reds', 'Greens']
    for i in range(n_exp):

        # Color index
        color_index = 1 - i/n_exp
            
        # Plot
        ax.plot(Z[i, :], F_an[i, 0, :], '-', color=plt.colormaps[colormaps[0]](color_index))
        ax.plot(Z[i, :], F_an[i, 1, :], '-', color=plt.colormaps[colormaps[1]](color_index))
        ax.plot(Z[i, :], F_an[i, 2, :], '-', color=plt.colormaps[colormaps[2]](color_index))
        ax.plot(Z[i, :], F_an[i, 3, :], '-', color=plt.colormaps[colormaps[3]](color_index))

        label = True if i == n_exp-1 else False
        ax.plot(Z[i, :], F_ode[i, 0, :], '-', color=plt.colormaps[colormaps[0]](color_index), label='Inert' if label else None)
        ax.plot(Z[i, :], F_ode[i, 1, :], '-', color=plt.colormaps[colormaps[1]](color_index), label='A' if label else None)
        ax.plot(Z[i, :], F_ode[i, 2, :], '-', color=plt.colormaps[colormaps[2]](color_index), label='B' if label else None)
        ax.plot(Z[i, :], F_ode[i, 3, :], '-', color=plt.colormaps[colormaps[3]](color_index), label='C' if label else None)

    ax.set(xlim=(0, w), ylim=(0, 1.0), title='Analytical vs ODE solution', xlabel='Reactor bed', ylabel='Molar flowrate')

    fig.savefig('figures/pfr_analytic_ode_supp.png', dpi=300)


    # Plot structure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = np.ravel(ax)

    for i in range(n_exp):

        # Color index
        color_index = 1 - i/n_exp

        # Plot analytical solution
        ax[0].plot(Z[i, :], F_an[i, 0, :], '-', color=plt.colormaps[colormaps[0]](color_index))
        ax[0].plot(Z[i, :], F_an[i, 1, :], '-', color=plt.colormaps[colormaps[1]](color_index))
        ax[0].plot(Z[i, :], F_an[i, 2, :], '-', color=plt.colormaps[colormaps[2]](color_index))
        ax[0].plot(Z[i, :], F_an[i, 3, :], '-', color=plt.colormaps[colormaps[3]](color_index))

        # Plot ode solution
        label = True if i == n_exp-1 else False
        ax[1].plot(Z[i, :], F_ode[i, 0, :], '-', color=plt.colormaps[colormaps[0]](color_index), label='Inert' if label else None)
        ax[1].plot(Z[i, :], F_ode[i, 1, :], '-', color=plt.colormaps[colormaps[1]](color_index), label='A' if label else None)
        ax[1].plot(Z[i, :], F_ode[i, 2, :], '-', color=plt.colormaps[colormaps[2]](color_index), label='B' if label else None)
        ax[1].plot(Z[i, :], F_ode[i, 3, :], '-', color=plt.colormaps[colormaps[3]](color_index), label='C' if label else None)

    ax[0].set(xlim=(0, w), ylim=(0, 1), title='Analytical solution', xlabel='Reactor bed', ylabel='Molar flowrate')
    ax[1].set(xlim=(0, w), ylim=(0, 1), title='ODE solution', xlabel='Reactor bed', ylabel='Molar flowrate')
    ax[1].legend(loc='upper right')

    fig.savefig('figures/pfr_analytic_ode_comp.png', dpi=300)
