import numpy as np


# Simple kinetic model
def simple_kinetic_model(k_coef, pp):

    # Input:
    # k_coef of shape (n_k,)
    # pp of shape (n_species,)

    # Compute reaction rates
    r1 = k_coef[0] * pp[1]  # k_1 * P_A
    r2 = k_coef[1] * pp[1]  # k_2 * P_A

    # return reaction rates
    return np.array([r1, r2])  # (n_reac,)

