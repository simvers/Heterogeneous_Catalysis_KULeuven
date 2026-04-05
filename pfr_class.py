import numpy as np
from scipy import integrate


# ------------------------------------------------------

class Reactor_model:

    def __init__(self, v, kinetic_model, w, temp, ptot, F_in, F_out=None, T_ref=273):

        """ 
        This is a reactor simulator.

        Input:
            v: stoichiometric coefficients for each species and each reaction; numpy array with shape (n_species, n_reac)
            kinetic model: function that contains the microkinetic model, it should take k(T) (n_k,) and partial pressures (n_species,) as input and return the reaction rates (n_reac,)
            w: array of catalyst loadings with shape (n_exp,)
            temp: array of temperature with shape (n_exp,)
            ptot: array of pressures with shape (n_exp,)
            F_in: array of molar flowrates at inlet of reactor with shape (n_exp, n_species)
            F_out: array of molar flowrates at outlet of reactor with shape (n_exp, n_species)
            T_ref: reference temperature to compute k0 from k_ref (usually the mean temperature of the dataset)

        """

        # Safety checks
        assert v.shape[0] == F_in.shape[1]  # n_species
        # assert v.shape[1] == kinetic_model(np.ones(v.shape[1]), np.ones(F_in.shape[1])).shape[0]  # n_reac

        # Reaction conditions for n_exp
        self.w = w
        self.temp = temp
        self.ptot = ptot
        self.F_in = F_in
        self.n_exp = len(w)
        self.F_out = F_out
        
        # Kinetic model
        self.v = v
        self.kinetic_model = kinetic_model
        self.T_ref = T_ref  # K

        # Integration space for odeint
        self.n_eval = 1001

        # Product index
        self.reactant_index = (self.v.sum(axis=1) < 0)
        self.product_index = (self.v.sum(axis=1) > 0)


    def no_kinetic_model(self, k_coef, pp):

        """ 
        Kinetic model example with no reaction

        Input:
            k_coef: array of kinetic coefficients, shape (n_k,)
            pp: array of partial pressures, shape (n_species,)

        Output:
            reaction rates, shape (n_reac,)
        """

        # No kinetic model: return null reaction rates
        return np.zeros(k_coef.shape)  # (n_reac,)

    
    def pfr_ode(self, F, w, p_tot, k_coef, v, kinetic_model=no_kinetic_model):

        """ 
        ODE function for odeint solver
        Molar balance at location w in the reactor for a single experiment

        Input:
            F: array of molar flowrates at location w in the reactor, shape (n_species,)
            w: integration variable
            ptot: total pressure, float
            k_coef: array of kinetic coefficients, shape (n_reac,)
            v: stoichiometric coefficients, shape (n_species, n_reac)
            kinetic_model: my_homemade_kinetic_model

        Output:
            dFdw: consumption of F at location w
        """

        # Compute the partial pressure of each reaction species
        # from molar flowrates and total pressure
        pp = F/F.sum(axis=0) * p_tot  # (n_species,)

        # Determine reaction rates from kinetic coefficients and partial pressures
        rates = kinetic_model(k_coef, pp)  # (n_reac,)

        # Safety check
        # if (F > (v @ rates)).all(): # needs to multiply by dw
        #     print("Reaction consuming more reactant than available, no reaction")
        #     rates = np.zeros(rates.shape)

        # Return the consumption or producion of each reaction species 
        # considering reaction stoichiometry
        dFdw = v @ rates  # (n_species,) = (n_species, n_reac) @ (n_reac,)

        # Sanity check
        assert F.shape == dFdw.shape

        return dFdw  # (n_species,)


    def predict_F_profile(self, k0, ea):

        """
        Solve ODE over reactor using the reaction conditions specified at class initialization

        Input:
            k0: preexponential factors, shape (n_k,)
            ea: activation energy/enthalpy of reaction, shape (n_k,)
        
        Output:
            outlet molar flowrates, shape (n_exp, n_species, reac_coo)
        """

        assert len(k0) == len(ea)

        # Integration space
        Z = np.linspace(np.zeros(self.n_exp), self.w, self.n_eval).T  # (n_exp, n_eval)
    
        # Compute kinetic constants from temperature, k0, and Ea or delta_H 
        exp_term = -np.outer(1/self.temp, ea) / 8.314  # (n_exp, n_reac) = (n_exp,) x (n_reac,)
        k_t = k0[None, :] * np.exp(exp_term)  # (n_exp, n_reac) = (1, n_reac) * (n_exp, n_reac)
        
        # Initialize result array
        F_profile_pred = np.empty((self.n_exp, len(self.v), self.n_eval))  # (n_exp, n_species, reac_coo)

        # ODE solver
        # arguments: ode_function dydt, initial values y0, evaluation_coordinates t, args for ode_function
        # returns y array at every t evaluation (t_shape, y_shape)
        for i in range(self.n_exp):
            sol = integrate.odeint(self.pfr_ode, self.F_in[i, :], Z[i, :], args=(self.ptot[i], k_t[i, :], self.v, self.kinetic_model))  # (reac_coo, n_species)
            F_profile_pred[i, :, :] = sol.T  # (n_exp, n_species, reac_coo)
        
        return F_profile_pred
    
    
    def predict_F_out(self, k0, ea):

        # Return outlet molar flow rates of every species
        return self.predict_F_profile(k0, ea)[:, :, -1]  # (n_exp, n_species)

    
    def residuals(self, trans_kinetic_param):

        """
        Compute residuals for least_squares solver

        Input:
            trans_kinetic param: concatenated array of transformed k_ref and Ea values, shape (2*n_k,)
        
        Output:
            array of residuals between true outlet product molar flowrate and the one predicted from the kinetic parameters
        """

        # Number of parameters
        n_param = len(trans_kinetic_param)

        # Energy values: adapt the solver space
        ea_kj_guess = trans_kinetic_param[int(n_param/2):]
        ea_guess = ea_kj_guess * 10000  # 1/10000 space

        # k0 values: adapt the solver space
        log_kref_guess = trans_kinetic_param[:int(n_param/2)]
        log_k0_guess = log_kref_guess + ea_guess/8.314/self.T_ref  # k0 from k_ref
        k0_guess = np.exp(log_k0_guess)  # log scale

        # Return product residuals
        # return self.F_out[:, self.product_index].flatten() - self.predict_F_out(k_guess, ea_guess)[:, self.product_index].flatten()
        return self.F_out[:, -1].flatten() - self.predict_F_out(k0_guess, ea_guess)[:, -1].flatten()

