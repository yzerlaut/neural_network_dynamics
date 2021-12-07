F_iPSP = lambda Q_j, tau_j, tau_m, E_j, mu_V, C_m, a_j, V0: Q_j*tau_j*tau_m*(E_j + V0*a_j - V0 - a_j*mu_V)/C_m
F_iiPSP = lambda Q_j, tau_j, tau_m, E_j, mu_V, C_m, a_j, V0: Q_j**2*tau_j**2*tau_m**2*(E_j + V0*a_j - V0 - a_j*mu_V)**2/(2*C_m**2*(tau_j + tau_m))
F_iiiPSP = lambda Q_j, tau_j, tau_m, E_j, mu_V, C_m, a_j, V0: 2*Q_j**3*tau_j**3*tau_m**3*(E_j + V0*a_j - V0 - a_j*mu_V)**3/(3*C_m**3*(tau_j + 2*tau_m)*(2*tau_j + tau_m))
F_iiiiPSP = lambda Q_j, tau_j, tau_m, E_j, mu_V, C_m, a_j, V0: 3*Q_j**4*tau_j**4*tau_m**4*(E_j + V0*a_j - V0 - a_j*mu_V)**4/(4*C_m**4*(tau_j + tau_m)*(tau_j + 3*tau_m)*(3*tau_j + tau_m))
F_numTv = lambda Q_j, tau_j, tau_m, E_j, mu_V, C_m, a_j, V0: Q_j**2*tau_j**2*tau_m**2*(E_j + V0*a_j - V0 - a_j*mu_V)**2/(2*C_m**2*(tau_j + tau_m))
F_denomTv = lambda Q_j, tau_j, tau_m, E_j, mu_V, C_m, a_j, V0: Q_j**2*tau_j**2*tau_m**2*(E_j + V0*a_j - V0 - a_j*mu_V)**2/C_m**2
