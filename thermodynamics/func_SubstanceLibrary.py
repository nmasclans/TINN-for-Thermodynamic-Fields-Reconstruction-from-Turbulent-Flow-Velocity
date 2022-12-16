import sys

def SubstanceLibrary(Substance):

    if Substance == 'N2':
        MW     = 2.80134e-2;      # Molecular weight kg/mol
        Tc     = 126.19;          # Critical temperature [k]
        pc     = 3.3958e+6;       # Critical pressure [Pa]
        p_inf  = 4.0e+6;          # Pressure infinity (liquid stiffness) [Pa]
        rhoc   = 313.3;           # Critical density [kg/m3]
        vc_bar = 8.9412E-5;       # Critical molar volume [m^3/mol]
        omega  = 0.03720;         # Acentric factor
        gamma  = 1.4;             # Heat capacity ratio (ideal-gas) [-]
        e_0    = 0.222;           # Internal energy zero point [J/kg]
        c_v    = 743.0;           # Specific isochoric heat capacity [J/(kg·K)]
        NASA_coefficients =  [ 2.952576370000000000000,
                               0.001396900400000000000,
                               -0.000000492631603000000,
                               0.000000000078601019000,
                               -0.000000000000004607552,
                               -923.9486880000000000000,
                               5.871887620000000000000,
                               3.531005280000000000000,
                               -0.000123660980000000000,
                               -0.000000502999433000000,
                               0.000000002435306120000,
                               -0.000000000001408812400,
                               -1046.976280000000000000,
                               2.967470380000000000000,
                               0.000000000000000000000] # NASA 7-coefficient polynomial (15 values)

        mu_0 = 0.00001663		  # Reference dynamic viscosity [Pa·s]
        kappa_0 = 0.0242		  # Reference thermal conductivity [W/(m·K)]
        T_0 = 273.0		     	  # Reference temperature [K]
        S_mu = 107.0			  # Sutherland's dynamic viscosity constant [K]
        S_kappa = 150.0		      # Sutherland's thermal conductivity constant [K]
        dipole_moment = 0.0       # Dipole moment [D]
        association_factor = 0.0  # Association factor [-]
    
    else:
        sys.exit("Not implemented Substance. Set Substance to 'N2'")

    return MW, Tc, pc, p_inf, rhoc, vc_bar, omega, gamma, e_0, c_v, NASA_coefficients, \
           mu_0, kappa_0, T_0, S_mu, S_kappa, dipole_moment, association_factor