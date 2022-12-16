"""
Thermodynamics equations from article:
    Title:   Transcritical diffuse-interface hydrodynamics of propellants in 
             high-pressure combustors of chemical propulsion systems 
    Authors: Lluís Jofre, Javier Urzay
or article:
    Title: Microconfined high-pressure transcritical fluids turbulence
    Authors: Marc Bernades, Francesco Capuano, Lluis Jofre
"""

import sys
import numpy as np

from scipy.optimize import newton

from thermodynamics.func_PengRobinson import PengRobinson

Ru = 8.314    # R universal
newton_maxiter = 1000
newton_tol     = 1e-8

def GasEquation_find_P(T, rho, bSolver, Substance):

    # PengRobinson
    a, b, R, _, _, _ = PengRobinson(T, Substance)

    # Calculate P from Gas Equation P = P(rho, T)
    v = 1/rho
    if bSolver == "Real":
        P = R * T / (v - b) - a / (v**2 + 2*b*v - b**2)
    elif bSolver == "Ideal": 
        P = R * T / v
    else:
        sys.exit(f"ErrorValue in bSolver = {bSolver}; admissible values of bSolver: 'Real', 'Ideal'")

    return P


def GasEquation_find_rho(T, P, bSolver, Substance, rho_initial_guess):

    func_GasEquationError = lambda rho: GasEquationError(T, P, rho, bSolver, Substance)

    # Solve gas equation to find rho
    rho, r = newton(func = func_GasEquationError, x0 = rho_initial_guess, tol = newton_tol, maxiter = newton_maxiter, full_output=True)

    if r.converged:
        return rho
    else:
        sys.exit("Newton solver to find 'rho' did not converge")


def GasEquation_find_T(P, rho, bSolver, Substance, T_initial_guess):

    func_GasEquationError = lambda T: GasEquationError(T, P, rho, bSolver, Substance)

    # Solve gas equation to find T
    T, r = newton(func = func_GasEquationError, x0 = T_initial_guess, tol = newton_tol, maxiter = newton_maxiter, full_output=True)

    if r.converged:
        return T
    else:
        sys.exit("Newton solver to find 'T' did not converge")


def GasEquationError(T, P, rho, bSolver, Substance):

    P_equation = GasEquation_find_P(T, rho, bSolver, Substance)
    error      = P_equation - P 

    return error

def GasEquationRelativeError(T, P, rho, bSolver, Substance):

    P_equation     = GasEquation_find_P(T, rho, bSolver, Substance)
    relative_error = abs(P_equation - P)/P_equation 

    return relative_error