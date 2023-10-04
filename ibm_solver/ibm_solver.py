import numpy as np

from scipy.integrate import quad

spectral_density = lambda omega: omega**3

beta = 1

def integrand(omega,t):
    integrand = spectral_density(omega) * (np.cosh(beta * omega / 2)/np.sinh(beta * omega / 2)) * \
            (1-np.cos(omega * t)) / omega**2
    
    return integrand


def integral(t):
    integral = quad(integrand,0,np.inf,args=(t))
    return integral.y


def gamma(t):
    gamma = 0.5 * integral(t)
    return gamma

# gamma(1)