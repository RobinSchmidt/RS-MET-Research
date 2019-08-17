"""
A couple of curve fittings.

@author: Robin Schmidt, www.rs-met.com

see:
http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.optimize.curve_fit.html
"""

from matplotlib import pylab as plt
from numpy import linspace, tanh
from scipy.optimize import curve_fit


def rat_odd_3_4(x, a1, a3, b0, b2, b4):
    """
    Rational function of order (3,4) with odd symmetry
    """
    return (a1*x + a3*x**3) / (b0 + b2*x**2 + b4*x**4)
    

def fit_tanh_rat34(x_max = 4, N = 200):
    """
    Rational fit to tanh
    
    Fits (a1*x + a3*x^3) / (b0 + b2*x^2 + b4*x^4) to tanh(x)
    """
    x  = linspace(0, x_max, N)
    y  = tanh(x)
    p, c = curve_fit(rat_odd_3_4, x, y)
    ya = rat_odd_3_4(x, *p)
    #plt.plot(x, y)
    #plt.plot(x, ya)
    return p


if __name__ == "__main__":
    #p = fit_tanh_rat34(4, 195)
    #p = fit_tanh_rat34(4.5, 175)
    #p  = fit_tanh_rat34(5, 165)
    p  = fit_tanh_rat34(7, 165)
    x  = linspace(0, 10, 1000)
    y  = tanh(x)
    ya = rat_odd_3_4(x, *p)
    plt.plot(x, y)
    plt.plot(x, ya)
    
