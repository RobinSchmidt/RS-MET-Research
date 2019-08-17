"""
Infinite Impulse Response Filter Design

@author: Robin Schmidt, www.rs-met.com
"""

from math import pi, sin, cos, exp


def dampedSinCoeffs(fn, dn, a=1, phi=0):
    """
    Coefficients for damped sinusoid filter.

    Computes coefficients for a filter that has as impulse response the damped
    sinusoid given by:

    h[n] = a * exp(-n/dn) * sin(w*n + phi)

    where w = pi*fn is the normalized radian frequency and dn is the normalized
    decay time.

    Parameters
    ----------
    fn : number
      normalized frequency from 0 to 1, where 1 corresponds the Nyquist 
	  frequency fs, so we have fn = 2*f/fs
    dn : number
      normalized decay time: number of samples for the amplitude to fall to
      a/e = 0.3678*a (e is the Euler number here)
    a : number
      amplitude as raw multiplier
    phi : number
      initial phase in radians

    Returns
    -------
    b : list of floats
      feedforward coefficients
    a : list of floats
      feedback coefficients

    Example
    -------
    Compute coefficients for a filter that produces a damped sinusoid with
    normalized frequency 0.1 (period = 2/0.1 = 20 samples), normalized decay
    time of 200 samples
    >>> b, a = dampedSinCoeffs(0.1, 200, 0.5, pi/4)
    """
    w = pi * fn              # normalized radian frequency
    P = exp(-1/dn)           # pole radius
    b0 =  a * sin(phi)
    b1 =  a * P * sin(w-phi)
    a1 = -2 * P * cos(w)
    a2 =  P * P
    return ([b0, b1], [1, a1, a2])


# maybe include an optional parameter that determines the value to fall to 
# instead of fixing it to 1/e - maybe to let it fall to 1/2, we would have
# to use 2**(-1/dn) instead of e**(-1/dn) or in general 
# (1/g0)**(-1/dn) == g0**(1/n)
# maybe rename to damped_sinor dampedSineFilter 

# write a couple of simple filter analysis tools, such as
# impResp(b, a, N): impulse response, stepResp(b, a, N), zpk(b, a, n),


def main():
    pass    # just a placeholder at the moment do some demos or unit tests here

# conditionally run the main function, if the module is executed on it own:
if __name__ == "__main__":
   main()



#def filterDF1(b, a, x):
#    N = len(x)
#    Na = len(a)
#    Nb = len(b)
#    y = [0 for n in range(N)] 
#    #xx = [0 for n in range(len(b)-1)]   # past input values
#    #yy = [0 for n in range(len(a)-1)]   # past output values
#    for n in range(N):
#        y[n] = 0
#        for k in range(Nb)
#    return 0





#def protoButter(N, w0, gc=1/sqrt(2), g=1, g0=0)
#    retrun 0

# def protoCheby1, protoCheby2, protoEllip, protoBessel, protoPapoul, 
# protoHalp, protoGauss
# def cookbookCoeffs(f, bAbs=0, bOct=0, bRel=0, g, type, constSkirt=False)
