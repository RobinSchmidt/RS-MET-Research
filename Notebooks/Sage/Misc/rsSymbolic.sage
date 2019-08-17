# Returns a list with all the n nth complex roots of a given number. This 
# function works for symbolic expressions in sage but may not for numeric plain
# python code because of using I instead of j for the imaginary unit (check 
# this)
def nthRoots(z, n):
    """
    Returns a list with all the n nth complex roots of a complex number z or
    a symbolic expression.
    """	
    r = abs(z)^(1/n)  # radius of output
    p = arg(z)/n      # argument/angle of first output (main branch)
    return [r*exp(I*(p + 2*k*pi/n)) for k in [0..n-1]]

# ToDo: provide symbolic filter design functions that return symbolic formulas
# for Butterworth, Chebychev, and elliptic filters. also for the damped sine 
# filter

def butterworthPrototypeZPK(n):
	return nthRoots(-1, 2*n)  # not yet quite correct - we must select left halfplane poles
