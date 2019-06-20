from fractions import Fraction
from functools import reduce

def toFractions(intList):
	return [Fraction(intList[k], 1) for k in range(0, len(intList))]

def gcd(a, b):
	while b != 0:
		t = b 
		b = a % b
		a = t
	return a

def gcdList(arr):
	"""Greatest common divisor for a list of numbers."""
	return reduce(gcd, arr)
# maybe this works also for tuples and should be named more generally? like
# gcdArray or something?

def lcm(a, b):
	#return a//gcd(a,b) * b
	return abs(a)//gcd(a,b) * abs(b)

def lcmList(arr):
	"""Least common multiple for a list of numbers. check, if this is is 
	actually a correct algorithm"""
	return reduce(lcm, arr)

# for two fractions a/b, c/d, this computes two numbers p,q by which to 
# multiply the numerators and denominators such that we can do
# n=(a/b)/(p/q), m=(c/d)/(p/q) such that n, m are integers that represent
# the same pair of ratios in thes sense that (a/b)/(c/d)=n/m and these
# integers are as small as possible (i.e. contain no common factors)
def reductor(a, b, c, d):
    return gcd(a,c), lcm(b,d) # p,q

def reductorList(a, b, c, d):
	return gcdList(a,c), lcmList(b,d)

#def makeFractionsInteger(a, b, c, d):
#	p, q = reductorList(a, b, c, d)
#	A = [a[k]/
# move these functions into a a file rsFractionTools