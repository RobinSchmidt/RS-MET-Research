from __future__ import print_function, division
from fractions import Fraction
from rsTools import rsCheck, printTestResult
from rsIntegerFunctions import toFractions

# see here for the future imports:
# https://docs.python.org/2/library/__future__.html

#==============================================================================
# The actual algorithms, operating on lists (maybe turn them into static
# functions of the class):

def polyEval(p, x):
	"""Evaluates the polynomial p a the given x using Horner's algorithm"""
	k = len(p)-1       # last valid index
	if(k < 0):
		return 0
	y = p[k]
	while(k > 0):
		k -= 1
		y = y*x + p[k]
	return y

def polyTrunc(p):
	""""Truncates trailing zeros of the list p - maybe rename to 
	rsTruncateTrailingZeros"""
	i = len(p)-1
	#while i > 0:  old - buggy!
  while i > 1:
		if p[i] != 0:
			break
		i -= 1
	return p[0:i+1]

def polyAdd(p, q, wp=1, wq=1):
	"""Forms a weighted sum of the two coefficient lists p and q with weights 
	wp and wq respectively. If the resulting list will have trailing zeros, 
	these will be truncated."""
	np = len(p)
	nq = len(q)
	if(np >= nq):
		r = wp*p
		for i in range(nq):
			r[i] += wq*q[i]
		return polyTrunc(r)
	else:
		r = wq*q
		for i in range(np):
			r[i] += wp*p[i]
		return polyTrunc(r)
# maybe rename to polyWeightedSum, implement polyAdd similar to polySub

def polySub(p, q):
	"""Subtracts the coefficient list q from the coefficient list p. If the 
	result has trailing zeros, these will be truncated."""
	return polyAdd(p, q, 1, -1)

def polyMul(x, h):
	""""Multiplies two lists of polynomial coefficients by convolution."""
	L = len(x)+len(h)-1   # length of result
	y = []
	for n in range(0, L):
		yn = 0
		for k in range(max(0, n-len(x)+1), min(n+1,len(h))):
			yn += h[k] * x[n-k]
		y.append(yn)    # why? wouldn't it be better to preallocate the list y?
	return polyTrunc(y) # trailing zeros may occur in the case when one of the
                        # inputs is the zero polynomial

def polyDivMod(p, d):
	""""Divides polynomial p (product) by polynomial d (divisor) and returns 
	the quotient and remainder (in that order) as tuple."""
	r = p[:]         # init remainder with copy of product
	q = [0]*len(p)   # init quotient to all zeros
	k = len(p)-len(d)
	while k >= 0:
		q[k] = r[len(d)-1+k] / d[len(d)-1]
		j = len(d)+k-2
		while j >= k:
			r[j] -= q[k] * d[j-k]
			j -= 1
		k -= 1
	for i in range(len(d)-1, len(p)):
		r[i] = 0
	return polyTrunc(q), polyTrunc(r)

def polyDiv(p, d):
	""""Quotient of polynomial division - this corresponds to the integer part 
	of the division of natural numbers."""
	q, r = polyDivMod(p, d) # r is just a dummy
	return q

def polyMod(p, d):
	""""Remainder of polynomial division"""
	q, r = polyDivMod(p, d) # q is just a dummy
	return r

def rsPolyLess(p, q):
	"""Less-than comparison for lists of polynomial coefficients. A polynomial 
	p is considered to be less than another polynomial q, if p has smaller 
	degree than q. If the degrees are equal, the comparison will be based on 
	the leading coefficient, i.e. the coefficient for the highest power of 
	x)"""
	lp, lq = len(p), len(q)
	if lp == lq:
		return p[lp-1] < q[lq-1]
	else:
		return lp < lq

def isZeroPoly(p):
	"""Checks, if polynomial p is the zero polynomial."""
	if len(p) > 1:
		return False
	else:
		return p[0] == 0

def makeMonic(p):
	""""Makes the polynomial a monic, i.e. divides all coefficients by the 
	leading coefficient to make the leading coefficient 1. Will result in
	division by zero error, if p is the zero polynomial. It works in place and 
	will return the leading coefficient (which may or may not be of interest to 
	the caller)"""
	lc = p[len(p)-1]
	for k in range(0, len(p)):
		p[k] /= lc
	return lc

def polyGcd(p, q, monic=True):
	"""Computes the greatest common divisor of polynomials p and q which is 
	defined as the polynomial of highest degree that divides both p and q. Such
	a polynomial is unique only up to multiplication by a constant, so it is 
	often additionally required to be a monic polynomial to make it unique. 
	This normalization can be controlled by by the monic parameter."""
	a, b = p[:], q[:]
	while not isZeroPoly(b):
		t = b 
		b = polyMod(a, b)
		a = t
	if(monic):
		makeMonic(a)
	return a
# maybe rename to polyGcd
# https://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor
# ToDo: figure out the meaning of the leading coeff when it's not forced to 1

def polyNest(a, b):
	"""Given the coefficient lists of two polynomials a(x) and b(x), this 
	function computes the coefficient list of the polynomial c(x) that results
	from nesting a(x) and b(x) where a(x) is the inner and b(x) the outer 
	polynomial such that: c(x) = b(a(x))"""
	aN = len(a)-1        # degree of a
	bN = len(b)-1        # degree of b
	cN = aN*bN           # degree of result
	an = [1]             # coeffs of successive powers of a
	c  = [0]*(cN+1)      # coeffs for result
	c[0] = b[0]
	K = 1
	for n in range(1, bN+1):
		an = polyMul(an, a)
		K += aN
		for k in range(0, K):
			c[k] += b[n] * an[k]
	return c

# maybe implement LCM, too - not really important but nice to have for 
# completeness
# implement polyInt, polyDiff
# https://en.wikipedia.org/wiki/Polynomial_long_division
# https://de.wikipedia.org/wiki/Polynomdivision

# maybe when declaring varaibles in sage via
# var("a", 5) we directly get a list of a-variables usable
# as coefficients for a 4th degree polynomial

#==============================================================================
# The class for convenience:

class rsPolynomial():
	""""
	A class for representing polynomials via their coefficient lists. It's 
	useful for doing all the typical operations on coefficient arrays. There 
	are several places where functions rely on exact comparison of 
	coefficients, so this class is supposed to work right only for exactly 
	representable coefficient types such as rational numbers (class Fraction) 
	or symbolic expressions (class Symbol from sympy). It *may* work in some 
	cases with floats, but this will then be unreliable and really depend on 
	whether two particular floats come out exactly equal in a numeric 
	computation and not only equal up to roundoff error. Sooo...better just 
	don't use it with floats.
	ToDo: maybe make it work with floats by introducing an error-tolerance
	"""

	def __init__(self, coeffs=None, intsToFractions=True):
		if type(coeffs) is type(None):
			self.setCoeffs([0], intsToFractions)
		elif type(coeffs) is list:
			self.setCoeffs(coeffs, intsToFractions)
		elif type(coeffs) is rsPolynomial:
			self.setCoeffs(coeffs.coeffs(), intsToFractions)
	# allow tuple like (a, 5) be passed which automatically creates the list
	# of symbolic variable a0..a5
	# it seems, the list of coefficients is not being copied when a 
	# polynomial is assigned to another polynomial
	# perhaps we should avoid automatic conversion to float by always 
	# casting integers to fractions
	# btw: a python fraction is always in reduced form:
	# https://www.geeksforgeeks.org/fraction-module-python/

	# setup:

	def setCoeffs(self, coeffs, intsToFractions=True):
		"""Sets the coefficients to the new values given by a list. If you pass
		a list of integers, they wil by default be converted to fractions 
		unless you set intsToFractions to false. The reason is that some 
		operations may otherwise let python automatically convert integers to
		floats which is undesirable, because this class does not work for 
		floats. Fractions are fine, though."""
		if intsToFractions and type(coeffs[0]) is int:
			self.cofs = toFractions(coeffs)
		else:
			self.cofs = list(coeffs) # a[0]*x^0 + a[1]*x^1 + ...

	def setRoots(self, newRoots, intsToFractions=True):
		"""Sets the coefficients of the polynomial according to the passed 
		roots. ToDo: let the user specify a leading coefficient."""
		if intsToFractions and type(newRoots[0]) is int:
			self.cofs = [Fraction(1,1)]
			self.addRoots(toFractions(newRoots))
		else:
			self.cofs = [type(newRoots[0])(1)]
			self.addRoots(newRoots)

	def addRoots(self, newRoots):
		"""Adds new roots into the polynomial such that the set of roots will 
		be augmented by given roots."""
		for k in range(0, len(newRoots)):
			self.augment(newRoots[k])

	def augment(self, newZero):
		"""Multiplies in a new zero crossing at given value"""
		self.cofs = polyMul(self.cofs, [-newZero, 1])
	# rename to addRoot

	# inquiry:

	def eval(self, x):
		"""Evaluates the polynomial at the given x"""
		return polyEval(self.cofs, x)

	def degree(self):
		"""Returns the degree of the polynomial, i.e. the highest power of x."""
		return len(self.cofs)-1

	def coeffs(self):
		"""Returns the coefficients of the polynomial as list in ascending 
		powers of x starting at x^0 ending at x^degree"""
		return self.cofs

	def leadingCoeff(self):
		"""Returns the leading coefficient, i.e. the coefficient for the highst 
		power of x."""
		return cofs[len(cofs)-1]

	def constantTerm(self):
		"""Returns the constant term, i.e. the coefficient fo x^0."""
		return cofs[0]

	def greatestCommonDivisor(self, q):
		""""Returns the greatest common divisor of this polynomial with the 
		passed polynomial q."""
		return rsPolynomial(polyGcd(self.coeffs(), q.coeffs()))

	#def hasEvenSymmetry():
	#	for i in range(1, len(cofs)-1, 2): # odd coeffs must be all zero for even symmetry
	#		if cofs[i] != 0:
	#			return False
	#	return True

	#def hasOddSymmetry():
	#	for i in range(0, len(cofs), 2):   # even coeffs must be all zero for odd symmetry
	#		if cofs[i] != 0:
	#			return False
	#	return True

	# operators:

	def __neg__(self):
		return rsPolynomial(polySub([0], self.cofs))

	def __add__(self, other):
		return rsPolynomial(polyAdd(self.cofs, other.cofs))

	def __sub__(self, other):
		return rsPolynomial(polySub(self.cofs, other.cofs))

	def __mul__(self, other):
		return rsPolynomial(polyMul(self.cofs, other.cofs))

	def __truediv__(self, other):
		return rsPolynomial(polyDiv(self.cofs, other.cofs))

	def __div__(self, other):
		return rsPolynomial(polyDiv(self.cofs, other.cofs))

	def __mod__(self, other):
		return rsPolynomial(polyMod(self.cofs, other.cofs))

	def __eq__(self, other):
		return self.coeffs() == other.coeffs()

	def __ne__(self, other):
		return self.coeffs() != other.coeffs()

	def __lt__(self, other):
		return rsPolyLess(self.coeffs(), other.coeffs())

	def __gt__(self, other):
		return rsPolyLess(other.coeffs(), self.coeffs())

	def __len__(self):
		return len(self.cofs)

	def __getitem__(self, index):
		return self.cofs[index]

	def __setitem__(self, index, value):
		self.cofs[index] = value

	def __repr__(self):
		return str(self.coeffs())
	# return "Polynomial of degree " + str(self.degree()) + " with coefficients " + str(self.coeffs())
	# maybe do something more 

	# __rmul__, _iter__, __repr__
	# https://docs.python.org/3/library/operator.html

#ToDo: add static methods
#@staticmethod
#def chebychev(degree):
#	"""Returns a Chebychev polynomial of given degree"""
#  return rsPolynomial([0]) # preliminary
# Bessel, Legendre, ..but probably sage has them already onboard - we'll see
# maybe all these free functions above polyMul, etc. could be moved as static 
# function into this class - we'll see...
# look at this for reference:
# https://docs.sympy.org/latest/modules/polys/index.html

#==============================================================================
# Unit Test:

def testRsPolynomial():

	testResult = True

	# create polynomial with coefficients 1, 2, 3, i.e. 1 + 2*x + 3*x^2:
	p = rsPolynomial([1,2,3])
	rsCheck(p == rsPolynomial([1, 2, 3]), "init failed")

	# create polynomial with roots at 1, 2, 3, 4:
	p = rsPolynomial()
	p.setRoots([1,2,3,4])
	testResult = testResult and\
	rsCheck(p == rsPolynomial([24, -50, 35, -10, 1]), "setRoots failed")

	# create polynomial with roots at -1, -2, -3, -4:
	q = rsPolynomial()
	q.setRoots([-1,-2,-3,-4])
	testResult = testResult and\
	rsCheck(q == rsPolynomial([24, 50, 35, 10, 1]), "setRoots failed")

	# multiply both polynomials together:
	r = p * q
	s = q * p
	testResult = testResult and r==s and\
	rsCheck(r == rsPolynomial([576, 0, -820, 0, 273, 0, -30, 0, 1]),\
		 "multiplication failed")

	# re-assign p and q and check addition and multiplication:
	p.setCoeffs([3, 7, 2, 4])
	q.setCoeffs([2, 1, 6])
	r = p + q
	testResult = testResult and\
	rsCheck(r == rsPolynomial([5, 8, 8, 4]), "addition failed")
	r = p - q
	testResult = testResult and\
	rsCheck(r == rsPolynomial([1, 6, -4, 4]), "subtraction failed")

	# check division and remainder:
	r = p * q
	p2 = r / q  # p2 contains floats now
	q2 = r / p  # q2 is wrong!!!
	testResult = testResult and	rsCheck(p == p2, "division failed")
	testResult = testResult and	rsCheck(q == q2, "division failed")
	rem = r%p  # remainder should be zero
	rsCheck(rem == rsPolynomial([0]), "modulo failed")
	rem = r%q  # this one, too
	rsCheck(rem == rsPolynomial([0]), "modulo failed")

	# try a product that has a nonzero remainder:
	r = rsPolynomial([4, 2])
	s = p * q + r
	p2 = s / q
	r2 = s % q
	testResult = testResult and	rsCheck(p == p2, "division failed")
	testResult = testResult and	rsCheck(r == r2, "modulo failed")
	q2 = s / p
	r2 = s % p
	testResult = testResult and	rsCheck(q == q2, "division failed")
	testResult = testResult and	rsCheck(r == r2, "modulo failed")

	# Test greatest common divisor by forming the two products:
	# s = p*r, t = q*r and then finding the gcd of s and t. This should give 
	# back r (if the sets of roots of p and q are disjoint, which they are):
	p.setRoots([ 1,  2])
	q.setRoots([-1, -2])
	r.setRoots([ 3, -3])
	s = p * r
	t = q * r
	g = s.greatestCommonDivisor(t)
	testResult = testResult and rsCheck(g == r, "gcd failed")
	# should be r ...comes out as 2*r - close, but why the factor 2?
	# but maybe that's ok, since here
	# https://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor
	# it says that the gcd is defined only up to a multiplicative constant

	## example from https://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor#Euclidean_algorithm
	#p = rsPolynomial([ 6,  7, 1])		# p(x) = x^2 + 7*x + 6 =  (x^2 - 5*x - 6) + 12*(x+1)
	#q = rsPolynomial([-6, -5, 1])		# q(x) = x^2 - 5*x - 6 =  (x+1) * (x-6)
	#g = rsGreatestCommonDivisor(p, q)

	# Test gcd with coeffs of type Fraction (this case would fail with floats 
	# due to numerical errors).
	f = [Fraction(k, 1) for k in range(1, 9)]
	p.setRoots([f[0], f[1]])
	q.setRoots([f[2], f[3], f[4]])
	r.setRoots([f[5], f[6]])
	s = p * r
	t = q * r
	g = s.greatestCommonDivisor(t)
	testResult = testResult and	rsCheck(g == r, "gcd failed")

	# test nesting:
	p.setCoeffs([1, 2,-3, 1])
	q.setCoeffs([2,-1, 2])
	r = rsPolynomial(polyNest(p.coeffs(), q.coeffs()))
	s = rsPolynomial(polyNest(q.coeffs(), p.coeffs()))
	rsCheck(r == rsPolynomial([3,  6, -1, -21, 26, -12, 2]), "nesting failed")
	rsCheck(s == rsPolynomial([1, -2,  7, -13, 18, -12, 8]), "nesting failed")

	printTestResult(testResult, "rsPolynomial")
	return testResult

if __name__ == "__main__":
	result = testRsPolynomial()
