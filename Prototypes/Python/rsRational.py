from __future__ import print_function, division
from rsTools import rsCheck, printTestResult, scaleList
from rsPolynomial import rsPolynomial, polyMul, polyDiv, polyAdd, polyGcd
from fractions import Fraction
from rsIntegerFunctions import gcd, lcm, gcdList, lcmList, toFractions

#==============================================================================
# The actual algorithms, operating on lists (maybe turn them into static
# functions of the class):

def ratReduce(p, q):
	""""Reduces rational function p/q to the lowest possible denominator."""
	gcd = polyGcd(p, q)
	return polyDiv(p, gcd), polyDiv(q, gcd)

def ratMul(p, q, r, s, reduced=True):
	""""Multiplies two rational functions represented as lists of coefficients 
	for	numerator and denominator. Computes n/d = p/q * r/s and returns them as 
	tuple. By default, it will reduce the result to the lowest possible 
	denominator but you can turn that off via the reduced parameter."""
	if reduced:
		return ratReduce(polyMul(p, r), polyMul(q, s))
	else:
		return polyMul(p, r), polyMul(q, s)
# maybe a better algorithm can be found that first finds the lowest common 
# denominator and thereby avoids blowing up the inetrmediate result (before
# reducing) unnecssarrily
# consider 9/2 * 8/3 = 12 - this has 1 as reduced denominator even though
# the two denominators have no common factors - similar thigs are supposed
# to happen with rational functions

def ratDiv(p, q, r, s, reduced=True):
	return ratMul(p, q, s, r, reduced)  # r and s are swapped

def ratAdd(n1, d1, n2, d2, w1=1, w2=1):
	"""Adds two rational functions represented as lists of coefficients for 
	numerator and denominator with optional weighting. It computes numerator 
	and denominator of N/D = w1*n1/d1 + w2*n2/d2 and returns them as tuple."""
	gcd = polyGcd(d1, d2)
	f1 = polyDiv(d2, gcd)
	f2 = polyDiv(d1, gcd)
	dr = polyMul(f1, d1)
	s1 = polyMul(f1, n1)         # 1st summand in numerator of result
	s2 = polyMul(f2, n2)         # 2nd summand
	nr = polyAdd(s1, s2, w1, w2) # numerator of result
	return nr, dr

def addFractions(n1, d1, n2, d2, w1=1, w2=1):
	"""Adds two rational numbers a.k.a. fractions with optional weighting. 
	Computes n/d = w1*n1/d1 + w2*n2/d2. 
	For example: n, d = addFractions(3, 8, 5, 12) 
	produces the output tuple (19, 24). That stands for: 3/8 + 5/12 = 19/24
	The function is mainly meant as prototype for the ratAdd function above."""
	g  = gcd(d1, d2)   # greatest common divisor of both denominators
	f1 = d2 / g        # factor to multiply n1,d1 with to obtain lowest common denominator
	f2 = d1 / g        # factor to multiply n2,d2 with
	dr = f1 * d1       # denominator of result, equals also f2 * d2
	nr = w1*f1*n1 + w2*f2*n2 # numerator of result
	return nr, dr

def ratPolyNest(ni, di, po):  # ni,di: inner num/den, no outer polynomial
	"""Computes lists of numerator and denominator coefficients that result 
	from from nesting an inner rational function with an outer polynomial."""
	nr = [po[0]]   # numerator of result
	dr = [1]       # denominator of result
	nt = ni        # temporary numerator (for convolutive accumulation)
	for k in range(1, len(po)):
		dr = polyMul(dr, di)
		nr = polyMul(nr, di)
		nr = polyAdd(nr, nt, 1, po[k])
		nt = polyMul(nt, ni)
	return nr, dr

def ratNest(ni, di, no, do): # ni,di: inner num/den, no,do outer num/den
	nu,	du = ratPolyNest(ni, di, no)  # upper num/den
	nl, dl = ratPolyNest(ni, di, do)  # lower num/den
	return ratDiv(nu, du, nl, dl);
# It's not optimal to call ratPolyNest two times - inside this function, 
# there are values that a calculated just the same in both calls, namely
# the successive powers of ni - but this is not meant to be optimized
# high performance code. Maybe in a port to C++, this optimization should
# be done - but maybe it's not worth the hassle.

# ToDo: partial fraction expansion, derivative, integral

#==============================================================================
# The class for convenience:

class rsRationalFunction():
	""""
	A class for reprenting rational functions, i.e. a ratio of two 
	polynomials. It's implemented using two instances of class rsPolynomial
	"""

	def __init__(self, numerator, denominator, intsToFractions=True):
		self.num = rsPolynomial(numerator, intsToFractions)
		self.den = rsPolynomial(denominator, intsToFractions)
		self.autoReduce = True

	# setup:

	def setCoeffs(self, newNumeratorCoeffs, newDenominatorCoeffs):
		self.num.setCoeffs(newNumeratorCoeffs)
		self.den.setCoeffs(newDenominatorCoeffs)

	def setFractionCoeffs(self, newNumeratorCoeffs, newDenominatorCoeffs):
		num = [Fraction(newNumeratorCoeffs[k])\
		 for k in range(0, len(newNumeratorCoeffs))]
		den = [Fraction(newDenominatorCoeffs[k])\
		 for k in range(0, len(newDenominatorCoeffs))]
		self.setCoeffs(num, den)

	#def reduce(self):
	#	gcd = self.num.greatestCommonDivisor(den)
	#	num = num / gcd
	#	den = den / gcd
	# it's actually always reduced on the fly

	def scaleCoeffs(self, s):
		""""Scales all coefficients by the same factor s. Because this scaling
	    is applied to numerator and denominator coefficients, it doesn't change
		the function."""
		scaleList(self.num.coeffs(), s)
		scaleList(self.den.coeffs(), s)

	def makeCoeffsSmallIntegers(self):
		"""Scales all coefficients by a factor that ensures that all 
		coefficients are integers and as small as possible.
		...This function needs more tests...
		"""

		# find lowest common multiple of all coefficient denominators:
		nn, nd, dn, dd = self.coeffLists()
		ln = lcmList(nd)   # lcm of all numerator coeff-denominators
		ld = lcmList(dd)   # lcm of all denominator coeff-denominators
		l  = lcm(ln, ld)   # lcm of all coeff-denominators

		# find greatest common divisor of all coefficient numerators:
		gn = gcdList(nn)   # gcd of all numerator coeff-numerators
		gd = gcdList(dn)   # gcd of all denominator coeff-numerators
		g  = gcd(gn, gd)   # gcd of all coeff-numerators

		# multiply all coefficients by l/g
		r = Fraction(l, g)
		self.scaleCoeffs(r)
		return r # applied scale factor may be of interest to caller

	#def setNumeratorCoeffs(self, newCoeffs):
	#	self.num.setCoeffs(newCoeffs)
	# setNumeratorCoeffs, setDenominatorCoeffs, setPoles, setZeros, setScaler

	# inquiry:

	def numeratorPolynomial(self):
		return self.num

	def numeratorCoeffs(self):
		return self.num.coeffs()

	def denominatorPolynomial(self):
		return self.den

	def denominatorCoeffs(self):
		return self.den.coeffs()

	def numeratorDegree(self):
		return self.num.degree()

	def denominatorDegree(self):
		return self.den.degree()

	def coeffLists(self):
		nn = [self.num[k].numerator   for k in range(0, len(self.num))]
		nd = [self.num[k].denominator for k in range(0, len(self.num))]
		dn = [self.den[k].numerator   for k in range(0, len(self.den))]
		dd = [self.den[k].denominator for k in range(0, len(self.den))]
		return nn, nd, dn, dd

	# misc:

	def composeWith(self, inner):
		"""Composes (i.e. nests, chains) the rational function represented by 
		self with another rational function which yields another rational 
		function. The rational function passed as parameter becomes the inner 
		function and self becomes the outer function. So, if self is f(x) and 
		inner is g(x), this method returns the rational function object that 
		represents h(x) = f(g(x)"""
		num, den = ratNest(inner.num.coeffs(), inner.den.coeffs(),\
					 self.num.coeffs(), self.den.coeffs())
		return rsRationalFunction(num, den)
	# behavior is consistent with sage's compose:
	# http://doc.sagemath.org/html/en/reference/misc/sage/misc/misc.html#sage.misc.misc.compose

	# operators:

	def __add__(self, other):
		return ratAdd(self.num.coeffs(), self.den.coeffs(),\
				other.num.coeffs(), other.den.coeffs())

	def __sub__(self, other):
		return ratAdd(self.num.coeffs(), self.den.coeffs(),\
				other.num.coeffs(), other.den.coeffs(), 1, -1)

	def __mul__(self, other):
		return ratMul(self.num.coeffs(), self.den.coeffs(),\
				other.num.coeffs(), other.den.coeffs(), self.autoReduce)
		#num = self.num * other.num
		#den = self.den * other.den
		#return rsRationalFunction(num, den)

	def __truediv__(self, other):
		return ratDiv(self.num.coeffs(), self.den.coeffs(),\
				other.num.coeffs(), other.den.coeffs(), self.autoReduce)
		#num = self.num * other.den
		#den = self.den * other.num
		#return rsRationalFunction(num, den)

	def __eq__(self, other):
		return self.num.coeffs() == other[0] and self.den.coeffs() == other[1]
		#return self.num == other.num and self.den == other.den
		# for some reason, this obvious commented version of the code doesn't 
		# work because for some weird reason passes "other" as tuple and not as 
		# rsPolynomial object - that is really weird!
		#print(self.num)
		#print(other.num) # other seems to get passed as tuple - wtf?

	def __ne__(self, other):
		return not self == other

#==============================================================================
# Unit Test:

def testRsRationalFunction():

	testResult = True

	r = rsRationalFunction([1,2,3],[4,5,6,7])
	s = rsRationalFunction([5,6],[5,7,11])

	r.setFractionCoeffs([1,2,3],[4,5,6,7])
	s.setFractionCoeffs([5,6],[5,7,11])

	# test arithmetic operators:
	t = r * s
	target = rsRationalFunction([5,16,27,18],[20,53,109,132,115,77])
	testResult = testResult and	rsCheck(t == target, "multiplication failed")
	u = r / s
	target = rsRationalFunction([5,17,40,43,33],[20,49,60,71,42])
	testResult = testResult and	rsCheck(u == target, "division failed")
	v = r + s
	target = rsRationalFunction([25,66,100,114,75],[20,53,109,132,115,77])
	testResult = testResult and	rsCheck(v == target, "addition failed")
	w = r - s
	target = rsRationalFunction([-15,-32,-20,-28,-9],[20,53,109,132,115,77])
	testResult = testResult and	rsCheck(w == target, "subtraction failed")

	# test nesting of a polynomial as outer function with a rational function
	# as inner function:
	ni, di = [2,3], [5,-2]        # inner numerator and denominator
	no, do = [2,-3,4,-5], [2, 3]  # outer numerator and denominator
	nu, du = ratPolyNest(ni, di, no)
	target = [140, -377, 90, -259]
	testResult = testResult and	rsCheck(nu == target, "ratPolyNest failed")
	target = [125, -150, 60, -8]
	testResult = testResult and	rsCheck(du == target, "ratPolyNest failed")
	nl, dl = ratPolyNest(ni, di, do)
	target = [16, 5]
	testResult = testResult and	rsCheck(nl == target, "ratPolyNest failed")
	target = [5, -2]
	testResult = testResult and	rsCheck(dl == target, "ratPolyNest failed")

	# test nesting of rational functions as outer function with another 
	# rational function as inner function:
	nr, dr = ratNest(ni, di, no, do) # results in (wrong) floating point numbers
	nr, dr = ratDiv(toFractions(nu), toFractions(du),\
				 toFractions(nl), toFractions(dl))
	# This result seems to be consistent with the result from sage - but the coeff
	# lists are not unique - maybe we should divide all coeffs by the gcd of 
	# all coeffs to make it unique
	# https://stackoverflow.com/questions/16628088/euclidean-algorithm-gcd-with-multiple-numbers
	# https://www.geeksforgeeks.org/gcd-two-array-numbers/

	# OK, due to the fact that python may automatically convert to floating 
	# point numbers during calculations, we need to instantiate objects of 
	# class rsRationalFunction which make sure that it automatically converts 
	# the given lists of integers to fractions:
	f = rsRationalFunction([2,-3,4,-5], [2, 3])  # outer function f(x)
	g = rsRationalFunction([2,3], [5,-2])        # inner function g(x)
	h = f.composeWith(g)
	#h.normalize() # there may be different ways of normalization, for example,
	# in a digital filter, the a0 coeff should be 1, we may want to have a 
	# monic numerator or denominator or we may want to have the smallest 
	# possible integers for all coeffs
	h.makeCoeffsSmallIntegers()

	printTestResult(testResult, "rsRationalFunction")
	return testResult

if __name__ == "__main__":
	testRsRationalFunction()

# visual studio tabs vs spaces bug:
#https://github.com/Microsoft/TypeScript/issues/4286