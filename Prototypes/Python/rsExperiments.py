from __future__ import print_function, division
from fractions import Fraction, gcd


#==============================================================================

# maybe rename to TelescopicRationalNumber or short TelRatNum..or maybe 
# LeveledNumber - normal numbers are level-0, zeros are level -1, ...
class rsTelescopicNumber():
	""" A class for representing rational numbers that are augmented by another
	integer that indicates the order of infinity that this number has. A 
	regular, finite	number has order of infinity 0, regular infinity has 
	order 1, infinity-squared has order 2, 0 has order -1, 0*0 has -2, etc. 
	There is no single zero anymore, the unit-zero is just 1 with infinity 
	order -1 (denoted a 1_-1 (the order is written as an index))
	... tbc
	"""

	def __init__(self, num = 0, den = 1, ord = 0):
		""" Initializes numerator, denominator and order with given values (which 
		should be integers). Without any arguments, a 1st order zero is 
		created, i.e. the regular zero that we know already. The constructor takes 
		care that the resulting number will be in canonical representaion, i.e. 
		den > 0 and gcd(num,den) == 1"""
		if num == 0:			# catch 0/a cases (a may or may not be zero)
			num = 1
			if den == 0:
				den = 1			# 0/0 = 1_-1
			else:
				ord = ord - 1	# 0/a = (1/a)_-1

		if den == 0:			# catch a/0 cases (a is nonzero)
			den = 1
			ord = ord + 1		# a/0 = (a/1)_1, (a/0)_n = (a/1)_n+1

		# code duplicated form canonical - refactor:
		g = gcd(num, den)
		num = num // g
		den = den // g
		if den < 0:
			den = -den
			num = -num

		self.num = num            # numerator
		self.den = den            # denominator
		self.ord = ord            # order of infinity
		assert self.isCanonical()

	def __add__(self, other):
		"""Adds two telescopic numbers. The rules are:
		a_n + b_m = a_n        if n > m -> a absorbs b
		a_n + b_m = b_m        if m > n -> b absorbs a
		a_n + b_m = (a+b)_n    if n = m and a+b != 0 -> adds to finite value
		a_n + b_m = 1_n-1      if n = m and a+b  = 0 -> 1st order unit zero in Q_n (unity in Q_n-1)
		"""
		if self.ord > other.ord:   # a absorbs b
			return self.canonical(self.num, self.den, self.ord)
		if self.ord < other.ord:   # b absorbs a
			return self.canonical(other.num, other.den, other.ord)
		# order of summands the same - no absorption
		num = self.num * other.den + other.num * self.den
		den = self.den * other.den
		if num == 0:
			return self.canonical(1, den, self.ord-1)
		else:
			return self.canonical(num, den, self.ord)

	def __sub__(self, other):
		""" ..code is actually almost the same as in add - refactor """
		if self.ord > other.ord:
			return self.canonical(self.num, self.den, self.ord)
		if self.ord < other.ord:
			return self.canonical(other.num, other.den, other.ord)
		num = self.num * other.den - other.num * self.den
		den = self.den * other.den
		if num == 0:
			return self.canonical(1, den, self.ord-1)
		else:
			return self.canonical(num, den, self.ord)

	def __mul__(self, other):
		"""Multiplies two telescopic numbers. Numerator and denominator are 
		computed just like in regular rational number multiplication and the 
		orders of the two inputs are added."""
		return self.canonical(self.num*other.num, self.den*other.den, 
						self.ord+other.ord)

	def __truediv__(self, other):
		"""Divides two telescopic numbers. Numerator and denominator are 
		computed just like in regular rational number division and the 
		orders of the two inputs are subtracted."""
		return self.canonical(self.num*other.den, self.den*other.num, 
						self.ord-other.ord)
		
	def __eq__(self, other):
		"""Compares two telescopic numbers for equality."""
		return self.num == other.num and self.den == other.den \
				and self.ord == other.ord

	def isCanonical(self):
		"""Tests wether or not self is in canonical representation. It should 
		always be by the way how this class works, so the purpose of this 
		function is mainly for debugging."""
		g = gcd(self.num, self.den)
		return self.den > 0 and g == 1
		# todo: maybe make it a non-member function taking 3 arguments - that 
		# may make it useful for other purposes as well.

	def canonical(self, num, den, ord):
		"""Creates and returns a telescopic number in canonical 
		representation, i.e. numerator and denominator are reduced to 
		lowest terms and the sign is in the numerator"""
		g = gcd(num, den)
		num = num // g
		den = den // g
		if den < 0:
			den = -den
			num = -num
		return rsTelescopicNumber(num, den, ord)
		# make this a class function that returns a triple of numbers and can be
		# used in the constructor


def commutativeAdd(i, j, k, p, q, r):
	a = rsTelescopicNumber(i,j,k)
	b = rsTelescopicNumber(p,q,r)
	c = a + b
	d = b + a
	return c == d

def commutativeMul(i, j, k, p, q, r):
	a = rsTelescopicNumber(i,j,k)
	b = rsTelescopicNumber(p,q,r)
	c = a * b
	d = b * a
	return c == d

def associativeAdd(i, j, k, p, q, r, x, y, z):
	a = rsTelescopicNumber(i,j,k)
	b = rsTelescopicNumber(p,q,r)
	c = rsTelescopicNumber(x,y,z)
	# this fails for a = (1/1)_-1, b = (1/1)_0, c = (-1/1)_0
	e = a + (b + c)   # (2/1)_-1 ..hmm - but that actually should be (1/1)_-1, too -> bug in addition?
	d = (a + b) + c   # (1/1)_-1
	assert(e == d)
	return e == d


"""Tests the algebraic properties of the telescopic rational numbers 
experimentally. The purpose is not to replace formal proofs but to figure out
obvious errors by finding counter-expamples. If some are found, trying to proof
a property would be wasted time. But if none are found, attempting proof may 
make sense. It returns true if all the tested algebraic properties could be 
verified experimentall, false otherwise"""
def testTelNum():
	good = True   # result of the test


	# a few preliminary manual tests:

	# test a + (b + c) vs (a + b) + c for
	# a = (1/1)_-1, b = (1/1)_0, c = (-1/1)_0
	a = rsTelescopicNumber( 1,1,-1)
	b = rsTelescopicNumber( 1,1, 0)
	c = rsTelescopicNumber(-1,1, 0)

	t = b + c			# should be 1_-1, ok
	d = a + t			# 

	d = a + (b + c)		# (2/1)_-1
	e = (a + b) + c		# (1/1)_-1
	# addition is NOT associative! when we first have an absorption of a 
	# -1 order term by a 0-order term and add the result to another to a 0-order 
	# term, we get a different result from doing the cancellation of the two 
	# 0-order terms first (giving a -1-order unit zero) and add the result to 
	# the other -1-order number - it matters if absorption or cancellation
	# happens first - can this be fixed? i don't think so! we could introduce
	# precedence rules like always doing cancellations first (add higher order
	# terms first) - but that that doesn't really affect associativity
	# ...hmm - can this struture be interesting anyway, even without 
	# associative addition? ...it is still conditionally associative, though..
	# ..or can we fix the absorption rules to restore associativity?
	# maybe the lower order terms should not be absorbed but rather carried
	# along such that they may come back to life (lifted up in order) later
	# when divided by zero? ..that would require a more complex datastructure
	# (maybe a list) to carry around terms of all orders - maybe only 
	# unit-zeros can be absorbed (since it is a unit zero that is created in
	# a cancellation) 3_0 - 3_0 creates a unit zero 1_-1 - that could be 
	# absorbed but not a 2_-1? a 2_-1 would have to be carried along


	#a = rsTelescopicNumber(-1,-1,-1) # to check canonical init in constructor


	a = rsTelescopicNumber(2,3,1)
	b = rsTelescopicNumber(5,7,0)
	one   = rsTelescopicNumber(1,1,0)   # first order unity
	zero  = rsTelescopicNumber(0,1,0)   # first order zero
	three = rsTelescopicNumber(3,1,0)
	c = a * b
	d = c / b

	ai = one / a
	y  = a * ai
	good = good and y == one

	x = three / zero;
	y = zero * x; 
	good = good and y == three  # yes, computes back to 3 again

	x = zero / zero;
	y = zero * x
	good = good and y == zero



	# loop limits
	min = -1
	max = +1
	for i in range(min, max):
		for j in range(min, max):
			for k in range(min, max):
				for p in range(min, max):
					for q in range(min, max):
						for r in range(min, max):
							good = good and commutativeAdd(i, j, k, p, q, r)
							good = good and commutativeMul(i, j, k, p, q, r)
							for x in range(min, max):
								for y in range(min, max):
									for z in range(min, max):
										good = good and associativeAdd(i, j, k, p, q, r, x, y,z)
										#good = good and associativeMul(i, j, k)
										# distributive laws (left/right, mul/div), 
										# inverses, neutral elements
	return good




if __name__ == "__main__":


	good = testTelNum()





	# todo: make random tests if associativity, distributivity and 
	# commutativity works
	# the multiplicative inverse of (a/b)_n should come out as (b/a)_-n - yep
	# ...how would we define exponentiation


	print("Experiments")

