from __future__ import print_function, division
#from rsPolynomial import rsPolynomial
from rsRational import rsRationalFunction
from sympy import Symbol


class rsSymbolicRationalFunction(rsRationalFunction):

	def __init__(self, numeratorCoeffNames, numeratorDegree,\
			denominatorCoeffNames, denominatorDegree, varName = "x"):

		self.varName = varName # preliminary
		x = Symbol(varName) # seems liek we don't actually need this - maybe get rid

		# create list of numerator coeffs:
		a = []
		for n in range(0, numeratorDegree+1):
			c = Symbol(numeratorCoeffNames + str(n))
			a.append(c)

		# create list of denominator coeffs:
		b = []
		for n in range(0, denominatorDegree+1):
			c = Symbol(denominatorCoeffNames + str(n))
			b.append(c)

		rsRationalFunction.__init__(self, a, b, False)



def testSymbolicRationalFunction():

	result = True

	r1 = rsSymbolicRationalFunction("a", 3, "b", 4)
	r2 = rsSymbolicRationalFunction("c", 2, "d", 2)

	#r3 = r1.composeWith(r2) # takes very long
	#print(r3)
	#r3 = r1+r2
	#print(r3)
	#r3 = r1-r2
	#print(r3)
	#r3 = r1*r2
	#print(r3)
	#r3 = r1/r2
	#print(r3)

	# it's very slow - maybe it can be improved by bringing the intermediate 
	# results into a canoncial form after each step?

	return result

if __name__ == "__main__":
	testSymbolicRationalFunction()