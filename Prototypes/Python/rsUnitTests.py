from __future__ import print_function
from rsPolynomial import testRsPolynomial
from rsRational import testRsRationalFunction
from rsSympyStuff import testSymbolicRationalFunction

if __name__ == "__main__":

	r = True  # test result

	r = r and testRsPolynomial()
	r = r and testRsRationalFunction()
	r = r and testSymbolicRationalFunction()

	if r == True:
		print("All unit tests have passed")
	else:
		print("WARNING!!! At least one unit test has FAILED!!!")
	# maybe wrap this code into a function printTestResult(testName, hasPassed)

