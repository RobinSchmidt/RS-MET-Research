from __future__ import print_function

def rsCheck(condition, errorMessage):
	if not condition:
		print(errorMessage)
	return condition
# maybe rename to rsTest

def printTestResult(hasPassed, testName):
	if(hasPassed):
		print(testName, "has passed all tests")
	else:
		print("WARNING!!! ", testName, " has FAILED at least one test!!!")

def scaleList(lst, scaler):
	assert type(lst) is list
	for k in range(0, len(lst)):
		lst[k] *= scaler
# rename to rsScaleList


