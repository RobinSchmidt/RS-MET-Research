import rsPy as rs
import numpy as np

if __name__ == "__main__":
    hi = rs.hello("Germany");
    s  = rs.invite(hi)
    print(s)
    cnVal  = rs.ellipj_cn(2.1231, 0.9) # 2nd argument must be in [0,1]
    expVal = rs.exp(3.14j)             # almost -1
    print(cnVal, expVal)
    a = np.array([1.,2,3])             # 1 float makes the whole array float
    b = np.array([2.,3,4])
    c = a+b


    testArray = rs.npArrayCreate(3, 5.0)

    norm = rs.eucnorm(a)                # computes euclidean norm (3.7416..)
    rs.scale(a, 2)                      # scale a by factor 2

  
    dummy = 0                          # to allow a breakpoint here




            
"""

from ctypes import windll
import sys, os.path  

if __name__ == "__main__":
	scriptName = sys.argv[0]  
	pathName = os.path.dirname(scriptName)  
	dllFile = "rsPy.dll"  
	locatedAt = (pathName + "/" + dllFile).replace("\\","/") 
	dll = windll.LoadLibrary(locatedAt) # works, if the dll file is here
	inviteFunc = dll.invite             # function is not found
	inviteFunc()                        # call function from dll
	print(dll)
	# the dll is not found - maybe it looks in the wrong directory?

"""

# oh wait - it seems like this way of using a dll via ctypes is a different
# way - we have a python extension that is used in a different way