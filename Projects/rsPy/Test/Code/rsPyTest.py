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

    arrayApi = rs.arrayAPI()
    rs.initArrayAPI()
    arrayApi = rs.arrayAPI()

    #testarray = rs.npArrayCreate(3, 5.0)

    #rs.mul(a, 2)                      # should multiply a by 2 - crashes!
    norm = rs.eucnorm(a)               # should compute euclidean norm - crashes
    #test = rs.npArrayTest(a);
    # the crash is with a message: The debug adapter exited unexpectedly
    # https://github.com/Microsoft/PTVS/issues/3812
    # when switching the conig to release, the error is:
    # Stream does not support reading
    # i also once got: cannot acces disposed object
    # ...wtf? ...maybe numpy is not yet properly initialized? ...maybe try a debug build and
    # set breakpoints in the c++ source ...i don't know how to hit the breakpoints..however, when
    # not calling numpy::initialize(); we get the same error, so it seems plausible that somehow
    # the call to initialize() has no actual effect?
    #https://stackoverflow.com/questions/49522024/python-extension-debug-adapter-process-has-terminated-unexpectedly
    #https://devblogs.microsoft.com/visualstudio/adding-support-for-debug-adapters-to-visual-studio-ide/
    
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