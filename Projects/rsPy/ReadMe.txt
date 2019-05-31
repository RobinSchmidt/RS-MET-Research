Project to create python bindings for the RS-MET signal processing libraries 
rapt and rosic.

To build it successfully, the rsPy juce module needs to include the header file
pyconfig.h which is supposed to reside somewhere in the python 
installation folder. The folder where this file resides needs to be added to
the "Header Search Paths" field in the exporter settings in the .jucer file.
On my machine, the path is:

C:\Users\Rob\Anaconda3\include

For numpy, the following files are included: 

<numpy/arrayobject.h>, <numpy/ufuncobject.h> which can be found on my machine in:

C:\Users\Rob\Anaconda3\Lib\site-packages\numpy\core\include

The paths may obviously be different on other people's machines. So if you get 
an include error for these files, change the path in the Projucer and 
re-generate the visual studio project. If you don't use Projucer, change the 
path directly in Visual Studio.

We also need to link to

C:\Users\Rob\Anaconda3\libs\python34.lib

so that path must be added to the linker paths (or do we? try without..)


To make it possible to import the .dll in python via a statement like:

import rsPy as rs

one way to do it is to rename the resulting built rsPy.dll into rsPy.pyd and 
copy it into an appropriate location in the python installation. On my 
machine, this location is:

C:\Users\Rob\Anaconda3\DLLs

I actually have a post-build command that creates a copy of the dll with 
appropriate pyd extension. If it's too inconvenient to copy the pyd file 
manually into the python folder, you may either modify this post-build step, 
such that the file will be copied into your desired target folder or add the
folder where the .pyd file ends up to the PYTHONPATH environment variable (which
you may have to create, if it doesn't exist already)

To test, if it works as intended, try the python VS project in the "Test" 
subfolder. If the compiled .dll is copied and renamed into the right place, it
should be possible to run the script "rsPyTest.py" from the "Code" folder.





todo: maybe at some stage, distribute it as a proper python module...maybe 
AudiPy or something - should work together with NumPy/SciPy
the name SigPy is already taken - maybe have a look at it - perhaps it's cool?